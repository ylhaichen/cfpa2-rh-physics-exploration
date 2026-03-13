#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User-tunable defaults
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"
RUN_TAG="${RUN_TAG:-physics_pipeline_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$HOME/physics_runs/${RUN_TAG}}"
VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"

BASE_CONFIG="${BASE_CONFIG:-configs/base.yaml}"
PLANNER_CONFIG="${PLANNER_CONFIG:-configs/planner_rh_cfpa2.yaml}"
PLANNER_NAME="${PLANNER_NAME:-rh_cfpa2}"
PREDICTOR_TYPE="${PREDICTOR_TYPE:-path_follow}"
ENV_CFGS_CSV="${ENV_CFGS_CSV:-configs/env_maze.yaml,configs/env_go2w_like.yaml,configs/env_narrow_t_branches.yaml}"

SEED_START="${SEED_START:-0}"
NUM_SEEDS="${NUM_SEEDS:-4000}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-1}"
MAX_STEPS="${MAX_STEPS:-400}"
SHARD_SIZE="${SHARD_SIZE:-200000}"
HARD_OVERSAMPLE="${HARD_OVERSAMPLE:-0.70}"
HARD_MAP_TYPES_CSV="${HARD_MAP_TYPES_CSV:-sharp_turn_corridor,narrow_t_branches,bottleneck_rooms,interaction_cross,branching_deadend}"

NUM_TASKS="${NUM_TASKS:-64}"

COLLECT_CORES="${COLLECT_CORES:-1}"
COLLECT_MEM="${COLLECT_MEM:-3G}"
COLLECT_TMPFS="${COLLECT_TMPFS:-8G}"
COLLECT_WALLTIME="${COLLECT_WALLTIME:-24:00:00}"

MERGE_CORES="${MERGE_CORES:-1}"
MERGE_MEM="${MERGE_MEM:-2G}"
MERGE_TMPFS="${MERGE_TMPFS:-4G}"
MERGE_WALLTIME="${MERGE_WALLTIME:-02:00:00}"

TRAIN_USE_GPU="${TRAIN_USE_GPU:-1}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_ALLOW_TAG="${GPU_ALLOW_TAG:-}"   # e.g. L for A100-40G, EF for V100

TRAIN_CORES="${TRAIN_CORES:-8}"
TRAIN_MEM="${TRAIN_MEM:-8G}"
TRAIN_TMPFS="${TRAIN_TMPFS:-30G}"
TRAIN_WALLTIME="${TRAIN_WALLTIME:-24:00:00}"

TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}" # cuda / auto / cpu
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8192}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-5}"
TRAIN_HIDDEN_DIMS="${TRAIN_HIDDEN_DIMS:-256,256}"
TRAIN_VAL_RATIO="${TRAIN_VAL_RATIO:-0.1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_MAX_SHARDS="${TRAIN_MAX_SHARDS:-}"  # optional, empty means full dataset

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/jobs" "${RUN_ROOT}/dataset" "${RUN_ROOT}/models"

COLLECT_SCRIPT="${REPO_DIR}/jobs/myriad_collect_array.sh"
MERGE_SCRIPT="${REPO_DIR}/jobs/myriad_merge_manifest.sh"
TRAIN_SCRIPT="${REPO_DIR}/jobs/myriad_train_eval_gpu.sh"

for f in "${COLLECT_SCRIPT}" "${MERGE_SCRIPT}" "${TRAIN_SCRIPT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "missing script: ${f}" >&2
    exit 2
  fi
done

chmod +x "${COLLECT_SCRIPT}" "${MERGE_SCRIPT}" "${TRAIN_SCRIPT}"

build_vlist() {
  local out=""
  local kv=""
  for kv in "$@"; do
    if [[ -z "${out}" ]]; then
      out="${kv}"
    else
      out="${out},${kv}"
    fi
  done
  printf "%s" "${out}"
}

COMMON_VARS=(
  "REPO_DIR=${REPO_DIR}"
  "RUN_ROOT=${RUN_ROOT}"
  "VENV_PATH=${VENV_PATH}"
)

COLLECT_VARS=(
  "${COMMON_VARS[@]}"
  "BASE_CONFIG=${BASE_CONFIG}"
  "PLANNER_CONFIG=${PLANNER_CONFIG}"
  "PLANNER_NAME=${PLANNER_NAME}"
  "PREDICTOR_TYPE=${PREDICTOR_TYPE}"
  "ENV_CFGS_CSV=${ENV_CFGS_CSV}"
  "SEED_START=${SEED_START}"
  "NUM_SEEDS=${NUM_SEEDS}"
  "EPISODES_PER_SEED=${EPISODES_PER_SEED}"
  "MAX_STEPS=${MAX_STEPS}"
  "SHARD_SIZE=${SHARD_SIZE}"
  "HARD_OVERSAMPLE=${HARD_OVERSAMPLE}"
  "HARD_MAP_TYPES_CSV=${HARD_MAP_TYPES_CSV}"
  "NUM_TASKS=${NUM_TASKS}"
)
COLLECT_VLIST="$(build_vlist "${COLLECT_VARS[@]}")"

MERGE_VLIST="$(build_vlist "${COMMON_VARS[@]}")"

TRAIN_VARS=(
  "${COMMON_VARS[@]}"
  "TRAIN_DEVICE=${TRAIN_DEVICE}"
  "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
  "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
  "TRAIN_LR=${TRAIN_LR}"
  "TRAIN_WEIGHT_DECAY=${TRAIN_WEIGHT_DECAY}"
  "TRAIN_HIDDEN_DIMS=${TRAIN_HIDDEN_DIMS}"
  "TRAIN_VAL_RATIO=${TRAIN_VAL_RATIO}"
  "TRAIN_SEED=${TRAIN_SEED}"
  "TRAIN_MAX_SHARDS=${TRAIN_MAX_SHARDS}"
)
TRAIN_VLIST="$(build_vlist "${TRAIN_VARS[@]}")"

echo "=== Submit Physics Residual Pipeline ==="
echo "repo: ${REPO_DIR}"
echo "run_root: ${RUN_ROOT}"
echo "num_tasks: ${NUM_TASKS}"
echo "env_cfgs_csv: ${ENV_CFGS_CSV}"

COLLECT_JOB_ID="$(qsub -terse \
  -cwd \
  -t "1-${NUM_TASKS}" \
  -pe smp "${COLLECT_CORES}" \
  -l "h_rt=${COLLECT_WALLTIME},mem=${COLLECT_MEM},tmpfs=${COLLECT_TMPFS}" \
  -o "${RUN_ROOT}/logs" \
  -e "${RUN_ROOT}/logs" \
  -v "${COLLECT_VLIST}" \
  "${COLLECT_SCRIPT}")"

MERGE_JOB_ID="$(qsub -terse \
  -cwd \
  -hold_jid "${COLLECT_JOB_ID}" \
  -pe smp "${MERGE_CORES}" \
  -l "h_rt=${MERGE_WALLTIME},mem=${MERGE_MEM},tmpfs=${MERGE_TMPFS}" \
  -o "${RUN_ROOT}/logs" \
  -e "${RUN_ROOT}/logs" \
  -v "${MERGE_VLIST}" \
  "${MERGE_SCRIPT}")"

TRAIN_CMD=(
  qsub -terse
  -cwd
  -hold_jid "${MERGE_JOB_ID}"
  -pe smp "${TRAIN_CORES}"
  -l "h_rt=${TRAIN_WALLTIME},mem=${TRAIN_MEM},tmpfs=${TRAIN_TMPFS}"
  -o "${RUN_ROOT}/logs"
  -e "${RUN_ROOT}/logs"
  -v "${TRAIN_VLIST}"
)

if [[ "${TRAIN_USE_GPU}" == "1" ]]; then
  TRAIN_CMD+=(-l "gpu=${GPU_COUNT}")
  if [[ -n "${GPU_ALLOW_TAG}" ]]; then
    TRAIN_CMD+=(-ac "allow=${GPU_ALLOW_TAG}")
  fi
fi

TRAIN_CMD+=("${TRAIN_SCRIPT}")
TRAIN_JOB_ID="$("${TRAIN_CMD[@]}")"

cat <<EOF
collect_job_id=${COLLECT_JOB_ID}
merge_job_id=${MERGE_JOB_ID}
train_job_id=${TRAIN_JOB_ID}
logs_dir=${RUN_ROOT}/logs
dataset_manifest=${RUN_ROOT}/dataset/manifest_merged.jsonl
checkpoint=${RUN_ROOT}/models/physics_residual_mlp.pt
monitor: qstat -u \$USER
EOF
