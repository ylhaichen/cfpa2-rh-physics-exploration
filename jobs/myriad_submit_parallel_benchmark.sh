#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cfpa2rh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
PHYSICS_WEIGHT_FILE="${PHYSICS_WEIGHT_FILE:-$HOME/physics_runs/full_20260312_172122/models/physics_residual_mlp.pt}"

BENCHMARK_TAG="${BENCHMARK_TAG:-parallel_bench_$(date +%Y%m%d_%H%M%S)}"
COMPARE_RUN_ID="${COMPARE_RUN_ID:-${BENCHMARK_TAG}_compare}"
PREDICTOR_RUN_ID="${PREDICTOR_RUN_ID:-${BENCHMARK_TAG}_predictors}"

NUM_TASKS="${NUM_TASKS:-36}"
SEED_START="${SEED_START:-0}"
NUM_SEEDS="${NUM_SEEDS:-10}"
MAX_STEPS="${MAX_STEPS:-5000}"

RUN_COMPARE="${RUN_COMPARE:-1}"
RUN_PREDICTORS="${RUN_PREDICTORS:-1}"

COMPARE_PLANNERS_CSV="${COMPARE_PLANNERS_CSV:-cfpa2;rh_cfpa2;physics_rh_cfpa2}"
COMPARE_ENV_CONFIGS_CSV="${COMPARE_ENV_CONFIGS_CSV:-configs/env_narrow_t_branches.yaml;configs/env_narrow_t_dense_branches.yaml;configs/env_narrow_t_asymmetric_branches.yaml;configs/env_narrow_t_loop_branches.yaml}"
PREDICTOR_ENV_CONFIG="${PREDICTOR_ENV_CONFIG:-configs/env_narrow_t_branches.yaml}"
PREDICTOR_PLANNERS_CSV="${PREDICTOR_PLANNERS_CSV:-rh_cfpa2;physics_rh_cfpa2}"
PREDICTOR_TYPES_CSV="${PREDICTOR_TYPES_CSV:-path_follow;physics_residual}"
ROLLOUT_HORIZONS_CSV="${ROLLOUT_HORIZONS_CSV:-3;5;7}"

TASK_CORES="${TASK_CORES:-1}"
TASK_MEM="${TASK_MEM:-6G}"
TASK_TMPFS="${TASK_TMPFS:-20G}"
TASK_WALLTIME="${TASK_WALLTIME:-48:00:00}"

REQUEST_GPU="${REQUEST_GPU:-0}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_ALLOW_TAG="${GPU_ALLOW_TAG:-L}"

MERGE_CORES="${MERGE_CORES:-1}"
MERGE_MEM="${MERGE_MEM:-4G}"
MERGE_TMPFS="${MERGE_TMPFS:-8G}"
MERGE_WALLTIME="${MERGE_WALLTIME:-06:00:00}"

LOG_ROOT="${LOG_ROOT:-$HOME/physics_runs/benchmarks_logs/${BENCHMARK_TAG}}"
mkdir -p "${LOG_ROOT}"

TASK_SCRIPT="${REPO_DIR}/jobs/myriad_benchmark_array_task.sh"
MERGE_SCRIPT="${REPO_DIR}/jobs/myriad_merge_benchmark_results.sh"
chmod +x "${TASK_SCRIPT}" "${MERGE_SCRIPT}"

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

VLIST="$(build_vlist \
  "REPO_DIR=${REPO_DIR}" \
  "CONDA_SH=${CONDA_SH}" \
  "CONDA_ENV=${CONDA_ENV}" \
  "OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "PHYSICS_WEIGHT_FILE=${PHYSICS_WEIGHT_FILE}" \
  "COMPARE_RUN_ID=${COMPARE_RUN_ID}" \
  "PREDICTOR_RUN_ID=${PREDICTOR_RUN_ID}" \
  "NUM_TASKS=${NUM_TASKS}" \
  "SEED_START=${SEED_START}" \
  "NUM_SEEDS=${NUM_SEEDS}" \
  "MAX_STEPS=${MAX_STEPS}" \
  "RUN_COMPARE=${RUN_COMPARE}" \
  "RUN_PREDICTORS=${RUN_PREDICTORS}" \
  "COMPARE_PLANNERS_CSV=${COMPARE_PLANNERS_CSV}" \
  "COMPARE_ENV_CONFIGS_CSV=${COMPARE_ENV_CONFIGS_CSV}" \
  "PREDICTOR_ENV_CONFIG=${PREDICTOR_ENV_CONFIG}" \
  "PREDICTOR_PLANNERS_CSV=${PREDICTOR_PLANNERS_CSV}" \
  "PREDICTOR_TYPES_CSV=${PREDICTOR_TYPES_CSV}" \
  "ROLLOUT_HORIZONS_CSV=${ROLLOUT_HORIZONS_CSV}")"

ARRAY_CMD=(
  qsub -terse
  -N "pr_bench36"
  -cwd
  -t "1-${NUM_TASKS}"
  -pe smp "${TASK_CORES}"
  -l "h_rt=${TASK_WALLTIME},mem=${TASK_MEM},tmpfs=${TASK_TMPFS}"
  -o "${LOG_ROOT}"
  -e "${LOG_ROOT}"
  -v "${VLIST}"
)

if [[ "${REQUEST_GPU}" == "1" ]]; then
  ARRAY_CMD+=(-l "gpu=${GPU_COUNT}")
  if [[ -n "${GPU_ALLOW_TAG}" ]]; then
    ARRAY_CMD+=(-ac "allow=${GPU_ALLOW_TAG}")
  fi
fi

ARRAY_CMD+=("${TASK_SCRIPT}")
ARRAY_RAW="$("${ARRAY_CMD[@]}")"
ARRAY_ID="${ARRAY_RAW%%.*}"

MERGE_RAW="$(qsub -terse \
  -N "pr_bmerge" \
  -cwd \
  -hold_jid "${ARRAY_ID}" \
  -pe smp "${MERGE_CORES}" \
  -l "h_rt=${MERGE_WALLTIME},mem=${MERGE_MEM},tmpfs=${MERGE_TMPFS}" \
  -o "${LOG_ROOT}" \
  -e "${LOG_ROOT}" \
  -v "${VLIST}" \
  "${MERGE_SCRIPT}")"

cat <<EOF
array_job_id=${ARRAY_RAW}
merge_job_id=${MERGE_RAW}
log_root=${LOG_ROOT}
compare_run_id=${COMPARE_RUN_ID}
predictor_run_id=${PREDICTOR_RUN_ID}
monitor=qstat -u \$USER
compare_outputs=${REPO_DIR}/${OUTPUT_ROOT}/benchmarks/${COMPARE_RUN_ID}
predictor_outputs=${REPO_DIR}/${OUTPUT_ROOT}/benchmarks/${PREDICTOR_RUN_ID}
EOF
