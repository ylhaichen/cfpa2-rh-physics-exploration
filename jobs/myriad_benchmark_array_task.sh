#!/bin/bash -l
set -euo pipefail

: "${REPO_DIR:?REPO_DIR is required}"
: "${CONDA_SH:?CONDA_SH is required}"
: "${CONDA_ENV:?CONDA_ENV is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${COMPARE_RUN_ID:?COMPARE_RUN_ID is required}"
: "${PREDICTOR_RUN_ID:?PREDICTOR_RUN_ID is required}"
: "${NUM_TASKS:?NUM_TASKS is required}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${JOB_ID:-na}_${SGE_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"

TASK_INDEX=0
if [[ -n "${SGE_TASK_ID:-}" ]]; then
  TASK_INDEX=$((SGE_TASK_ID - 1))
fi

IFS=';' read -r -a COMPARE_ENV_CONFIGS <<< "${COMPARE_ENV_CONFIGS_CSV:-configs/env_narrow_t_branches.yaml;configs/env_narrow_t_dense_branches.yaml;configs/env_narrow_t_asymmetric_branches.yaml;configs/env_narrow_t_loop_branches.yaml}"
IFS=';' read -r -a COMPARE_PLANNERS <<< "${COMPARE_PLANNERS_CSV:-cfpa2;rh_cfpa2;physics_rh_cfpa2}"
IFS=';' read -r -a PREDICTOR_PLANNERS <<< "${PREDICTOR_PLANNERS_CSV:-rh_cfpa2;physics_rh_cfpa2}"
IFS=';' read -r -a PREDICTOR_TYPES <<< "${PREDICTOR_TYPES_CSV:-path_follow;physics_residual}"
IFS=';' read -r -a ROLLOUT_HORIZONS <<< "${ROLLOUT_HORIZONS_CSV:-3;5;7}"

if [[ "${RUN_COMPARE:-1}" == "1" ]]; then
  python experiments/parallel_compare_planners_shard.py \
    --base-config "${BASE_CONFIG:-configs/base.yaml}" \
    --planners "${COMPARE_PLANNERS[@]}" \
    --env-configs "${COMPARE_ENV_CONFIGS[@]}" \
    --seed-start "${SEED_START:-0}" \
    --num-seeds "${NUM_SEEDS:-10}" \
    --max-steps "${MAX_STEPS:-5000}" \
    --run-id "${COMPARE_RUN_ID}" \
    --output-root "${OUTPUT_ROOT}" \
    --disable-animation \
    --physics-weight-file "${PHYSICS_WEIGHT_FILE:-}" \
    --task-index "${TASK_INDEX}" \
    --num-tasks "${NUM_TASKS}"
fi

if [[ "${RUN_PREDICTORS:-1}" == "1" ]]; then
  python experiments/parallel_compare_predictors_shard.py \
    --base-config "${BASE_CONFIG:-configs/base.yaml}" \
    --env-config "${PREDICTOR_ENV_CONFIG:-configs/env_narrow_t_branches.yaml}" \
    --planners "${PREDICTOR_PLANNERS[@]}" \
    --predictors "${PREDICTOR_TYPES[@]}" \
    --rollout-horizons "${ROLLOUT_HORIZONS[@]}" \
    --seed-start "${SEED_START:-0}" \
    --num-seeds "${NUM_SEEDS:-10}" \
    --max-steps "${MAX_STEPS:-5000}" \
    --run-id "${PREDICTOR_RUN_ID}" \
    --output-root "${OUTPUT_ROOT}" \
    --disable-animation \
    --physics-weight-file "${PHYSICS_WEIGHT_FILE:-}" \
    --task-index "${TASK_INDEX}" \
    --num-tasks "${NUM_TASKS}"
fi
