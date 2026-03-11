# Unified Multi-Robot Exploration Framework (CFPA2 / RH-CFPA2 / Physics-RH-CFPA2)

This repository now includes a unified planner framework for high-level multi-robot exploration under one simulator, one metric protocol, and one animation pipeline.

## Key Points

- Shared simulator + observation + frontier + path + metrics + animation pipeline
- Three planners under one interface:
  - `cfpa2`
  - `rh_cfpa2`
  - `physics_rh_cfpa2`
- Predictor abstraction with pluggable backends:
  - `constant_velocity`
  - `path_follow`
  - `physics_residual`
- Go2W-like planner-level realism approximation:
  - footprint/clearance-aware traversability
  - finite FOV + LOS + occlusion
  - heading/turning constraints
  - slowdown near obstacles
  - motion uncertainty and conflict proxies

## Directory Layout

```text
configs/
core/
planners/
predictors/
simulators/grid_sim/
training/
experiments/
tests/
```

## Quick Start

Single run with animation:

```bash
PYTHONPATH=. python experiments/run_single_experiment.py \
  --planner cfpa2 \
  --env-config configs/env_maze.yaml \
  --seed 0
```

Planner comparison across maps:

```bash
PYTHONPATH=. python experiments/compare_planners_across_maps.py \
  --seed-start 0 --num-seeds 2 --animate-first-seed-only
```

Predictor and rollout ablations:

```bash
PYTHONPATH=. python experiments/compare_predictors.py \
  --seed-start 0 --num-seeds 2 --disable-animation
```

Plot metrics from raw CSV:

```bash
PYTHONPATH=. python experiments/plot_metrics.py \
  --input outputs/benchmarks/<run_id>/results_csv/compare_planners_results.csv
```

Export one animation directly:

```bash
PYTHONPATH=. python experiments/export_animation.py \
  --planner rh_cfpa2 --env-config configs/env_go2w_like.yaml --seed 0
```

Export side-by-side comparison animation:

```bash
PYTHONPATH=. python experiments/export_side_by_side_animation.py \
  --env-config configs/env_maze.yaml --seed 0 --max-steps 240
```

Large-scale residual predictor pipeline (cloud-friendly task split):

```bash
PYTHONPATH=. python training/collect_physics_residual_dataset.py \
  --env-configs configs/env_maze.yaml configs/env_go2w_like.yaml \
  --seed-start 0 --num-seeds 2000 \
  --task-index 0 --num-tasks 8 \
  --output-dir training/datasets/physics_residual_dataset

PYTHONPATH=. python training/merge_dataset_manifests.py \
  --inputs "training/datasets/physics_residual_dataset/task*/manifest.jsonl" \
  --output training/datasets/physics_residual_dataset/manifest_merged.jsonl

PYTHONPATH=. python training/train_physics_residual_torch.py \
  --manifest training/datasets/physics_residual_dataset/manifest_merged.jsonl \
  --output training/models/physics_residual_mlp.pt

PYTHONPATH=. python training/evaluate_physics_residual_torch.py \
  --manifest training/datasets/physics_residual_dataset/manifest_merged.jsonl \
  --checkpoint training/models/physics_residual_mlp.pt
```

## Tests

```bash
PYTHONPATH=. pytest -q tests
```

## Notes

- This is still a planner-level approximation (not full rigid-body physics simulation).
- Adapter interfaces are provided in `core/adapters.py` for future Gazebo/Unitree integration.
