# CFPA-2 Demo

> Legacy compatibility module. This folder is kept as a reference/regression baseline.
> The default execution path for this repository is now the unified framework at the repo root (`main.py`, `experiments/run_compare.py`).

Centralized Complementary Frontier Pair Allocation (CFPA-2) demo for frontier-based exploration with one or two robots on a 2D occupancy grid.

## Overview

This project is a high-level algorithm validation platform for unknown-environment exploration with:

- Shared occupancy grid (`UNKNOWN=-1`, `FREE=0`, `OCCUPIED=1`)
- Frontier detection + clustering
- Centralized target assignment
- Grid A* planning
- Event-driven replanning
- Matplotlib visualization + optional GIF saving
- Reproducible experiments via random seeds

Assumptions:

- Perfect robot localization (no SLAM)
- Perfect centralized map fusion
- Focus on task allocation and exploration efficiency

## Method Background

The implementation is inspired by:

- **Yamauchi**: Frontier-based exploration (frontiers as free/unknown boundaries)
- **Burgard et al.**: Coordinated multi-robot utility/cost assignment with overlap reduction
- **Keidar & Kaminka (WFD/FFD motivation)**: Frequent frontier updates and stale-target handling

## Core Algorithm

Single-robot utility:

\[
U(r,f)=w_{ig}IG(f)-w_cC(r,f)-w_{sw}SwitchPenalty(r,f)
\]

Two-robot joint CFPA-2 score:

\[
J(f_i,f_j)=U(r_1,f_i)+U(r_2,f_j)-\lambda O(f_i,f_j)-\mu I(f_i,f_j)
\]

Overlap term (implemented):

\[
O(f_i,f_j)=\exp\left(-\frac{\|p_i-p_j\|^2}{2\sigma^2}\right)
\]

Path interference term is a v1 stub (`0.0`) with extension hooks.

## Project Structure

```
cfpa2_demo/
├── README.md
├── requirements.txt
├── main.py
├── config/
├── core/
├── maps/
├── viz/
├── experiments/
├── outputs/
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Single run:

```bash
python main.py --config config/default.yaml --mode dual_joint
```

Modes:

- `single`
- `dual_greedy`
- `dual_joint`

Disable live plot:

```bash
python main.py --mode dual_joint --no-viz
```

Save GIF animation:

```bash
python main.py --mode dual_joint --save-animation --no-viz
```

Save MP4 video:

```bash
python main.py --mode dual_joint --save-video
```

Run experiment comparison:

```bash
python experiments/run_compare.py
```

Summarize existing CSV results:

```bash
python experiments/summarize_results.py --input outputs/results_csv/compare_results.csv
```

## Config

Main config file: `config/default.yaml`.

Key sections:

- `environment`: map size/type/seed/density
- `robots`: robot count, starts, sensing
- `frontier`: clustering + utility weights
- `allocator`: CFPA-2 penalties + hysteresis + reservation TTL
- `replanning`: event triggers and periodic backup
- `termination`: coverage threshold + max steps
- `visualization`: plotting and animation controls

Map presets:

- `config/map_open.yaml`
- `config/map_rooms.yaml`
- `config/map_maze.yaml`

## Metrics Output

Per run:

- completion steps
- coverage curve (`coverage vs step`)
- replan count and reasons
- repeated coverage ratio
- idle steps
- conflict count (dual greedy baseline)
- target invalidation events

Saved outputs:

- `outputs/results_csv/*.csv`
- `outputs/figures/*.png`
- `outputs/animations/*.gif` (if enabled)

## Notes and Extensions

Current implementation is specialized for 1-2 robots with centralized planning, but key modules are isolated to support:

- N-robot allocator strategies (Hungarian/Auction)
- non-zero path interference models
- richer sensing (occlusion/raycast)
- additional coordination policies
