from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "cos_heading",
    "sin_heading",
    "cos_goal_dir",
    "sin_goal_dir",
    "local_obstacle_density",
    "teammate_distance_norm",
]


TARGETS = ["target_dx", "target_dy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual predictor linear head (numpy LS)")
    parser.add_argument("--dataset", type=str, default="training/predictor_dataset.csv")
    parser.add_argument("--output", type=str, default="training/physics_residual_weights.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset)

    x = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGETS].to_numpy(dtype=float)

    # Add bias term.
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=float)], axis=1)
    theta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)

    w = theta[:-1, :].T
    b = theta[-1, :]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, w=w, b=b)
    print(f"weights_saved: {out}")


if __name__ == "__main__":
    main()
