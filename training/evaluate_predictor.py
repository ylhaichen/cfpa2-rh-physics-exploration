from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="Evaluate residual predictor linear head")
    parser.add_argument("--dataset", type=str, default="training/predictor_dataset.csv")
    parser.add_argument("--weights", type=str, default="training/physics_residual_weights.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset)
    data = np.load(args.weights)
    w = np.asarray(data["w"], dtype=float)
    b = np.asarray(data["b"], dtype=float)

    x = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGETS].to_numpy(dtype=float)

    pred = (x @ w.T) + b
    mse = float(np.mean((pred - y) ** 2))
    print(f"mse: {mse:.8f}")


if __name__ == "__main__":
    main()
