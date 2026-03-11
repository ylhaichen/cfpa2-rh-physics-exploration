from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect synthetic predictor training tuples from step logs")
    parser.add_argument("--step-log-csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="training/predictor_dataset.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.step_log_csv)
    required = {"step", "robot_poses", "robot_headings_deg", "robot_goals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"step log csv missing required columns: {sorted(missing)}")

    rows = []
    for i in range(len(df) - 1):
        cur = df.iloc[i]
        nxt = df.iloc[i + 1]

        cur_poses = ast.literal_eval(str(cur["robot_poses"]))
        nxt_poses = ast.literal_eval(str(nxt["robot_poses"]))
        cur_headings = ast.literal_eval(str(cur["robot_headings_deg"]))
        cur_goals = ast.literal_eval(str(cur["robot_goals"]))

        for rid, pose in cur_poses.items():
            rid = int(rid)
            if rid not in nxt_poses:
                continue
            x, y = pose
            nx, ny = nxt_poses[rid]
            heading = float(cur_headings.get(rid, 0.0))

            goal = cur_goals.get(rid)
            if goal is None:
                goal_dx = 0.0
                goal_dy = 0.0
            else:
                gx, gy = goal
                goal_dx = float(gx) - float(x)
                goal_dy = float(gy) - float(y)

            goal_norm = math.hypot(goal_dx, goal_dy)
            if goal_norm > 1e-6:
                cos_goal_dir = goal_dx / goal_norm
                sin_goal_dir = goal_dy / goal_norm
            else:
                cos_goal_dir = 0.0
                sin_goal_dir = 0.0

            teammate_dist = 10.0
            for rid2, pose2 in cur_poses.items():
                rid2 = int(rid2)
                if rid2 == rid:
                    continue
                teammate_dist = min(teammate_dist, math.hypot(float(pose2[0]) - float(x), float(pose2[1]) - float(y)))

            rows.append(
                {
                    "cos_heading": math.cos(math.radians(heading)),
                    "sin_heading": math.sin(math.radians(heading)),
                    "cos_goal_dir": cos_goal_dir,
                    "sin_goal_dir": sin_goal_dir,
                    "local_obstacle_density": 0.0,
                    "teammate_distance_norm": min(1.0, teammate_dist / 10.0),
                    "target_dx": float(nx) - float(x),
                    "target_dy": float(ny) - float(y),
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved_dataset: {out}")


if __name__ == "__main__":
    main()
