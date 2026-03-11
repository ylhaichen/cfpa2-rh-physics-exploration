from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge distributed dataset manifest shards")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["training/datasets/physics_residual_dataset/task*/manifest.jsonl"],
        help="Input glob patterns for per-task manifest files",
    )
    parser.add_argument("--output", type=str, default="training/datasets/physics_residual_dataset/manifest_merged.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths: list[str] = []
    for p in args.inputs:
        paths.extend(sorted(glob.glob(p)))

    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    total_samples = int(sum(int(r.get("num_samples", 0)) for r in rows))

    print("=== Manifest Merge Done ===")
    print(f"input_manifests: {len(paths)}")
    print(f"shards: {len(rows)}")
    print(f"total_samples: {total_samples}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
