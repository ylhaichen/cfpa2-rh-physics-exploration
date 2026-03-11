from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for this evaluation script. Please install torch in your environment.") from exc


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [input_dim] + [int(h) for h in hidden_dims] + [2]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Physics residual predictor checkpoint")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def _choose_device(flag: str) -> torch.device:
    if flag == "cpu":
        return torch.device("cpu")
    if flag == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_manifest(path: str, max_shards: int | None = None) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line)["shard_path"])
    if max_shards is not None:
        out = out[: int(max_shards)]
    return out


def main() -> None:
    args = parse_args()

    paths = _read_manifest(args.manifest, max_shards=args.max_shards)
    if not paths:
        raise RuntimeError("No shards available for evaluation")

    device = _choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    input_dim = int(ckpt["input_dim"])
    hidden_dims = [int(v) for v in ckpt["hidden_dims"]]
    feature_mean = torch.from_numpy(np.asarray(ckpt["feature_mean"], dtype=np.float32)).to(device)
    feature_std = torch.from_numpy(np.asarray(ckpt["feature_std"], dtype=np.float32)).to(device)

    model = ResidualMLP(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    n_total = 0

    with torch.no_grad():
        for p in paths:
            data = np.load(p)
            x = torch.from_numpy(data["X"].astype(np.float32)).to(device)
            y = torch.from_numpy(data["y"].astype(np.float32)).to(device)

            x = (x - feature_mean) / feature_std
            pred = model(x)

            err = pred - y
            mse_sum += float((err * err).sum().item())
            mae_sum += float(err.abs().sum().item())
            n_total += int(np.prod(y.shape))

    mse = mse_sum / float(max(1, n_total))
    mae = mae_sum / float(max(1, n_total))

    out_payload = {
        "mse": mse,
        "mae": mae,
        "num_shards": len(paths),
        "num_values": n_total,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "device": str(device),
    }

    if args.output is None:
        out_path = Path(args.checkpoint).with_suffix(".eval.json")
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, sort_keys=True)

    print("=== Evaluation Done ===")
    print(json.dumps(out_payload, indent=2, sort_keys=True))
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
