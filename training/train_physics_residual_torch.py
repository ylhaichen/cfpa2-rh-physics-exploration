from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for this training script. Please install torch in your environment.") from exc


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
    parser = argparse.ArgumentParser(description="Train Physics residual predictor (torch MLP) from shard manifest")
    parser.add_argument("--manifest", type=str, required=True, help="Merged manifest jsonl")
    parser.add_argument("--output", type=str, default="training/models/physics_residual_mlp.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dims", type=str, default="256,256")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def _read_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stream_shard_paths(rows: list[dict], max_shards: int | None = None) -> list[str]:
    paths = [str(r["shard_path"]) for r in rows]
    if max_shards is not None:
        paths = paths[: int(max_shards)]
    return paths


def _compute_feature_stats(paths: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    total_n = 0
    mean = None
    m2 = None
    input_dim = -1

    for p in paths:
        data = np.load(p)
        x = data["X"].astype(np.float64)
        if input_dim < 0:
            input_dim = int(x.shape[1])
            mean = np.zeros((input_dim,), dtype=np.float64)
            m2 = np.zeros((input_dim,), dtype=np.float64)

        for row in x:
            total_n += 1
            delta = row - mean
            mean += delta / total_n
            delta2 = row - mean
            m2 += delta * delta2

    if total_n <= 1:
        std = np.ones((input_dim,), dtype=np.float32)
    else:
        var = m2 / float(max(1, total_n - 1))
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

    return mean.astype(np.float32), std, input_dim


def _choose_device(flag: str) -> torch.device:
    if flag == "cpu":
        return torch.device("cpu")
    if flag == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_one_epoch(
    model: nn.Module,
    optimizer,
    device: torch.device,
    paths: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int,
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    model.train()

    feature_mean_t = torch.from_numpy(feature_mean).to(device)
    feature_std_t = torch.from_numpy(feature_std).to(device)

    for p in paths:
        data = np.load(p)
        x = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

        n = x.shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)

        split = int(n * (1.0 - val_ratio))
        train_idx = idx[:split]
        val_idx = idx[split:]

        for start in range(0, len(train_idx), batch_size):
            bidx = train_idx[start : start + batch_size]
            if bidx.size == 0:
                continue

            xb = torch.from_numpy(x[bidx]).to(device)
            yb = torch.from_numpy(y[bidx]).to(device)
            xb = (xb - feature_mean_t) / feature_std_t

            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu()) * int(bidx.size)
            train_count += int(bidx.size)

        if len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                for start in range(0, len(val_idx), batch_size):
                    bidx = val_idx[start : start + batch_size]
                    xb = torch.from_numpy(x[bidx]).to(device)
                    yb = torch.from_numpy(y[bidx]).to(device)
                    xb = (xb - feature_mean_t) / feature_std_t
                    pred = model(xb)
                    loss = nn.functional.mse_loss(pred, yb)
                    val_loss_sum += float(loss.detach().cpu()) * int(bidx.size)
                    val_count += int(bidx.size)
            model.train()

    train_loss = train_loss_sum / float(max(1, train_count))
    val_loss = val_loss_sum / float(max(1, val_count))
    return train_loss, val_loss


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    manifest_rows = _read_manifest(args.manifest)
    shard_paths = _stream_shard_paths(manifest_rows, max_shards=args.max_shards)
    if not shard_paths:
        raise RuntimeError("No shards found in manifest.")

    feature_mean, feature_std, input_dim = _compute_feature_stats(shard_paths)

    hidden_dims = [int(v.strip()) for v in str(args.hidden_dims).split(",") if v.strip()]
    if not hidden_dims:
        hidden_dims = [256, 256]

    device = _choose_device(args.device)
    model = ResidualMLP(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    rng = np.random.default_rng(int(args.seed))

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        train_loss, val_loss = _train_one_epoch(
            model=model,
            optimizer=optimizer,
            device=device,
            paths=shard_paths,
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_size=int(args.batch_size),
            val_ratio=float(args.val_ratio),
            rng=rng,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"[epoch {epoch}] train_mse={train_loss:.6f} val_mse={val_loss:.6f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "input_dim": int(input_dim),
        "hidden_dims": hidden_dims,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "model_state": model.state_dict(),
        "history": history,
    }
    torch.save(ckpt, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_mse": best_val,
                "epochs": int(args.epochs),
                "num_shards": len(shard_paths),
                "input_dim": int(input_dim),
                "hidden_dims": hidden_dims,
                "device": str(device),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print("=== Training Done ===")
    print(f"checkpoint: {out_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
