from __future__ import annotations

from .base_predictor import BasePredictor
from .constant_velocity_predictor import ConstantVelocityPredictor
from .path_follow_predictor import PathFollowPredictor
from .physics_residual_predictor import PhysicsResidualPredictor


def build_predictor(cfg: dict) -> BasePredictor:
    pred_cfg = cfg.get("predictor", {})
    ptype = str(pred_cfg.get("type", "path_follow"))

    if ptype == "constant_velocity":
        cv_cfg = pred_cfg.get("constant_velocity", {})
        return ConstantVelocityPredictor(default_speed_cells_per_step=float(cv_cfg.get("default_speed_cells_per_step", 0.8)))

    if ptype == "physics_residual":
        phy_cfg = pred_cfg.get("physics_residual", {})
        hidden_dims_raw = phy_cfg.get("hidden_dims", [128, 128])
        hidden_dims = tuple(int(v) for v in hidden_dims_raw)
        return PhysicsResidualPredictor(
            enabled=bool(phy_cfg.get("enabled", True)),
            weight_file=phy_cfg.get("weight_file"),
            residual_scale=float(phy_cfg.get("residual_scale", 0.35)),
            occupancy_patch_radius=int(phy_cfg.get("occupancy_patch_radius", 4)),
            hidden_dims=hidden_dims,
        )

    return PathFollowPredictor()


__all__ = [
    "BasePredictor",
    "ConstantVelocityPredictor",
    "PathFollowPredictor",
    "PhysicsResidualPredictor",
    "build_predictor",
]
