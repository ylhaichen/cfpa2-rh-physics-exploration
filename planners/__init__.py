from __future__ import annotations

from .base_planner import BasePlanner
from .cfpa2_planner import CFPA2Planner
from .mui_tare_2d_planner import MUITARE2DPlanner
from .physics_rh_cfpa2_planner import PhysicsRHCFPA2Planner
from .rh_cfpa2_planner import RHCFPA2Planner


def build_planner(cfg: dict) -> BasePlanner:
    planner_name = str(cfg.get("planning", {}).get("planner_name", "cfpa2"))
    if planner_name == "cfpa2":
        return CFPA2Planner()
    if planner_name == "rh_cfpa2":
        return RHCFPA2Planner(cfg)
    if planner_name == "physics_rh_cfpa2":
        return PhysicsRHCFPA2Planner(cfg)
    if planner_name == "mui_tare_2d":
        return MUITARE2DPlanner(cfg)
    raise ValueError(f"Unsupported planner_name: {planner_name}")


__all__ = [
    "BasePlanner",
    "CFPA2Planner",
    "RHCFPA2Planner",
    "PhysicsRHCFPA2Planner",
    "MUITARE2DPlanner",
    "build_planner",
]
