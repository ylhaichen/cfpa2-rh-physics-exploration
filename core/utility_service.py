from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .map_manager import MapManager
from .path_service import astar_path, heading_delta_cost, path_cost
from .types import Cell, RobotState


@dataclass
class CandidateEvaluation:
    utility: float
    information_gain: float
    travel_cost: float
    switch_penalty: float
    turn_penalty: float
    path: list[Cell]


def information_gain(
    map_mgr: MapManager,
    frontier_cell: Cell,
    radius: int,
    known_grid: np.ndarray | None = None,
) -> float:
    return float(map_mgr.count_unknown_in_radius(frontier_cell, radius, grid=known_grid))


def switch_penalty(robot: RobotState, frontier: Cell) -> float:
    if robot.current_target is None:
        return 0.0
    return 0.0 if robot.current_target == frontier else 1.0


def evaluate_candidate(
    robot: RobotState,
    frontier: Cell,
    map_mgr: MapManager,
    cfg: dict,
    neighborhood: int = 8,
    known_grid: np.ndarray | None = None,
) -> CandidateEvaluation | None:
    weights = cfg["planning"]["weights"]
    clearance = int(cfg["robots"].get("clearance_cells", 0))

    ig = information_gain(map_mgr, frontier, radius=int(cfg["frontier"]["ig_radius"]), known_grid=known_grid)
    path = astar_path(map_mgr, robot.pose, frontier, neighborhood=neighborhood, clearance_cells=clearance)
    if path is None:
        return None

    travel = path_cost(path)
    sw = switch_penalty(robot, frontier)
    turn = heading_delta_cost(robot.heading_deg, path)

    score = (
        float(weights.get("w_ig", 1.0)) * ig
        - float(weights.get("w_cost", 1.0)) * travel
        - float(weights.get("w_switch", 0.0)) * sw
        - float(weights.get("w_turn", 0.0)) * turn
    )

    return CandidateEvaluation(
        utility=score,
        information_gain=ig,
        travel_cost=travel,
        switch_penalty=sw,
        turn_penalty=turn,
        path=path,
    )


def overlap_penalty(a: Cell, b: Cell, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    d2 = dx * dx + dy * dy
    return math.exp(-d2 / (2.0 * sigma * sigma))


def path_interference_penalty(path1: list[Cell], path2: list[Cell], distance_threshold: float = 2.5) -> float:
    if not path1 or not path2:
        return 0.0
    thr2 = distance_threshold * distance_threshold
    penalty = 0.0
    n = min(len(path1), len(path2))
    for i in range(n):
        p = path1[i]
        q = path2[i]
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        d2 = dx * dx + dy * dy
        if d2 <= thr2:
            penalty += (thr2 - d2) / max(thr2, 1e-6)
    return penalty / float(max(1, n))
