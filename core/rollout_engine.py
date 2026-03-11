from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from predictors.base_predictor import BasePredictor

from .predictor_features import local_context_for_predictor
from .types import Cell, FrontierCandidate, PredictorInput, RobotState


@dataclass
class RolloutResult:
    future_score: float
    predicted_paths: dict[int, list[Cell]]
    predictor_inference_times: dict[int, float]
    breakdown: dict[str, float]


def _clamp_cell(cell: tuple[float, float], width: int, height: int) -> Cell:
    x = int(round(cell[0]))
    y = int(round(cell[1]))
    x = min(width - 1, max(0, x))
    y = min(height - 1, max(0, y))
    return (x, y)


def _future_overlap_proxy(a: Cell, b: Cell, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    d2 = dx * dx + dy * dy
    return math.exp(-d2 / (2.0 * sigma * sigma))


def _best_virtual_frontier_gain(
    pose: Cell,
    candidates: list[FrontierCandidate],
    virtual_known: np.ndarray,
    map_mgr,
    reveal_radius: int,
    used_targets: set[Cell],
) -> float:
    best = 0.0
    for c in candidates:
        rep = c.representative
        if rep in used_targets:
            continue
        ig = map_mgr.count_unknown_in_radius(rep, reveal_radius, grid=virtual_known)
        dist = abs(rep[0] - pose[0]) + abs(rep[1] - pose[1])
        score = ig - 0.35 * dist
        if score > best:
            best = score
    return max(0.0, best)


def _best_virtual_frontier_target(
    pose: Cell,
    candidates: list[FrontierCandidate],
    virtual_known: np.ndarray,
    map_mgr,
    reveal_radius: int,
    used_targets: set[Cell],
) -> Cell | None:
    best_target: Cell | None = None
    best_score = float("-inf")
    for c in candidates:
        rep = c.representative
        if rep in used_targets:
            continue
        ig = float(map_mgr.count_unknown_in_radius(rep, reveal_radius, grid=virtual_known))
        dist = abs(rep[0] - pose[0]) + abs(rep[1] - pose[1])
        score = ig - 0.35 * dist
        if score > best_score:
            best_score = score
            best_target = rep
    return best_target


def _straight_line_cells(start: Cell, goal: Cell, max_steps: int) -> list[Cell]:
    if max_steps <= 0:
        return []
    sx, sy = start
    gx, gy = goal
    out: list[Cell] = []
    cur_x, cur_y = sx, sy
    for _ in range(max_steps):
        if cur_x == gx and cur_y == gy:
            out.append((cur_x, cur_y))
            continue
        step_x = 0 if gx == cur_x else (1 if gx > cur_x else -1)
        step_y = 0 if gy == cur_y else (1 if gy > cur_y else -1)
        cur_x += step_x
        cur_y += step_y
        out.append((cur_x, cur_y))
    return out


def _rewrite_future_path_on_reach(
    path_cells: list[Cell],
    initial_goal: Cell | None,
    candidates: list[FrontierCandidate],
    virtual_known: np.ndarray,
    map_mgr,
    reveal_radius: int,
    used_targets: set[Cell],
) -> tuple[list[Cell], int]:
    if not path_cells or initial_goal is None:
        return path_cells, 0

    rewritten = list(path_cells)
    current_goal = initial_goal
    reassign_count = 0

    for t, pose in enumerate(list(rewritten)):
        if pose != current_goal:
            continue
        remaining = len(rewritten) - (t + 1)
        if remaining <= 0:
            continue
        next_goal = _best_virtual_frontier_target(
            pose=pose,
            candidates=candidates,
            virtual_known=virtual_known,
            map_mgr=map_mgr,
            reveal_radius=reveal_radius,
            used_targets=used_targets,
        )
        if next_goal is None:
            continue
        used_targets.add(next_goal)
        suffix = _straight_line_cells(start=pose, goal=next_goal, max_steps=remaining)
        rewritten[t + 1 :] = suffix
        current_goal = next_goal
        reassign_count += 1

    return rewritten, reassign_count


def rollout_pair_score(
    map_mgr,
    cfg: dict,
    robot1: RobotState,
    robot2: RobotState,
    goal1: Cell | None,
    goal2: Cell | None,
    path1: list[Cell],
    path2: list[Cell],
    candidates: list[FrontierCandidate],
    predictor: BasePredictor,
) -> RolloutResult:
    rollout_cfg = cfg["planning"]["rollout"]
    horizon = int(rollout_cfg.get("horizon", 4))
    gamma = float(rollout_cfg.get("gamma", 0.9))
    reveal_radius = int(rollout_cfg.get("virtual_reveal_radius", 4))
    gain_decay = float(rollout_cfg.get("virtual_gain_decay", 0.92))
    reassign_on_reach = bool(cfg["planning"].get("reassign_on_reach", True))

    if horizon <= 0:
        return RolloutResult(0.0, {}, {}, {"future_gain": 0.0})

    local_context_r1 = local_context_for_predictor(map_mgr=map_mgr, robot=robot1, teammate=robot2, cfg=cfg)
    local_context_r2 = local_context_for_predictor(map_mgr=map_mgr, robot=robot2, teammate=robot1, cfg=cfg)

    horizon_steps = int(cfg.get("predictor", {}).get("horizon_steps", horizon))
    step_dt = float(cfg["termination"].get("step_dt", 1.0))

    pred1 = predictor.predict(
        PredictorInput(
            robot_state=robot1,
            goal=goal1,
            current_path=path1,
            local_context=local_context_r1,
            horizon_steps=horizon_steps,
            step_dt=step_dt,
        )
    )
    pred2 = predictor.predict(
        PredictorInput(
            robot_state=robot2,
            goal=goal2,
            current_path=path2,
            local_context=local_context_r2,
            horizon_steps=horizon_steps,
            step_dt=step_dt,
        )
    )

    path_cells1 = [_clamp_cell((p.x, p.y), map_mgr.width, map_mgr.height) for p in pred1.trajectory[:horizon]]
    path_cells2 = [_clamp_cell((p.x, p.y), map_mgr.width, map_mgr.height) for p in pred2.trajectory[:horizon]]

    if not path_cells1:
        path_cells1 = [robot1.pose]
    if not path_cells2:
        path_cells2 = [robot2.pose]

    virtual_known = map_mgr.clone_known()

    sensor_range = float(cfg["robots"].get("sensor_range", 6))
    penalties = cfg["planning"]["penalties"]
    sigma = penalties.get("sigma_overlap")
    if sigma is None:
        sigma = 2.0 * sensor_range
    sigma = float(sigma)
    lambda_overlap = float(penalties.get("lambda_overlap", 0.0))

    future_gain = 0.0
    future_overlap_penalty = 0.0

    used_targets: set[Cell] = set(c for c in (goal1, goal2) if c is not None)

    reassign_count_r1 = 0
    reassign_count_r2 = 0
    if reassign_on_reach:
        path_cells1, reassign_count_r1 = _rewrite_future_path_on_reach(
            path_cells=path_cells1,
            initial_goal=goal1,
            candidates=candidates,
            virtual_known=virtual_known,
            map_mgr=map_mgr,
            reveal_radius=reveal_radius,
            used_targets=used_targets,
        )
        path_cells2, reassign_count_r2 = _rewrite_future_path_on_reach(
            path_cells=path_cells2,
            initial_goal=goal2,
            candidates=candidates,
            virtual_known=virtual_known,
            map_mgr=map_mgr,
            reveal_radius=reveal_radius,
            used_targets=used_targets,
        )

    for t in range(horizon):
        c1 = path_cells1[min(t, len(path_cells1) - 1)]
        c2 = path_cells2[min(t, len(path_cells2) - 1)]

        g1 = float(map_mgr.count_unknown_in_radius(c1, reveal_radius, grid=virtual_known))
        g2 = float(map_mgr.count_unknown_in_radius(c2, reveal_radius, grid=virtual_known))

        if reassign_on_reach:
            if goal1 is not None and c1 == goal1:
                g1 += 0.5 * _best_virtual_frontier_gain(c1, candidates, virtual_known, map_mgr, reveal_radius, used_targets)
            if goal2 is not None and c2 == goal2:
                g2 += 0.5 * _best_virtual_frontier_gain(c2, candidates, virtual_known, map_mgr, reveal_radius, used_targets)

        w = (gamma**t) * (gain_decay**t)
        future_gain += w * (g1 + g2)

        overlap = _future_overlap_proxy(c1, c2, sigma)
        future_overlap_penalty += w * lambda_overlap * overlap

        map_mgr.mark_virtual_revealed(c1, reveal_radius, virtual_known)
        map_mgr.mark_virtual_revealed(c2, reveal_radius, virtual_known)

    future_score = future_gain - future_overlap_penalty

    return RolloutResult(
        future_score=float(future_score),
        predicted_paths={robot1.robot_id: path_cells1, robot2.robot_id: path_cells2},
        predictor_inference_times={robot1.robot_id: pred1.inference_time_sec, robot2.robot_id: pred2.inference_time_sec},
        breakdown={
            "future_gain": float(future_gain),
            "future_overlap_penalty": float(future_overlap_penalty),
            "future_score": float(future_score),
            "virtual_reassign_count_r1": float(reassign_count_r1),
            "virtual_reassign_count_r2": float(reassign_count_r2),
        },
    )
