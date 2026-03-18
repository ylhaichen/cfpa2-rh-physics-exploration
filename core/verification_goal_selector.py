from __future__ import annotations

from dataclasses import dataclass

from .path_service import astar_path, path_cost
from .submap_manager import SubmapManager
from .transform_hypothesis import TransformHypothesis, apply_transform
from .types import Cell, RobotState


@dataclass
class VerificationGoal:
    active_robot_id: int
    passive_robot_id: int
    target: Cell | None
    path: list[Cell]
    score: float
    strategy: str
    debug: dict


class VerificationGoalSelector:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _fallback_projected_history(
        self,
        active_robot: RobotState,
        passive_robot_id: int,
        hypothesis: TransformHypothesis,
        submaps: SubmapManager,
        score_radius: int,
        lambda_dist: float,
        lambda_risk: float,
    ) -> VerificationGoal | None:
        active_map = submaps.get_local_submap(active_robot.robot_id)
        passive_history = submaps.get_recent_local_trajectory(passive_robot_id, max_len=30)
        if not passive_history:
            return None

        best: VerificationGoal | None = None
        seen: set[Cell] = set()
        for cell in reversed(passive_history):
            projected = apply_transform(cell, hypothesis.rotation_deg, hypothesis.dx, hypothesis.dy)
            if projected in seen:
                continue
            seen.add(projected)
            candidate = active_map.nearest_known_free(projected, max_radius=6)
            if candidate is None:
                continue
            if candidate == active_robot.pose:
                continue
            path = astar_path(active_map, active_robot.pose, candidate, neighborhood=int(self.cfg["frontier"].get("neighborhood", 8)), clearance_cells=int(self.cfg["robots"].get("clearance_cells", 0)))
            if path is None:
                continue
            unknown_gain = float(active_map.count_unknown_in_radius(candidate, score_radius))
            if unknown_gain <= 0.0:
                continue
            risk = float(active_map.obstacle_count_around(candidate, radius=1))
            dist = path_cost(path)
            score = unknown_gain - lambda_dist * dist - lambda_risk * risk
            goal = VerificationGoal(
                active_robot_id=int(active_robot.robot_id),
                passive_robot_id=int(passive_robot_id),
                target=candidate,
                path=path,
                score=float(score),
                strategy="projected_history",
                debug={"projected_cell": projected, "unknown_gain": unknown_gain, "risk": risk, "distance": dist},
            )
            if best is None or goal.score > best.score:
                best = goal
        return best

    def select_goal(
        self,
        robot_states: list[RobotState],
        active_robot_id: int,
        passive_robot_id: int,
        hypothesis: TransformHypothesis,
        submaps: SubmapManager,
    ) -> VerificationGoal | None:
        active_robot = next((r for r in robot_states if r.robot_id == active_robot_id), None)
        if active_robot is None:
            return None

        vcfg = self.cfg.get("verification", {})
        score_radius = int(vcfg.get("score_radius", 5))
        lambda_dist = float(vcfg.get("lambda_dist", 0.2))
        lambda_risk = float(vcfg.get("lambda_risk", 0.5))
        strategy = str(vcfg.get("strategy", "disagreement_region"))

        # Simplified disagreement-region version: use projected passive history points
        # as candidate overlap probes in the active robot frame.
        goal = self._fallback_projected_history(
            active_robot=active_robot,
            passive_robot_id=passive_robot_id,
            hypothesis=hypothesis,
            submaps=submaps,
            score_radius=score_radius,
            lambda_dist=lambda_dist,
            lambda_risk=lambda_risk,
        )
        if goal is None:
            return None
        if strategy == "disagreement_region":
            goal.strategy = "disagreement_region"
        return goal
