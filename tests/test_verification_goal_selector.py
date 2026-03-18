from __future__ import annotations

from core.map_manager import FREE, OCCUPIED
from core.submap_manager import SubmapManager
from core.transform_hypothesis import TransformHypothesis
from core.types import RobotState
from core.verification_goal_selector import VerificationGoalSelector


def test_projected_history_goal_is_reachable() -> None:
    submaps = SubmapManager(world_width=40, world_height=40, padding=10, robot_ids=[1, 2])
    active = submaps.get_local_submap(1)

    for x in range(-5, 6):
        for y in range(-2, 3):
            active.set_known((x, y), FREE)
    active.set_known((2, 1), OCCUPIED)

    for cell in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]:
        submaps.record_local_pose(2, cell)

    selector = VerificationGoalSelector(
        {
            "frontier": {"neighborhood": 8},
            "robots": {"clearance_cells": 0},
            "verification": {"score_radius": 3, "lambda_dist": 0.1, "lambda_risk": 0.1, "strategy": "projected_history"},
        }
    )
    hypothesis = TransformHypothesis(2, 1, 0, 0, 0, overlap_cells=10, free_agree=10, occ_agree=0, mismatch=0, normalized_score=0.8)
    goal = selector.select_goal([RobotState(1, (0, 0), 0.0), RobotState(2, (0, 0), 0.0)], 1, 2, hypothesis, submaps)

    assert goal is not None
    assert goal.target is not None
    assert goal.target != (0, 0)
    assert goal.path
