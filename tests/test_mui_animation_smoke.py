from __future__ import annotations

import numpy as np

from core.animation_renderer import AnimationRenderer
from core.map_manager import MapManager
from core.types import GoalAssignment, RobotState


def test_mui_animation_renderer_mp4_smoke(tmp_path) -> None:
    truth = np.zeros((20, 20), dtype=np.int8)
    m = MapManager(truth)
    m.known[:, :] = truth

    renderer = AnimationRenderer(
        {
            "experiment": {"enable_live_plot": False, "save_animation": True},
            "animation": {"fps": 4, "figsize": [8.0, 5.0], "show_frontier_cells": False, "show_sensor_fov": False, "save_gif": False, "save_mp4": True},
        }
    )
    robots = [RobotState(robot_id=1, pose=(4, 4), heading_deg=0.0), RobotState(robot_id=2, pose=(12, 12), heading_deg=180.0)]
    assignments = {
        1: GoalAssignment(robot_id=1, target=(8, 8), path=[(5, 5), (6, 6), (7, 7), (8, 8)], utility=1.0, valid=True),
        2: GoalAssignment(robot_id=2, target=(10, 10), path=[(11, 11), (10, 10)], utility=1.0, valid=True),
    }
    renderer.update(
        step_idx=0,
        map_mgr=m,
        robots=robots,
        frontier_cells=[],
        frontier_candidates=[],
        assignments=assignments,
        coverage=0.5,
        planner_name="mui_tare_2d",
        seed=0,
        sim_time=0.0,
        replan_count=1,
        joint_score=2.0,
        last_replan_reason="initial",
        sensor_range=6,
        sensor_fov_deg=270.0,
        per_robot_observed_cells={1: [(3, 3), (4, 4)], 2: [(11, 11), (12, 12)]},
        mui_debug={"merge_state": "VERIFYING", "verification_goal": (8, 8), "merge_attempt_count": 1, "verification_count": 1},
    )
    gif_path, mp4_path = renderer.finalize(tmp_path / "mui_anim")
    assert gif_path is None
    assert mp4_path is None or mp4_path.endswith(".mp4")
