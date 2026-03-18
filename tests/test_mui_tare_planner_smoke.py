from __future__ import annotations

import numpy as np

from core.config import load_experiment_config
from core.map_manager import MapManager, OCCUPIED
from core.submap_manager import SubmapManager
from core.transform_hypothesis import rotate_cell
from core.types import GoalAssignment, PlannerInput, RobotState
from planners import build_planner


class _Frame:
    def __init__(self, origin: tuple[int, int], rot: int):
        self.origin = origin
        self.rot = rot

    def world_to_local(self, cell: tuple[int, int]) -> tuple[int, int]:
        return rotate_cell((cell[0] - self.origin[0], cell[1] - self.origin[1]), (360 - self.rot) % 360)


def _snap(angle: float, allowed: list[int]) -> int:
    angle %= 360.0
    return int(min(allowed, key=lambda r: min(abs(angle - r), 360.0 - abs(angle - r))))


def test_mui_tare_direct_merge_smoke() -> None:
    truth = np.zeros((32, 48), dtype=np.int8)
    truth[0, :] = truth[-1, :] = truth[:, 0] = truth[:, -1] = OCCUPIED
    truth[12:20, 20:28] = OCCUPIED
    truth[8:24, 10:12] = OCCUPIED
    truth[8:24, 36:38] = OCCUPIED

    map_mgr = MapManager(truth)
    robots = [RobotState(robot_id=1, pose=(8, 16), heading_deg=0.0), RobotState(robot_id=2, pose=(40, 16), heading_deg=180.0)]
    cfg = load_experiment_config("configs/base.yaml", "configs/planner_mui_tare.yaml")
    cfg["matching"]["search_dx"] = 5
    cfg["matching"]["search_dy"] = 5
    cfg["matching"]["min_overlap_cells"] = 5
    cfg["matching"]["min_steps_before_matching"] = 0
    cfg["matching"]["min_known_cells_per_robot"] = 0
    cfg["matching"]["min_occupied_cells_per_robot"] = 0
    cfg["matching"]["accept_min_overlap"] = 5
    cfg["matching"]["reject_min_overlap"] = 1
    cfg["matching"]["accept_score_threshold"] = 0.8
    cfg["matching"]["reject_score_threshold"] = 0.1
    cfg["matching"]["ambiguity_gap"] = 0.0
    cfg["matching"]["accept_min_occ_agree"] = 0
    cfg["matching"]["accept_min_occ_ratio"] = 0.0
    cfg["matching"]["accept_max_mismatch_ratio"] = 1.0
    frames = {r.robot_id: _Frame(r.pose, _snap(r.heading_deg, cfg["mapping"]["allowed_rotations_deg"])) for r in robots}
    submaps = SubmapManager(world_width=48, world_height=32, padding=10, robot_ids=[1, 2])
    rng = np.random.default_rng(0)

    for r in robots:
        observed = map_mgr.observe_from(r.pose, r.heading_deg, 10, 360.0, True, 0.0, rng)
        local_obs = [(frames[r.robot_id].world_to_local(c), map_mgr.get_truth(c)) for c in observed]
        submaps.update_from_observation(r.robot_id, local_obs)
        submaps.record_local_pose(r.robot_id, frames[r.robot_id].world_to_local(r.pose))

    planner_states = [RobotState(robot_id=r.robot_id, pose=frames[r.robot_id].world_to_local(r.pose), heading_deg=0.0) for r in robots]
    planner = build_planner(cfg)
    out = planner.plan(
        PlannerInput(
            shared_map=submaps,
            robot_states=planner_states,
            frontier_candidates=[],
            current_assignments={1: GoalAssignment(1, None, [], float("-inf"), False, {}), 2: GoalAssignment(2, None, [], float("-inf"), False, {})},
            reservation_state={},
            step_idx=0,
            sim_time=0.0,
            config=cfg,
        )
    )

    assert out.planner_name == "mui_tare_2d"
    assert out.debug["merge_state"] == "POST_MERGE"
    assert submaps.is_merged()
    assert out.assignments
