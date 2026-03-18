from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.assignment_solver import compute_candidate_utilities, solve_single_robot
from core.frontier_manager import build_frontier_candidates
from core.map_manager import OCCUPIED
from core.map_matching import search_transform_hypotheses
from core.merge_manager import MergeDecision, MergeManager
from core.submap_manager import LocalSubmap, SubmapManager
from core.transform_hypothesis import TransformHypothesis, invert_transform
from core.types import GoalAssignment, PlannerInput, PlannerOutput, RobotState
from core.verification_goal_selector import VerificationGoal, VerificationGoalSelector
from planners.cfpa2_planner import CFPA2Planner
from planners.rh_cfpa2_planner import RHCFPA2Planner

from .base_planner import BasePlanner


@dataclass
class VerificationContext:
    active_robot_id: int
    passive_robot_id: int
    hypothesis: TransformHypothesis
    goal: VerificationGoal
    start_step: int
    start_observed_count: int
    attempts: int


class MUITARE2DPlanner(BasePlanner):
    """MUI-TARE-inspired 2D planner with unknown relative pose and submap merging.

    This planner is intentionally discrete and planner-level only:
    - each robot keeps a private local submap before merge
    - relative pose is searched over discrete rotations/translations
    - ambiguous matches trigger an active verification phase
    - after merge, an existing cooperative planner is reused on the merged frame
    """

    name = "mui_tare_2d"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.merge_state = "PRE_MERGE"
        self.merge_manager = MergeManager(cfg.get("matching", {}))
        self.verification_selector = VerificationGoalSelector(cfg)
        self.verification_ctx: VerificationContext | None = None
        self.verification_count = 0
        self.verification_total_steps = 0
        self.merge_step: int | None = None
        self.accepted_transform_score: float | None = None
        self.accepted_transform_overlap: int | None = None
        self.post_merge_planner = self._build_post_merge_planner(cfg)

    def _build_post_merge_planner(self, cfg: dict) -> BasePlanner:
        post_name = str(cfg.get("post_merge", {}).get("planner_name", "cfpa2"))
        if post_name == "rh_cfpa2":
            post_cfg = dict(cfg)
            post_cfg["planning"] = dict(cfg.get("planning", {}))
            post_cfg["planning"]["planner_name"] = "rh_cfpa2"
            return RHCFPA2Planner(post_cfg)
        return CFPA2Planner()

    def _idle_assignment(self, robot_id: int) -> GoalAssignment:
        return GoalAssignment(robot_id=int(robot_id), target=None, path=[], utility=float("-inf"), valid=False, breakdown={})

    def _single_agent_frontier_plan(self, robot: RobotState, local_map: LocalSubmap, cfg: dict) -> tuple[GoalAssignment, dict[str, Any]]:
        _, candidates = build_frontier_candidates(local_map, cfg)
        utilities = compute_candidate_utilities(robot, candidates, cfg, local_map, reservation_state={})
        assignment = solve_single_robot(robot, utilities)
        debug = {
            "frontier_candidate_count": len(candidates),
            "utility_count": len(utilities),
        }
        return assignment, debug

    def _independent_assignments(self, robots: list[RobotState], submaps: SubmapManager, cfg: dict) -> tuple[dict[int, GoalAssignment], dict[str, Any]]:
        assignments: dict[int, GoalAssignment] = {}
        debug: dict[str, Any] = {"mode": "independent"}
        total = 0.0
        for robot in robots:
            local_map = submaps.get_local_submap(robot.robot_id)
            assignment, robot_debug = self._single_agent_frontier_plan(robot, local_map, cfg)
            assignments[robot.robot_id] = assignment
            debug[f"robot_{robot.robot_id}"] = robot_debug
            if assignment.valid:
                total += float(assignment.utility)
        debug["joint_score"] = total
        return assignments, debug

    def _matching_decision(self, robots: list[RobotState], submaps: SubmapManager, step_idx: int) -> tuple[list[TransformHypothesis], MergeDecision]:
        if len(robots) < 2:
            return [], MergeDecision(status="reject", best=None, second=None, debug={"reason": "single_robot"})
        matching_cfg = dict(self.cfg.get("matching", {}))
        min_steps = int(matching_cfg.get("min_steps_before_matching", 0))
        if int(step_idx) < min_steps:
            return [], MergeDecision(status="reject", best=None, second=None, debug={"reason": "matching_warmup"})

        min_known = int(matching_cfg.get("min_known_cells_per_robot", 0))
        min_occ = int(matching_cfg.get("min_occupied_cells_per_robot", 0))
        for robot in robots[:2]:
            local_map = submaps.get_local_submap(robot.robot_id)
            if local_map.known_cell_count() < min_known:
                return [], MergeDecision(status="reject", best=None, second=None, debug={"reason": "insufficient_known_cells"})
            occ_count = sum(1 for value in local_map.export_sparse_cells().values() if value == OCCUPIED)
            if occ_count < min_occ:
                return [], MergeDecision(status="reject", best=None, second=None, debug={"reason": "insufficient_occupied_cells"})

        anchor = robots[0].robot_id
        source = robots[1].robot_id
        matching_cfg.setdefault("allowed_rotations_deg", self.cfg.get("mapping", {}).get("allowed_rotations_deg", [0, 90, 180, 270]))
        hypotheses = search_transform_hypotheses(
            source_robot_id=source,
            target_robot_id=anchor,
            source_submap=submaps.get_local_submap(source),
            target_submap=submaps.get_local_submap(anchor),
            matching_cfg=matching_cfg,
            blacklist=self.merge_manager.blacklist_keys(step_idx),
        )
        return hypotheses, self.merge_manager.classify(hypotheses, step_idx=step_idx)

    def _choose_verifier(
        self,
        robots: list[RobotState],
        submaps: SubmapManager,
        hypothesis: TransformHypothesis,
    ) -> VerificationGoal | None:
        # Try target-frame verification first using the searched hypothesis.
        goal = self.verification_selector.select_goal(
            robot_states=robots,
            active_robot_id=int(hypothesis.target_robot_id),
            passive_robot_id=int(hypothesis.source_robot_id),
            hypothesis=hypothesis,
            submaps=submaps,
        )
        if goal is not None:
            return goal

        inv_rot, inv_dx, inv_dy = invert_transform(hypothesis.rotation_deg, hypothesis.dx, hypothesis.dy)
        inverse = TransformHypothesis(
            source_robot_id=int(hypothesis.target_robot_id),
            target_robot_id=int(hypothesis.source_robot_id),
            rotation_deg=inv_rot,
            dx=inv_dx,
            dy=inv_dy,
            overlap_cells=hypothesis.overlap_cells,
            free_agree=hypothesis.free_agree,
            occ_agree=hypothesis.occ_agree,
            mismatch=hypothesis.mismatch,
            normalized_score=hypothesis.normalized_score,
            confidence_gap=hypothesis.confidence_gap,
            status=hypothesis.status,
        )
        return self.verification_selector.select_goal(
            robot_states=robots,
            active_robot_id=int(inverse.target_robot_id),
            passive_robot_id=int(inverse.source_robot_id),
            hypothesis=inverse,
            submaps=submaps,
        )

    def _start_verification(self, robots: list[RobotState], submaps: SubmapManager, hypothesis: TransformHypothesis, step_idx: int) -> bool:
        goal = self._choose_verifier(robots, submaps, hypothesis)
        if goal is None or goal.target is None:
            return False
        attempts = 0 if self.verification_ctx is None else int(self.verification_ctx.attempts)
        self.verification_ctx = VerificationContext(
            active_robot_id=int(goal.active_robot_id),
            passive_robot_id=int(goal.passive_robot_id),
            hypothesis=hypothesis,
            goal=goal,
            start_step=int(step_idx),
            start_observed_count=int(submaps.newly_observed_counts.get(goal.active_robot_id, 0)),
            attempts=attempts + 1,
        )
        self.verification_count += 1
        self.merge_state = "VERIFYING"
        return True

    def _verification_assignments(self, robots: list[RobotState], submaps: SubmapManager, cfg: dict) -> tuple[dict[int, GoalAssignment], dict[str, Any]]:
        if self.verification_ctx is None:
            return self._independent_assignments(robots, submaps, cfg)

        assignments, debug = self._independent_assignments(robots, submaps, cfg)
        ctx = self.verification_ctx
        active_robot = next((r for r in robots if r.robot_id == ctx.active_robot_id), None)
        if active_robot is not None and ctx.goal.target is not None:
            assignments[ctx.active_robot_id] = GoalAssignment(
                robot_id=int(ctx.active_robot_id),
                target=ctx.goal.target,
                path=list(ctx.goal.path),
                utility=float(ctx.goal.score),
                valid=True,
                breakdown={"verification_score": float(ctx.goal.score)},
            )
        debug["mode"] = "verification"
        debug["verification_goal"] = None if ctx.goal.target is None else tuple(ctx.goal.target)
        debug["verification_active_robot_id"] = int(ctx.active_robot_id)
        debug["verification_passive_robot_id"] = int(ctx.passive_robot_id)
        debug["verification_strategy"] = str(ctx.goal.strategy)
        return assignments, debug

    def _verification_finished(self, planner_input: PlannerInput, submaps: SubmapManager) -> bool:
        if self.verification_ctx is None:
            return True
        vcfg = planner_input.config.get("verification", {})
        ctx = self.verification_ctx
        active_robot = next((r for r in planner_input.robot_states if r.robot_id == ctx.active_robot_id), None)
        if active_robot is not None and ctx.goal.target is not None and active_robot.pose == ctx.goal.target:
            return True
        if planner_input.step_idx - ctx.start_step >= int(vcfg.get("max_steps", 30)):
            return True
        observed_gain = int(submaps.newly_observed_counts.get(ctx.active_robot_id, 0)) - int(ctx.start_observed_count)
        if observed_gain >= int(vcfg.get("obs_threshold", 20)):
            return True
        return False

    def _handle_verification_transition(self, planner_input: PlannerInput, submaps: SubmapManager) -> tuple[dict[int, GoalAssignment] | None, dict[str, Any]]:
        if self.verification_ctx is None:
            return None, {}
        ctx = self.verification_ctx
        self.verification_total_steps += max(0, int(planner_input.step_idx) - int(ctx.start_step))
        hypotheses, decision = self._matching_decision(planner_input.robot_states, submaps, planner_input.step_idx)
        debug: dict[str, Any] = {
            "matching_count": len(hypotheses),
            "verification_attempt": int(ctx.attempts),
            "verification_goal": None if ctx.goal.target is None else tuple(ctx.goal.target),
        }
        if decision.best is not None:
            debug["best_hypothesis"] = decision.best.to_debug_dict()
        if decision.second is not None:
            debug["second_hypothesis"] = decision.second.to_debug_dict()

        if decision.status == "accept" and decision.best is not None:
            vcfg = planner_input.config.get("verification", {})
            verified_accept_score = float(vcfg.get("accept_score_threshold", 0.90))
            verified_accept_overlap = int(vcfg.get("accept_min_overlap", 260))
            verified_accept_occ = int(vcfg.get("accept_min_occ_agree", 64))
            if (
                float(decision.best.normalized_score) < verified_accept_score
                or int(decision.best.overlap_cells) < verified_accept_overlap
                or int(decision.best.occ_agree) < verified_accept_occ
            ):
                self.merge_manager.register_rejected(decision.best, planner_input.step_idx)
                self.merge_state = "PRE_MERGE"
                self.verification_ctx = None
                debug.update(
                    {
                        "verified_accept_guard": False,
                        "verified_accept_score_threshold": verified_accept_score,
                        "verified_accept_min_overlap": verified_accept_overlap,
                        "verified_accept_min_occ_agree": verified_accept_occ,
                        "merge_state": self.merge_state,
                    }
                )
                return None, debug
            merged = self.merge_manager.accept_and_merge(
                submap_manager=submaps,
                anchor_robot_id=int(decision.best.target_robot_id),
                source_robot_id=int(decision.best.source_robot_id),
                hypothesis=decision.best,
            )
            self.merge_state = "POST_MERGE"
            self.verification_ctx = None
            self.merge_step = int(planner_input.step_idx)
            self.accepted_transform_score = float(decision.best.normalized_score)
            self.accepted_transform_overlap = int(decision.best.overlap_cells)
            debug.update({"merged": merged.merged, "merge_state": self.merge_state, "verified_accept_guard": True})
            return None, debug

        max_attempts = int(planner_input.config.get("verification", {}).get("max_attempts_per_pair", 3))
        if decision.status == "ambiguous" and decision.best is not None and ctx.attempts < max_attempts:
            if self._start_verification(planner_input.robot_states, submaps, decision.best, planner_input.step_idx):
                debug["verification_restarted"] = True
                return None, debug

        if decision.best is not None:
            self.merge_manager.register_rejected(decision.best, planner_input.step_idx)
        self.merge_state = "PRE_MERGE"
        self.verification_ctx = None
        debug["merge_state"] = self.merge_state
        return None, debug

    def _post_merge_plan(self, planner_input: PlannerInput, submaps: SubmapManager) -> PlannerOutput:
        merged_map = submaps.get_merged_map()
        if merged_map is None:
            self.merge_state = "PRE_MERGE"
            assignments, debug = self._independent_assignments(planner_input.robot_states, submaps, planner_input.config)
            return PlannerOutput(
                planner_name=self.name,
                assignments=assignments,
                joint_score=float(sum(a.utility for a in assignments.values() if a.valid)),
                predicted_paths={rid: a.path for rid, a in assignments.items() if a.valid},
                debug={"merge_state": self.merge_state, **debug},
            )

        _, merged_frontiers = build_frontier_candidates(merged_map, planner_input.config)
        post_input = PlannerInput(
            shared_map=merged_map,
            robot_states=planner_input.robot_states,
            frontier_candidates=merged_frontiers,
            current_assignments=planner_input.current_assignments,
            reservation_state={},
            step_idx=planner_input.step_idx,
            sim_time=planner_input.sim_time,
            config=planner_input.config,
        )
        out = self.post_merge_planner.plan(post_input)
        debug = dict(out.debug)
        debug.update(
            {
                "merge_state": "POST_MERGE",
                "post_merge_planner": out.planner_name,
                "merge_step": self.merge_step,
                "merge_attempt_count": self.merge_manager.merge_attempt_count,
                "merge_success_count": self.merge_manager.merge_success_count,
                "verification_count": self.verification_count,
                "verification_total_steps": self.verification_total_steps,
                "rejected_hypothesis_count": self.merge_manager.rejected_hypothesis_count,
                "accepted_transform_score": self.accepted_transform_score,
                "accepted_transform_overlap": self.accepted_transform_overlap,
            }
        )
        return PlannerOutput(
            planner_name=self.name,
            assignments=out.assignments,
            joint_score=out.joint_score,
            score_breakdown=out.score_breakdown,
            predicted_paths=out.predicted_paths,
            debug=debug,
        )

    def plan(self, planner_input: PlannerInput) -> PlannerOutput:
        submaps = planner_input.shared_map
        if not isinstance(submaps, SubmapManager):
            raise TypeError("mui_tare_2d expects PlannerInput.shared_map to be a SubmapManager")

        robots = planner_input.robot_states
        if not robots:
            return PlannerOutput(planner_name=self.name, assignments={}, joint_score=float("-inf"), debug={"reason": "no_robot"})

        if submaps.is_merged():
            self.merge_state = "POST_MERGE"

        if self.merge_state == "POST_MERGE":
            return self._post_merge_plan(planner_input, submaps)

        debug: dict[str, Any] = {
            "merge_state": self.merge_state,
            "merge_attempt_count": self.merge_manager.merge_attempt_count,
            "merge_success_count": self.merge_manager.merge_success_count,
            "verification_count": self.verification_count,
            "verification_total_steps": self.verification_total_steps,
            "rejected_hypothesis_count": self.merge_manager.rejected_hypothesis_count,
            "merge_step": self.merge_step,
            "accepted_transform_score": self.accepted_transform_score,
            "accepted_transform_overlap": self.accepted_transform_overlap,
        }

        if self.merge_state == "VERIFYING" and self._verification_finished(planner_input, submaps):
            _, transition_debug = self._handle_verification_transition(planner_input, submaps)
            debug.update(transition_debug)
            if self.merge_state == "POST_MERGE":
                return self._post_merge_plan(planner_input, submaps)

        if self.merge_state == "PRE_MERGE":
            hypotheses, decision = self._matching_decision(robots, submaps, planner_input.step_idx)
            debug.update(
                {
                    "matching_count": len(hypotheses),
                    "best_hypothesis": decision.best.to_debug_dict() if decision.best is not None else None,
                    "second_hypothesis": decision.second.to_debug_dict() if decision.second is not None else None,
                    "merge_decision": decision.status,
                }
            )
            if decision.status == "accept" and decision.best is not None:
                self.merge_manager.accept_and_merge(
                    submap_manager=submaps,
                    anchor_robot_id=int(decision.best.target_robot_id),
                    source_robot_id=int(decision.best.source_robot_id),
                    hypothesis=decision.best,
                )
                self.merge_state = "POST_MERGE"
                self.merge_step = int(planner_input.step_idx)
                self.accepted_transform_score = float(decision.best.normalized_score)
                self.accepted_transform_overlap = int(decision.best.overlap_cells)
                return self._post_merge_plan(planner_input, submaps)
            if decision.status == "ambiguous" and bool(planner_input.config.get("verification", {}).get("enabled", True)) and decision.best is not None:
                started = self._start_verification(robots, submaps, decision.best, planner_input.step_idx)
                debug["verification_started"] = bool(started)
                if started:
                    assignments, verify_debug = self._verification_assignments(robots, submaps, planner_input.config)
                    debug.update(verify_debug)
                    return PlannerOutput(
                        planner_name=self.name,
                        assignments=assignments,
                        joint_score=float(sum(a.utility for a in assignments.values() if a.valid)),
                        predicted_paths={rid: a.path for rid, a in assignments.items() if a.valid},
                        debug=debug,
                    )

        assignments, mode_debug = (
            self._verification_assignments(robots, submaps, planner_input.config)
            if self.merge_state == "VERIFYING"
            else self._independent_assignments(robots, submaps, planner_input.config)
        )
        debug.update(mode_debug)
        return PlannerOutput(
            planner_name=self.name,
            assignments=assignments,
            joint_score=float(sum(a.utility for a in assignments.values() if a.valid)),
            predicted_paths={rid: a.path for rid, a in assignments.items() if a.valid},
            debug=debug,
        )
