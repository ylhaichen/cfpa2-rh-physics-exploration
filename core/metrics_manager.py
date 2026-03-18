from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pvariance
from typing import Any

import pandas as pd

from .types import GoalAssignment, RobotState


@dataclass
class EpisodeMetrics:
    planner_name: str
    map_name: str
    seed: int
    rollout_horizon: int
    predictor_type: str

    coverage_curve: list[float] = field(default_factory=list)
    frontier_cell_curve: list[int] = field(default_factory=list)
    frontier_candidate_curve: list[int] = field(default_factory=list)

    replan_count: int = 0
    replan_reasons: dict[str, int] = field(default_factory=dict)
    reassignment_count: int = 0
    switching_count: int = 0

    conflict_count: int = 0
    congestion_count: int = 0

    planner_compute_times: list[float] = field(default_factory=list)
    predictor_inference_times: list[float] = field(default_factory=list)

    target_invalidation_events: int = 0
    success: bool = False
    failure_reason: str = ""
    completion_steps: int = 0
    completion_time: float = 0.0

    step_logs: list[dict[str, Any]] = field(default_factory=list)

    prediction_error_sums: dict[int, float] = field(default_factory=dict)
    prediction_error_counts: dict[int, int] = field(default_factory=dict)
    _pending_predictions: list[dict[str, Any]] = field(default_factory=list)

    decision_probe_pair_count: int = 0
    decision_divergence_count: int = 0
    chosen_frontier_difference_count: int = 0
    predictor_rollout_score_variances: list[float] = field(default_factory=list)

    merge_attempt_count: int = 0
    merge_success: int = 0
    merge_step: int = -1
    verification_count: int = 0
    verification_total_steps: int = 0
    rejected_hypothesis_count: int = 0
    accepted_transform_score: float = 0.0
    accepted_transform_overlap: int = 0
    false_merge_count: int = 0
    true_merge_count: int = 0
    merge_transform_error_translation: float = 0.0
    merge_transform_error_rotation: float = 0.0
    pre_merge_coverage: float = 0.0
    post_merge_coverage: float = 0.0
    duplicate_exploration_proxy_pre_merge: float = 0.0
    duplicate_exploration_proxy_post_merge: float = 0.0

    _last_targets: dict[int, tuple[int, int] | None] = field(default_factory=dict)

    def log_step(
        self,
        step_idx: int,
        sim_time: float,
        coverage: float,
        frontier_cells: int,
        frontier_candidates: int,
        planner_score: float | None,
    ) -> None:
        self.coverage_curve.append(float(coverage))
        self.frontier_cell_curve.append(int(frontier_cells))
        self.frontier_candidate_curve.append(int(frontier_candidates))
        self.step_logs.append(
            {
                "step": int(step_idx),
                "sim_time": float(sim_time),
                "coverage": float(coverage),
                "frontier_cells": int(frontier_cells),
                "frontier_candidates": int(frontier_candidates),
                "planner_score": float(planner_score) if planner_score is not None else None,
            }
        )

    def log_replan(self, reason: str, planner_compute_time: float) -> None:
        self.replan_count += 1
        self.replan_reasons[reason] = self.replan_reasons.get(reason, 0) + 1
        self.planner_compute_times.append(float(planner_compute_time))
        if "target_invalidated" in reason:
            self.target_invalidation_events += 1

    def log_assignments(self, assignments: dict[int, GoalAssignment]) -> None:
        for rid, a in assignments.items():
            target = a.target if a.valid else None
            prev = self._last_targets.get(rid)
            if prev is not None and target is not None and prev != target:
                self.switching_count += 1
            if target is not None and prev != target:
                self.reassignment_count += 1
            self._last_targets[rid] = target

    def log_conflict(self) -> None:
        self.conflict_count += 1

    def log_congestion(self) -> None:
        self.congestion_count += 1

    def log_predictor_times(self, times: dict[int, float]) -> None:
        for t in times.values():
            self.predictor_inference_times.append(float(t))

    def log_decision_probe(
        self,
        base_predictor: str,
        decision_signatures: dict[str, tuple[Any, ...]],
        predictor_scores: dict[str, float],
    ) -> None:
        if not decision_signatures:
            return
        base_sig = decision_signatures.get(base_predictor)
        if base_sig is None:
            base_predictor = sorted(decision_signatures.keys())[0]
            base_sig = decision_signatures[base_predictor]

        score_values = [float(v) for v in predictor_scores.values()]
        if len(score_values) >= 2:
            self.predictor_rollout_score_variances.append(float(pvariance(score_values)))
        else:
            self.predictor_rollout_score_variances.append(0.0)

        for name, sig in decision_signatures.items():
            if name == base_predictor:
                continue
            self.decision_probe_pair_count += 1
            diff = 0
            n = min(len(base_sig), len(sig))
            for i in range(n):
                if base_sig[i] != sig[i]:
                    diff += 1
            diff += abs(len(base_sig) - len(sig))
            if diff > 0:
                self.decision_divergence_count += 1
            self.chosen_frontier_difference_count += int(diff)

    def register_predictions(self, step_idx: int, predicted_paths: dict[int, list[tuple[int, int]]]) -> None:
        for rid, path in predicted_paths.items():
            if not path:
                continue
            self._pending_predictions.append(
                {
                    "start_step": int(step_idx),
                    "robot_id": int(rid),
                    "path": list(path),
                }
            )

    def update_prediction_error(self, step_idx: int, robots: list[RobotState]) -> None:
        pose_by_id = {r.robot_id: r.pose for r in robots}
        keep: list[dict[str, Any]] = []

        for rec in self._pending_predictions:
            rid = rec["robot_id"]
            if rid not in pose_by_id:
                continue
            elapsed = step_idx - int(rec["start_step"])
            if elapsed < 0:
                keep.append(rec)
                continue

            path = rec["path"]
            if not path:
                continue

            idx = min(elapsed, len(path) - 1)
            pred = path[idx]
            cur = pose_by_id[rid]
            err = ((pred[0] - cur[0]) ** 2 + (pred[1] - cur[1]) ** 2) ** 0.5

            horizon = idx + 1
            self.prediction_error_sums[horizon] = self.prediction_error_sums.get(horizon, 0.0) + float(err)
            self.prediction_error_counts[horizon] = self.prediction_error_counts.get(horizon, 0) + 1

            if elapsed < len(path) - 1:
                keep.append(rec)

        self._pending_predictions = keep

    def finalize(self, robots: list[RobotState], steps: int, sim_time: float, success: bool, reason: str) -> None:
        self.completion_steps = int(steps)
        self.completion_time = float(sim_time)
        self.success = bool(success)
        self.failure_reason = reason

        total_move = int(sum(r.total_move_steps for r in robots))
        revisited = int(sum(r.revisited_move_steps for r in robots))

        self._summary_cache = {
            "per_robot_path_length": {f"r{r.robot_id}": float(r.path_length) for r in robots},
            "total_path_length": float(sum(r.path_length for r in robots)),
            "total_move_steps": total_move,
            "revisited_move_steps": revisited,
            "duplicated_exploration_proxy": float(revisited / total_move) if total_move > 0 else 0.0,
            "idle_steps": int(sum(r.idle_steps for r in robots)),
        }

    def update_mui_metrics(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        self.merge_attempt_count = max(self.merge_attempt_count, int(payload.get("merge_attempt_count", self.merge_attempt_count)))
        self.merge_success = max(self.merge_success, int(payload.get("merge_success", self.merge_success)))
        merge_step = payload.get("merge_step")
        if merge_step is not None:
            self.merge_step = int(merge_step)
        self.verification_count = max(self.verification_count, int(payload.get("verification_count", self.verification_count)))
        self.verification_total_steps = max(
            self.verification_total_steps,
            int(payload.get("verification_total_steps", self.verification_total_steps)),
        )
        self.rejected_hypothesis_count = max(
            self.rejected_hypothesis_count,
            int(payload.get("rejected_hypothesis_count", self.rejected_hypothesis_count)),
        )
        if payload.get("accepted_transform_score") is not None:
            self.accepted_transform_score = float(payload.get("accepted_transform_score", self.accepted_transform_score))
        if payload.get("accepted_transform_overlap") is not None:
            self.accepted_transform_overlap = int(payload.get("accepted_transform_overlap", self.accepted_transform_overlap))
        self.false_merge_count = max(self.false_merge_count, int(payload.get("false_merge_count", self.false_merge_count)))
        self.true_merge_count = max(self.true_merge_count, int(payload.get("true_merge_count", self.true_merge_count)))
        if payload.get("merge_transform_error_translation") is not None:
            self.merge_transform_error_translation = float(
                payload.get("merge_transform_error_translation", self.merge_transform_error_translation)
            )
        if payload.get("merge_transform_error_rotation") is not None:
            self.merge_transform_error_rotation = float(
                payload.get("merge_transform_error_rotation", self.merge_transform_error_rotation)
            )
        if payload.get("pre_merge_coverage") is not None:
            self.pre_merge_coverage = float(payload.get("pre_merge_coverage", self.pre_merge_coverage))
        if payload.get("post_merge_coverage") is not None:
            self.post_merge_coverage = float(payload.get("post_merge_coverage", self.post_merge_coverage))
        if payload.get("duplicate_exploration_proxy_pre_merge") is not None:
            self.duplicate_exploration_proxy_pre_merge = float(
                payload.get("duplicate_exploration_proxy_pre_merge", self.duplicate_exploration_proxy_pre_merge)
            )
        if payload.get("duplicate_exploration_proxy_post_merge") is not None:
            self.duplicate_exploration_proxy_post_merge = float(
                payload.get("duplicate_exploration_proxy_post_merge", self.duplicate_exploration_proxy_post_merge)
            )

    def prediction_error_by_horizon(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for h, s in self.prediction_error_sums.items():
            c = self.prediction_error_counts.get(h, 0)
            out[h] = float(s / c) if c > 0 else 0.0
        return out

    def to_summary_row(self) -> dict[str, Any]:
        planner_ms_mean = (sum(self.planner_compute_times) / len(self.planner_compute_times) * 1000.0) if self.planner_compute_times else 0.0
        planner_ms_p95 = 0.0
        if self.planner_compute_times:
            vals = sorted(self.planner_compute_times)
            idx = int(0.95 * (len(vals) - 1))
            planner_ms_p95 = vals[idx] * 1000.0

        pred_ms_mean = (sum(self.predictor_inference_times) / len(self.predictor_inference_times) * 1000.0) if self.predictor_inference_times else 0.0

        final_cov = float(self.coverage_curve[-1]) if self.coverage_curve else 0.0
        avg_frontier_candidates = (
            sum(self.frontier_candidate_curve) / len(self.frontier_candidate_curve)
            if self.frontier_candidate_curve
            else 0.0
        )

        prediction_error = self.prediction_error_by_horizon()
        score_var_mean = (
            float(sum(self.predictor_rollout_score_variances) / len(self.predictor_rollout_score_variances))
            if self.predictor_rollout_score_variances
            else 0.0
        )
        score_var_p95 = 0.0
        if self.predictor_rollout_score_variances:
            vals = sorted(self.predictor_rollout_score_variances)
            idx = int(0.95 * (len(vals) - 1))
            score_var_p95 = float(vals[idx])
        decision_div_rate = (
            float(self.decision_divergence_count / self.decision_probe_pair_count)
            if self.decision_probe_pair_count > 0
            else 0.0
        )
        frontier_diff_mean = (
            float(self.chosen_frontier_difference_count / self.decision_probe_pair_count)
            if self.decision_probe_pair_count > 0
            else 0.0
        )

        base = {
            "planner_name": self.planner_name,
            "map_name": self.map_name,
            "seed": self.seed,
            "success": self.success,
            "completion_steps": self.completion_steps,
            "completion_time": self.completion_time,
            "final_coverage": final_cov,
            "replan_count": self.replan_count,
            "reassignment_count": self.reassignment_count,
            "switching_count": self.switching_count,
            "conflict_count": self.conflict_count,
            "congestion_count": self.congestion_count,
            "planner_compute_time_ms_mean": planner_ms_mean,
            "planner_compute_time_ms_p95": planner_ms_p95,
            "predictor_inference_time_ms_mean": pred_ms_mean,
            "target_invalidation_events": self.target_invalidation_events,
            "rollout_horizon": self.rollout_horizon,
            "predictor_type": self.predictor_type,
            "avg_frontier_candidates": avg_frontier_candidates,
            "failure_reason": self.failure_reason,
            "prediction_error_by_horizon": json.dumps(prediction_error, sort_keys=True),
            "prediction_error_h1": float(prediction_error.get(1, 0.0)),
            "prediction_error_h3": float(prediction_error.get(3, 0.0)),
            "prediction_error_h5": float(prediction_error.get(5, 0.0)),
            "decision_probe_pair_count": self.decision_probe_pair_count,
            "decision_divergence_count": self.decision_divergence_count,
            "decision_divergence_rate": decision_div_rate,
            "chosen_frontier_difference_count": self.chosen_frontier_difference_count,
            "chosen_frontier_difference_mean": frontier_diff_mean,
            "predictor_rollout_score_variance_mean": score_var_mean,
            "predictor_rollout_score_variance_p95": score_var_p95,
            "replan_reasons": json.dumps(self.replan_reasons, sort_keys=True),
            "merge_attempt_count": self.merge_attempt_count,
            "merge_success": self.merge_success,
            "merge_step": self.merge_step,
            "verification_count": self.verification_count,
            "verification_total_steps": self.verification_total_steps,
            "rejected_hypothesis_count": self.rejected_hypothesis_count,
            "accepted_transform_score": self.accepted_transform_score,
            "accepted_transform_overlap": self.accepted_transform_overlap,
            "false_merge_count": self.false_merge_count,
            "true_merge_count": self.true_merge_count,
            "merge_transform_error_translation": self.merge_transform_error_translation,
            "merge_transform_error_rotation": self.merge_transform_error_rotation,
            "pre_merge_coverage": self.pre_merge_coverage,
            "post_merge_coverage": self.post_merge_coverage,
            "duplicate_exploration_proxy_pre_merge": self.duplicate_exploration_proxy_pre_merge,
            "duplicate_exploration_proxy_post_merge": self.duplicate_exploration_proxy_post_merge,
        }
        base.update(self._summary_cache)
        return base


def save_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def save_step_logs_csv(path: str | Path, logs: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(logs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def save_coverage_csv(path: str | Path, coverage_curve: list[float]) -> None:
    rows = [{"step": i, "coverage": float(c)} for i, c in enumerate(coverage_curve)]
    save_step_logs_csv(path, rows)
