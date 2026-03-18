from __future__ import annotations

from dataclasses import dataclass, field

from .submap_manager import SubmapManager
from .transform_hypothesis import TransformHypothesis


@dataclass
class MergeDecision:
    status: str
    best: TransformHypothesis | None
    second: TransformHypothesis | None
    merged: bool = False
    merged_anchor_robot_id: int | None = None
    merged_source_robot_id: int | None = None
    debug: dict = field(default_factory=dict)


class MergeManager:
    def __init__(self, matching_cfg: dict):
        self.accept_min_overlap = int(matching_cfg.get("accept_min_overlap", 60))
        self.reject_min_overlap = int(matching_cfg.get("reject_min_overlap", 20))
        self.accept_score_threshold = float(matching_cfg.get("accept_score_threshold", 0.75))
        self.reject_score_threshold = float(matching_cfg.get("reject_score_threshold", 0.35))
        self.ambiguity_gap = float(matching_cfg.get("ambiguity_gap", 0.10))
        self.ambiguity_neighbor_radius = int(matching_cfg.get("ambiguity_neighbor_radius", 3))
        self.blacklist_neighbor_radius = int(matching_cfg.get("blacklist_neighbor_radius", self.ambiguity_neighbor_radius))
        self.accept_min_occ_agree = int(matching_cfg.get("accept_min_occ_agree", 18))
        self.accept_min_occ_ratio = float(matching_cfg.get("accept_min_occ_ratio", 0.10))
        self.accept_max_mismatch_ratio = float(matching_cfg.get("accept_max_mismatch_ratio", 0.08))
        self.blacklist_ttl = int(matching_cfg.get("blacklist_ttl", 50))
        self.blacklist: dict[tuple[int, int, int, int, int], int] = {}
        self.rejected_hypothesis_count = 0
        self.merge_attempt_count = 0
        self.merge_success_count = 0

    def _prune(self, step_idx: int) -> None:
        expired = [k for k, ttl in self.blacklist.items() if ttl <= int(step_idx)]
        for key in expired:
            self.blacklist.pop(key, None)

    def blacklist_keys(self, step_idx: int) -> set[tuple[int, int, int, int, int]]:
        self._prune(step_idx)
        return set(self.blacklist.keys())

    def register_rejected(self, hypothesis: TransformHypothesis | None, step_idx: int) -> None:
        if hypothesis is None:
            return
        ttl = int(step_idx) + self.blacklist_ttl
        for ddx in range(-self.blacklist_neighbor_radius, self.blacklist_neighbor_radius + 1):
            for ddy in range(-self.blacklist_neighbor_radius, self.blacklist_neighbor_radius + 1):
                key = (
                    int(hypothesis.source_robot_id),
                    int(hypothesis.target_robot_id),
                    int(hypothesis.rotation_deg),
                    int(hypothesis.dx + ddx),
                    int(hypothesis.dy + ddy),
                )
                self.blacklist[key] = ttl
        self.rejected_hypothesis_count += 1

    def _distinct_runner_up(
        self,
        best: TransformHypothesis,
        filtered: list[TransformHypothesis],
    ) -> TransformHypothesis | None:
        for hypothesis in filtered[1:]:
            same_rot = int(hypothesis.rotation_deg) == int(best.rotation_deg)
            near_translation = (
                abs(int(hypothesis.dx) - int(best.dx)) <= self.ambiguity_neighbor_radius
                and abs(int(hypothesis.dy) - int(best.dy)) <= self.ambiguity_neighbor_radius
            )
            if not (same_rot and near_translation):
                return hypothesis
        return None

    def classify(self, hypotheses: list[TransformHypothesis], step_idx: int) -> MergeDecision:
        self._prune(step_idx)
        filtered = [h for h in hypotheses if h.key() not in self.blacklist]
        self.merge_attempt_count += 1
        if not filtered:
            return MergeDecision(status="reject", best=None, second=None, debug={"reason": "no_hypothesis"})

        best = filtered[0]
        second = self._distinct_runner_up(best, filtered)
        gap = best.normalized_score - (second.normalized_score if second is not None else 0.0)
        best.confidence_gap = float(gap)
        occ_ratio = float(best.occ_agree) / float(max(best.overlap_cells, 1))
        mismatch_ratio = float(best.mismatch) / float(max(best.overlap_cells, 1))

        if (
            best.overlap_cells >= self.accept_min_overlap
            and best.normalized_score >= self.accept_score_threshold
            and gap >= self.ambiguity_gap
            and best.occ_agree >= self.accept_min_occ_agree
            and occ_ratio >= self.accept_min_occ_ratio
            and mismatch_ratio <= self.accept_max_mismatch_ratio
        ):
            best.status = "accept"
            self.merge_success_count += 1
            return MergeDecision(
                status="accept",
                best=best,
                second=second,
                debug={"confidence_gap": gap, "occ_ratio": occ_ratio, "mismatch_ratio": mismatch_ratio},
            )

        if best.overlap_cells < self.reject_min_overlap or best.normalized_score < self.reject_score_threshold:
            best.status = "reject"
            self.register_rejected(best, step_idx)
            return MergeDecision(
                status="reject",
                best=best,
                second=second,
                debug={"confidence_gap": gap, "occ_ratio": occ_ratio, "mismatch_ratio": mismatch_ratio},
            )

        best.status = "ambiguous"
        return MergeDecision(
            status="ambiguous",
            best=best,
            second=second,
            debug={"confidence_gap": gap, "occ_ratio": occ_ratio, "mismatch_ratio": mismatch_ratio},
        )

    def accept_and_merge(
        self,
        submap_manager: SubmapManager,
        anchor_robot_id: int,
        source_robot_id: int,
        hypothesis: TransformHypothesis,
    ) -> MergeDecision:
        submap_manager.merge_with_transform(anchor_robot_id, source_robot_id, hypothesis)
        return MergeDecision(
            status="accept",
            best=hypothesis,
            second=None,
            merged=True,
            merged_anchor_robot_id=int(anchor_robot_id),
            merged_source_robot_id=int(source_robot_id),
            debug={"merged": True},
        )
