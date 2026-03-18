from __future__ import annotations

from core.merge_manager import MergeManager
from core.transform_hypothesis import TransformHypothesis


def _hyp(score: float, overlap: int, dx: int = 1, dy: int = 2) -> TransformHypothesis:
    return TransformHypothesis(
        source_robot_id=2,
        target_robot_id=1,
        rotation_deg=0,
        dx=dx,
        dy=dy,
        overlap_cells=overlap,
        free_agree=overlap,
        occ_agree=0,
        mismatch=0,
        normalized_score=score,
    )


def test_accept_ambiguous_reject_and_blacklist_ttl() -> None:
    mgr = MergeManager(
        {
            "accept_min_overlap": 10,
            "reject_min_overlap": 3,
            "accept_score_threshold": 0.8,
            "reject_score_threshold": 0.3,
            "ambiguity_gap": 0.1,
            "accept_min_occ_agree": 0,
            "accept_min_occ_ratio": 0.0,
            "accept_max_mismatch_ratio": 1.0,
            "blacklist_ttl": 5,
        }
    )

    accept = mgr.classify([_hyp(0.9, 12, dx=1), _hyp(0.7, 12, dx=8)], step_idx=0)
    assert accept.status == "accept"

    ambiguous = mgr.classify([_hyp(0.85, 12, dx=1), _hyp(0.80, 12, dx=8)], step_idx=1)
    assert ambiguous.status == "ambiguous"

    reject = mgr.classify([_hyp(0.2, 12)], step_idx=2)
    assert reject.status == "reject"
    assert reject.best is not None
    assert reject.best.key() in mgr.blacklist_keys(step_idx=2)
    assert reject.best.key() not in mgr.blacklist_keys(step_idx=8)


def test_nearby_runner_up_does_not_force_ambiguity() -> None:
    mgr = MergeManager(
        {
            "accept_min_overlap": 10,
            "reject_min_overlap": 3,
            "accept_score_threshold": 0.8,
            "reject_score_threshold": 0.3,
            "ambiguity_gap": 0.1,
            "ambiguity_neighbor_radius": 3,
            "accept_min_occ_agree": 0,
            "accept_min_occ_ratio": 0.0,
            "accept_max_mismatch_ratio": 1.0,
            "blacklist_ttl": 5,
        }
    )
    decision = mgr.classify([_hyp(0.9, 12, dx=1, dy=2), _hyp(0.86, 12, dx=2, dy=3)], step_idx=0)
    assert decision.status == "accept"
