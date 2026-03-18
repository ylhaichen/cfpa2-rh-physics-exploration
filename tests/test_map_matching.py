from __future__ import annotations

from core.map_matching import search_transform_hypotheses
from core.map_manager import FREE, OCCUPIED
from core.submap_manager import LocalSubmap


def _submap() -> LocalSubmap:
    return LocalSubmap(world_width=30, world_height=30, padding=10)


def test_known_transform_ranked_first() -> None:
    a = _submap()
    b = _submap()

    shape = {
        (0, 0): FREE,
        (1, 0): FREE,
        (2, 0): FREE,
        (0, 1): FREE,
        (2, 1): OCCUPIED,
        (0, 2): FREE,
        (1, 2): FREE,
        (2, 2): FREE,
        (3, 1): OCCUPIED,
    }
    for cell, value in shape.items():
        a.set_known(cell, value)

    # b is a translated copy of a; matching should recover dx=3, dy=-2.
    for (x, y), value in shape.items():
        b.set_known((x - 3, y + 2), value)

    hyps = search_transform_hypotheses(
        source_robot_id=2,
        target_robot_id=1,
        source_submap=b,
        target_submap=a,
        matching_cfg={
            "allowed_rotations_deg": [0, 90, 180, 270],
            "search_dx": 5,
            "search_dy": 5,
            "min_overlap_cells": 5,
            "top_k_hypotheses": 5,
            "w_occ": 2.0,
            "w_free": 1.0,
            "w_mismatch": 3.0,
        },
    )

    assert hyps
    best = hyps[0]
    assert best.rotation_deg == 0
    assert best.dx == 3
    assert best.dy == -2
    assert best.mismatch == 0
