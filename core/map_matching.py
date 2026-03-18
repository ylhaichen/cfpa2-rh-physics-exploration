from __future__ import annotations

from collections import defaultdict

from .map_manager import FREE, OCCUPIED
from .submap_manager import LocalSubmap
from .transform_hypothesis import TransformHypothesis, apply_transform, rotate_cell


def _sample_cells(cells: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    if limit <= 0 or len(cells) <= limit:
        return list(cells)
    step = float(len(cells)) / float(limit)
    return [cells[min(len(cells) - 1, int(idx * step))] for idx in range(limit)]


def _evaluate_candidate(
    source_robot_id: int,
    target_robot_id: int,
    rot: int,
    dx: int,
    dy: int,
    source_cells: dict[tuple[int, int], int],
    target_cells: dict[tuple[int, int], int],
    min_overlap: int,
    w_occ: float,
    w_free: float,
    w_mismatch: float,
) -> TransformHypothesis | None:
    overlap = 0
    free_agree = 0
    occ_agree = 0
    mismatch = 0

    for source_cell, source_val in source_cells.items():
        target_cell = apply_transform(source_cell, rot, dx, dy)
        target_val = target_cells.get(target_cell)
        if target_val is None:
            continue
        overlap += 1
        if source_val == target_val == OCCUPIED:
            occ_agree += 1
        elif source_val == target_val == FREE:
            free_agree += 1
        else:
            mismatch += 1

    if overlap < min_overlap:
        return None

    raw_score = w_occ * occ_agree + w_free * free_agree - w_mismatch * mismatch
    normalized_score = float(raw_score) / float(max(overlap, 1))
    return TransformHypothesis(
        source_robot_id=source_robot_id,
        target_robot_id=target_robot_id,
        rotation_deg=rot,
        dx=dx,
        dy=dy,
        overlap_cells=overlap,
        free_agree=free_agree,
        occ_agree=occ_agree,
        mismatch=mismatch,
        normalized_score=normalized_score,
    )


def _is_near_duplicate(
    candidate: TransformHypothesis,
    kept: list[TransformHypothesis],
    radius: int,
) -> bool:
    for other in kept:
        if candidate.rotation_deg != other.rotation_deg:
            continue
        if abs(candidate.dx - other.dx) <= radius and abs(candidate.dy - other.dy) <= radius:
            return True
    return False


def search_transform_hypotheses(
    source_robot_id: int,
    target_robot_id: int,
    source_submap: LocalSubmap,
    target_submap: LocalSubmap,
    matching_cfg: dict,
    blacklist: set[tuple[int, int, int, int, int]] | None = None,
) -> list[TransformHypothesis]:
    blacklist = blacklist or set()
    source_cells = source_submap.export_sparse_cells()
    target_cells = target_submap.export_sparse_cells()
    if not source_cells or not target_cells:
        return []

    allowed_rotations = [int(r) for r in matching_cfg.get("allowed_rotations_deg", [0, 90, 180, 270])]
    search_dx = int(matching_cfg.get("search_dx", 40))
    search_dy = int(matching_cfg.get("search_dy", 40))
    min_overlap = int(matching_cfg.get("min_overlap_cells", 40))
    top_k = int(matching_cfg.get("top_k_hypotheses", 5))
    vote_top_k = int(matching_cfg.get("candidate_vote_top_k", max(24, top_k * 8)))
    occ_seed_limit = int(matching_cfg.get("candidate_seed_limit_occ", 64))
    free_seed_limit = int(matching_cfg.get("candidate_seed_limit_free", 96))
    refine_radius = int(matching_cfg.get("translation_refine_radius", 1))
    nms_radius = int(matching_cfg.get("translation_nms_radius", 2))
    w_occ = float(matching_cfg.get("w_occ", 2.0))
    w_free = float(matching_cfg.get("w_free", 1.0))
    w_mismatch = float(matching_cfg.get("w_mismatch", 3.0))

    source_occ = [cell for cell, value in source_cells.items() if value == OCCUPIED]
    target_occ = [cell for cell, value in target_cells.items() if value == OCCUPIED]
    source_free = [cell for cell, value in source_cells.items() if value == FREE]
    target_free = [cell for cell, value in target_cells.items() if value == FREE]

    source_occ_seed = _sample_cells(source_occ, occ_seed_limit)
    target_occ_seed = _sample_cells(target_occ, occ_seed_limit)
    source_free_seed = _sample_cells(source_free, free_seed_limit)
    target_free_seed = _sample_cells(target_free, free_seed_limit)

    scored: list[TransformHypothesis] = []
    evaluated: set[tuple[int, int, int]] = set()

    for rot in allowed_rotations:
        vote_map: dict[tuple[int, int], float] = defaultdict(float)

        for source_cell in source_occ_seed:
            rs = rotate_cell(source_cell, rot)
            for target_cell in target_occ_seed:
                dx = int(target_cell[0] - rs[0])
                dy = int(target_cell[1] - rs[1])
                if abs(dx) > search_dx or abs(dy) > search_dy:
                    continue
                vote_map[(dx, dy)] += float(w_occ)

        for source_cell in source_free_seed:
            rs = rotate_cell(source_cell, rot)
            for target_cell in target_free_seed:
                dx = int(target_cell[0] - rs[0])
                dy = int(target_cell[1] - rs[1])
                if abs(dx) > search_dx or abs(dy) > search_dy:
                    continue
                vote_map[(dx, dy)] += float(w_free) * 0.35

        if not vote_map:
            continue

        seed_translations = sorted(vote_map.items(), key=lambda item: item[1], reverse=True)[: max(1, vote_top_k)]
        candidate_translations: set[tuple[int, int]] = set()
        for (base_dx, base_dy), _ in seed_translations:
            for ddx in range(-refine_radius, refine_radius + 1):
                for ddy in range(-refine_radius, refine_radius + 1):
                    cand_dx = int(base_dx + ddx)
                    cand_dy = int(base_dy + ddy)
                    if abs(cand_dx) > search_dx or abs(cand_dy) > search_dy:
                        continue
                    candidate_translations.add((cand_dx, cand_dy))

        for dx, dy in candidate_translations:
            key = (int(rot), int(dx), int(dy))
            if key in evaluated:
                continue
            evaluated.add(key)
            full_key = (int(source_robot_id), int(target_robot_id), int(rot), int(dx), int(dy))
            if full_key in blacklist:
                continue
            hypothesis = _evaluate_candidate(
                source_robot_id=source_robot_id,
                target_robot_id=target_robot_id,
                rot=rot,
                dx=dx,
                dy=dy,
                source_cells=source_cells,
                target_cells=target_cells,
                min_overlap=min_overlap,
                w_occ=w_occ,
                w_free=w_free,
                w_mismatch=w_mismatch,
            )
            if hypothesis is None:
                continue
            scored.append(hypothesis)

    scored.sort(key=lambda h: (h.normalized_score, h.overlap_cells, h.occ_agree, -h.mismatch), reverse=True)

    hyps: list[TransformHypothesis] = []
    for hypothesis in scored:
        if _is_near_duplicate(hypothesis, hyps, radius=nms_radius):
            continue
        hyps.append(hypothesis)
        if len(hyps) >= max(1, top_k):
            break
    return hyps[: max(1, top_k)]
