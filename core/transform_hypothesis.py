from __future__ import annotations

from dataclasses import dataclass

from .types import Cell

ALLOWED_ROTATIONS = (0, 90, 180, 270)


def normalize_rotation_deg(rotation_deg: int) -> int:
    rot = int(rotation_deg) % 360
    if rot not in ALLOWED_ROTATIONS:
        raise ValueError(f"Unsupported discrete rotation: {rotation_deg}")
    return rot


def rotate_cell(cell: Cell, rotation_deg: int) -> Cell:
    x, y = int(cell[0]), int(cell[1])
    rot = normalize_rotation_deg(rotation_deg)
    if rot == 0:
        return (x, y)
    if rot == 90:
        return (-y, x)
    if rot == 180:
        return (-x, -y)
    return (y, -x)


def apply_transform(cell: Cell, rotation_deg: int, dx: int, dy: int) -> Cell:
    rx, ry = rotate_cell(cell, rotation_deg)
    return (int(rx + dx), int(ry + dy))


def invert_transform(rotation_deg: int, dx: int, dy: int) -> tuple[int, int, int]:
    rot = normalize_rotation_deg(rotation_deg)
    inv_rot = (360 - rot) % 360
    ix, iy = rotate_cell((-int(dx), -int(dy)), inv_rot)
    return int(inv_rot), int(ix), int(iy)


@dataclass
class TransformHypothesis:
    source_robot_id: int
    target_robot_id: int
    rotation_deg: int
    dx: int
    dy: int
    overlap_cells: int
    free_agree: int
    occ_agree: int
    mismatch: int
    normalized_score: float
    confidence_gap: float = 0.0
    status: str = "reject"

    def __post_init__(self) -> None:
        self.rotation_deg = normalize_rotation_deg(self.rotation_deg)
        self.dx = int(self.dx)
        self.dy = int(self.dy)
        self.overlap_cells = int(self.overlap_cells)
        self.free_agree = int(self.free_agree)
        self.occ_agree = int(self.occ_agree)
        self.mismatch = int(self.mismatch)
        self.normalized_score = float(self.normalized_score)
        self.confidence_gap = float(self.confidence_gap)
        self.status = str(self.status)

    def key(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.source_robot_id),
            int(self.target_robot_id),
            int(self.rotation_deg),
            int(self.dx),
            int(self.dy),
        )

    def transform_cell(self, cell: Cell) -> Cell:
        return apply_transform(cell, self.rotation_deg, self.dx, self.dy)

    def to_debug_dict(self) -> dict:
        return {
            "source_robot_id": int(self.source_robot_id),
            "target_robot_id": int(self.target_robot_id),
            "rotation_deg": int(self.rotation_deg),
            "dx": int(self.dx),
            "dy": int(self.dy),
            "overlap_cells": int(self.overlap_cells),
            "free_agree": int(self.free_agree),
            "occ_agree": int(self.occ_agree),
            "mismatch": int(self.mismatch),
            "normalized_score": float(self.normalized_score),
            "confidence_gap": float(self.confidence_gap),
            "status": str(self.status),
        }
