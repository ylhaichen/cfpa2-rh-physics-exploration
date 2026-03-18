from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .map_manager import FREE, OCCUPIED, UNKNOWN
from .transform_hypothesis import TransformHypothesis, apply_transform
from .types import Cell


class LocalSubmap:
    """Dense local occupancy grid backed by an oversized local frame array.

    Coordinates are expressed in the robot-local grid frame, not in world frame.
    The array is sized to safely contain any rotated/translated world projection.
    """

    def __init__(self, world_width: int, world_height: int, padding: int = 20, name: str = ""):
        max_dim = max(int(world_width), int(world_height))
        half_extent = max_dim + int(padding)
        side = int(2 * half_extent + 1)
        self.width = side
        self.height = side
        self._center = half_extent
        self._half_extent = half_extent
        self.name = name
        self.known = np.full((self.height, self.width), UNKNOWN, dtype=np.int8)
        self._known_cells: set[Cell] = set()

    def clone(self) -> "LocalSubmap":
        other = LocalSubmap(1, 1, padding=0, name=self.name)
        other.width = self.width
        other.height = self.height
        other._center = self._center
        other._half_extent = self._half_extent
        other.known = self.known.copy()
        other._known_cells = set(self._known_cells)
        return other

    def clone_known(self) -> np.ndarray:
        return self.known.copy()

    @property
    def center(self) -> int:
        return int(self._center)

    def _to_index(self, cell: Cell) -> tuple[int, int]:
        x, y = int(cell[0]), int(cell[1])
        return self._center + y, self._center + x

    def _from_index(self, iy: int, ix: int) -> Cell:
        return (int(ix - self._center), int(iy - self._center))

    def in_bounds(self, cell: Cell) -> bool:
        iy, ix = self._to_index(cell)
        return 0 <= iy < self.height and 0 <= ix < self.width

    def neighbors(self, cell: Cell, neighborhood: int = 8) -> list[Cell]:
        x, y = cell
        if neighborhood == 4:
            out = []
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nxt = (nx, ny)
                if self.in_bounds(nxt):
                    out.append(nxt)
            return out
        out: list[Cell] = []
        for ny in range(y - 1, y + 2):
            for nx in range(x - 1, x + 2):
                if nx == x and ny == y:
                    continue
                nxt = (nx, ny)
                if self.in_bounds(nxt):
                    out.append(nxt)
        return out

    def grid_value(self, cell: Cell, grid: np.ndarray) -> int:
        iy, ix = self._to_index(cell)
        return int(grid[iy, ix])

    def iter_cells_with_value(self, value: int, grid: np.ndarray | None = None) -> list[Cell]:
        arr = self.known if grid is None else grid
        ys, xs = np.where(arr == int(value))
        return [self._from_index(int(iy), int(ix)) for iy, ix in zip(ys.tolist(), xs.tolist())]

    def get_known(self, cell: Cell) -> int:
        iy, ix = self._to_index(cell)
        return int(self.known[iy, ix])

    def set_known(self, cell: Cell, value: int) -> bool:
        if not self.in_bounds(cell):
            return False
        iy, ix = self._to_index(cell)
        prev = int(self.known[iy, ix])
        self.known[iy, ix] = int(value)
        if value != UNKNOWN:
            self._known_cells.add((int(cell[0]), int(cell[1])))
        elif (int(cell[0]), int(cell[1])) in self._known_cells:
            self._known_cells.remove((int(cell[0]), int(cell[1])))
        return prev == UNKNOWN and value != UNKNOWN

    def update_cells(self, observations: Iterable[tuple[Cell, int]]) -> int:
        new_cells = 0
        for cell, value in observations:
            if self.set_known(cell, int(value)):
                new_cells += 1
        return new_cells

    def export_sparse_cells(self) -> dict[Cell, int]:
        return {cell: self.get_known(cell) for cell in self._known_cells}

    def known_cell_count(self) -> int:
        return len(self._known_cells)

    def known_free_count(self) -> int:
        return int(np.count_nonzero(self.known == FREE))

    def is_known_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.get_known(cell) == FREE

    def is_known_occupied(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.get_known(cell) == OCCUPIED

    def is_unknown(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.get_known(cell) == UNKNOWN

    def clearance_ok(self, cell: Cell, clearance: int) -> bool:
        if clearance <= 0:
            return True
        cx, cy = cell
        for y in range(cy - clearance, cy + clearance + 1):
            for x in range(cx - clearance, cx + clearance + 1):
                c = (x, y)
                if not self.in_bounds(c):
                    continue
                if self.get_known(c) == OCCUPIED:
                    return False
        return True

    def is_traversable(self, cell: Cell, clearance: int) -> bool:
        return self.is_known_free(cell) and self.clearance_ok(cell, clearance)

    def nearest_known_free(self, cell: Cell, max_radius: int = 20) -> Cell | None:
        if self.is_known_free(cell):
            return cell
        cx, cy = cell
        for radius in range(1, max_radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                for x in range(cx - radius, cx + radius + 1):
                    c = (x, y)
                    if self.in_bounds(c) and self.get_known(c) == FREE:
                        return c
        return None

    def obstacle_count_around(self, cell: Cell, radius: int = 1) -> int:
        cx, cy = cell
        count = 0
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                c = (x, y)
                if self.in_bounds(c) and self.get_known(c) == OCCUPIED:
                    count += 1
        return count

    def count_unknown_in_radius(self, center: Cell, radius: int, grid: np.ndarray | None = None) -> int:
        arr = self.known if grid is None else grid
        cx, cy = center
        rr = radius * radius
        gain = 0
        for y in range(cy - radius, cy + radius + 1):
            dy = y - cy
            for x in range(cx - radius, cx + radius + 1):
                dx = x - cx
                if dx * dx + dy * dy > rr:
                    continue
                if not self.in_bounds((x, y)):
                    continue
                iy, ix = self._to_index((x, y))
                if arr[iy, ix] == UNKNOWN:
                    gain += 1
        return gain

    def mark_virtual_revealed(self, center: Cell, radius: int, grid: np.ndarray) -> None:
        cx, cy = center
        rr = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            dy = y - cy
            for x in range(cx - radius, cx + radius + 1):
                dx = x - cx
                if dx * dx + dy * dy > rr:
                    continue
                if not self.in_bounds((x, y)):
                    continue
                iy, ix = self._to_index((x, y))
                if grid[iy, ix] == UNKNOWN:
                    grid[iy, ix] = FREE


@dataclass
class SubmapManager:
    world_width: int
    world_height: int
    padding: int
    robot_ids: list[int]
    submaps: dict[int, LocalSubmap] = field(init=False)
    local_trajectories: dict[int, list[Cell]] = field(init=False)
    newly_observed_counts: dict[int, int] = field(init=False)
    merged_anchor_robot_id: int | None = None
    merged_source_robot_id: int | None = None
    accepted_hypothesis: TransformHypothesis | None = None
    _merged_cache: LocalSubmap | None = None
    _merged_dirty: bool = True

    def __post_init__(self) -> None:
        self.submaps = {
            int(rid): LocalSubmap(self.world_width, self.world_height, padding=self.padding, name=f"robot_{rid}")
            for rid in self.robot_ids
        }
        self.local_trajectories = {int(rid): [] for rid in self.robot_ids}
        self.newly_observed_counts = {int(rid): 0 for rid in self.robot_ids}

    def update_from_observation(self, robot_id: int, observations: Iterable[tuple[Cell, int]]) -> int:
        rid = int(robot_id)
        new_cells = self.submaps[rid].update_cells(observations)
        self.newly_observed_counts[rid] += int(new_cells)
        self._merged_dirty = True
        return int(new_cells)

    def record_local_pose(self, robot_id: int, pose: Cell) -> None:
        rid = int(robot_id)
        self.local_trajectories.setdefault(rid, []).append((int(pose[0]), int(pose[1])))

    def get_local_submap(self, robot_id: int) -> LocalSubmap:
        return self.submaps[int(robot_id)]

    def get_local_frontiers(self, robot_id: int, cfg: dict):
        from .frontier_manager import build_frontier_candidates

        return build_frontier_candidates(self.get_local_submap(robot_id), cfg)

    def export_sparse_cells(self, robot_id: int) -> dict[Cell, int]:
        return self.get_local_submap(robot_id).export_sparse_cells()

    def get_recent_local_trajectory(self, robot_id: int, max_len: int = 40) -> list[Cell]:
        traj = self.local_trajectories.get(int(robot_id), [])
        if max_len <= 0:
            return list(traj)
        return list(traj[-max_len:])

    def accept_merge(self, anchor_robot_id: int, source_robot_id: int, hypothesis: TransformHypothesis) -> None:
        self.merged_anchor_robot_id = int(anchor_robot_id)
        self.merged_source_robot_id = int(source_robot_id)
        self.accepted_hypothesis = hypothesis
        self._merged_dirty = True

    def is_merged(self) -> bool:
        return self.accepted_hypothesis is not None and self.merged_anchor_robot_id is not None

    def get_merged_map(self) -> LocalSubmap | None:
        if not self.is_merged():
            return None
        if self._merged_cache is not None and not self._merged_dirty:
            return self._merged_cache

        assert self.merged_anchor_robot_id is not None
        assert self.merged_source_robot_id is not None
        assert self.accepted_hypothesis is not None

        anchor = self.get_local_submap(self.merged_anchor_robot_id)
        source = self.get_local_submap(self.merged_source_robot_id)
        merged = anchor.clone()
        for cell, value in source.export_sparse_cells().items():
            dst = apply_transform(cell, self.accepted_hypothesis.rotation_deg, self.accepted_hypothesis.dx, self.accepted_hypothesis.dy)
            if not merged.in_bounds(dst):
                continue
            cur = merged.get_known(dst)
            if cur == UNKNOWN:
                merged.set_known(dst, value)
            elif cur != value and value == OCCUPIED:
                merged.set_known(dst, OCCUPIED)
        self._merged_cache = merged
        self._merged_dirty = False
        return merged

    def merge_with_transform(self, anchor_robot_id: int, source_robot_id: int, hypothesis: TransformHypothesis) -> LocalSubmap:
        self.accept_merge(anchor_robot_id, source_robot_id, hypothesis)
        merged = self.get_merged_map()
        assert merged is not None
        return merged
