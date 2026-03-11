from __future__ import annotations

import heapq
import math

from .map_manager import MapManager
from .types import Cell


def _heuristic(a: Cell, b: Cell, neighborhood: int) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if neighborhood == 4:
        return float(dx + dy)
    return float((dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy))


def _neighbors(cell: Cell, neighborhood: int) -> list[tuple[Cell, float]]:
    x, y = cell
    if neighborhood == 4:
        return [((x + 1, y), 1.0), ((x - 1, y), 1.0), ((x, y + 1), 1.0), ((x, y - 1), 1.0)]
    c = math.sqrt(2.0)
    return [
        ((x + 1, y), 1.0),
        ((x - 1, y), 1.0),
        ((x, y + 1), 1.0),
        ((x, y - 1), 1.0),
        ((x + 1, y + 1), c),
        ((x + 1, y - 1), c),
        ((x - 1, y + 1), c),
        ((x - 1, y - 1), c),
    ]


def astar_path(
    map_mgr: MapManager,
    start: Cell,
    goal: Cell,
    neighborhood: int = 8,
    clearance_cells: int = 0,
) -> list[Cell] | None:
    if not map_mgr.in_bounds(start) or not map_mgr.in_bounds(goal):
        return None
    if not map_mgr.is_traversable(start, clearance_cells):
        return None
    if not map_mgr.is_traversable(goal, clearance_cells):
        return None
    if start == goal:
        return [start]

    open_heap: list[tuple[float, Cell]] = [(0.0, start)]
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    closed: set[Cell] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)

        closed.add(current)

        for nxt, step_cost in _neighbors(current, neighborhood):
            if not map_mgr.in_bounds(nxt):
                continue
            if not map_mgr.is_traversable(nxt, clearance_cells):
                continue

            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + _heuristic(nxt, goal, neighborhood)
                heapq.heappush(open_heap, (f, nxt))

    return None


def _reconstruct(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_cost(path: list[Cell] | None) -> float:
    if not path or len(path) <= 1:
        return 0.0 if path else float("inf")
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dx == 1 and dy == 1:
            total += math.sqrt(2.0)
        else:
            total += 1.0
    return total


def heading_delta_cost(heading_deg: float, path: list[Cell]) -> float:
    if len(path) < 2:
        return 0.0
    x0, y0 = path[0]
    x1, y1 = path[1]
    target_heading = math.degrees(math.atan2(y1 - y0, x1 - x0))
    diff = abs((target_heading - heading_deg + 180.0) % 360.0 - 180.0)
    return diff / 180.0
