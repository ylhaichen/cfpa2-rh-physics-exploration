from __future__ import annotations

import numpy as np

FREE = 0
OCCUPIED = 1


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _add_boundaries(grid: np.ndarray) -> None:
    grid[0, :] = OCCUPIED
    grid[-1, :] = OCCUPIED
    grid[:, 0] = OCCUPIED
    grid[:, -1] = OCCUPIED


def _carve_corridor(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], width: int = 2) -> None:
    x0, y0 = p0
    x1, y1 = p1
    if x0 == x1:
        x_min = max(1, x0 - width)
        x_max = min(grid.shape[1] - 2, x0 + width)
        y_min = min(y0, y1)
        y_max = max(y0, y1)
        grid[y_min : y_max + 1, x_min : x_max + 1] = FREE
    elif y0 == y1:
        y_min = max(1, y0 - width)
        y_max = min(grid.shape[0] - 2, y0 + width)
        x_min = min(x0, x1)
        x_max = max(x0, x1)
        grid[y_min : y_max + 1, x_min : x_max + 1] = FREE


def generate_corridor_maze(width: int, height: int, obstacle_density: float, seed: int) -> np.ndarray:
    rng = _rng(seed)
    grid = np.full((height, width), OCCUPIED, dtype=np.int8)
    _add_boundaries(grid)

    x_nodes = list(range(4, width - 4, max(8, width // 10)))
    y_nodes = list(range(4, height - 4, max(8, height // 8)))

    # Carve a connected corridor graph with bottlenecks and T-junction style turns.
    for y in y_nodes:
        _carve_corridor(grid, (x_nodes[0], y), (x_nodes[-1], y), width=1)

    for x in x_nodes:
        _carve_corridor(grid, (x, y_nodes[0]), (x, y_nodes[-1]), width=1)

    # Create dead-ends and side branches.
    for _ in range(max(6, len(x_nodes) + len(y_nodes))):
        x = int(rng.choice(x_nodes))
        y = int(rng.choice(y_nodes))
        length = int(rng.integers(4, 12))
        direction = int(rng.integers(0, 4))
        if direction == 0:
            _carve_corridor(grid, (x, y), (min(width - 2, x + length), y), width=1)
        elif direction == 1:
            _carve_corridor(grid, (x, y), (max(1, x - length), y), width=1)
        elif direction == 2:
            _carve_corridor(grid, (x, y), (x, min(height - 2, y + length)), width=1)
        else:
            _carve_corridor(grid, (x, y), (x, max(1, y - length)), width=1)

    # Add a few random internal blockers for bottleneck-heavy behavior.
    free_cells = np.argwhere(grid == FREE)
    n_blockers = int(obstacle_density * 0.15 * len(free_cells))
    if n_blockers > 0:
        picks = free_cells[rng.choice(len(free_cells), size=n_blockers, replace=False)]
        for y, x in picks:
            if x in (1, width - 2) or y in (1, height - 2):
                continue
            if rng.random() < 0.6:
                grid[y, x] = OCCUPIED

    # Guarantee a known-free launch zone.
    grid[2:10, 2:14] = FREE
    _add_boundaries(grid)
    return grid


def generate_bottleneck_rooms(width: int, height: int, obstacle_density: float, seed: int) -> np.ndarray:
    rng = _rng(seed)
    grid = np.full((height, width), FREE, dtype=np.int8)
    _add_boundaries(grid)

    room_w = max(10, width // 5)
    room_h = max(10, height // 4)

    v_walls = list(range(room_w, width - 1, room_w))
    h_walls = list(range(room_h, height - 1, room_h))

    for x in v_walls:
        grid[:, x] = OCCUPIED
    for y in h_walls:
        grid[y, :] = OCCUPIED

    # Narrow doors create realistic bottlenecks.
    for x in v_walls:
        for y0, y1 in zip([0] + h_walls, h_walls + [height - 1]):
            if y1 - y0 < 4:
                continue
            door_y = int(rng.integers(y0 + 2, y1 - 1))
            grid[door_y - 1 : door_y + 1, x] = FREE

    for y in h_walls:
        for x0, x1 in zip([0] + v_walls, v_walls + [width - 1]):
            if x1 - x0 < 4:
                continue
            door_x = int(rng.integers(x0 + 2, x1 - 1))
            grid[y, door_x - 1 : door_x + 1] = FREE

    interior = np.argwhere(grid[1:-1, 1:-1] == FREE)
    n_noise = int(obstacle_density * 0.10 * len(interior))
    if n_noise > 0:
        picks = interior[rng.choice(len(interior), size=n_noise, replace=False)]
        for y, x in picks:
            grid[y + 1, x + 1] = OCCUPIED

    grid[2:10, 2:16] = FREE
    _add_boundaries(grid)
    return grid


def generate_branching_deadend(width: int, height: int, obstacle_density: float, seed: int) -> np.ndarray:
    rng = _rng(seed)
    grid = np.full((height, width), OCCUPIED, dtype=np.int8)
    _add_boundaries(grid)

    # Depth-first maze backbone.
    if width % 2 == 0:
        width -= 1
    if height % 2 == 0:
        height -= 1

    local = np.full((height, width), OCCUPIED, dtype=np.int8)

    def neighbors(cx: int, cy: int):
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                yield nx, ny, dx, dy

    start = (1, 1)
    stack = [start]
    visited = {start}
    local[1, 1] = FREE
    while stack:
        cx, cy = stack[-1]
        cand = [(nx, ny, dx, dy) for nx, ny, dx, dy in neighbors(cx, cy) if (nx, ny) not in visited]
        if not cand:
            stack.pop()
            continue
        nx, ny, dx, dy = cand[int(rng.integers(0, len(cand)))]
        local[cy + dy // 2, cx + dx // 2] = FREE
        local[ny, nx] = FREE
        visited.add((nx, ny))
        stack.append((nx, ny))

    # Add sparse loops and dead-end blockers.
    walls = np.argwhere(local == OCCUPIED)
    n_open = max(1, int(0.02 * len(walls)))
    for y, x in walls[rng.choice(len(walls), size=n_open, replace=False)]:
        if 0 < x < width - 1 and 0 < y < height - 1:
            local[y, x] = FREE

    free_cells = np.argwhere(local == FREE)
    n_close = int(obstacle_density * 0.06 * len(free_cells))
    if n_close > 0:
        for y, x in free_cells[rng.choice(len(free_cells), size=n_close, replace=False)]:
            if x > 2 and y > 2:
                local[y, x] = OCCUPIED

    grid[: local.shape[0], : local.shape[1]] = local
    grid[2:10, 2:14] = FREE
    _add_boundaries(grid)
    return grid


def generate_map(map_type: str, width: int, height: int, obstacle_density: float, seed: int) -> np.ndarray:
    if map_type == "corridor_maze":
        return generate_corridor_maze(width, height, obstacle_density, seed)
    if map_type == "bottleneck_rooms":
        return generate_bottleneck_rooms(width, height, obstacle_density, seed)
    if map_type == "branching_deadend":
        return generate_branching_deadend(width, height, obstacle_density, seed)
    if map_type == "open":
        grid = np.full((height, width), FREE, dtype=np.int8)
        _add_boundaries(grid)
        return grid
    raise ValueError(f"Unsupported map_type: {map_type}")
