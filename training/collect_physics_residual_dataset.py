from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config import load_experiment_config
from core.predictor_features import build_physics_feature_vector
from core.types import RobotState
from simulators.grid_sim import GridSimulation


@dataclass
class _Snapshot:
    step_idx: int
    feature: np.ndarray
    pose: tuple[int, int]
    heading_deg: float
    robot_id: int
    teammate_distance: float
    local_obstacle_density: float
    goal: tuple[int, int] | None


class ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int, prefix: str):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1000, int(shard_size))
        self.prefix = prefix

        self.features: list[np.ndarray] = []
        self.targets: list[tuple[float, float]] = []
        self.robot_ids: list[int] = []
        self.step_ids: list[int] = []
        self.seed_ids: list[int] = []
        self.map_names: list[str] = []
        self.planner_names: list[str] = []
        self.scenario_tags: list[str] = []

        self.shard_count = 0
        self.total_samples = 0
        self.manifest_rows: list[dict[str, Any]] = []

    def add(
        self,
        feature: np.ndarray,
        target_dx: float,
        target_dy: float,
        robot_id: int,
        step_idx: int,
        seed: int,
        map_name: str,
        planner_name: str,
        scenario_tag: str,
        repeat: int = 1,
    ) -> None:
        n = max(1, int(repeat))
        for _ in range(n):
            self.features.append(feature.astype(np.float32))
            self.targets.append((float(target_dx), float(target_dy)))
            self.robot_ids.append(int(robot_id))
            self.step_ids.append(int(step_idx))
            self.seed_ids.append(int(seed))
            self.map_names.append(str(map_name))
            self.planner_names.append(str(planner_name))
            self.scenario_tags.append(str(scenario_tag))

            if len(self.features) >= self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self.features:
            return

        x = np.stack(self.features, axis=0).astype(np.float32)
        y = np.asarray(self.targets, dtype=np.float32)

        rid = np.asarray(self.robot_ids, dtype=np.int32)
        sid = np.asarray(self.step_ids, dtype=np.int32)
        seed_arr = np.asarray(self.seed_ids, dtype=np.int32)
        map_arr = np.asarray(self.map_names)
        planner_arr = np.asarray(self.planner_names)
        scenario_arr = np.asarray(self.scenario_tags)

        shard_path = self.output_dir / f"{self.prefix}_shard{self.shard_count:05d}.npz"
        np.savez_compressed(
            shard_path,
            X=x,
            y=y,
            robot_id=rid,
            step_idx=sid,
            seed=seed_arr,
            map_name=map_arr,
            planner_name=planner_arr,
            scenario_tag=scenario_arr,
        )

        self.manifest_rows.append(
            {
                "shard_path": str(shard_path),
                "num_samples": int(x.shape[0]),
                "input_dim": int(x.shape[1]),
            }
        )
        self.total_samples += int(x.shape[0])
        self.shard_count += 1

        self.features.clear()
        self.targets.clear()
        self.robot_ids.clear()
        self.step_ids.clear()
        self.seed_ids.clear()
        self.map_names.clear()
        self.planner_names.clear()
        self.scenario_tags.clear()

    def finalize(self) -> tuple[Path, Path]:
        self.flush()

        manifest_path = self.output_dir / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for row in self.manifest_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        meta_path = self.output_dir / "dataset_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_samples": self.total_samples,
                    "num_shards": self.shard_count,
                    "manifest_path": str(manifest_path),
                },
                f,
                indent=2,
                sort_keys=True,
            )

        return manifest_path, meta_path



def _angle_diff_deg(target: float, source: float) -> float:
    return ((float(target) - float(source) + 180.0) % 360.0) - 180.0



def _known_open_degree(map_mgr, cell: tuple[int, int]) -> int:
    from core.map_manager import FREE

    deg = 0
    for n in map_mgr.neighbors(cell, neighborhood=4):
        if map_mgr.in_bounds(n) and map_mgr.get_known(n) == FREE:
            deg += 1
    return deg



def _choose_hard_map_type(rng: np.random.Generator, map_types: list[str]) -> str:
    if not map_types:
        return "narrow_t_branches"
    idx = int(rng.integers(0, len(map_types)))
    return str(map_types[idx])



def _apply_hard_scenario_config(cfg: dict, hard_map_type: str, episode_seed: int) -> str:
    env = cfg["environment"]
    robots = cfg["robots"]

    env["map_type"] = str(hard_map_type)
    env["map_name"] = f"hard_{hard_map_type}"

    # Keep map size if provided, but enforce realistic minimums for hard scenarios.
    env["map_width"] = max(72, int(env.get("map_width", 96)))
    env["map_height"] = max(56, int(env.get("map_height", 72)))

    if hard_map_type in ("narrow_t_branches", "sharp_turn_corridor", "interaction_cross"):
        env["obstacle_density"] = max(0.05, float(env.get("obstacle_density", 0.0)))
    elif hard_map_type in ("bottleneck_rooms", "branching_deadend"):
        env["obstacle_density"] = max(0.04, float(env.get("obstacle_density", 0.0)))

    w = int(env["map_width"])
    h = int(env["map_height"])
    cx = w // 2
    cy = h // 2

    starts = list(robots.get("start_positions", [[4, 4], [8, 4]]))
    headings = list(robots.get("start_headings_deg", [0.0, 0.0]))

    if hard_map_type == "interaction_cross":
        starts = [[max(3, cx - 10), cy], [min(w - 4, cx + 10), cy]]
        headings = [0.0, 180.0]
    elif hard_map_type == "sharp_turn_corridor":
        starts = [[max(3, w // 5), max(3, h - 8)], [max(4, w // 5 + 4), max(3, h - 8)]]
        headings = [0.0, 0.0]
    elif hard_map_type == "narrow_t_branches":
        starts = [[cx - 2, max(3, h - 8)], [cx + 2, max(3, h - 8)]]
        headings = [90.0, 90.0]
    elif hard_map_type == "bottleneck_rooms":
        starts = [[4, 4], [8, 4]]
        headings = [0.0, 0.0]
    elif hard_map_type == "branching_deadend":
        starts = [[3, 3], [5, 3]]
        headings = [0.0, 0.0]

    robots["start_positions"] = starts
    robots["start_headings_deg"] = headings

    # Bias toward near-obstacle slowdown and interaction phenomena.
    robots["slowdown_near_obstacle"] = True
    robots["obstacle_slowdown_distance"] = max(1, int(robots.get("obstacle_slowdown_distance", 1)))
    robots["obstacle_slowdown_prob"] = max(0.30, float(robots.get("obstacle_slowdown_prob", 0.2)))
    robots["motion_uncertainty_prob"] = max(0.02, float(robots.get("motion_uncertainty_prob", 0.0)))

    return str(env["map_name"])


class PhysicsResidualSampleCollector:
    def __init__(
        self,
        cfg: dict,
        writer: ShardWriter,
        map_name: str,
        seed: int,
        planner_name: str,
        scenario_tag: str,
        sharp_turn_threshold_deg: float,
        near_obstacle_density_threshold: float,
        interaction_distance_threshold: float,
        max_repeat_factor: int,
    ):
        self.cfg = cfg
        self.writer = writer
        self.map_name = map_name
        self.seed = int(seed)
        self.planner_name = planner_name
        self.scenario_tag = str(scenario_tag)

        self.prev: dict[int, _Snapshot] = {}
        self.max_speed = float(cfg["robots"].get("max_speed_cells_per_step", 1.0))
        self.patch_radius = int(cfg.get("predictor", {}).get("physics_residual", {}).get("occupancy_patch_radius", 4))

        self.sharp_turn_threshold_deg = float(sharp_turn_threshold_deg)
        self.near_obstacle_density_threshold = float(near_obstacle_density_threshold)
        self.interaction_distance_threshold = float(interaction_distance_threshold)
        self.max_repeat_factor = max(1, int(max_repeat_factor))

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "step_begin":
            self._on_step_begin(payload)

    def _transition_tags(self, prev: _Snapshot, cur: RobotState, map_mgr) -> list[str]:
        tags: list[str] = []

        yaw_change = abs(_angle_diff_deg(cur.heading_deg, prev.heading_deg))
        if yaw_change >= self.sharp_turn_threshold_deg:
            tags.append("sharp_turn")

        if prev.local_obstacle_density >= self.near_obstacle_density_threshold:
            tags.append("near_obstacle")

        if prev.teammate_distance <= self.interaction_distance_threshold:
            tags.append("interaction")

        open_deg = _known_open_degree(map_mgr, prev.pose)
        if open_deg <= 1:
            tags.append("dead_end")
        elif open_deg == 2 and prev.local_obstacle_density >= 0.20:
            tags.append("bottleneck")

        if self.scenario_tag.startswith("hard_"):
            tags.append(self.scenario_tag)

        if not tags:
            tags.append("default")
        return sorted(set(tags))

    def _on_step_begin(self, payload: dict[str, Any]) -> None:
        step_idx = int(payload["step_idx"])
        map_mgr = payload["map_mgr"]
        robots: list[RobotState] = payload["robots"]
        assignments = payload["assignments"]

        by_id = {r.robot_id: r for r in robots}

        # Close previous transition with current pose as next-state label.
        for rid, prev in list(self.prev.items()):
            cur = by_id.get(rid)
            if cur is None:
                continue
            dx = float(cur.pose[0] - prev.pose[0])
            dy = float(cur.pose[1] - prev.pose[1])

            tags = self._transition_tags(prev, cur, map_mgr)
            hard_tags = [t for t in tags if t != "default"]
            repeat = min(self.max_repeat_factor, 1 + len(hard_tags))
            tag_join = "|".join(tags)

            self.writer.add(
                feature=prev.feature,
                target_dx=dx,
                target_dy=dy,
                robot_id=rid,
                step_idx=prev.step_idx,
                seed=self.seed,
                map_name=self.map_name,
                planner_name=self.planner_name,
                scenario_tag=tag_join,
                repeat=repeat,
            )

        # Record new snapshot for next step.
        next_prev: dict[int, _Snapshot] = {}
        for r in robots:
            teammate = None
            for rr in robots:
                if rr.robot_id != r.robot_id:
                    teammate = rr
                    break

            goal = None
            a = assignments.get(r.robot_id)
            if a is not None and a.valid:
                goal = a.target

            feat = build_physics_feature_vector(
                map_mgr=map_mgr,
                robot=r,
                teammate=teammate,
                goal=goal,
                patch_radius=self.patch_radius,
                max_speed=self.max_speed,
            )

            teammate_distance = 10.0
            if teammate is not None:
                teammate_distance = float(math.hypot(teammate.pose[0] - r.pose[0], teammate.pose[1] - r.pose[1]))

            local_obstacle_density = float(map_mgr.obstacle_count_around(r.pose, radius=1)) / 9.0

            next_prev[r.robot_id] = _Snapshot(
                step_idx=step_idx,
                feature=feat,
                pose=tuple(r.pose),
                heading_deg=float(r.heading_deg),
                robot_id=int(r.robot_id),
                teammate_distance=float(teammate_distance),
                local_obstacle_density=float(local_obstacle_density),
                goal=goal,
            )

        self.prev = next_prev



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect large-scale dataset for Physics-RH-CFPA2 residual predictor")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--planner-config", type=str, default="configs/planner_rh_cfpa2.yaml")
    parser.add_argument("--env-configs", nargs="+", default=["configs/env_maze.yaml", "configs/env_go2w_like.yaml"])
    parser.add_argument("--planner-name", type=str, default="rh_cfpa2", choices=["cfpa2", "rh_cfpa2", "physics_rh_cfpa2"])
    parser.add_argument("--predictor-type", type=str, default="path_follow", choices=["path_follow", "constant_velocity", "physics_residual"])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--task-index", type=int, default=0, help="Distributed collection: current task index")
    parser.add_argument("--num-tasks", type=int, default=1, help="Distributed collection: total tasks")
    parser.add_argument("--shard-size", type=int, default=200000)
    parser.add_argument("--output-dir", type=str, default="training/datasets/physics_residual_dataset")

    parser.add_argument("--hard-scenario-oversample-prob", type=float, default=0.70)
    parser.add_argument(
        "--hard-scenario-map-types",
        nargs="+",
        default=[
            "sharp_turn_corridor",
            "narrow_t_branches",
            "bottleneck_rooms",
            "interaction_cross",
            "branching_deadend",
        ],
    )
    parser.add_argument("--sharp-turn-threshold-deg", type=float, default=55.0)
    parser.add_argument("--near-obstacle-density-threshold", type=float, default=0.22)
    parser.add_argument("--interaction-distance-threshold", type=float, default=4.0)
    parser.add_argument("--max-repeat-factor", type=int, default=3)

    return parser.parse_args()



def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_tag = f"task{int(args.task_index):03d}_of_{int(args.num_tasks):03d}"
    writer = ShardWriter(output_dir=output_dir / task_tag, shard_size=args.shard_size, prefix="physics_residual")

    tasks: list[tuple[str, int, int]] = []
    for env_idx, env_cfg in enumerate(args.env_configs):
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            for ep in range(max(1, int(args.episodes_per_seed))):
                # episode-specific seed keeps each rollout distinct
                ep_seed = int(seed + 10000 * ep + 100000 * env_idx)
                tasks.append((env_cfg, ep_seed, ep))

    selected = [t for i, t in enumerate(tasks) if i % max(1, int(args.num_tasks)) == int(args.task_index)]

    sim = GridSimulation()
    run_count = 0

    for env_cfg_path, episode_seed, ep_idx in selected:
        cfg = load_experiment_config(args.base_config, planner_cfg_path=args.planner_config, env_cfg_path=env_cfg_path)
        cfg["planning"]["planner_name"] = args.planner_name
        cfg["predictor"]["type"] = args.predictor_type
        cfg["termination"]["max_steps"] = int(args.max_steps)
        cfg["experiment"]["save_animation"] = False
        cfg["experiment"]["enable_live_plot"] = False

        rng = np.random.default_rng(int(episode_seed) + 7919)
        scenario_tag = "default"
        if args.hard_scenario_oversample_prob > 0.0 and rng.random() < float(args.hard_scenario_oversample_prob):
            hard_map_type = _choose_hard_map_type(rng, [str(v) for v in args.hard_scenario_map_types])
            scenario_tag = _apply_hard_scenario_config(cfg, hard_map_type=hard_map_type, episode_seed=episode_seed)

        map_name = cfg["environment"].get("map_name", cfg["environment"].get("map_type", "map"))
        stem = f"collect_{args.planner_name}_{map_name}_seed{episode_seed}_ep{ep_idx}"

        collector = PhysicsResidualSampleCollector(
            cfg=cfg,
            writer=writer,
            map_name=map_name,
            seed=episode_seed,
            planner_name=args.planner_name,
            scenario_tag=scenario_tag,
            sharp_turn_threshold_deg=float(args.sharp_turn_threshold_deg),
            near_obstacle_density_threshold=float(args.near_obstacle_density_threshold),
            interaction_distance_threshold=float(args.interaction_distance_threshold),
            max_repeat_factor=int(args.max_repeat_factor),
        )

        sim.run_episode(
            cfg=cfg,
            planner_name=args.planner_name,
            seed=episode_seed,
            output_dir=output_dir / task_tag / "episodes" / stem,
            animation_stem=stem,
            sample_callback=collector.on_event,
        )

        run_count += 1
        if run_count % 10 == 0:
            print(
                f"[collector] task={task_tag} episodes={run_count} "
                f"samples_so_far={writer.total_samples + len(writer.features)}",
                flush=True,
            )

    manifest_path, meta_path = writer.finalize()

    print("=== Dataset Collection Done ===")
    print(f"task: {task_tag}")
    print(f"episodes_run: {run_count}")
    print(f"manifest: {manifest_path}")
    print(f"meta: {meta_path}")
    print(f"total_samples: {writer.total_samples}")


if __name__ == "__main__":
    main()
