#!/usr/bin/env python3
"""Diagnostic controllers for the single TRAINING map, never training examples.

waypoints uses a manually supplied map-specific route, so its success is only
evidence of physical task feasibility, NOT a learned or deployable baseline.
frontier uses only the robot's own known graph/utility and no remote truth.
"""
import argparse
import heapq
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ir2-finite-contact-matplotlib")
import numpy as np
from finite_contact.config import Config
from finite_contact.environment import Environment


def paths(coords, edges):
    lookup = {tuple(p): i for i, p in enumerate(coords)}
    distances = np.full(len(coords), np.inf)
    first = np.zeros(len(coords), dtype=int)
    distances[0] = 0.
    heap = [(0., 0)]
    while heap:
        distance, i = heapq.heappop(heap)
        if distance > distances[i]:
            continue
        for point in edges.get(tuple(coords[i]), {}):
            j = lookup[point]
            cost = distance + np.linalg.norm(coords[i] - coords[j])
            if i != j and cost < distances[j]:
                distances[j] = cost
                first[j] = j if i == 0 else first[i]
                heapq.heappush(heap, (cost, j))
    return distances, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["frontier", "waypoints"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != Path("/home/wyz/anaconda3/envs/ir2_train"):
        parser.error("Use ir2_train")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    cfg = Config()
    env = Environment(cfg)
    waypoints = np.array([[59, 98], [59, 157], [59, 216], [98, 59],
                          [137, 59], [196, 59], [196, 118], [196, 177],
                          [196, 216], [137, 216], [137, 157], [196, 118]])
    waypoint = [0, 0]
    history = []
    done = truncated = False
    while not (done or truncated):
        actions, diagnostics = [], []
        for robot in range(2):
            coords, edges, utility, visited = env.graph_data[robot]
            distance, first = paths(coords, edges)
            reachable = np.isfinite(distance)
            reachable[0] = False
            obs = env.observation(robot)
            candidates = obs[1][0]
            valid = obs[4][0] == 0
            if args.controller == "waypoints":
                if np.linalg.norm(env.positions[robot] - waypoints[waypoint[robot]]) < 16:
                    waypoint[robot] = min(waypoint[robot] + 1, len(waypoints) - 1)
                goal = waypoints[waypoint[robot]]
                score = -np.linalg.norm(coords - goal, axis=1)
            else:
                score = utility / (distance + 20.)
                score -= .01 * np.minimum(visited.ravel(), 1)
            score[~reachable] = -np.inf
            destination = int(score.argmax())
            slot = np.flatnonzero(valid & (candidates == first[destination]))
            if len(slot):
                action = int(slot[0])
            else:
                action = int(np.flatnonzero(valid)[0])
            actions.append(action)
            diagnostics.append({"nodes": len(coords), "reachable": int(reachable.sum()),
                                "max_utility": int(utility.max()), "waypoint": waypoint[robot]})
        reward, duration, done, truncated = env.step(actions)
        local, union = env.coverage()
        history.append({"time": env.time, "coverage": local.tolist(), "union_coverage": union,
                        "positions": env.positions.tolist(), "diagnostics": diagnostics})
    metrics = {"controller": args.controller, "privileged_waypoints": args.controller == "waypoints",
               "config": cfg.to_dict(), "map": env.map_info, **env.metrics(done)}
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.run_dir / "trajectory.json").write_text(json.dumps(history), encoding="utf-8")
    np.savez_compressed(args.run_dir / "maps.npz", truth=env.truth, beliefs=np.stack(env.beliefs))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
