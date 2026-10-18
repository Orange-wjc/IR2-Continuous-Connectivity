"""Small finite-capacity environment. Never calls IR2's instantaneous map merger."""
import heapq
import numpy as np
from skimage.measure import block_reduce

import parameter  # initializes the upstream TRAINING flag before graph imports
from env import Env as LegacyEnv
from graph_generator import Graph_generator
from sensor import sensor_work
from .communication import Radio, MAP
from .maps import configured_map
from .sensing import sensor_work_vectorized


class Environment:
    def __init__(self, config, split="train", map_index=0, scorer=None):
        config.validate()
        self.config = config
        self.truth, self.positions, self.map_info = configured_map(config, split, map_index)
        self.beliefs = [np.full(self.truth.shape, 127, np.uint8) for _ in range(2)]
        self.resolution = 4
        self.time = 0.0
        self.wait = np.zeros(2)
        self.distance = np.zeros(2)
        self.wait_seconds = np.zeros(2)
        self.wait_disconnected_seconds = np.zeros(2)
        self.wait_without_new_map_seconds = np.zeros(2)
        self.sync_idle_seconds = np.zeros(2)
        self.coverage_deficit = 0.0
        self.connected_seconds = 0.0
        self.union_completion_time = None
        self.reward_components = {name: np.zeros(2) for name in
                                  ("exploration", "delivery", "time", "distance", "completion", "frontier_shaping")}
        self.radio = Radio(self.truth.shape, config)
        self.scorer = scorer
        self.graphs = []
        self.graph_data = [None, None]
        self.feature_provider = None
        self._rank_cache = {}
        self.trajectory = [{"time": 0.0, "positions": self.positions.tolist()}]
        for robot in range(2):
            graph = Graph_generator(robot, self.truth.shape, config.neighbors,
                                    config.sensor_range, "synthetic_simple", plot=False)
            axis = np.linspace(0, config.map_size - 1, config.grid_points).round().astype(int)
            graph.uniform_points = np.array([(x, y) for x in axis for y in axis])
            graph.route_node = [self.positions[robot].copy()]
            self.graphs.append(graph)
            self.sense(robot)
        # Both starts are agreed mission configuration, not live free pose updates.
        for robot in range(2):
            self.radio.peer_pose[robot] = np.r_[self.positions[1 - robot], [0., 0.]]
            self.radio.peer_pose_time[robot] = 0.0
            self.radio.enqueue_updates(robot, self.beliefs[robot], self.time)
        self.refresh_graphs()
        local, union = self.coverage()
        self.trajectory[0].update({"coverage": local.tolist(), "union_coverage": union})

    def sense(self, robot):
        sense = sensor_work_vectorized if self.config.sensor_implementation == "vectorized" else sensor_work
        self.beliefs[robot] = sense(np.rint(self.positions[robot]).astype(int), self.config.sensor_range,
                                   self.beliefs[robot], self.truth)

    def frontiers(self, robot):
        downsampled = block_reduce(self.beliefs[robot], (4, 4), np.min)
        return LegacyEnv.find_frontier(self, downsampled)

    def refresh_graphs(self):
        for robot in range(2):
            graph = self.graphs[robot]
            graph.nodes_list = []  # generate_graph is rebuilt from local received knowledge only
            current = np.rint(self.positions[robot]).astype(int)
            # Exclude duplicate current node in the uniform lattice.
            saved = graph.uniform_points
            graph.uniform_points = saved[np.any(saved != current, axis=1)]
            coords, edges, utility, visited = graph.generate_graph(
                current, self.beliefs[robot], self.frontiers(robot))
            graph.uniform_points = saved
            self.graph_data[robot] = (coords, edges, utility, visited)

    def packet_features(self, sender, packets):
        """Only sender's local map, received poses and confirmed knowledge are inputs."""
        if not packets:
            return np.zeros((0, 12), dtype=np.float32), np.zeros(0, dtype=int)
        coords, _, utility, _ = self.graph_data[sender]
        peer = self.radio.peer_pose[sender]
        age = self.time - self.radio.peer_pose_time[sender]
        frontier = self.frontiers(sender)
        rows, node_indices = [], []
        for packet in packets:
            cells = packet.cells()
            ids = cells["index"].astype(int)
            center = np.array([(ids % self.config.map_size).mean(),
                               (ids // self.config.map_size).mean()])
            index = int(np.argmin(np.linalg.norm(coords - center, axis=1)))
            known = self.radio.known_peer[(sender, 1 - sender)].ravel()[ids]
            unknown_fraction = float(np.mean(known == 127))
            near_frontier = 0.0 if len(frontier) == 0 else float(
                np.min(np.linalg.norm(frontier - center, axis=1)) < 24)
            rows.append([
                len(cells) / 100., np.mean(cells["value"] == 255),
                center[0] / 256., center[1] / 256., unknown_fraction,
                np.linalg.norm(center - peer[:2]) / 256., min(age / 60., 4.),
                min((self.time - packet.created) / 60., 4.),
                packet.size / 512., near_frontier, utility[index] / 50.,
                np.linalg.norm(center - self.positions[sender]) / 256.])
            node_indices.append(index)
        return np.array(rows, dtype=np.float32), np.array(node_indices, dtype=int)

    def packet_scores(self, sender, packets, mode=None):
        mode = mode or self.config.content
        if mode == "fifo":
            # FIFO novelty needs no frontier search or graph-to-packet matching.
            # Retain the original float32 fraction before multiplying, including
            # its rounding, so historical policies see exactly the same scores.
            known = self.radio.known_peer[(sender, 1 - sender)].ravel()
            return np.array([len(p.cells()) * np.float32(np.mean(
                known[p.cells()["index"]] == 127)) for p in packets])
        features, indices = self.packet_features(sender, packets)
        if mode == "learned":
            if self.scorer is None or self.feature_provider is None:
                raise RuntimeError("Learned content requires a trained scorer and local graph features")
            context = self.feature_provider(sender, indices)
            scores = np.maximum(0., self.scorer(features, context)) * 100.
            if self.config.content_score_bound == "sender":
                known = self.radio.known_peer[(sender, 1 - sender)].ravel()
                upper = np.array([np.count_nonzero(known[p.cells()["index"]] == 127) for p in packets])
                scores = np.minimum(scores, upper)
            return scores
        # Strong rule: estimated novelty with a modest local-frontier preference.
        return np.array([len(p.cells()) for p in packets]) * features[:, 4] * (1. + .25 * features[:, 9])

    def rank_packet(self, sender, packets):
        if self.config.content == "fifo":
            return 0
        key = (self.radio.tick_index, sender)
        if key not in self._rank_cache:
            scores = self.packet_scores(sender, packets)
            self._rank_cache = {k: v for k, v in self._rank_cache.items()
                                if k[0] == self.radio.tick_index}
            self._rank_cache[key] = {p.seq: score / p.size for p, score in zip(packets, scores)}
        scores = self._rank_cache[key]
        return max(range(len(packets)), key=lambda i: scores[packets[i].seq])

    def collect_labels(self, sender):
        """Privileged training labels. No labels are passed to ranking or observations."""
        packets = self.radio.pending(sender)
        if self.config.label_sampling == "spread" and len(packets) > 64:
            packets = [packets[i] for i in np.linspace(0, len(packets) - 1, 64).astype(int)]
        else:
            packets = packets[:64]
        if not packets or self.feature_provider is None:
            return []
        features, indices = self.packet_features(sender, packets)
        context = self.feature_provider(sender, indices)
        target = self.beliefs[1 - sender].ravel()
        labels = np.array([np.count_nonzero(target[p.cells()["index"]] == 127) / 100.
                           for p in packets], dtype=np.float32)
        return [(f, c, y) for f, c, y in zip(features, context, labels)]

    def expected_delivery(self, sender):
        packets = self.radio.pending(sender)
        active_map = None
        if self.config.delivery_queue_order == "service":
            pair = (sender, 1 - sender)
            active = self.radio.active[pair]
            if active is not None and active.kind == MAP:
                active_map = active
            known = self.radio.known_peer[pair].ravel()
            queued = [p for p in self.radio.queues[pair] if p.offset > 0 or not np.all(
                known[p.cells()["index"]] == p.cells()["value"])]
            packets = ([active_map] if active_map is not None else []) + queued
        if not packets:
            return 0.0
        seconds = self.config.estimated_contact_seconds
        peer = self.radio.peer_pose[sender]
        relative_speed = np.linalg.norm(peer[2:])
        if relative_speed > 1e-3:
            seconds = min(seconds, 2 * self.config.communication_range / relative_speed)
        # Estimate control overhead, not the true future delivery budget.
        budget = max(0., self.config.bytes_per_second * seconds - 64.)
        mode = self.config.content if self.config.coupled else "fifo"
        scores = self.packet_scores(sender, packets, mode)
        order = list(range(len(packets)))
        if mode != "fifo":
            fixed = [0] if active_map is not None else []
            order = fixed + sorted(order[len(fixed):], key=lambda i: scores[i] / packets[i].size, reverse=True)
        gain = 0.0
        for i in order:
            remaining = packets[i].size - packets[i].offset
            if remaining <= budget:
                budget -= remaining
                gain += scores[i]
            elif budget > 0:
                # Conservative: an incomplete small packet yields no immediately usable map.
                break
        age = max(0., self.time - self.radio.peer_pose_time[sender])
        return gain * np.exp(-age / 60.)

    def action_inputs(self, robot):
        """Exact action candidates/mask without constructing unused map features."""
        cfg = self.config
        coords, edges, _, _ = self.graph_data[robot]
        if len(coords) > cfg.nodes:
            raise RuntimeError("Node padding overflow; episode is not silently discarded")
        lookup = {tuple(p): i for i, p in enumerate(coords)}
        neighbors = [0] + [lookup[p] for p in edges.get(tuple(coords[0]), {}) if lookup[p] != 0]
        neighbors = neighbors[:cfg.neighbors]
        action_mask = np.ones((1, cfg.neighbors), np.int64)
        action_mask[0, :len(neighbors)] = 0
        if self.wait[robot] >= cfg.max_wait_seconds and len(neighbors) > 1:
            action_mask[0, 0] = 1
        candidates = np.zeros((1, cfg.neighbors), np.int64)
        candidates[0, :len(neighbors)] = neighbors

        return candidates, action_mask

    def observation(self, robot, include_rendezvous=True):
        cfg = self.config
        coords, edges, utility, visited = self.graph_data[robot]
        count = len(coords)
        if count > cfg.nodes:
            raise RuntimeError("Node padding overflow; episode is not silently discarded")
        lookup = {tuple(p): i for i, p in enumerate(coords)}
        adjacency = np.ones((cfg.nodes, cfg.nodes), np.uint8)
        adjacency[np.arange(cfg.nodes), np.arange(cfg.nodes)] = 0
        for i, point in enumerate(coords):
            for destination in edges.get(tuple(point), {}):
                adjacency[i, lookup[destination]] = 0
        candidates, action_mask = self.action_inputs(robot)

        peer = self.radio.peer_pose[robot]
        age = max(0., self.time - self.radio.peer_pose_time[robot])
        pending = self.radio.pending(robot)
        if not include_rendezvous:
            benefit = 0.
        elif cfg.variant in {"A", "AB"}:
            benefit = self.expected_delivery(robot) / 500.
        else:
            known_peer = self.radio.known_peer[(robot, 1 - robot)]
            benefit = np.count_nonzero((self.beliefs[robot] != 127) & (known_peer == 127)) / 500.
        # Back-propagate value along locally reachable graph paths toward the last known peer.
        destination = int(np.argmin(np.linalg.norm(coords - peer[:2], axis=1)))
        reverse = [[] for _ in range(count)]
        for i in range(count):
            for j in np.flatnonzero(adjacency[i, :count] == 0):
                if i != j:
                    reverse[j].append((i, np.linalg.norm(coords[i] - coords[j])))
        distances = np.full(count, np.inf)
        distances[destination] = 0.
        heap = [(0., destination)]
        while heap:
            distance, i = heapq.heappop(heap)
            if distance > distances[i]:
                continue
            for j, length in reverse[i]:
                cost = distance + length
                if cost < distances[j]:
                    distances[j] = cost
                    heapq.heappush(heap, (cost, j))
        rendezvous = np.clip(benefit, 0, 4) * np.exp(-distances / 100.)
        if cfg.variant in {"A", "AB"}:
            rendezvous = np.maximum(0., rendezvous - .002 * distances / cfg.speed)
        occupancy = np.zeros(count)
        occupancy[0] = -1
        if age < 30 and np.min(np.linalg.norm(coords - peer[:2], axis=1)) < 20:
            occupancy[destination] = 1 if destination != 0 else -1
        queue = min(sum(p.size - p.offset for p in pending) / 10000., 4.)
        constant = np.ones(count)
        features = np.column_stack([
            coords[:, 0] / 256., coords[:, 1] / 256., utility / 50., rendezvous,
            np.minimum(visited.ravel(), 1), occupancy, constant * queue,
            constant * min(age / 60., 4.), constant * (cfg.bytes_per_second / 1024.),
            constant * min(self.wait[robot] / cfg.max_wait_seconds, 2.),
            constant * (np.linalg.norm(self.positions[robot] - peer[:2]) / 256.),
            constant * min(self.time / cfg.horizon_seconds, 1.)]).astype(np.float32)
        padded = np.zeros((cfg.nodes, cfg.input_dim), np.float32)
        padded[:count] = features
        node_mask = np.ones((1, cfg.nodes), np.int64)
        node_mask[0, :count] = 0
        return (padded, candidates, np.array([[0]], np.int64), node_mask, action_mask, adjacency)

    def coverage(self):
        free = self.truth == 255
        local = np.array([np.count_nonzero((b == 255) & free) / free.sum() for b in self.beliefs])
        union = np.count_nonzero(((self.beliefs[0] == 255) | (self.beliefs[1] == 255)) & free) / free.sum()
        return local, float(union)

    def frontier_potential(self, robot):
        """Negative distance to useful local graph nodes; no remote state or truth."""
        scale = self.config.frontier_shaping_scale
        if scale == 0:
            return 0.
        coords, edges, utility, _ = self.graph_data[robot]
        lookup = {tuple(point): i for i, point in enumerate(coords)}
        distances = np.full(len(coords), np.inf)
        distances[0] = 0.
        heap = [(0., 0)]
        while heap:
            distance, i = heapq.heappop(heap)
            if distance > distances[i]:
                continue
            if utility[i] > 0:
                return -scale * min(distance / self.config.map_size, 1.)
            for destination in edges.get(tuple(coords[i]), {}):
                j = lookup[destination]
                cost = distance + np.linalg.norm(coords[i] - coords[j])
                if cost < distances[j]:
                    distances[j] = cost
                    heapq.heappush(heap, (cost, j))
        return 0.

    def step(self, actions):
        """Synchronous graph decisions, with speed-limited substeps and explicit idle accounting."""
        cfg = self.config
        potential_before = np.array([self.frontier_potential(i) for i in range(2)])
        start = self.positions.copy()
        novel_before = self.radio.unique_cells
        connected_before = self.connected_seconds
        targets = start.copy()
        for robot, action in enumerate(actions):
            # Validation only needs candidate indices and masks. Recomputing
            # rendezvous would rescore the entire packet queue without using it.
            candidates, action_mask = self.action_inputs(robot)
            if not 0 <= action < cfg.neighbors or action_mask[0, action]:
                raise ValueError("Invalid or masked movement action")
            targets[robot] = self.graph_data[robot][0][candidates[0, action]]
        lengths = np.linalg.norm(targets - start, axis=1)
        travel_times = lengths / cfg.speed
        duration = min(max(cfg.dt, float(travel_times.max())), cfg.horizon_seconds - self.time)
        rewards = np.zeros(2)
        elapsed = 0.
        terminated = False
        while elapsed < duration - 1e-8:
            tick = min(cfg.dt, duration - elapsed)
            before_local, before_union = self.coverage()
            old_positions = self.positions.copy()
            fraction = np.minimum(1., (elapsed + tick) / np.maximum(travel_times, 1e-9))
            self.positions = start + (targets - start) * fraction[:, None]
            moved = np.linalg.norm(self.positions - old_positions, axis=1)
            velocity = (self.positions - old_positions) / tick
            self.distance += moved
            for robot in range(2):
                self.sense(robot)
                self.radio.enqueue_updates(robot, self.beliefs[robot], self.time)
                idle = max(0., elapsed + tick - max(elapsed, travel_times[robot]))
                if lengths[robot] == 0:
                    self.wait[robot] += tick
                    self.wait_seconds[robot] += tick
                else:
                    self.wait[robot] = 0.
                    self.sync_idle_seconds[robot] += idle
            # Check an entire substep, conservatively: both endpoints must be connected.
            connected = (np.linalg.norm(old_positions[0] - old_positions[1]) <= cfg.communication_range
                         and np.linalg.norm(self.positions[0] - self.positions[1]) <= cfg.communication_range)
            gain = self.radio.service(self.positions.copy(), velocity, self.time, tick,
                                       connected, self.beliefs, self.rank_packet)
            for robot in range(2):
                if lengths[robot] == 0:
                    self.wait_disconnected_seconds[robot] += tick * (not connected)
                    self.wait_without_new_map_seconds[robot] += tick * (gain.sum() == 0)
            self.connected_seconds += tick * connected
            self.time += tick
            elapsed += tick
            local, union = self.coverage()
            discount = cfg.gamma ** ((elapsed - tick) / cfg.dt)
            progress = ((local.mean() - before_local.mean()) if cfg.coverage_reward == "local_mean"
                        else union - before_union)
            components = {"exploration": np.full(2, progress * 20.),
                          "delivery": cfg.delivery_reward * gain / 500.,
                          "time": np.full(2, -cfg.time_penalty * tick),
                          "distance": -.002 * moved}
            for name, value in components.items():
                rewards += discount * value
                self.reward_components[name] += discount * value
            self.coverage_deficit += (1. - float(local.min())) * tick
            if union >= cfg.completion and self.union_completion_time is None:
                self.union_completion_time = self.time
            terminated = bool(np.all(local >= cfg.completion))
            if terminated:
                rewards += discount * 20.
                self.reward_components["completion"] += discount * 20.
                break
        # Decisions terminate at a node, unless a true termination/time limit interrupts an edge.
        for robot in range(2):
            self.graphs[robot].route_node.append(np.rint(self.positions[robot]).astype(int))
        self.refresh_graphs()
        local, union = self.coverage()
        self.trajectory.append({"time": self.time, "positions": self.positions.tolist(),
                                "actions": actions, "requested_lengths": lengths.tolist(),
                                "duration": elapsed, "coverage": local.tolist(), "union_coverage": union,
                                "novel_received_cells": self.radio.unique_cells - novel_before,
                                "connected_seconds": self.connected_seconds - connected_before})
        truncated = self.time >= cfg.horizon_seconds - 1e-8 and not terminated
        potential_after = (np.zeros(2) if terminated or truncated else
                           np.array([self.frontier_potential(i) for i in range(2)]))
        shaping = cfg.gamma ** (elapsed / cfg.dt) * potential_after - potential_before
        rewards += shaping
        self.reward_components["frontier_shaping"] += shaping
        return rewards, elapsed, terminated, truncated

    def metrics(self, terminated=False):
        local, union = self.coverage()
        radio = self.radio
        return {"success": bool(terminated), "time": self.time,
                "coverage": local.tolist(), "union_coverage": union,
                "coverage_deficit": self.coverage_deficit / self.config.horizon_seconds,
                "union_completion_time": self.union_completion_time,
                "total_distance": float(self.distance.sum()),
                "wait_seconds": self.wait_seconds.tolist(),
                "wait_disconnected_seconds": self.wait_disconnected_seconds.tolist(),
                "wait_without_new_map_seconds": self.wait_without_new_map_seconds.tolist(),
                "synchronization_idle_seconds": self.sync_idle_seconds.tolist(),
                "sent_bytes": radio.sent_bytes, "delivered_bytes": radio.delivered_bytes,
                "control_bytes": radio.control_bytes, "novel_received_cells": radio.unique_cells,
                "duplicate_received_cells": radio.duplicate_cells,
                "connected_seconds": self.connected_seconds,
                "queue_bytes": [radio.queue_bytes(i) for i in range(2)],
                "reward_components": {k: v.tolist() for k, v in self.reward_components.items()}}
