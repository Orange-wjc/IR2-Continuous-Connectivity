"""Narrow integration boundary between original IR2 and finite-contact logic."""
import heapq
import numpy as np
import torch
from skimage.measure import block_reduce
from .radio import Radio, CELL, HEADER, MAP
from .learning import build_worker_models


class ContactAdapter:
    def __init__(self, config, policy, auxiliary, episode):
        self.config = config
        self.content_ready = bool(auxiliary and auxiliary.get('ready', False))
        self.encoder = self.head = None
        if config.learned:
            if auxiliary is None:
                raise ValueError('finite_learned needs synchronized content head weights')
            self.encoder, self.head = build_worker_models(policy, auxiliary)
        self.contexts = {}
        self.labels = []
        self.label_count = 0
        self.rng = np.random.RandomState(20261018 + episode)
        self.time = 0.
        self.waits = 0
        self._rank_cache = {}
        self.reward_sums = dict(frontier=0., exploration=0., rendezvous=0., distance_cost=0.,
                                time_cost=0., delivery=0.)
        self.empty_waits = self.positive_empty_waits = 0
        self.last_delivery_novel = self.last_delivery_reward = 0.
        self.empty_wait_details = []
        self.last_utility_pose = {}

    def reset(self, env):
        self.radio = Radio(env.ground_truth_size, env.n_agent, self.config)
        # The shared deployment/start prior is known initially, not future data.
        # Break original list aliases so receive(sender->receiver) is local.
        env.all_robot_belief = [[env.all_robot_belief[i][j].copy()
                                for j in range(env.n_agent)] for i in range(env.n_agent)]
        for i, j in self.radio.pairs:
            self.radio.known_peer[(i,j)][:] = env.all_robot_belief[i][j]
        self.env = env

    def observe(self, robot, observations):
        if self.encoder is None:
            return
        nodes, _, _, padding, _, adjacency = observations
        count = len(self.env.all_node_coords[robot])
        with torch.no_grad():
            local = nodes.detach().cpu().clone()
            local[:, :, 3] = 0.  # No recursive rendezvous/content feedback.
            embedding = self.encoder.encode_graph(local, padding.cpu(), adjacency.cpu())
        self.contexts[robot] = (self.env.all_node_coords[robot].copy(), embedding[0,:count].numpy())

    def features(self, pair, packets):
        sender, receiver = pair
        env = self.env
        coords = env.all_node_coords[sender]
        height, width = env.ground_truth_size
        scale = float(max(height, width))
        position = env.all_robot_positions_belief[sender][sender]
        peer = self.radio.peer_pose.get(pair)
        # Initial common start is the sole shared position prior.
        peer_position = env.start_position if peer is None else peer[:2]
        age = self.time - self.radio.peer_pose_time.get(pair, 0.)
        # Env.find_frontier already returns full-resolution pixel coordinates.
        frontier = env.all_frontiers[sender]
        cached = self.contexts.get(sender)
        context_dim = env.contact_embedding_dim
        fs, cs = [], []
        for p in packets:
            cells = p.cells()
            ids = cells['index'].astype(int)
            center = np.array([(ids % width).mean(), (ids // width).mean()])
            index = int(np.argmin(np.linalg.norm(coords-center, axis=1)))
            known = self.radio.known_peer[pair].ravel()[ids]
            near_frontier = float(len(frontier) > 0 and np.min(np.linalg.norm(frontier-center,axis=1)) < 24.)
            fs.append([len(cells)/100., np.mean(cells['value']==255), center[0]/scale,
                       center[1]/scale, np.mean(known==127), np.linalg.norm(center-peer_position)/scale,
                       min(age/60.,4.), min((self.time-p.created)/60.,4.), p.size/512.,
                       near_frontier, env.all_node_utility[sender][index]/50.,
                       np.linalg.norm(center-position)/scale])
            if cached is None:
                cs.append(np.zeros(context_dim, dtype=np.float32))
            else:
                cindex = int(np.argmin(np.linalg.norm(cached[0]-center,axis=1)))
                cs.append(cached[1][cindex])
        return np.asarray(fs,dtype=np.float32), np.asarray(cs,dtype=np.float32)

    def scores(self, pair, packets, mode=None):
        if not packets:
            return np.zeros(0)
        mode = mode or self.config.method
        if mode == 'finite_learned' and not self.content_ready:
            # Collect labels using a legal rule, never a random content head.
            mode = 'finite_rule'
        known = self.radio.known_peer[pair].ravel()
        upper = np.array([np.count_nonzero(known[p.cells()['index']]==127) for p in packets],dtype=float)
        if mode == 'finite_fifo':
            return upper
        fs, cs = self.features(pair,packets)
        if mode == 'finite_rule':
            return upper * (1. + .25 * fs[:,9])
        with torch.no_grad():
            values = self.head(torch.from_numpy(fs),torch.from_numpy(cs)).numpy()*100.
        return np.minimum(np.maximum(values,0.),upper)

    @staticmethod
    def candidates(packets):
        # Same deterministic spread bound as archived label sampling. Make
        # selection cost bounded rather than score thousands of packets/step.
        ids = np.linspace(0,len(packets)-1,min(64,len(packets))).astype(int)
        return ids, [packets[i] for i in ids]

    def rank(self, pair, packets):
        if self.config.method == 'finite_fifo':
            return 0
        ids, candidates = self.candidates(packets)
        cache = self._rank_cache.setdefault(pair,{})
        missing = [p for p in candidates if p.seq not in cache]
        if missing:
            cache.update({p.seq:s/p.size for p,s in zip(missing,self.scores(pair,missing))})
        return int(ids[max(range(len(candidates)),key=lambda i:cache[candidates[i].seq])])

    def collect_labels(self):
        if self.head is None:
            return
        # Receiver state is read HERE only for auxiliary training targets.
        # A bounded independent reservoir avoids changing global SAC RNG.
        for pair in self.radio.pairs:
            pending = self.radio.pending(pair)
            if not pending:
                continue
            ids = np.linspace(0,len(pending)-1,min(8,len(pending))).astype(int)
            packets = [pending[i] for i in ids]
            fs, cs = self.features(pair,packets)
            target = self.env.all_robot_belief[pair[1]][pair[1]].ravel()
            for p,f,c in zip(packets,fs,cs):
                y = np.float32(np.count_nonzero(target[p.cells()['index']]==127)/100.)
                item = (f.copy(),c.copy(),y)
                self.label_count += 1
                if len(self.labels) < 256:
                    self.labels.append(item)
                else:
                    i = self.rng.randint(self.label_count)
                    if i < len(self.labels):
                        self.labels[i] = item

    def exchange(self, env, positions, distance, robot, step):
        # Conservative endpoint model: movement consumes time but cannot earn
        # capacity using a link observed only at the destination. A paid dwell
        # follows each action; WAIT consumes that dwell without movement.
        movement = distance/self.config.speed
        duration = movement + self.config.wait_seconds
        self.waits += int(distance < 1e-8)
        self._rank_cache.clear()
        beliefs = [env.all_robot_belief[i][i] for i in range(env.n_agent)]
        for pair in self.radio.pairs:
            self.radio.enqueue(pair,beliefs[pair[0]],self.time)
        self.collect_labels()
        links = np.zeros((env.n_agent,env.n_agent),dtype=bool)
        for i,j in self.radio.pairs:
            # Actual link truth is used by the simulator, not by scoring.
            links[i,j] = tuple(positions[j]) in env.graph_dict.get(tuple(positions[i]),[])
        before_novel = self.radio.unique_cells
        gains = self.radio.service(beliefs,positions,self.time+movement,
                                   self.config.wait_seconds,links,self.rank)
        self.last_delivery_novel = self.radio.unique_cells-before_novel
        self.last_delivery_reward = float(gains[robot]/1000.)
        self.time += duration
        for own in range(env.n_agent):
            env.all_downsampled_belief[own] = block_reduce(beliefs[own].copy(),
                block_size=(env.resolution,env.resolution),func=np.min)
            for other in range(env.n_agent):
                if own == other:
                    continue
                # ACK-backed conservative map estimate, never peer truth.
                env.all_robot_belief[own][other] = self.radio.known_peer[(own,other)].copy()
                pose = self.radio.peer_pose.get((own,other))
                if pose is not None:
                    # IR2 graph cropping uses integer grid positions as slices.
                    point = np.rint(pose[:2]).astype(int)
                    x,y = point
                    # A paid pose can lie beyond the receiver's delivered map.
                    # IR2's local graph assumes every supplied pose is on known
                    # free space. Retain the full pose in radio state/utility,
                    # but only project locally known free poses into that graph.
                    legal = (0 <= y < beliefs[own].shape[0] and 0 <= x < beliefs[own].shape[1]
                             and beliefs[own][y,x] == 255)
                    if legal:
                        points = env.all_graph_generator[own].uniform_points
                        nearby = points[np.all(np.abs(points-point)<env.contact_peer_radius,axis=1)]
                        legal = bool(len(nearby) and np.any(beliefs[own][nearby[:,1],nearby[:,0]]==255))
                    env.all_robot_positions_belief[own][other] = point if legal else None
                    env.all_robot_positions_step_updated[own][other] = self.radio.peer_pose_time[(own,other)]
        env.all_explored_rate[robot] = env.evaluate_exploration_rate(robot)
        frontier = env.find_frontier(env.all_downsampled_belief[robot])
        success = env.update_graph(robot,frontier,eps=step,step=step)
        # Same real-delivery/time terms for all finite communication methods.
        reward = gains[robot]/1000. - self.config.time_penalty*duration
        return success, reward

    def expected_delivery(self, pair):
        packets = self.radio.pending(pair)
        if not packets:
            return 0.
        if self.config.method == 'finite_fifo' or not self.config.coupled:
            # Uncoupled baseline: conservative outgoing map delta, not the
            # learned/capacity-aware service gain. Bounded by map area.
            return float(self.scores(pair,packets,'finite_fifo').sum())
        active = self.radio.active[pair]
        _, candidates = self.candidates(self.radio.queues[pair]) if self.radio.queues[pair] else ([],[])
        values = self.scores(pair,candidates)
        order = sorted(range(len(candidates)),key=lambda i:values[i]/candidates[i].size,reverse=True)
        queue = ([active] if active is not None else []) + [candidates[i] for i in order]
        # Equal-share estimate respects the node's aggregate budget when there
        # are multiple peers. Actual service still accounts every byte exactly.
        budget = max(0.,self.config.bytes_per_second*self.config.contact_seconds/
                     max(1,self.env.n_agent-1) - 64.)
        gain = 0.
        lookup = {p.seq:s for p,s in zip(candidates,values)}
        for packet in queue:
            remaining = packet.size-packet.offset
            if remaining > budget:
                break
            budget -= remaining
            if packet.kind == MAP:
                gain += (lookup[packet.seq] if packet.seq in lookup
                         else float(self.scores(pair,[packet])[0]))
        return gain

    @staticmethod
    def path_distances(coords, edges, position):
        """Distances TO the paid pose's nearest local node, on directed local edges.

        Reverse Dijkstra preserves edge direction; disconnected nodes stay inf.
        No map truth, peer route history, or graph repair is used here.
        """
        nodes = {tuple(p): i for i, p in enumerate(coords)}
        reverse = [[] for _ in coords]
        for source, neighbors in edges.items():
            if source not in nodes:
                continue
            for edge in neighbors.values():
                target = tuple(edge.to_node)
                if target in nodes:
                    reverse[nodes[target]].append((nodes[source], float(edge.length)))
        distances = np.full(len(coords), np.inf)
        target = int(np.argmin(np.linalg.norm(coords-position, axis=1)))
        distances[target] = 0.
        queue = [(0., target)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for source, length in reverse[node]:
                candidate = distance + length
                if candidate < distances[source]:
                    distances[source] = candidate
                    heapq.heappush(queue, (candidate, source))
        return distances

    def utility(self, env, robot):
        coords = env.all_node_coords[robot]
        utility = np.zeros((len(coords),1))
        delta = np.zeros(env.n_agent)
        for peer in range(env.n_agent):
            if peer == robot:
                continue
            pair = (robot,peer)
            gain = self.expected_delivery(pair)
            delta[peer] = gain
            pose = self.radio.peer_pose.get(pair)
            position = env.start_position if pose is None else pose[:2]
            age = self.time-self.radio.peer_pose_time.get(pair,0.)
            # Reuse original single rendezvous input channel, with local
            # stale pose confidence and no ground-truth positions/maps.
            d = self.path_distances(coords, env.all_graph[robot], position)
            benefit = np.clip(gain/self.config.gain_scale, 0., 4.) * np.exp(-age/60.)
            utility[:,0] += np.maximum(0., benefit*np.exp(-d/self.config.path_decay)
                                       - .002*d/self.config.speed)
        # Match IR2's own-pose exclusion. Otherwise WAIT can repeatedly collect
        # a positive predicted rendezvous reward without any real delivery.
        own = env.all_robot_positions_belief[robot][robot]
        self.last_utility_pose[robot] = np.asarray(own).copy()
        utility[np.linalg.norm(coords-own,axis=1) <= self.config.self_exclusion_radius] = 0.
        return delta,utility,True

    def record_reward(self, distance, known_gain, reward, frontier, exploration, rendezvous,
                      robot=None, step=None, position=None):
        # Diagnostics only; these privileged outcome counters never feed scores.
        for key,value in dict(frontier=frontier,exploration=exploration,rendezvous=rendezvous,
                distance_cost=distance/512.,time_cost=self.config.time_penalty*(
                    distance/self.config.speed+self.config.wait_seconds),delivery=self.last_delivery_reward).items():
            self.reward_sums[key] += float(value)
        if distance < 1e-8 and known_gain == 0 and self.last_delivery_novel == 0:
            self.empty_waits += 1
            self.positive_empty_waits += int(reward > 1e-8)
            if reward > 1e-8 and len(self.empty_wait_details)<16:
                center=self.last_utility_pose.get(robot)
                self.empty_wait_details.append(dict(robot=robot,step=step,reward=float(reward),
                    frontier=float(frontier),exploration=float(exploration),rendezvous=float(rendezvous),
                    delivery=float(self.last_delivery_reward),known_gain=int(known_gain),
                    delivered_novel=int(self.last_delivery_novel),
                    position=None if position is None else np.asarray(position).tolist(),
                    utility_center=None if center is None else center.tolist()))

    def metrics(self):
        stats = dict(contact_seconds=self.time,contact_sent_bytes=self.radio.sent_bytes,
                    contact_control_bytes=self.radio.control_bytes,
                    contact_unique_cells=self.radio.unique_cells,
                    contact_duplicate_cells=self.radio.duplicate_cells,
                    contact_map_packets=self.radio.completed_maps,contact_wait_actions=self.waits,
                    contact_empty_waits=self.empty_waits,contact_positive_empty_waits=self.positive_empty_waits)
        stats.update({'contact_reward_'+k:v for k,v in self.reward_sums.items()})
        # Ground truth is used ONLY for terminal reporting, never observations.
        env = self.env
        local = [env.evaluate_exploration_rate(i) for i in range(env.n_agent)]
        union = np.any(np.stack([env.all_robot_belief[i][i] == 255 for i in range(env.n_agent)]),axis=0)
        stats['contact_min_local_coverage'] = float(min(local))
        stats['contact_union_coverage'] = float(np.count_nonzero(union & (env.ground_truth == 255))/
                                               np.count_nonzero(env.ground_truth == 255))
        return stats
