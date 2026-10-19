"""Three-or-more robot version of the archived finite_contact packet protocol.

Direct links only. Each node has one aggregate TX and one aggregate RX budget
(full duplex); six directed links do not each get a free independent budget.
Payloads/poses/ACKs are serialized. A packet is visible only after completion.
No same-service forwarding, no free route-history or group map propagation.
"""
from collections import deque
from dataclasses import dataclass
import struct
import numpy as np

HEADER = struct.Struct('!BBBIfH')
POSE = struct.Struct('!ffff')
ACK = struct.Struct('!I')
CELL = np.dtype([('index', '>u4'), ('value', 'u1')])
MAP, STATE, CONFIRM = 0, 1, 2


@dataclass
class Packet:
    kind: int
    sender: int
    receiver: int
    seq: int
    created: float
    wire: bytes
    offset: int = 0

    @property
    def size(self):
        return len(self.wire)

    def cells(self):
        return np.frombuffer(self.wire, dtype=CELL, offset=HEADER.size)


class Radio:
    def __init__(self, shape, robots, config):
        self.shape, self.robots, self.config = shape, robots, config
        self.pairs = [(i, j) for i in range(robots) for j in range(robots) if i != j]
        self.queues = {p: [] for p in self.pairs}
        self.controls = {p: deque() for p in self.pairs}
        self.active = {p: None for p in self.pairs}
        self.offered = {p: np.full(shape, 127, np.uint8) for p in self.pairs}
        self.known_peer = {p: np.full(shape, 127, np.uint8) for p in self.pairs}
        self.last_pose_sent = {p: -float('inf') for p in self.pairs}
        self.peer_pose, self.peer_pose_time = {}, {}
        self.awaiting_ack = {}
        self.seq = [0] * robots
        self.credit = np.zeros(robots)
        self.tick = 0
        self.sent_bytes = self.control_bytes = self.unique_cells = self.duplicate_cells = 0
        self.completed_maps = 0
        self.last_tx = self.last_rx = np.zeros(robots, dtype=int)
        # Local receiver observations, updated only after complete paid packets.
        self.last_receive_at = np.full(robots, -np.inf)
        self.last_novel_receive_at = np.full(robots, -np.inf)
        self.last_novel_receive_cells = np.zeros(robots)
        self.last_novel_receive_free = np.zeros(robots)

    def packet(self, kind, pair, now, payload, count=0):
        sender, receiver = pair
        seq = self.seq[sender]
        self.seq[sender] += 1
        return Packet(kind, sender, receiver, seq, now,
                      HEADER.pack(kind, sender, receiver, seq, now, count) + payload)

    def enqueue(self, pair, belief, now):
        mask = ((belief != 127) & (belief != self.offered[pair]) &
                (belief != self.known_peer[pair]))
        yy, xx = np.where(mask)
        limit = (self.config.packet_bytes - HEADER.size) // CELL.itemsize
        tile_width = (self.shape[1] + self.config.tile_size - 1) // self.config.tile_size
        tiles = (yy // self.config.tile_size) * tile_width + xx // self.config.tile_size
        # Sort once rather than scanning every cell once per tile.
        order = np.argsort(tiles, kind='stable')
        boundaries = np.r_[0, np.flatnonzero(np.diff(tiles[order])) + 1, len(order)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            for start in range(lo, hi, limit):
                take = order[start:min(start + limit, hi)]
                data = np.empty(len(take), dtype=CELL)
                data['index'] = yy[take] * self.shape[1] + xx[take]
                data['value'] = belief[yy[take], xx[take]]
                self.queues[pair].append(self.packet(MAP, pair, now, data.tobytes(), len(take)))
        self.offered[pair][mask] = belief[mask]

    def pending(self, pair):
        active = self.active[pair]
        return ([active] if active is not None and active.kind == MAP else []) + self.queues[pair]

    def service(self, beliefs, positions, now, duration, links, rank=None):
        if duration <= 0:
            raise ValueError('Communication consumes positive simulated time')
        for pair in self.pairs:
            sender, receiver = pair
            self.enqueue(pair, beliefs[sender], now)
            known = self.known_peer[pair].ravel()
            self.queues[pair] = [p for p in self.queues[pair]
                                 if p.offset or not np.all(known[p.cells()['index']] == p.cells()['value'])]
            if now - self.last_pose_sent[pair] >= self.config.pose_period:
                if not any(p.kind == STATE for p in self.controls[pair]):
                    self.controls[pair].append(self.packet(
                        STATE, pair, now, POSE.pack(*positions[sender], 0., 0.)))
                    self.last_pose_sent[pair] = now
        # Reliable link model: ACKs can themselves wait behind a finite budget.
        # Preserve the archived explicit retry mechanism, with a longer timeout.
        for key, (packet, sent_time) in list(self.awaiting_ack.items()):
            if now - sent_time >= 120.:
                packet.offset = 0
                self.queues[(packet.sender, packet.receiver)].insert(0, packet)
                del self.awaiting_ack[key]
        connected = np.any(links, axis=1)
        self.credit[~connected] = 0.
        amount = self.config.bytes_per_second * duration + self.credit
        budget = np.floor(amount).astype(int)
        self.credit = np.where(connected, amount - budget, 0.)
        tx, rx = budget.copy(), budget.copy()
        deliveries = []
        offset = self.tick % len(self.pairs)
        for pair in self.pairs[offset:] + self.pairs[:offset]:
            sender, receiver = pair
            if not links[sender, receiver]:
                continue
            while tx[sender] > 0 and rx[receiver] > 0:
                packet = self.active[pair]
                if packet is None:
                    if self.controls[pair]:
                        packet = self.controls[pair].popleft()
                    elif self.queues[pair]:
                        index = 0 if rank is None else rank(pair, self.queues[pair])
                        packet = self.queues[pair].pop(index)
                    else:
                        break
                    self.active[pair] = packet
                count = int(min(packet.size - packet.offset, tx[sender], rx[receiver]))
                packet.offset += count
                tx[sender] -= count
                rx[receiver] -= count
                self.sent_bytes += count
                self.control_bytes += count if packet.kind != MAP else 0
                if packet.offset == packet.size:
                    deliveries.append(packet)
                    self.active[pair] = None
        gains = np.zeros(self.robots)
        received_novel = np.zeros(self.robots)
        received_free = np.zeros(self.robots)
        # Delivery begins only after all links have consumed this time slice.
        for packet in deliveries:
            sender, receiver = packet.sender, packet.receiver
            self.last_receive_at[receiver] = now + duration
            if packet.kind == STATE:
                self.peer_pose[(receiver, sender)] = np.array(POSE.unpack_from(packet.wire, HEADER.size))
                self.peer_pose_time[(receiver, sender)] = packet.created
            elif packet.kind == CONFIRM:
                seq = ACK.unpack_from(packet.wire, HEADER.size)[0]
                pending = self.awaiting_ack.pop((receiver, sender, seq), None)
                if pending:
                    data = pending[0].cells()
                    self.known_peer[(receiver, sender)].ravel()[data['index']] = data['value']
            else:
                data = packet.cells()
                target = beliefs[receiver].ravel()
                novel = target[data['index']] == 127
                received_novel[receiver] += int(novel.sum())
                received_free[receiver] += int(np.count_nonzero(novel & (data['value'] == 255)))
                gains[sender] += int(novel.sum())
                self.unique_cells += int(novel.sum())
                self.duplicate_cells += len(data) - int(novel.sum())
                target[data['index']] = data['value']
                self.known_peer[(receiver, sender)].ravel()[data['index']] = data['value']
                self.completed_maps += 1
                self.awaiting_ack[(sender, receiver, packet.seq)] = (packet, now + duration)
                self.controls[(receiver, sender)].appendleft(self.packet(
                    CONFIRM, (receiver, sender), now + duration, ACK.pack(packet.seq)))
        updated = received_novel > 0
        self.last_novel_receive_at[updated] = now + duration
        self.last_novel_receive_cells[updated] = received_novel[updated]
        self.last_novel_receive_free[updated] = received_free[updated]
        self.last_tx, self.last_rx = budget - tx, budget - rx
        self.tick += 1
        return gains
