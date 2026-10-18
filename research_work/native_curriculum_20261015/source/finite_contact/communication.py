"""Serialized unicast, finite capacity, acknowledged incremental maps and paid poses.

Only complete packets are visible to receivers. Partial progress is retained at
the link endpoints during a disconnection. All deliveries are applied after the
time slice's service, so no same-slice forwarding or knowledge shortcut occurs.
"""
from collections import deque
from dataclasses import dataclass
import struct
import numpy as np

HEADER = struct.Struct("!BBBIfH")  # kind, sender, receiver, ID, creation time, count
POSE = struct.Struct("!ffff")
ACK = struct.Struct("!I")
CELL = np.dtype([("index", ">u4"), ("value", "u1")])
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
    def __init__(self, shape, config):
        self.config = config
        self.shape = shape
        self.queues = {(i, 1 - i): [] for i in range(2)}
        self.controls = {pair: deque() for pair in self.queues}
        self.active = {pair: None for pair in self.queues}
        self.awaiting_ack = {}
        self.offered = {pair: np.full(shape, 127, np.uint8) for pair in self.queues}
        self.known_peer = {pair: np.full(shape, 127, np.uint8) for pair in self.queues}
        self.peer_pose = [None, None]
        self.peer_pose_time = [None, None]
        self.last_pose_sent = [-float("inf"), -float("inf")]
        self.seq = [0, 0]
        self.tick_index = 0
        self.credit = [0.0, 0.0]
        self.sent_bytes = self.delivered_bytes = self.control_bytes = 0
        self.unique_cells = self.duplicate_cells = self.completed_maps = 0
        self.events = []

    def packet(self, kind, sender, time, payload, count=0):
        seq = self.seq[sender]
        self.seq[sender] += 1
        receiver = 1 - sender
        wire = HEADER.pack(kind, sender, receiver, seq, time, count) + payload
        return Packet(kind, sender, receiver, seq, time, wire)

    def enqueue_updates(self, sender, belief, time):
        pair = (sender, 1 - sender)
        confirmed = self.known_peer[pair]
        mask = ((belief != 127) & (belief != self.offered[pair])
                & (belief != confirmed))
        yy, xx = np.where(mask)
        if len(xx) == 0:
            return
        tiles = (yy // self.config.tile_size) * 10000 + xx // self.config.tile_size
        limit = (self.config.max_packet_bytes - HEADER.size) // CELL.itemsize
        for tile in np.unique(tiles):
            ids = np.flatnonzero(tiles == tile)
            for start in range(0, len(ids), limit):
                take = ids[start:start + limit]
                data = np.empty(len(take), dtype=CELL)
                data["index"] = yy[take] * self.shape[1] + xx[take]
                data["value"] = belief[yy[take], xx[take]]
                self.queues[pair].append(self.packet(MAP, sender, time, data.tobytes(), len(take)))
        self.offered[pair][mask] = belief[mask]

    def prune_known(self, sender):
        """Common protocol optimization for ALL baselines, independent of content ranking."""
        pair = (sender, 1 - sender)
        known = self.known_peer[pair].ravel()
        self.queues[pair] = [p for p in self.queues[pair]
                             if p.offset > 0 or not np.all(
                                 known[p.cells()["index"]] == p.cells()["value"])]

    def pending(self, sender):
        pair = (sender, 1 - sender)
        packets = list(self.queues[pair])
        if self.active[pair] is not None and self.active[pair].kind == MAP:
            packets.append(self.active[pair])
        return packets

    def queue_bytes(self, sender):
        pair = (sender, 1 - sender)
        return sum(p.size - p.offset for p in self.pending(sender)) + sum(
            p.size for p in self.controls[pair])

    def queue_pose(self, sender, position, velocity, time):
        pair = (sender, 1 - sender)
        # At most one unsent pose; never replace a partially transmitted packet.
        if time - self.last_pose_sent[sender] >= self.config.pose_period:
            if not any(p.kind == STATE for p in self.controls[pair]):
                payload = POSE.pack(*position, *velocity)
                self.controls[pair].append(self.packet(STATE, sender, time, payload))
                self.last_pose_sent[sender] = time

    def service(self, positions, velocities, time, duration, connected, beliefs,
                rank=None, log_events=True):
        """Service frozen sender queues; connected is a simulator-only link result."""
        for sender in range(2):
            self.prune_known(sender)
            self.queue_pose(sender, positions[sender], velocities[sender], time)
        # Reliability in the no-loss first profile; ACKs also consume capacity.
        for key, (packet, sent_time) in list(self.awaiting_ack.items()):
            if time - sent_time >= 12.0:
                packet.offset = 0
                self.queues[(packet.sender, packet.receiver)].insert(0, packet)
                del self.awaiting_ack[key]
        if not connected:
            self.credit = [0.0, 0.0]  # no capacity accumulated across disconnections
            self.tick_index += 1
            return np.zeros(2, dtype=np.float64)
        budgets = []
        for sender in range(2):
            amount = self.config.bytes_per_second * duration + self.credit[sender]
            budgets.append(int(amount))
            self.credit[sender] = amount - int(amount)
        rx_budgets = budgets.copy()
        deliveries = []
        for sender in ((self.tick_index % 2), (1 - self.tick_index % 2)):
            receiver = 1 - sender
            pair = (sender, receiver)
            while budgets[sender] > 0 and rx_budgets[receiver] > 0:
                packet = self.active[pair]
                if packet is None:
                    if self.controls[pair]:
                        packet = self.controls[pair].popleft()
                    elif self.queues[pair]:
                        index = 0 if rank is None else int(rank(sender, self.queues[pair]))
                        packet = self.queues[pair].pop(index)
                    else:
                        break
                    self.active[pair] = packet
                count = min(packet.size - packet.offset, budgets[sender], rx_budgets[receiver])
                packet.offset += count
                budgets[sender] -= count
                rx_budgets[receiver] -= count
                self.sent_bytes += count
                if packet.kind != MAP:
                    self.control_bytes += count
                if log_events:
                    self.events.append({"event": "transmit", "time": time, "sender": sender, "receiver": receiver,
                                        "packet": packet.seq, "kind": packet.kind, "bytes": count,
                                        "complete": packet.offset == packet.size})
                if packet.offset == packet.size:
                    deliveries.append(packet)
                    self.active[pair] = None
        gains = np.zeros(2, dtype=np.float64)
        for packet in deliveries:
            sender, receiver = packet.sender, packet.receiver
            self.delivered_bytes += packet.size
            if packet.kind == STATE:
                self.peer_pose[receiver] = np.array(POSE.unpack_from(packet.wire, HEADER.size), dtype=float)
                self.peer_pose_time[receiver] = packet.created
            elif packet.kind == CONFIRM:
                confirmed_seq = ACK.unpack_from(packet.wire, HEADER.size)[0]
                key = (receiver, sender, confirmed_seq)
                pending = self.awaiting_ack.pop(key, None)
                if pending is not None:
                    data = pending[0].cells()
                    self.known_peer[(receiver, sender)].ravel()[data["index"]] = data["value"]
            else:
                data = packet.cells()
                target = beliefs[receiver].ravel()
                novel = target[data["index"]] == 127
                gains[sender] += np.count_nonzero(novel)
                self.unique_cells += int(novel.sum())
                self.duplicate_cells += len(data) - int(novel.sum())
                target[data["index"]] = data["value"]
                # Receiving these cells is legal evidence that the sender knows them.
                self.known_peer[(receiver, sender)].ravel()[data["index"]] = data["value"]
                self.completed_maps += 1
                if log_events:
                    self.events.append({"event": "map_delivery", "time": time + duration,
                                        "sender": sender, "receiver": receiver, "packet": packet.seq,
                                        "new_cells": int(novel.sum()), "duplicate_cells": len(data) - int(novel.sum())})
                self.awaiting_ack[(sender, receiver, packet.seq)] = (packet, time)
                ack = self.packet(CONFIRM, receiver, time + duration, ACK.pack(packet.seq))
                self.controls[(receiver, sender)].appendleft(ack)
        self.tick_index += 1
        return gains
