"""Receiver/sender-local communication state, separate from IR2 graph inputs.

The actor never reads a peer map, a true link or a simulation outcome label.
The six fields are stamped only on the current node and stored with replay.
"""
import math
import numpy as np
import torch
from torch import nn

FEATURE_NAMES = ('expected_outgoing', 'queued_wire_bytes', 'paid_receive_freshness',
                 'novel_free_receive', 'novel_known_receive', 'paid_pose_age')
STATE_DIM = len(FEATURE_NAMES)


def freshness(now, received, period):
    return 0. if not math.isfinite(float(received)) else math.exp(-max(0., now-received)/period)


def local_state(contact, env, robot):
    radio, cfg, now = contact.radio, contact.config, contact.time
    own_pairs = [pair for pair in radio.pairs if pair[0] == robot]
    remaining = 0
    for pair in own_pairs:
        active = radio.active[pair]
        if active is not None:
            remaining += active.size-active.offset
        remaining += sum(packet.size-packet.offset for packet in radio.queues[pair])
        remaining += sum(packet.size-packet.offset for packet in radio.controls[pair])
    # Already computed from this sender's queue, ACK knowledge and frozen head
    # by generate_rendezvous_utility_layer. No second head pass or peer truth.
    delta = np.asarray(env.all_robot_map_belief_area_diff[robot], dtype=float)
    outgoing = float(np.maximum(delta, 0.).sum()) / cfg.gain_scale
    received = freshness(now, radio.last_receive_at[robot], cfg.pose_period)
    novel = freshness(now, radio.last_novel_receive_at[robot], cfg.pose_period)
    pose_age = min(max(0., now-radio.peer_pose_time.get(pair, 0.)) for pair in own_pairs) if own_pairs else 0.
    values = np.asarray([
        outgoing, math.log1p(remaining)/math.log1p(cfg.bytes_per_second*cfg.contact_seconds),
        received, radio.last_novel_receive_free[robot]/cfg.gain_scale*novel,
        radio.last_novel_receive_cells[robot]/cfg.gain_scale*novel, pose_age/60.
    ], dtype=np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError('Non-finite local communication state')
    return np.clip(values, 0., 4.)


def attach_state(observations, values):
    nodes, edges, current, node_padding, edge_padding, adjacency = observations
    if nodes.shape[0] != 1 or nodes.shape[-1] != 6 or len(values) != STATE_DIM:
        raise ValueError('Communication packing requires one original six-field graph')
    extra = nodes.new_zeros(nodes.shape[0], nodes.shape[1], STATE_DIM)
    extra[0, int(current.item())] = torch.as_tensor(values, dtype=nodes.dtype, device=nodes.device)
    return torch.cat((nodes, extra), dim=-1), edges, current, node_padding, edge_padding, adjacency


def make_projection(state_dim, embedding_dim):
    if state_dim != STATE_DIM:
        raise ValueError('Unexpected communication-state width')
    # Preserve the original policy/Q initialization and SAC sampler RNG.
    with torch.random.fork_rng(devices=[]):
        projection = nn.Linear(state_dim, embedding_dim, bias=False)
        nn.init.zeros_(projection.weight)
    return projection


def graph_embedding(model, inputs):
    base = model.initial_embedding.in_features
    width = inputs.shape[-1]
    if width not in (base, base+model.contact_state_dim):
        raise ValueError('Unexpected graph/communication observation width')
    embedded = model.initial_embedding(inputs[..., :base])
    if model.contact_state_dim and width != base:
        embedded = embedded + model.communication_embedding(inputs[..., base:])
    return embedded


def validate_actor_input(model, inputs):
    if inputs.shape[-1] != model.initial_embedding.in_features+model.contact_state_dim:
        raise ValueError('Finite actor/critic requires its recorded communication state')


def zero_projection_import(model, saved):
    """Explicit fresh legacy initialization only; ordinary resume stays strict."""
    expected = model.state_dict()
    missing = set(expected)-set(saved)
    if missing not in (set(), {'communication_embedding.weight'}):
        raise ValueError('Legacy initialization differs beyond the new projection')
    if set(saved)-set(expected):
        raise ValueError('Unexpected parameters in legacy initialization')
    imported = saved.copy()
    if missing:
        imported['communication_embedding.weight'] = torch.zeros_like(expected['communication_embedding.weight'])
    return imported
