"""Archived 12-feature + frozen local IR2 context content-value head.

The original movement SAC is not replaced. Auxiliary labels are simulator-only
training data, never an input to sender ranking or rendezvous observations.
"""
from collections import deque
import copy
import random
import numpy as np
import torch
from torch import nn


class ContentHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12 + embedding_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, features, context):
        return self.net(torch.cat((features, context), dim=-1)).squeeze(-1)


class ContentTrainer:
    def __init__(self, policy, config, saved=None, provenance=None):
        self.config = config
        # An independent CPU auxiliary model avoids modifying global SAC RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261018)
            self.head = ContentHead(policy.initial_embedding.out_features)
        self.encoder = {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=3e-4)
        self.replay = deque(maxlen=20000)
        self.rng = random.Random(20261018)
        self.updates = 0
        self.ready = False
        self.provenance = provenance
        if saved is not None:
            self.head.load_state_dict(saved['head'])
            self.encoder = saved['encoder']
            self.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
            self.replay.extend(saved['replay'])
            self.rng.setstate(saved['rng'])
            self.updates = saved['updates']
            self.ready = saved.get('ready', False)
            self.provenance = saved.get('provenance')

    def weights(self):
        return dict(head={k:v.detach().cpu().clone() for k,v in self.head.state_dict().items()},
                    encoder=self.encoder, ready=self.ready, provenance=self.provenance)

    def state_dict(self):
        return dict(**self.weights(), optimizer=self.optimizer.state_dict(),
                    replay=list(self.replay), rng=self.rng.getstate(), updates=self.updates)

    def update(self, labels, steps=4):
        self.replay.extend(labels)
        if len(self.replay) < 128:
            return None
        losses = []
        for _ in range(steps):
            batch = self.rng.sample(list(self.replay), 128)
            features, context, target = (torch.from_numpy(np.stack(a)).float() for a in zip(*batch))
            loss = nn.functional.mse_loss(self.head(features, context), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite content loss')
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.head.parameters(), 5.)
            self.optimizer.step()
            self.updates += 1
            losses.append(loss.item())
        return float(np.mean(losses))


def build_worker_models(policy, weights):
    encoder = copy.deepcopy(policy).cpu().eval().requires_grad_(False)
    encoder.load_state_dict(weights['encoder'])
    with torch.random.fork_rng(devices=[]):
        head = ContentHead(policy.initial_embedding.out_features)
    head.load_state_dict(weights['head'])
    return encoder, head.eval().requires_grad_(False)
