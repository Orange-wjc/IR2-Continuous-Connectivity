"""IR2 discrete SAC plus a separately supervised content-value head."""
import copy
import random
from collections import deque
import numpy as np
import torch
from torch import nn
from model import PolicyNet, QNet
from finite_contact.memory import PackedObservation


def compact_observation(observation):
    """Pack only the binary adjacency mask; retain all other values and dtypes."""
    adjacency = observation[-1]
    if adjacency.ndim == 1:
        return observation
    if adjacency.dtype != np.uint8 or not np.all((adjacency == 0) | (adjacency == 1)):
        raise ValueError('Only binary uint8 adjacency masks can be compacted')
    return (*observation[:-1], np.packbits(adjacency.reshape(-1)))


def dense_observation(observation):
    if isinstance(observation, PackedObservation):
        observation = observation.unpack()
    adjacency = observation[-1]
    if adjacency.ndim == 2:
        return observation
    nodes = observation[0].shape[0]
    adjacency = np.unpackbits(adjacency, count=nodes * nodes).reshape(nodes, nodes)
    return (*observation[:-1], adjacency)


def tensors(observations, device):
    observations = [dense_observation(o) for o in observations]
    return tuple(torch.as_tensor(np.stack(parts), device=device)
                 for parts in zip(*observations))


def bellman_target(config, rewards, durations, terminated, truncated, next_value):
    # This is a finite task with remaining time in the observation, rather than
    # an external training timeout. A failed deadline has no continuation value.
    ended = torch.maximum(terminated, truncated) if config.horizon_target == "terminal" else terminated
    return rewards + config.gamma ** (durations / config.dt) * (1 - ended) * next_value


def success_tail_start(episode):
    """First two-robot decision overlapping the final quarter of physical time."""
    transitions = episode["transitions"]
    elapsed = 0.
    for index in range(0, len(transitions), 2):
        elapsed += transitions[index][3]
        if elapsed > .75 * episode["duration"]:
            return index
    return max(0, len(transitions) - 2)


def sample_success_demonstrations(episodes, batch_size, mode):
    """Use only stored successful training transitions; no additional action labels."""
    if mode == "pooled":
        examples = [x for episode in episodes for x in episode["transitions"]]
        return random.sample(examples, min(batch_size, len(examples)))
    if mode != "map_tail":
        raise ValueError("Unknown success demonstration sampling")
    if not episodes:
        return []
    batch, starts = [], {}
    for _ in range(batch_size):
        index = random.randrange(len(episodes))
        episode = episodes[index]
        transitions = episode["transitions"]
        start = 0
        if random.random() < .5:
            if index not in starts:
                starts[index] = success_tail_start(episode)
            start = starts[index]
        batch.append(transitions[random.randrange(start, len(transitions))])
    return batch


class ContentHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12 + embedding_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, features, context):
        return self.net(torch.cat((features, context), dim=-1)).squeeze(-1)


class Learner:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.policy = PolicyNet(config.input_dim, config.embedding_dim, allow_wait=True,
                                center_pointer_logits=config.pointer_mode == "centered").to(self.device)
        self.critics = [QNet(config.input_dim, config.embedding_dim, allow_wait=True).to(self.device)
                        for _ in range(2)]
        self.targets = [copy.deepcopy(q).requires_grad_(False) for q in self.critics]
        self.context_encoder = copy.deepcopy(self.policy).requires_grad_(False).eval()
        self.policy.pointer.track_stats = True
        self.content = ContentHead(config.embedding_dim).to(self.device)
        self.log_alpha = torch.tensor([-2.], device=self.device, requires_grad=True)
        policy_rate = config.learning_rate if config.policy_learning_rate is None else config.policy_learning_rate
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_rate)
        self.q_optimizers = [torch.optim.Adam(q.parameters(), lr=config.learning_rate) for q in self.critics]
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_learning_rate)
        self.content_optimizer = torch.optim.Adam(self.content.parameters(), lr=3e-4)
        self.replay = deque(maxlen=config.replay_size)
        self.content_replay = deque(maxlen=20000)
        self.success_episodes = []
        self.imitation_updates = 0
        self.updates = 0
        self.content_updates = 0
        self.on_update = None
        self.on_content_update = None

    @torch.no_grad()
    def act(self, observations, greedy=False, random_action=False):
        inputs = tensors(observations, self.device)
        if random_action:
            probabilities = (inputs[4].squeeze(1) == 0).float()
        else:
            probabilities = self.policy(*inputs).exp()
        actions = (probabilities.argmax(dim=-1) if greedy
                   else torch.multinomial(probabilities, 1).squeeze(1))
        return actions.cpu().tolist()

    @torch.no_grad()
    def contexts(self, env):
        result = []
        for robot in range(2):
            inputs = tensors([env.observation(robot, include_rendezvous=False)], self.device)
            embedding = self.context_encoder.encode_graph(inputs[0], inputs[3], inputs[5])
            result.append(embedding[0].cpu().numpy())
        return result

    @torch.no_grad()
    def score(self, features, context):
        f = torch.as_tensor(features, device=self.device)
        c = torch.as_tensor(context, device=self.device)
        return self.content(f, c).cpu().numpy()

    def remember_success(self, transitions, duration, map_key=None):
        """Keep fast complete self-generated episodes, never timeout/partial traces."""
        if (len(transitions) < 2 or len(transitions) % 2 or duration <= 0 or
                not all(x[5] and not x[6] for x in transitions[-2:])):
            raise ValueError("A success example must end in a completed two-robot task")
        if self.config.success_memory_mode == "per_map":
            if map_key is None:
                raise ValueError("Per-map success memory requires a training layout identifier")
            previous = [e for e in self.success_episodes if e.get("map_key") == map_key]
            if previous and previous[0]["duration"] <= duration:
                return
            self.success_episodes = [e for e in self.success_episodes if e.get("map_key") != map_key]
        self.success_episodes.append({"duration": float(duration), "transitions": list(transitions),
                                      "map_key": map_key})
        self.success_episodes.sort(key=lambda episode: episode["duration"])
        del self.success_episodes[self.config.success_episode_limit:]

    def update(self):
        if len(self.replay) < max(self.config.batch_size, self.config.warmup):
            return {}
        batch = random.sample(self.replay, self.config.batch_size)
        obs = tensors([x[0] for x in batch], self.device)
        nxt = tensors([x[4] for x in batch], self.device)
        actions = torch.tensor([x[1] for x in batch], device=self.device).view(-1, 1, 1)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device).view(-1, 1)
        durations = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=self.device).view(-1, 1)
        terminated = torch.tensor([x[5] for x in batch], dtype=torch.float32, device=self.device).view(-1, 1)
        truncated = torch.tensor([x[6] for x in batch], dtype=torch.float32, device=self.device).view(-1, 1)
        with torch.no_grad():
            next_logp = self.policy(*nxt)
            next_q = torch.minimum(*(q(*nxt)[0].squeeze(-1) for q in self.targets))
            next_value = (next_logp.exp() * (next_q - self.log_alpha.exp() * next_logp)).sum(-1, keepdim=True)
            target = bellman_target(self.config, rewards, durations, terminated, truncated, next_value)
        q_losses = []
        for q, optimizer in zip(self.critics, self.q_optimizers):
            prediction = q(*obs)[0].gather(1, actions).squeeze(-1)
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.)
            optimizer.step()
            q_losses.append(loss.item())
        logp = self.policy(*obs)
        with torch.no_grad():
            q_values = torch.minimum(*(q(*obs)[0].squeeze(-1) for q in self.critics))
        sac_policy_loss = (logp.exp() * (self.log_alpha.exp().detach() * logp - q_values)).sum(-1).mean()
        pointer_stats = self.policy.pointer.last_stats.copy()
        imitation_loss = logp.new_zeros(())
        imitation_saturation = 0.
        example_count = sum(len(e["transitions"]) for e in self.success_episodes)
        demonstration = []
        if self.config.success_imitation_weight > 0 and example_count:
            demonstration = sample_success_demonstrations(
                self.success_episodes, self.config.batch_size, self.config.success_sampling)
            success_obs = tensors([x[0] for x in demonstration], self.device)
            success_actions = torch.tensor([x[1] for x in demonstration], device=self.device).view(-1, 1)
            success_logp = self.policy(*success_obs)
            imitation_loss = -success_logp.gather(1, success_actions).mean()
            imitation_saturation = self.policy.pointer.last_stats["pointer_saturation_fraction"].item()
            self.imitation_updates += 1
        policy_loss = sac_policy_loss + self.config.success_imitation_weight * imitation_loss
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 5.)
        self.policy_optimizer.step()
        entropy = -(logp.exp() * logp).sum(-1)
        # Use the actual valid action count, including WAIT, rather than padded K.
        action_count = (obs[4] == 0).sum(-1).squeeze(1).float()
        entropy_target = .05 * action_count.log()
        alpha_loss = (self.log_alpha * (entropy.detach() - entropy_target)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-8., 2.)
            for target_net, q in zip(self.targets, self.critics):
                for target_parameter, parameter in zip(target_net.parameters(), q.parameters()):
                    target_parameter.lerp_(parameter, .01)
        self.updates += 1
        result = {"policy_loss": policy_loss.item(), "q1_loss": q_losses[0],
                  "sac_policy_loss": sac_policy_loss.item(), "imitation_loss": imitation_loss.item(),
                  "imitation_updates": self.imitation_updates, "success_examples": example_count,
                  "imitation_terminal_fraction": (float(np.mean([x[5] for x in demonstration]))
                                                  if demonstration else 0.),
                  "success_episodes": len(self.success_episodes),
                  "imitation_pointer_saturation_fraction": imitation_saturation,
                  "q2_loss": q_losses[1], "entropy": entropy.mean().item(),
                  "alpha": self.log_alpha.exp().item(), "updates": self.updates,
                  "deadline_fraction": truncated.mean().item(),
                  "policy_grad_norm": policy_grad_norm.item(),
                  "wait_probability": logp[:, 0].exp().mean().item(),
                  **{k: v.item() for k, v in pointer_stats.items()}}
        if not all(np.isfinite(value) for value in result.values()):
            raise FloatingPointError("Non-finite training result")
        if self.on_update is not None:
            self.on_update(result)
        return result

    def update_content(self, batch=None):
        if batch is None:
            if len(self.content_replay) < self.config.batch_size:
                return {}
            batch = random.sample(self.content_replay, self.config.batch_size)
        elif len(batch) != self.config.batch_size:
            raise ValueError("Explicit content batches must match the configured batch size")
        f, c, y = (torch.as_tensor(np.stack(values), device=self.device) for values in zip(*batch))
        prediction = self.content(f, c)
        loss = nn.functional.mse_loss(prediction, y)
        self.content_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.content.parameters(), 5.)
        self.content_optimizer.step()
        self.content_updates += 1
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite content supervision loss")
        result = {"content_loss": loss.item(), "content_updates": self.content_updates,
                  "content_target_mean": y.mean().item(),
                  "content_positive_fraction": (y > 0).float().mean().item(),
                  "content_prediction_mean": prediction.mean().item()}
        if self.on_content_update is not None:
            self.on_content_update(result)
        return result

    def state(self):
        return {"policy": self.policy.state_dict(),
                "critics": [q.state_dict() for q in self.critics],
                "targets": [q.state_dict() for q in self.targets],
                "context_encoder": self.context_encoder.state_dict(),
                "content": self.content.state_dict(), "log_alpha": self.log_alpha.detach(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "q_optimizers": [q.state_dict() for q in self.q_optimizers],
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "content_optimizer": self.content_optimizer.state_dict(),
                "updates": self.updates, "content_updates": self.content_updates,
                "imitation_updates": self.imitation_updates}

    def load(self, state, optimizers=False):
        self.policy.load_state_dict(state["policy"])
        for q, weights in zip(self.critics, state["critics"]):
            q.load_state_dict(weights)
        for q, weights in zip(self.targets, state["targets"]):
            q.load_state_dict(weights)
        self.context_encoder.load_state_dict(state["context_encoder"])
        self.content.load_state_dict(state["content"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.updates, self.content_updates = state["updates"], state["content_updates"]
        self.imitation_updates = state.get("imitation_updates", 0)
        if optimizers:
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
            for optimizer, weights in zip(self.q_optimizers, state["q_optimizers"]):
                optimizer.load_state_dict(weights)
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
            self.content_optimizer.load_state_dict(state["content_optimizer"])
