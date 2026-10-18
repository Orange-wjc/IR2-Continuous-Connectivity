"""Validated SAC/SIL recipe on native graphs with bounded microbatches.

Q first/clip10, then actor with updated Q and successful self-demonstrations;
legal-action entropy, bounded alpha, tau=.01 targets. One optimizer step per
full effective batch, regardless of the graph-dependent microbatch size.
"""
import random
import numpy as np
import torch
from torch import nn
from finite_contact.learning import Learner, tensors, bellman_target, sample_success_demonstrations
from finite_contact.memory import PackedObservation


class NativeLearner(Learner):
    def __init__(self, config, device='cuda', microbatch=16):
        super().__init__(config, device)
        if microbatch < 1:
            raise ValueError('microbatch must be positive')
        self.microbatch = microbatch
        self.lossless_observations = True

    def update(self):
        cfg = self.config
        if len(self.replay) < max(cfg.warmup, cfg.batch_size):
            return {}
        batch = random.sample(self.replay, cfg.batch_size)
        def size(obs):
            return obs.schema[0][1][0] if isinstance(obs, PackedObservation) else obs[0].shape[0]
        example_count = sum(len(e['transitions']) for e in self.success_episodes)
        demo = (sample_success_demonstrations(self.success_episodes, cfg.batch_size, cfg.success_sampling)
                if cfg.success_imitation_weight > 0 and example_count else [])
        largest = max([size(x[i]) for x in batch for i in (0, 4)] + [size(x[0]) for x in demo])
        chunk = max(1, min(self.microbatch, 2097152//max(1, largest*largest)))
        values = {k: 0. for k in ('q1_loss', 'q2_loss', 'sac_policy_loss', 'imitation_loss',
                                 'entropy', 'deadline_fraction', 'wait_probability')}
        for opt in self.q_optimizers:
            opt.zero_grad(set_to_none=True)
        for start in range(0, len(batch), chunk):
            rows = batch[start:start+chunk]; weight = len(rows)/len(batch)
            obs = tensors([x[0] for x in rows], self.device)
            nxt = tensors([x[4] for x in rows], self.device)
            actions = torch.tensor([x[1] for x in rows], device=self.device).view(-1, 1, 1)
            def column(index):
                return torch.tensor([x[index] for x in rows], dtype=torch.float32, device=self.device).view(-1, 1)
            with torch.no_grad():
                lp = self.policy(*nxt)
                q = torch.minimum(*(net(*nxt)[0].squeeze(-1) for net in self.targets))
                v = (lp.exp()*(q-self.log_alpha.exp()*lp)).sum(-1, keepdim=True)
                target = bellman_target(cfg, column(2), column(3), column(5), column(6), v)
            for i, net in enumerate(self.critics):
                loss = nn.functional.mse_loss(net(*obs)[0].gather(1, actions).squeeze(-1), target)
                (loss*weight).backward(); values['q%d_loss'%(i+1)] += weight*loss.detach().item()
        qnorm = [nn.utils.clip_grad_norm_(net.parameters(), 10.) for net in self.critics]
        for opt in self.q_optimizers:
            opt.step()
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        stats = {}
        for start in range(0, len(batch), chunk):
            rows = batch[start:start+chunk]; weight = len(rows)/len(batch)
            obs = tensors([x[0] for x in rows], self.device)
            lp = self.policy(*obs)
            for k, v in self.policy.pointer.last_stats.items():
                stats[k] = stats.get(k, 0.) + weight*v.detach().item()
            with torch.no_grad():
                q = torch.minimum(*(net(*obs)[0].squeeze(-1) for net in self.critics))
            loss = (lp.exp()*(self.log_alpha.exp().detach()*lp-q)).sum(-1).mean()
            entropy = -(lp.exp()*lp).sum(-1)
            valid = (obs[4] == 0).sum(-1).squeeze(1).float()
            alpha_loss = (self.log_alpha*(entropy.detach()-.05*valid.log())).mean()
            (loss*weight).backward(); (alpha_loss*weight).backward()
            values['sac_policy_loss'] += weight*loss.detach().item()
            values['entropy'] += weight*entropy.detach().mean().item()
            values['wait_probability'] += weight*lp[:, 0].exp().detach().mean().item()
            values['deadline_fraction'] += weight*float(np.mean([x[6] for x in rows]))
        imitation_saturation = 0.
        for start in range(0, len(demo), chunk):
            rows = demo[start:start+chunk]; weight = len(rows)/len(demo)
            obs = tensors([x[0] for x in rows], self.device)
            actions = torch.tensor([x[1] for x in rows], device=self.device).view(-1, 1)
            lp = self.policy(*obs)
            loss = -lp.gather(1, actions).mean()
            (cfg.success_imitation_weight*loss*weight).backward()
            values['imitation_loss'] += weight*loss.detach().item()
            imitation_saturation += weight*self.policy.pointer.last_stats['pointer_saturation_fraction'].item()
        if demo:
            self.imitation_updates += 1
        pnorm = nn.utils.clip_grad_norm_(self.policy.parameters(), 5.)
        self.policy_optimizer.step(); self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-8., 2.)
            for target, net in zip(self.targets, self.critics):
                for a, b in zip(target.parameters(), net.parameters()):
                    a.lerp_(b, .01)
        self.updates += 1
        values.update(policy_loss=values['sac_policy_loss']+cfg.success_imitation_weight*values['imitation_loss'],
                      imitation_updates=self.imitation_updates, success_examples=example_count,
                      success_episodes=len(self.success_episodes),
                      imitation_terminal_fraction=float(np.mean([x[5] for x in demo])) if demo else 0.,
                      imitation_pointer_saturation_fraction=imitation_saturation,
                      updates=self.updates, alpha=self.log_alpha.exp().item(), policy_grad_norm=float(pnorm),
                      q_grad_norm=max(float(v) for v in qnorm), effective_batch=len(batch),
                      microbatch=chunk, max_nodes=largest, **stats)
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError('Non-finite native SAC/SIL update')
        if self.on_update is not None:
            self.on_update(values)
        return values
