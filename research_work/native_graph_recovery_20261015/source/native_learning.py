"""IR2 driver SAC update, with lossless storage and bounded microbatches.

The eight updates per completed rollout, replay warmup, optimizer rates,
gradient clips and 64-update hard target copies follow the original driver.
Physical-duration discounting is retained for the finite-contact task.
"""
import random
import numpy as np
import torch
from torch import nn
from finite_contact.learning import Learner,tensors,bellman_target
from finite_contact.memory import PackedObservation


class NativeLearner(Learner):
    def __init__(self,config,device='cuda',microbatch=32):
        super().__init__(config,device)
        self.microbatch=microbatch
        self.lossless_observations=True

    def update(self):
        if len(self.replay)<max(self.config.warmup,self.config.batch_size):return {}
        batch=random.sample(self.replay,self.config.batch_size)
        def nodes(obs):return obs.schema[0][1][0] if isinstance(obs,PackedObservation) else obs[0].shape[0]
        largest=max(nodes(x[i]) for x in batch for i in (0,4))
        # Bound quadratic attention workspace even when a graph grows late in training.
        actual_microbatch=max(1,min(self.microbatch,2097152//max(1,largest*largest)))
        for opt in [self.policy_optimizer,*self.q_optimizers,self.alpha_optimizer]:opt.zero_grad(set_to_none=True)
        values={k:0. for k in ('policy_loss','q1_loss','q2_loss','entropy','alpha_loss','wait_probability')}
        for start in range(0,len(batch),actual_microbatch):
            items=batch[start:start+actual_microbatch];weight=len(items)/len(batch)
            obs=tensors([x[0] for x in items],self.device)
            nxt=tensors([x[4] for x in items],self.device)
            actions=torch.tensor([x[1] for x in items],device=self.device).view(-1,1,1)
            rewards=torch.tensor([x[2] for x in items],dtype=torch.float32,device=self.device).view(-1,1)
            durations=torch.tensor([x[3] for x in items],dtype=torch.float32,device=self.device).view(-1,1)
            ended=torch.tensor([x[5] for x in items],dtype=torch.float32,device=self.device).view(-1,1)
            truncated=torch.tensor([x[6] for x in items],dtype=torch.float32,device=self.device).view(-1,1)
            with torch.no_grad():
                q=torch.minimum(*(net(*obs)[0].squeeze(-1) for net in self.critics))
                next_logp=self.policy(*nxt)
                next_q=torch.minimum(*(net(*nxt)[0].squeeze(-1) for net in self.targets))
                next_value=(next_logp.exp()*(next_q-self.log_alpha.exp()*next_logp)).sum(-1,keepdim=True)
                target=bellman_target(self.config,rewards,durations,ended,truncated,next_value)
            logp=self.policy(*obs)
            policy_loss=(logp.exp()*(self.log_alpha.exp().detach()*logp-q)).sum(-1).mean()
            q_losses=[nn.functional.mse_loss(net(*obs)[0].gather(1,actions).squeeze(-1),target)
                      for net in self.critics]
            signed_entropy=(logp.exp()*logp).sum(-1)
            entropy_target=.05*np.log(self.config.neighbors)
            alpha_loss=-(self.log_alpha*(signed_entropy.detach()+entropy_target)).mean()
            (policy_loss*weight).backward()
            for loss in q_losses:(loss*weight).backward()
            (alpha_loss*weight).backward()
            for key,value in zip(values,(policy_loss,q_losses[0],q_losses[1],-signed_entropy.mean(),alpha_loss,logp[:,0].exp().mean())):
                values[key]+=weight*value.detach().item()
        pn=nn.utils.clip_grad_norm_(self.policy.parameters(),5.)
        qn=[nn.utils.clip_grad_norm_(net.parameters(),2000.) for net in self.critics]
        for opt in [self.policy_optimizer,*self.q_optimizers,self.alpha_optimizer]:opt.step()
        self.updates+=1
        if self.updates%64==0:
            for target,net in zip(self.targets,self.critics):target.load_state_dict(net.state_dict())
        values.update(updates=self.updates,alpha=self.log_alpha.exp().item(),policy_grad_norm=float(pn),
                      q_grad_norm=max(float(x) for x in qn),effective_batch=len(batch),microbatch=actual_microbatch,max_nodes=largest)
        if not all(np.isfinite(v) for v in values.values()):raise FloatingPointError('Non-finite native SAC update')
        if self.on_update is not None:self.on_update(values)
        return values
