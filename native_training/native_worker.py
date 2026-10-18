"""CPU simulation workers using original maps/graphs and the audited byte radio."""
import gzip
import json
import os
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch
from model import PolicyNet
from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.learning import Learner,ContentHead,compact_observation
from finite_contact.memory import pack_observation
from finite_contact_stability import audit_record


class RolloutPolicy:
    act=Learner.act
    contexts=Learner.contexts
    score=Learner.score
    def __init__(self,cfg,weights,device='cpu'):
        self.device=torch.device(device)
        self.policy=PolicyNet(cfg.input_dim,cfg.embedding_dim,allow_wait=True,center_pointer_logits=True).to(device).eval()
        self.context_encoder=PolicyNet(cfg.input_dim,cfg.embedding_dim,allow_wait=True,center_pointer_logits=True).to(device).eval()
        self.content=ContentHead(cfg.embedding_dim).to(device).eval()
        for key in ('policy','context_encoder','content'):getattr(self,key).load_state_dict(weights[key])


def initialize_worker():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def seed_episode(seed,episode):
    value=(seed+episode*10007)%2**32
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    return value


def cpu_weights(learner):
    return {key:{k:v.detach().cpu().clone() for k,v in getattr(learner,key).state_dict().items()}
            for key in ('policy','context_encoder','content')}


def run_episode(job):
    started=time.monotonic();cfg=Config.from_checkpoint(job['config'])
    action_seed=seed_episode(cfg.seed,job['episode'])
    weights=torch.load(job['weights'],map_location='cpu')
    policy=RolloutPolicy(cfg,weights,job.get('device','cpu'))
    # Policy construction consumes Torch RNG; restore the action stream explicitly.
    seed_episode(cfg.seed,job['episode'])
    env=Environment(cfg,job.get('split','train'),job['map_index'],scorer=policy.score)
    transitions=[];labels=[];returns=np.zeros(2);decisions=0;done=False;cached=None
    label_limit=job.get('max_label_samples',2048)
    label_rng=np.random.RandomState(action_seed);labels_seen=0
    peak_nodes=max(len(g[0]) for g in env.graph_data)
    terminated=truncated=False
    while not done:
        if time.monotonic()-started>job.get('episode_wall_seconds',1800):
            raise RuntimeError('A native rollout exceeded its wall-time guard; no skipped or fabricated episode')
        if cached is None:
            if cfg.content=='learned':
                context=policy.contexts(env)
                env.feature_provider=lambda robot,ids:context[robot][ids]
            observations=[env.observation(i) for i in range(2)]
        else:observations=cached
        if job.get('collect_labels') and decisions%4==0:
            context=policy.contexts(env)
            env.feature_provider=lambda robot,ids:context[robot][ids]
            for robot in range(2):
                # Bounded deterministic reservoir, privileged targets only in training.
                for sample in env.collect_labels(robot):
                    labels_seen+=1
                    if len(labels)<label_limit:
                        labels.append(sample)
                    else:
                        index=label_rng.randint(labels_seen)
                        if index<label_limit:labels[index]=sample
        actions=policy.act(observations,greedy=job.get('greedy',False))
        reward,duration,terminated,truncated=env.step(actions)
        returns+=reward;decisions+=1;done=terminated or truncated
        if cfg.content=='learned':
            context=policy.contexts(env)
            env.feature_provider=lambda robot,ids:context[robot][ids]
        cached=[env.observation(i) for i in range(2)]
        peak_nodes=max(peak_nodes,max(len(g[0]) for g in env.graph_data))
        if not job.get('evaluation'):
            a=[pack_observation(compact_observation(o)) for o in observations]
            b=[pack_observation(compact_observation(o)) for o in cached]
            transitions.extend((a[i],actions[i],float(reward[i]),float(duration),b[i],float(terminated),bool(truncated)) for i in range(2))
    metric=env.metrics(terminated)
    metric.update(map=env.map_info,episode=job['episode'],action_seed=action_seed,decisions=decisions,
                  truncated=bool(truncated),interrupted=False,return_values=returns.tolist(),
                  stop_reason='success' if terminated else ('decision_limit' if decisions>=cfg.max_decisions else 'physical_time_limit'),
                  capped_time=env.time if terminated else cfg.horizon_seconds,
                  wall_seconds=time.monotonic()-started,peak_graph_nodes=peak_nodes)
    trace={'map':env.map_info,'episode':job['episode'],'trajectory':env.trajectory}
    audit=audit_record(metric,trace,env.truth,np.stack(env.beliefs),cfg,geometry_details=True)
    sent=[0,0];novel=duplicate=control=0
    for e in env.radio.events:
        if e['event']=='transmit':
            sent[e['sender']]+=e['bytes']
            if e['kind']!=0:control+=e['bytes']
        elif e['event']=='map_delivery':novel+=e['new_cells'];duplicate+=e['duplicate_cells']
    assert sum(sent)==metric['sent_bytes'] and control==metric['control_bytes']
    assert novel==metric['novel_received_cells'] and duplicate==metric['duplicate_received_cells']
    assert all(x<=cfg.bytes_per_second*metric['connected_seconds']+1e-6 for x in sent)
    audit.update(radio_events=len(env.radio.events),radio_totals_and_capacity_passed=True)
    destination=Path(job['output'])
    destination.mkdir(parents=True,exist_ok=True)
    stem=destination/('%08d'%job['episode'])
    with gzip.open(str(stem)+'.json.gz','wt') as f:
        json.dump(dict(metric=metric,trace=trace,radio=env.radio.events,audit=audit),f)
    np.savez_compressed(str(stem)+'.npz',truth=env.truth,beliefs=np.stack(env.beliefs),positions=env.positions)
    return dict(metric=metric,transitions=transitions,labels=labels,audit=audit,
                worker_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
