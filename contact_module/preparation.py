"""Explicit shared movement initialization and pre-joint content preparation.

Preparation uses only the ten training maps. Its rollout budget is recorded
separately; it is neither movement SAC training nor held-out-map evaluation.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import ray
import torch

from baseline_support import write_json, memory_available_gib


def load_initialization(path, manifest):
    path = Path(path).resolve()
    state = torch.load(path, map_location='cpu')
    if state.get('contact_config', {'method': 'original'}).get('method') != 'original':
        raise ValueError('IR2_INIT_FROM must reference the shared original IR2 movement checkpoint')
    if state.get('training_updates', 0) <= 0:
        raise ValueError('The initialization must contain a trained movement policy')
    if set(state.get('baseline_progress', {}).get('visits', {})) != {m['name'] for m in manifest['maps']}:
        raise ValueError('Initialization must use the same ten training maps')
    source_manifest = Path('experiments')/path.parent.name/'map_manifest.json'
    if not source_manifest.exists():
        raise ValueError('Initialization requires its recorded training-map manifest')
    recorded = json.loads(source_manifest.read_text())
    fingerprints = lambda m: {item['name']: item['sha256'] for item in m['maps']}
    if fingerprints(recorded) != fingerprints(manifest):
        raise ValueError('Initialization map content differs from the new training maps')
    provenance = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      completed_episodes=state['completed_episodes'],
                      training_updates=state['training_updates'],
                      transfer='actor, both critics and temperature; fresh optimizers/counters/replay',
                      comparison_requirement='Use this identical initialization for finite FIFO control')
    return state, provenance


def validation_loss(head, labels):
    squared = []
    with torch.no_grad():
        for start in range(0, len(labels), 256):
            f, c, y = (torch.from_numpy(np.stack(a)).float()
                       for a in zip(*labels[start:start+256]))
            squared.extend(((head(f, c)-y)**2).tolist())
    return float(np.mean(squared))


def load_content_initialization(path, config, initialization, movement):
    """Reuse a prepared v2 head with unchanged features for a controlled v3 run.

    Only a zero-joint-update preparation snapshot with exactly the shared actor
    is accepted. This does not import the741-episode movement policy or replay.
    """
    path = Path(path).resolve()
    state = torch.load(path,map_location='cpu')
    previous = state.get('contact_config',{})
    if previous.get('schema_version') not in (2,3):
        raise ValueError('Content initialization must use the corrected v2/v3 feature contract')
    allowed = {'schema_version','self_exclusion_radius'}
    current = config.to_dict()
    if previous.get('method') != 'finite_learned' or any(
            previous.get(k) != v for k,v in current.items() if k not in allowed):
        raise ValueError('Content initialization feature/protocol contract differs')
    if state.get('completed_episodes') != 0 or state.get('training_updates') != 0:
        raise ValueError('Only the pre-joint preparation snapshot may initialize this controlled run')
    aux = state.get('contact_learning',{})
    if not aux.get('ready') or not aux.get('updates'):
        raise ValueError('Content initialization is not prepared')
    if aux.get('provenance',{}).get('sha256') != initialization['sha256']:
        raise ValueError('Content context source differs from shared movement initialization')
    if not all(torch.equal(v.detach().cpu(),aux['encoder'][k]) and
               torch.equal(v.detach().cpu(),state['policy_model'][k]) for k,v in movement.items()):
        raise ValueError('Content context does not match the shared trained actor')
    aux = copy.deepcopy(aux)
    audit = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 source_schema=previous.get('schema_version'),head_updates=aux['updates'],
                 feature_contract='unchanged 12 local features + frozen128-dimensional context')
    aux['provenance'] = dict(aux['provenance'],content_import=audit)
    return aux,audit


def prepare_content(agents, weights, auxiliary, writer, directory, stopping,
                    map_count=10, rounds=2, updates=1000):
    """Frozen movement + rule ranking -> labels -> head fit -> learned ranking.

    Each map appears once in fit and once in validation (separate episodes).
    No returned movement transition is added to SAC replay. Only training-map
    content validation chooses the head; independent evaluation remains undone.
    """
    if auxiliary.ready:
        return
    if auxiliary.provenance is None:
        raise ValueError('A trained movement/context initialization is required before content preparation')
    count = map_count * rounds
    if rounds != 2:
        raise ValueError('Preparation requires one fit and one validation pass')
    train, validation, episodes = [], [], []
    pending = {}
    next_episode = 1
    report = dict(stage='collecting_content_labels', movement_sac_updates=0,
                  initialization=auxiliary.provenance, planned_episodes=count,
                  ranking='finite_rule until preparation is complete', episodes=episodes)
    write_json(Path(directory)/'preparation.json', report)
    for agent in agents[:min(len(agents), count)]:
        pending[agent.job.remote(weights, next_episode)] = agent
        next_episode += 1
    while pending:
        if stopping[0]:
            raise InterruptedError('Stop requested during content preparation; joint training has not started')
        if memory_available_gib() < 4:
            raise MemoryError('Available host RAM below 4 GiB during content preparation')
        done, _ = ray.wait(list(pending), timeout=1)
        if not done:
            continue
        ref = done[0]
        agent = pending.pop(ref)
        valid, _, metrics, info = ray.get(ref)
        episode = info['episode_number']
        labels = info.get('contact_labels', [])
        if not valid or len(labels) < 128:
            raise RuntimeError('Invalid or insufficient content preparation episode: {}'.format(episode))
        (train if episode <= map_count else validation).extend(labels)
        episodes.append(dict(episode=episode, split='fit' if episode <= map_count else 'validation',
                             labels=len(labels), task_success=bool(metrics['success_rate'])))
        write_json(Path(directory)/'preparation.json', report)
        print(json.dumps(dict(stage='content_preparation', completed=len(episodes), total=count,
                              fit_labels=len(train), validation_labels=len(validation))), flush=True)
        writer.add_scalar('Preparation/Collected Episodes', len(episodes), len(episodes))
        if next_episode <= count:
            pending[agent.job.remote(weights, next_episode)] = agent
            next_episode += 1
        del ref
    # Discard any incomplete preparation replay when retrying after interruption.
    auxiliary.replay.clear()
    auxiliary.replay.extend(train)
    constant = float(np.mean([x[2] for x in train]))
    constant_mse = float(np.mean([(float(x[2])-constant)**2 for x in validation]))
    before = validation_loss(auxiliary.head, validation)
    best = None
    best_mse = float('inf')
    for step in range(0, updates, 100):
        if stopping[0]:
            raise InterruptedError('Stop requested during content head fitting')
        loss = auxiliary.update([], steps=min(100, updates-step))
        mse = validation_loss(auxiliary.head, validation)
        writer.add_scalar('Preparation/Fit Loss', loss, auxiliary.updates)
        writer.add_scalar('Preparation/Validation MSE', mse, auxiliary.updates)
        if mse < best_mse:
            best_mse = mse
            best = (copy.deepcopy(auxiliary.head.state_dict()),
                    copy.deepcopy(auxiliary.optimizer.state_dict()), auxiliary.updates)
    report.update(stage='content_fit_checked', fit_labels=len(train), validation_labels=len(validation),
                  fit_updates=updates, validation_constant_mse=constant_mse,
                  validation_initial_mse=before, validation_best_mse=best_mse,
                  selected_optimizer_step=best[2],
                  validation_scope='Separate rollout episodes on the same training maps, NOT held-out maps')
    write_json(Path(directory)/'preparation.json', report)
    if not np.isfinite(best_mse) or best_mse >= constant_mse:
        raise RuntimeError('Content head failed training-map episode validation; joint training withheld')
    auxiliary.head.load_state_dict(best[0])
    auxiliary.optimizer.load_state_dict(best[1])
    auxiliary.ready = True
    report['stage'] = 'ready_for_joint_training'
    auxiliary.provenance = dict(auxiliary.provenance, content_preparation=copy.deepcopy(report))
    write_json(Path(directory)/'preparation.json', report)
    print(json.dumps(dict(stage='content_ready', validation_mse=best_mse,
                          constant_mse=constant_mse, content_updates=auxiliary.updates)), flush=True)
