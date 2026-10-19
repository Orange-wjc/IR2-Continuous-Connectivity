"""Explicit v6->v7 state expansion; never used for ordinary checkpoint load."""
import copy


PROJECTION = 'communication_embedding.weight'


def expand_checkpoint(saved, config, maps, audit):
    previous, current = saved.get('contact_config', {}), config.to_dict()
    if previous.get('schema_version') != 6 or current['schema_version'] != 7 or config.state_dim != 6:
        raise ValueError('Only the explicit finite v6->v7 state expansion is supported')
    if any(previous.get(k) != v for k,v in current.items() if k != 'schema_version'):
        raise ValueError('Protocol/reward configuration changed beyond state expansion')
    if set(saved['baseline_progress']['visits']) != set(maps):
        raise ValueError('Training-map roster changed')
    state = copy.deepcopy(saved)
    for model_key, optimizer_key in (('policy_model','policy_optimizer'),
            ('q_net1_model','q_net1_optimizer'),('q_net2_model','q_net2_optimizer')):
        weights, optimizer = state[model_key], state[optimizer_key]
        if PROJECTION in weights:
            raise ValueError('Source already has a communication projection')
        groups = optimizer['param_groups']
        if len(groups) != 1 or len(groups[0]['params']) != len(weights):
            raise ValueError('Unexpected v6 optimizer/model parameter contract')
        ids = groups[0]['params']
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate source optimizer parameter IDs')
        base = weights['initial_embedding.weight']
        weights[PROJECTION] = base.new_zeros(base.shape[0], config.state_dim)
        new_id = max(ids)+1
        if new_id in optimizer['state']:
            raise ValueError('New projection collides with an old Adam state')
        ids.append(new_id)  # Old moments/steps remain exact; new state starts naturally.
    if config.learned:
        encoder = state['contact_learning']['encoder']
        if PROJECTION in encoder:
            raise ValueError('Source context already has a communication projection')
        base = encoder['initial_embedding.weight']
        encoder[PROJECTION] = base.new_zeros(base.shape[0], config.state_dim)
    state['contact_config'] = current
    state['checkpoint_upgrade'] = audit
    state['initialization'] = dict(path=audit['source'], sha256=audit['source_sha256'],
        completed_episodes=saved['completed_episodes'], training_updates=saved['training_updates'],
        transfer='Continue v6 learned tensors/moments with zero local-communication projections',
        parent_initialization=saved.get('initialization'), continued_from=audit)
    state['episode'] = state['completed_episodes'] = state['invalid_episodes'] = 0
    state['baseline_progress'] = dict(recent=[],visits={m:0 for m in maps},
                                    per_map={m:[] for m in maps},acceptance_recent=[])
    state['automatic_review_control'] = True
    config.validate_checkpoint(state)
    return state
