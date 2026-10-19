"""Explicit v4->v5 local utility-cache repair; preserve all learned states."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; os.chdir(root); sys.path.insert(0, str(root))
    os.environ['IR2_METHOD'] = 'finite_learned'; os.environ['IR2_RUN_NAME'] = args.run
    import torch
    import parameter as config
    from baseline_support import prepare_maps, write_json
    torch.set_num_threads(1)
    source = Path(args.source).resolve()
    # Keep native tensor device tags for the ordinary driver resume path.
    state = torch.load(source)
    previous, current = state.get('contact_config', {}), config.CONTACT_CONFIG.to_dict()
    if previous.get('schema_version') != 4 or current['schema_version'] != 5:
        raise ValueError('Only the explicit v4->v5 local utility-cache repair is supported')
    if any(previous.get(k) != v for k, v in current.items() if k != 'schema_version'):
        raise ValueError('Configuration changed beyond the cache repair')
    target = Path(config.MODEL_PATH)
    if target.exists():
        raise FileExistsError(target)
    if not state['contact_learning'].get('ready'):
        raise ValueError('Source head is not prepared')
    manifest = prepare_maps(config)
    parent = json.loads((Path('experiments')/source.parent.name/'map_manifest.json').read_text())
    fingerprint = lambda m: {r['name']: r['sha256'] for r in m['maps']}
    if fingerprint(parent) != fingerprint(manifest):
        raise ValueError('Training maps changed')
    audit = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        from_schema=4, to_schema=5,
        reason='Refresh unvisited zero frontier utility from current own map after paid map/graph update',
        inherited_completed_episodes=state['completed_episodes'],
        inherited_training_updates=state['training_updates'],
        inherited_head_updates=state['contact_learning']['updates'],
        preserved='actor, both critics, temperature, optimizers/schedulers, frozen context, head and RNG',
        reset='new-condition episode/visits/acceptance counters and auxiliary replay with stale utility features',
        new_conditions_only_acceptance=True)
    maps = [r['name'] for r in manifest['maps']]
    state['contact_config'] = current
    state['checkpoint_upgrade'] = audit
    state['initialization'] = dict(path=str(source), sha256=audit['source_sha256'],
        completed_episodes=state['completed_episodes'], training_updates=state['training_updates'],
        transfer='Continue learned v4 state; only new-condition progress/acceptance reset',
        parent_initialization=state.get('initialization'), continued_from=audit)
    state['episode'] = state['completed_episodes'] = state['invalid_episodes'] = 0
    state['baseline_progress'] = dict(recent=[], visits={m: 0 for m in maps},
                                    per_map={m: [] for m in maps}, acceptance_recent=[])
    state['contact_learning']['replay'] = []
    state['contact_learning']['provenance'] = dict(state['contact_learning']['provenance'], utility_upgrade=audit)
    state['automatic_review_control'] = True
    config.CONTACT_CONFIG.validate_checkpoint(state)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp'); torch.save(state, temp); temp.replace(target)
    write_json(Path(config.BASELINE_DIR)/'checkpoint_upgrade.json', audit)
    write_json(Path(config.BASELINE_DIR)/'run_state.json', dict(status='prepared_for_resume',
        completed_episodes=0, training_updates=state['training_updates'], checkpoint=str(target), upgrade=audit))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
