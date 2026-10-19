"""Prepare an explicit v6->v7 continuation; never starts training."""
from pathlib import Path
import argparse,hashlib,json,os,sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--source-manifest')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root))
    os.environ['IR2_METHOD']='finite_learned';os.environ['IR2_RUN_NAME']=args.run
    import torch
    import parameter as config
    from baseline_support import prepare_maps,write_json
    from contact_module.state_upgrade import expand_checkpoint
    torch.set_num_threads(1);source=Path(args.source).resolve();target=Path(config.MODEL_PATH)
    if target.exists():raise FileExistsError(target)
    saved=torch.load(source)  # Preserve native device tags for ordinary driver resume.
    manifest=prepare_maps(config)
    parent=Path(args.source_manifest) if args.source_manifest else Path('experiments')/source.parent.name/'map_manifest.json'
    recorded=json.loads(parent.read_text());fingerprints=lambda m:{x['name']:x['sha256'] for x in m['maps']}
    if fingerprints(manifest)!=fingerprints(recorded):raise ValueError('Original training map bytes changed')
    audit=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),from_schema=6,to_schema=7,
        reason='Add explicit own paid-receive/queue state through zero6->128 projections; graph6, actions, rewards, physics and SAC equations unchanged',
        inherited_completed_episodes=saved['completed_episodes'],inherited_training_updates=saved['training_updates'],
        inherited_head_updates=saved['contact_learning']['updates'],
        preserved='all old actor/critic/temperature tensors and moments/schedulers, frozen graph context, head/replay/RNG',
        added='one zero projection and fresh Adam parameter ID per movement model; zero graph-context projection',
        reset='new-condition episode/visits/acceptance; movement replay rebuilds as original driver',
        new_conditions_only_acceptance=True)
    state=expand_checkpoint(saved,config.CONTACT_CONFIG,[x['name'] for x in manifest['maps']],audit)
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix('.tmp')
    torch.save(state,temporary);temporary.replace(target)
    write_json(Path(config.BASELINE_DIR)/'checkpoint_upgrade.json',audit)
    write_json(Path(config.BASELINE_DIR)/'run_state.json',dict(status='prepared_for_resume',completed_episodes=0,
        training_updates=state['training_updates'],checkpoint=str(target),upgrade=audit))
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
