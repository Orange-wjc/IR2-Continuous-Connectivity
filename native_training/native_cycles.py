"""Fixed 100-map, unlimited-episode training from a pure 100-map checkpoint."""
import copy
import fcntl
from pathlib import Path
import shutil
import torch
from finite_contact.config import Config
from finite_contact.memory import load_checkpoint,semantic_digest
from native_data import digest
from native_continuation import ALGORITHM_FILES,preserved_state


def prepare_cycles(root,args):
    from native_driver import read,write,identity,now,SOURCE,GROUPS
    if not args.from_run:raise ValueError('--continuous requires --from-run for the reviewed 100-map start')
    if args.train_maps!=100:raise ValueError('The authorized continuous pool is exactly 100 original easy maps')
    if args.pilot_episodes is not None:raise ValueError('Continuous training has no episode cap')
    if args.rollout_wave!=1 or args.microbatch<1:raise ValueError('Keep one rollout before updates and a positive microbatch')
    if not args.bootstrap_prepared:raise ValueError('Launch via repository driver.py --finite-contact')
    assert {p.name for p in root.iterdir()}=={'source','bootstrap_identity.json','entry_provenance'}
    assert identity()==read(root/'bootstrap_identity.json')
    parent=args.from_run.resolve();assert parent!=root
    lock=(parent/'.coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(parent/'pipeline.json')
    if old['status']!='pilot_complete' or bool(old['preflight'])!=bool(args.preflight):
        raise ValueError('Need a completed 100-map parent of the same engineering/real scope')
    assert old['train_maps']==100
    start=4 if args.preflight else 100
    assert old['completed_episodes']=={name:start for name in GROUPS}
    assert len(set(old['sac_updates'].values()))==1
    assert old['test_policy_evaluations']==0 and not old['stage2_started']
    assert args.seed==old['seed'] and args.microbatch==old['microbatch']
    checks={str(p.relative_to(parent)):digest(p) for p in [parent/'pipeline.json',parent/'source_identity.json',
        parent/'dataset/native_dataset.json',parent/'common_start.pth',parent/'head_complete.json',
        *(parent/name/'latest.pth' for name in GROUPS),*(parent/name/'training.jsonl' for name in GROUPS)]}
    source=read(parent/'source_identity.json')
    assert source==old['source_identity']
    assert all(digest(parent/'source'/name)==sha for name,sha in source.items())
    assert all(digest(SOURCE/name)==source[name] for name in ALGORITHM_FILES)
    manifest=read(parent/'dataset/native_dataset.json')
    assert digest(parent/'dataset/native_dataset.json')==old['dataset_sha256']
    assert digest(args.dataset)==old['dataset_sha256'],'Use the pure 100-map parent manifest'
    train=manifest['families']['easy']['splits']['train'];dev=manifest['families']['easy']['splits']['validation']
    assert len(train)==100 and len(dev)==100
    assert not ({e['geometry_group'] for e in train}&{e['geometry_group'] for e in dev})
    for entry in train+dev:assert digest(Path(manifest['repository'])/entry['path'])==entry['file_sha256']
    (root/'dataset').mkdir();shutil.copy2(parent/'dataset/native_dataset.json',root/'dataset/native_dataset.json')
    for name in ['common_start.pth','head_complete.json']:shutil.copy2(parent/name,root/name)
    assert digest(root/'common_start.pth')==read(root/'head_complete.json')['sha256']
    inherited=root/'inherited';inherited.mkdir()
    audit=dict(time=now(),parent_run=str(parent),parent_checksums=checks,start_episode=start,
        allowed_configuration_changes=['dataset_manifest'],algorithm_files_byte_identical=list(ALGORITHM_FILES),
        pure_100_map_parent=True,shared_prefix_retrained=False,head_retrained=False,groups={})
    for name in GROUPS:
        raw=inherited/(name+'.pth');shutil.copy2(parent/name/'latest.pth',raw)
        assert digest(raw)==checks[name+'/latest.pth']
        state=load_checkpoint(raw);assert state['episode']==start and state['role']==name
        assert state['learner']['updates']==old['sac_updates'][name]
        original=Config.from_checkpoint(state['config']).to_dict()
        assert (original['variant'],original['content'])==GROUPS[name]
        assert original['train_maps']==100 and original['success_episode_limit']==100
        cfg=Config.from_checkpoint(state['config']);cfg.dataset_manifest=str((root/'dataset/native_dataset.json').resolve());cfg.validate()
        changes={k:dict(previous=original[k],new=cfg.to_dict()[k]) for k in original if original[k]!=cfg.to_dict()[k]}
        assert set(changes)=={'dataset_manifest'}
        expected=semantic_digest(preserved_state(state))
        state['config']=cfg.to_dict();state['time']=now()
        state['inherited_from']=dict(path=str(raw),sha256=digest(raw),parent_run=str(parent))
        directory=root/name;directory.mkdir()
        tmp=directory/'initial.tmp';initial=directory/'initial.pth'
        torch.save(state,tmp,pickle_protocol=4);tmp.replace(initial);del state
        check=load_checkpoint(initial);actual=semantic_digest(preserved_state(check));assert actual==expected
        audit['groups'][name]=dict(original_sha256=digest(raw),converted_sha256=digest(initial),
            inherited_state_semantic_sha256=actual,configuration_changes=changes,state_preserved=True,
            start_episode=check['episode'],start_sac_updates=check['learner']['updates'],
            retained_replay=len(check['replay']),retained_success_episodes=len(check['success_episodes']))
        del check
        shutil.copy2(initial,directory/'latest.pth')
        records=[__import__('json').loads(line) for line in (parent/name/'training.jsonl').read_text().splitlines() if line.strip()]
        assert [r['episode'] for r in records]==list(range(1,start+1))
        for filename in ['training.jsonl','updates.jsonl']:
            if (parent/name/filename).exists():shutil.copy2(parent/name/filename,directory/filename)
        write(directory/'status.json',dict(status='prepared_continuous',episodes=start,sac_updates=old['sac_updates'][name],updated=now()))
    assert checks=={p:digest(parent/p) for p in checks}
    write(root/'inheritance_audit.json',audit)
    plan=copy.deepcopy(old)
    for key in ['evaluation_progress','comparison_gate_passed','last_review_episode','error','parallelism_changes']:
        plan.pop(key,None)
    plan.update(status='prepared',stage='easy_fixed_pool_training',created=now(),updated=now(),
        parent_run=str(parent),continuation=True,continuous_training=True,
        continuation_start_episode=start,continuation_start_sac_updates=old['sac_updates'],
        train_maps=100,target_unique_train_maps=100,pilot_episodes=None,wall_guard_hours=None,
        workers=args.workers,NUM_META_AGENT=args.workers,microbatch=args.microbatch,rollout_wave=1,
        completed_episodes={name:start for name in GROUPS},sac_updates=old['sac_updates'].copy(),
        automatic_development_evaluation=False,endpoint_development_evaluation=False,evaluation_policy='manual_only',
        success_rate_goal=.99,stop_on_success_goal=False,stop_after_cycles=None,
        cross_method_prefetch=not args.no_prefetch,paired_learning_enabled=True,
        source_identity=identity(),dataset_sha256=digest(root/'dataset/native_dataset.json'),
        inheritance_audit_sha256=digest(root/'inheritance_audit.json'),
        training_status='Cycle the same 100 original easy maps without a planned episode/time/success stop; user watches TensorBoard and requests evaluation',
        inherited_log_episodes=start,large_scale_started=False,full_dataset_started=False,stage2_started=False,notifications=False)
    write(root/'source_identity.json',plan['source_identity']);write(root/'pipeline.json',plan)
    print(__import__('json').dumps(dict(stage='continuous_prepared',fixed_maps=100,from_episode=start,
        episode_limit=None,wall_guard_hours=None,automatic_evaluation=False,stop_on_99_percent=False)),flush=True)
    return plan
