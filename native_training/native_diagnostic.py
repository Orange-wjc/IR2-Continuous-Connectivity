"""Repeated-ten-map learnability diagnostic from the equal pretrained start.

Only subsequent adaptation is restricted to ten maps. The common policy was
already pretrained on 100 native maps; this is not training from random weights.
No replay, success trajectories or adaptation logs from those maps are imported.
"""
import copy
import fcntl
from pathlib import Path
import shutil
import torch
from finite_contact.config import Config
from finite_contact.memory import load_checkpoint,semantic_digest
from native_data import digest
from native_continuation import ALGORITHM_FILES,preserved_state


def prepare_diagnostic(root,args):
    from native_driver import read,write,identity,now,SOURCE,GROUPS
    if not args.from_run or args.train_maps!=10 or not args.continuous:
        raise ValueError('The ten-map diagnostic requires --continuous --train-maps 10 --from-run')
    if args.pilot_episodes is not None or args.rollout_wave!=1 or args.microbatch<1:
        raise ValueError('Keep unlimited episodes, wave=1 and a positive microbatch')
    assert args.bootstrap_prepared
    assert {p.name for p in root.iterdir()}=={'source','bootstrap_identity.json','entry_provenance'}
    assert identity()==read(root/'bootstrap_identity.json')
    parent=args.from_run.resolve();assert parent!=root
    lock=(parent/'.coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(parent/'pipeline.json')
    assert old['status']=='pilot_complete' and not old['preflight'] and old['train_maps']==100
    assert old['completed_episodes']=={name:100 for name in GROUPS}
    assert args.seed==old['seed'] and args.microbatch==old['microbatch']
    assert old['test_policy_evaluations']==0 and not old['stage2_started']
    paths=['pipeline.json','source_identity.json','dataset/native_dataset.json','common_start.pth','head_complete.json',
           *(name+'/latest.pth' for name in GROUPS)]
    before={name:digest(parent/name) for name in paths}
    source=read(parent/'source_identity.json');assert source==old['source_identity']
    assert all(digest(parent/'source'/name)==sha for name,sha in source.items())
    assert all(digest(SOURCE/name)==source[name] for name in ALGORITHM_FILES)
    assert digest(args.dataset)==old['dataset_sha256']==before['dataset/native_dataset.json']
    manifest=read(args.dataset)
    splits=manifest['families']['easy']['splits'];assert len(splits['train'])==100 and len(splits['validation'])==100
    selected=copy.deepcopy(splits['train'][:10]);splits['train']=selected
    assert not ({e['geometry_group'] for e in selected}&{e['geometry_group'] for e in splits['validation']})
    for e in selected+splits['validation']:
        assert digest(Path(manifest['repository'])/e['path'])==e['file_sha256']
    manifest['pilot']=dict(parent_manifest=str(args.dataset.resolve()),parent_sha256=digest(args.dataset),
        training_selection='First 10 entries of the reviewed 100-map pool; no performance-based selection',
        selected_train_maps=10,generated_maps=0,development_maps=100,
        diagnostic_only=True,prior_shared_pretraining_maps=100,adaptation_only_uses_selected_maps=True)
    (root/'dataset').mkdir();write(root/'dataset/native_dataset.json',manifest)
    inherited=root/'inherited';inherited.mkdir()
    raw=inherited/'common_start.pth';shutil.copy2(parent/'common_start.pth',raw)
    marker=read(parent/'head_complete.json');assert digest(raw)==marker['sha256']
    saved=load_checkpoint(raw)
    assert saved['episode']==0 and saved['role']=='common_native_easy'
    assert not saved['replay'] and not saved['content_replay'] and not saved['success_episodes']
    assert saved['learner']['updates']==9000 and saved['learner']['content_updates']==5000
    expected=semantic_digest(preserved_state(saved))
    cfg=Config.from_checkpoint(saved['config']);original=cfg.to_dict()
    cfg.dataset_manifest=str((root/'dataset/native_dataset.json').resolve())
    cfg.train_maps=10;cfg.success_episode_limit=10
    if args.preflight:cfg.max_decisions=8
    cfg.validate()
    changes={key:dict(previous=value,new=cfg.to_dict()[key]) for key,value in original.items() if value!=cfg.to_dict()[key]}
    allowed={'dataset_manifest','train_maps','success_episode_limit'}|({'max_decisions'} if args.preflight else set())
    assert set(changes)==allowed
    saved['config']=cfg.to_dict();saved['time']=now()
    tmp=root/'common_start.tmp';torch.save(saved,tmp,pickle_protocol=4);tmp.replace(root/'common_start.pth');del saved
    check=load_checkpoint(root/'common_start.pth')
    assert semantic_digest(preserved_state(check))==expected;del check
    marker.update(sha256=digest(root/'common_start.pth'),inherited_from=str(raw),
                  diagnostic_adaptation_maps=10,prior_shared_pretraining_maps=100)
    write(root/'head_complete.json',marker)
    audit=dict(time=now(),parent_run=str(parent),parent_checksums=before,
        initialization='Identical common pretrained learner, optimizers, targets, RNG and counters for both methods; no own-method adaptation inherited',
        common_state_semantic_sha256=expected,common_state_preserved=True,
        prior_shared_pretraining_maps=100,prior_shared_sac_updates=9000,prior_head_updates=5000,
        imported_replay=0,imported_content_replay=0,imported_success_episodes=0,imported_adaptation_log_episodes=0,
        adaptation_start_episode=0,configuration_changes=changes,algorithm_files_byte_identical=list(ALGORITHM_FILES),
        selected_maps=[e['path'] for e in selected],generated_maps=0,preflight=args.preflight,
        shared_prefix_retrained=False,head_retrained=False)
    assert before=={name:digest(parent/name) for name in before}
    write(root/'inheritance_audit.json',audit)
    plan=copy.deepcopy(old)
    for key in ['evaluation_progress','comparison_gate_passed','last_review_episode','error','parallelism_changes',
                'training_progress','active_group','paired_warmup_episodes']:
        plan.pop(key,None)
    plan.update(status='prepared',created=now(),updated=now(),stage='easy_fixed_pool_training',
        parent_run=str(parent),continuation=False,continuous_training=True,learnability_diagnostic=True,
        prior_shared_pretraining_maps=100,continuation_start_episode=0,
        continuation_start_sac_updates={name:9000 for name in GROUPS},
        train_maps=10,target_unique_train_maps=10,pilot_episodes=None,wall_guard_hours=None,
        workers=args.workers,NUM_META_AGENT=args.workers,microbatch=args.microbatch,rollout_wave=1,
        completed_episodes={name:0 for name in GROUPS},sac_updates={name:9000 for name in GROUPS},
        automatic_development_evaluation=False,endpoint_development_evaluation=False,evaluation_policy='manual_only',
        success_rate_goal=.99,stop_on_success_goal=False,stop_after_cycles=None,
        cross_method_prefetch=not args.no_prefetch,paired_learning_enabled=True,preflight=args.preflight,
        updates_per_completed_episode=2 if args.preflight else 20,
        source_identity=identity(),dataset_sha256=digest(root/'dataset/native_dataset.json'),
        inheritance_audit_sha256=digest(root/'inheritance_audit.json'),
        training_status='Repeated-ten-map diagnostic from the equal pretrained start; empty replay and success cache; manual evaluation only',
        inherited_log_episodes=0,large_scale_started=False,full_dataset_started=False,stage2_started=False,notifications=False)
    write(root/'source_identity.json',plan['source_identity']);write(root/'pipeline.json',plan)
    print(__import__('json').dumps(dict(stage='diagnostic_prepared',fixed_maps=10,prior_shared_pretraining_maps=100,
        from_episode=0,from_sac_updates=9000,imported_replay=0,episode_limit=None,automatic_evaluation=False)),flush=True)
    return plan
