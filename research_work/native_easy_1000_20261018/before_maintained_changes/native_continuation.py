"""Audited, lossless per-method continuation into a larger original easy subset."""
import copy
import fcntl
from pathlib import Path
import shutil
import torch
from finite_contact.config import Config
from finite_contact.memory import load_checkpoint, semantic_digest
from native_data import digest

CONFIG_CHANGES={'dataset_manifest','train_maps','success_episode_limit'}
ALGORITHM_FILES=('native_learning.py','native_worker.py','native_environment.py','model.py',
    'graph_generator.py','native_data.py','finite_contact/config.py','finite_contact/environment.py',
    'finite_contact/learning.py','finite_contact/communication.py','finite_contact/memory.py',
    'finite_contact/maps.py','finite_contact/sensing.py')
STATE_KEYS=('learner','replay','content_replay','success_episodes','episode','role','rng')


def preserved_state(state):
    return {key:state[key] for key in STATE_KEYS}


def prepare_continuation(root,args):
    from native_driver import read,write,identity,now,SOURCE,GROUPS
    parent=args.from_run.resolve()
    if root==parent:raise ValueError('Continuation must use a new output; parent results are immutable')
    if not args.bootstrap_prepared:raise ValueError('Launch via repository driver.py --finite-contact')
    assert {p.name for p in root.iterdir()}=={'source','bootstrap_identity.json','entry_provenance'}
    assert identity()==read(root/'bootstrap_identity.json')
    if args.rollout_wave!=1 or args.microbatch<1:raise ValueError('Preserve wave=1 and use a positive microbatch')
    if args.train_maps!=500:raise ValueError('This continuation is authorized for exactly 500 original easy training files')
    expected_target=6 if args.preflight else 500
    if args.pilot_episodes!=expected_target:
        raise ValueError('Expected endpoint 500 (engineering continuation preflight: 6)')
    lock=(parent/'.coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(parent/'pipeline.json')
    if old['status']!='pilot_complete' or bool(old['preflight'])!=bool(args.preflight):
        raise ValueError('Need a completed parent of the same engineering/real scope')
    start=4 if args.preflight else 100
    assert old['completed_episodes']=={name:start for name in GROUPS}
    assert len(set(old['sac_updates'].values()))==1
    assert old['test_policy_evaluations']==0 and not old['stage2_started']
    if args.seed!=old['seed']:raise ValueError('Continuation must preserve the parent training seed')
    if args.microbatch!=old['microbatch']:raise ValueError('Preserve the effective/microbatch settings')
    before={str(p.relative_to(parent)):digest(p) for p in [parent/'pipeline.json',parent/'source_identity.json',
        parent/'dataset/native_dataset.json',parent/'common_start.pth',parent/'head_complete.json',
        *(parent/name/'latest.pth' for name in GROUPS)]}
    source=read(parent/'source_identity.json')
    assert source==old['source_identity']
    assert all(digest(parent/'source'/name)==sha for name,sha in source.items())
    for name in ALGORITHM_FILES:
        assert digest(SOURCE/name)==source[name], 'Do not change learning/physical semantics during scale continuation: '+name
    manifest=read(args.dataset)
    parent_manifest=read(parent/'dataset/native_dataset.json')
    assert digest(parent/'dataset/native_dataset.json')==old['dataset_sha256']
    assert manifest['repository']==parent_manifest['repository']
    entries=manifest['families']['easy']['splits']['train']
    selected=entries[:args.train_maps]
    assert len(selected)==args.train_maps
    previous=parent_manifest['families']['easy']['splits']['train']
    assert selected[:start]==previous[:start]
    validation=manifest['families']['easy']['splits']['validation']
    assert validation==parent_manifest['families']['easy']['splits']['validation'] and len(validation)==100
    assert manifest['original_test_files']==parent_manifest['original_test_files']
    assert not ({e['geometry_group'] for e in selected}&{e['geometry_group'] for e in validation})
    for entry in selected+validation:
        assert digest(Path(manifest['repository'])/entry['path'])==entry['file_sha256']
    manifest['families']['easy']['splits']['train']=selected
    manifest['pilot']=dict(parent_manifest=str(args.dataset.resolve()),parent_sha256=digest(args.dataset),
        parent_run=str(parent),training_selection='First 500 entries of the frozen original train ordering; no development-based selection',
        available_original_easy_training_entries=len(entries),selected_train_maps=len(selected),
        unused_training_maps=len(entries)-len(selected),development_maps=100,generated_maps=0,
        continued_from_episode=start,additional_training_maps=expected_target-start)
    (root/'dataset').mkdir();write(root/'dataset/native_dataset.json',manifest)
    for name in ('common_start.pth','head_complete.json'):
        shutil.copy2(parent/name,root/name)
    assert digest(root/'common_start.pth')==read(root/'head_complete.json')['sha256']
    audit=dict(time=now(),parent_run=str(parent),parent_checksums=before,
        allowed_configuration_changes=sorted(CONFIG_CHANGES),algorithm_files_byte_identical=list(ALGORITHM_FILES),
        start_episode=start,endpoint=expected_target,additional_maps_per_method=expected_target-start,
        additional_sac_per_method=(expected_target-start)*(2 if args.preflight else 20),
        shared_prefix_retrained=False,head_retrained=False,generated_maps=0,groups={})
    inherited=root/'inherited';inherited.mkdir()
    for name in GROUPS:
        raw=inherited/(name+'.pth');shutil.copy2(parent/name/'latest.pth',raw)
        assert digest(raw)==before[name+'/latest.pth']
        state=load_checkpoint(raw)
        assert state['episode']==start and state['learner']['updates']==old['sac_updates'][name]
        assert state['role']==name
        original_config=Config.from_checkpoint(state['config']).to_dict()
        assert (original_config['variant'],original_config['content'])==GROUPS[name]
        cfg=Config.from_checkpoint(state['config'])
        cfg.dataset_manifest=str((root/'dataset/native_dataset.json').resolve())
        cfg.train_maps=args.train_maps;cfg.success_episode_limit=args.train_maps;cfg.validate()
        changes={k:dict(previous=original_config[k],new=cfg.to_dict()[k]) for k in original_config if original_config[k]!=cfg.to_dict()[k]}
        assert set(changes)<=CONFIG_CHANGES and all(k in changes for k in ('train_maps','success_episode_limit'))
        expected=semantic_digest(preserved_state(state))
        state['config']=cfg.to_dict()
        state['inherited_from']=dict(path=str(raw),sha256=digest(raw),parent_run=str(parent))
        state['time']=now()
        directory=root/name;directory.mkdir()
        initial=directory/'initial.pth';tmp=directory/'initial.tmp'
        torch.save(state,tmp,pickle_protocol=4);tmp.replace(initial)
        del state
        verified=load_checkpoint(initial)
        actual=semantic_digest(preserved_state(verified))
        assert actual==expected,'Continuation changed weights, Adam, replay, success demonstrations, counters, role or RNG'
        audit['groups'][name]=dict(original_sha256=digest(raw),converted_sha256=digest(initial),
            inherited_state_semantic_sha256=actual,configuration_changes=changes,
            start_episode=verified['episode'],start_sac_updates=verified['learner']['updates'],
            retained_replay=len(verified['replay']),retained_success_episodes=len(verified['success_episodes']),
            retained_imitation_updates=verified['learner']['imitation_updates'],state_preserved=True)
        del verified
        shutil.copy2(initial,directory/'latest.pth')
        write(directory/'status.json',dict(status='prepared_continuation',episodes=start,
            sac_updates=old['sac_updates'][name],updated=now()))
    assert before=={p:digest(parent/p) for p in before}
    write(root/'inheritance_audit.json',audit)
    plan=copy.deepcopy(old)
    for key in ('evaluation_progress','comparison_gate_passed','last_review_episode','error','parallelism_changes'):
        plan.pop(key,None)
    plan.update(status='prepared',created=now(),updated=now(),stage='easy_paired_training',
        parent_run=str(parent),continuation=True,continuation_start_episode=start,
        continuation_start_sac_updates=old['sac_updates'],
        target_unique_train_maps=args.train_maps,additional_training_episodes_per_method=expected_target-start,
        additional_sac_updates_per_method=audit['additional_sac_per_method'],
        workers=args.workers,NUM_META_AGENT=args.workers,microbatch=args.microbatch,
        train_maps=args.train_maps,pilot_episodes=expected_target,rollout_wave=1,
        completed_episodes={name:start for name in GROUPS},sac_updates=old['sac_updates'].copy(),
        source_identity=identity(),dataset_sha256=digest(root/'dataset/native_dataset.json'),
        inheritance_audit_sha256=digest(root/'inheritance_audit.json'),
        automatic_development_evaluation=False,endpoint_development_evaluation=True,
        cross_method_prefetch=not args.no_prefetch,wall_guard_hours=args.max_hours,
        training_status='Continue matched per-method full checkpoints from 100 to cumulative 500 original easy maps; endpoint development review only',
        large_scale_started=False,full_dataset_started=False,stage2_started=False,notifications=False)
    write(root/'source_identity.json',plan['source_identity']);write(root/'pipeline.json',plan)
    print(__import__('json').dumps(dict(stage='continuation_prepared',from_episode=start,to_episode=expected_target,
        new_maps_per_method=expected_target-start,new_sac_updates_per_method=audit['additional_sac_per_method'])),flush=True)
    return plan


def compare_parent(root,report):
    """Compare fixed development endpoints, keeping every failed map in main metrics."""
    from native_driver import read,write
    import numpy as np
    root=Path(root);plan=read(root/'pipeline.json')
    if not plan.get('continuation'):return None,''
    parent=Path(plan['parent_run']);old=read(parent/'latest_comparison.json')
    pairs=read(root/'analysis/paired_maps.json')
    old_pairs=read(parent/'analysis/paired_maps.json')
    assert [p['map'] for p in pairs]==[p['map'] for p in old_pairs]
    common=[(a,b) for a,b in zip(old_pairs,pairs) if all(p[g]['success'] for p in (a,b) for g in ('fifo','ab'))]
    by_round=[]
    for title,r in (('parent',old),('current',report)):
        a=r['groups']['IR2_FIFO'];b=r['groups']['AB_learned']
        by_round.append(dict(round=title,episode=r['easy_episodes_per_method'],fifo_successes=a['successes'],
            ab_successes=b['successes'],fifo_capped_mean=a['mean_capped_time'],ab_capped_mean=b['mean_capped_time'],
            capped_reduction_percent=100*(a['mean_capped_time']-b['mean_capped_time'])/a['mean_capped_time'],
            common_success_count=r['common_success_maps'],common_success_mean_delta=r['common_mean_time_delta'],
            four_model_common_fifo_mean=float(np.mean([x[title=='current']['fifo']['time'] for x in common])) if common else None,
            four_model_common_ab_mean=float(np.mean([x[title=='current']['ab']['time'] for x in common])) if common else None))
    result=dict(parent_run=str(parent),same_development_files=True,rounds=by_round,
        four_model_common_success_count=len(common),
        caveat='The ordinary common-success subset changes across checkpoints. Four-model intersection is auxiliary; all-map capped means remain primary. One continued seed, not independent replications.')
    write(root/'analysis/round_progress.json',result)
    text='\n## 与上一轮固定开发集端点对比\n\n|训练累计图数|基线成功|AB成功|基线罚时均值/秒|AB罚时均值/秒|AB降低比例|\n|---|---:|---:|---:|---:|---:|\n'
    for row in by_round:
        text+=f"|{row['episode']}|{row['fifo_successes']}|{row['ab_successes']}|{row['fifo_capped_mean']:.3f}|{row['ab_capped_mean']:.3f}|{row['capped_reduction_percent']:.2f}%|\n"
    text+=f"\n这是同一训练种子的延续，不是新的独立重复。两轮共同成功集合不同，不能直接将各自条件子集的耗时百分比解释为纵向改善；另保存四模型共同成功{len(common)}图的相同子集对比，见analysis/round_progress.json。全部失败均保留在主指标中。\n"
    return result,text
