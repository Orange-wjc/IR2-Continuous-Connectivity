from pathlib import Path
p=Path('native_training/native_driver.py');s=p.read_text()
def replace(a,b):
 global s
 assert a in s,a
 s=s.replace(a,b)
replace('"""Native-map, paired IR2 SAC training; easy first, difficult gated by review.\n\nOriginal maps/ordering, original hierarchical graph, replay warmup 2000,\neffective batch 128, 8 updates/finished episode, hard targets every 64 updates.\nFinite-contact environment and the paired control share byte/time constraints.\n"""','"""Bounded native easy pilot: restored validated SAC/SIL recipe, then paired development review.\n\nUse 100 original training images, shared 100x90 movement updates, 100 label maps\nand 5000 head updates, followed by 100x20 paired adaptation updates. No scaling\nor difficult/test execution. CPU workers and policy-refresh wave are separate.\n"""')
replace('batch_size=128,warmup=2000,replay_size=10000','batch_size=16,warmup=16,replay_size=12000')
replace('learning_rate=1e-5,policy_learning_rate=1e-5','learning_rate=1e-4,policy_learning_rate=None')
replace("success_imitation_weight=0.,success_memory_mode='fastest',success_episode_limit=1000", "success_imitation_weight=1.,success_memory_mode='per_map',success_episode_limit=len(d['train'])")
replace('microbatch=32','microbatch=16')
replace("collect_labels=collect_labels,output=str(directory/'episodes')", "collect_labels=collect_labels,max_label_samples=2048,output=str(directory/'episodes')")
replace('writer=None,enable_updates=True):','writer=None,enable_updates=True,updates=20):')
replace("before=x.updates;x.replay.extend(item['transitions'])\n    for _ in range(8):", "before=x.updates;x.replay.extend(item['transitions'])\n    if item['metric']['success'] and cfg.success_imitation_weight>0:\n        x.remember_success(item['transitions'],item['metric']['time'],item['metric']['map']['layout_id'])\n    for _ in range(updates):")
replace("'head_updates':x.content_updates,'worker_rss_mib':item['worker_rss_mib']", "'head_updates':x.content_updates,'imitation_updates':x.imitation_updates,\n         'success_episodes':len(x.success_episodes),'worker_rss_mib':item['worker_rss_mib']")
replace("target=2 if plan['preflight'] else 32", "target=2 if plan['preflight'] else plan['train_maps']")
replace("if stop[0]:raise InterruptedError('Stop requested at warmup boundary')", "if stop[0] or (root/'STOP').exists():raise InterruptedError('Stop requested at warmup boundary')\n        memory_guard(root)")
replace("completed+1,min(plan['workers'],target-completed)", "completed+1,min(plan['rollout_wave'],target-completed)")
replace('completed,warmup_writer)',"completed,warmup_writer,updates=2 if plan['preflight'] else 90)")
replace("label_count=2 if plan['preflight'] else 16", "label_count=2 if plan['preflight'] else plan['train_maps']")
replace("if stop[0]:raise InterruptedError('Stop requested during label collection')", "if stop[0] or (root/'STOP').exists():raise InterruptedError('Stop requested during label collection')\n            memory_guard(root)")
replace("if stop[0]:\n                save_state(head_latest", "if stop[0] or (root/'STOP').exists():\n                save_state(head_latest")
replace("x.replay.clear()\n        save_state(root/'common_start.pth'", "x.replay.clear();x.success_episodes=[]\n        cfg.policy_learning_rate=1e-5\n        for group in x.policy_optimizer.param_groups:group['lr']=cfg.policy_learning_rate\n        save_state(root/'common_start.pth'")
a=s.index('def prepare(root,args):');b=s.index('\ndef evaluate_pair(',a)
s=s[:a]+'''def memory_guard(root):
    """Stop with saved boundaries before exhausting host RAM; never skip maps."""
    info={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    sample=dict(time=now(),available_memory_gib=info['MemAvailable']/1048576,
                used_memory_gib=(info['MemTotal']-info['MemAvailable'])/1048576,
                cuda_allocated_mib=torch.cuda.memory_allocated()/1048576 if torch.cuda.is_available() else 0,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576 if torch.cuda.is_available() else 0)
    append(root/'resources.jsonl',sample)
    if sample['available_memory_gib']<3:
        raise MemoryError('Available host RAM below 3 GiB; resume from saved boundary after freeing memory')


def prepare(root,args):
    if args.train_maps!=100 and not args.preflight:
        raise ValueError('This pilot is limited to the authorized 100 native easy training maps')
    if args.pilot_episodes!=100 and not args.preflight:
        raise ValueError('This pilot is limited to 100 adaptation episodes per method')
    if args.rollout_wave!=1:
        raise ValueError('This repair pilot preserves one map then updates; training wave must be 1')
    if args.microbatch<1:raise ValueError('microbatch must be positive')
    if args.bootstrap_prepared:
        allowed={'source','bootstrap_identity.json','entry_provenance'}
        assert {p.name for p in root.iterdir()}==allowed
        assert identity()==read(root/'bootstrap_identity.json')
    else:
        if root.exists() and any(root.iterdir()):raise ValueError('Fresh native output required')
        root.mkdir(parents=True)
        snapshot=root/'source'
        for name in identity():
            p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,p)
        raise ValueError('Launch a new pilot through the repository driver.py --finite-contact')
    manifest=read(args.dataset)
    original=manifest['families']['easy']['splits']['train']
    selected=original[:args.train_maps]
    assert len(selected)==args.train_maps
    validation=manifest['families']['easy']['splits']['validation']
    assert len(validation)==100
    assert not ({e['geometry_group'] for e in selected}&{e['geometry_group'] for e in validation})
    manifest['families']['easy']['splits']['train']=selected
    manifest['pilot']=dict(parent_manifest=str(args.dataset.resolve()),parent_sha256=digest(args.dataset),
        training_selection='First 100 easy entries in the already frozen upstream reverse file ordering; no development-informed selection',
        available_original_easy_training_entries=len(original),selected_train_maps=len(selected),
        unused_training_maps=len(original)-len(selected),development_maps=len(validation),generated_maps=0)
    (root/'dataset').mkdir();write(root/'dataset/native_dataset.json',manifest)
    plan=dict(status='prepared',created=now(),stage='easy_shared_warmup',seed=args.seed,
        groups=list(GROUPS),workers=args.workers,NUM_META_AGENT=args.workers,microbatch=args.microbatch,
        rollout_wave=args.rollout_wave,train_maps=args.train_maps,pilot_episodes=4 if args.preflight else args.pilot_episodes,
        effective_batch=16,replay_warmup=16,updates_per_completed_episode=20,
        warmup_updates_per_map=2 if args.preflight else 90,
        source_identity=identity(),dataset_sha256=digest(root/'dataset/native_dataset.json'),
        stage2_requires_easy_review=True,preflight=args.preflight,notifications=False,
        learning_recipe='validated_finite_contact_SAC_SIL_tau001_Qfirst_legal_entropy_bounded_alpha',
        original_training_file_counts={k:v['original_training_files'] for k,v in manifest['families'].items()},
        development_maps={k:len(v['splits']['validation']) for k,v in manifest['families'].items()},
        test_policy_evaluations=0,training_status='Bounded pilot; one sweep of 100 original easy maps per method',
        review_interval=args.review_interval,wall_guard_hours=args.max_hours,automatic_development_evaluation=False,
        cross_method_prefetch=not args.no_prefetch,paired_learning_enabled=True,
        endpoint_development_evaluation=True,large_scale_started=False,stage2_started=False,
        evaluation_gate='Audits pass, AB success count >= FIFO, strictly lower failure-capped mean time, strictly lower common-success mean and median time; one-seed pilot evidence only',
        task='two robots; identical paid finite-contact protocol; original native hierarchical graphs and 128D capacity',
        methods_note='IR2_FIFO is the capacity-matched finite-contact/FIFO control, not unlimited upstream map merger')
    write(root/'pipeline.json',plan);write(root/'source_identity.json',plan['source_identity'])
    return plan
''' +s[b:]
# Replace evaluation with restartable fixed held-out review and offline result analysis.
a=s.index('def evaluate_pair(');b=s.index('\ndef automatic_review_due',a)
s=s[:a]+'''def evaluate_pair(root,plan,pool,learners,configs,completed):
    from native_analysis import analyze
    import gzip
    point=min(completed.values());assert len(set(completed.values()))==1
    plan.update(stage='pilot_development_evaluation',updated=now());write(root/'pipeline.json',plan)
    before={name:digest(root/name/'latest.pth') for name in learners}
    for name,x in learners.items():
        cfg=configs[name];directory=root/name/('validation_%08d'%point)
        count=min(2,cfg.validation_maps) if plan['preflight'] else cfg.validation_maps
        records=[]
        for start in range(0,count,plan['workers']):
            memory_guard(root)
            indices=list(range(start,min(start+plan['workers'],count)))
            jobs=jobs_for(cfg,x,directory,start+1,len(indices),evaluation=True,indices=indices)
            if plan['preflight']:
                for job in jobs:job['split']='train'
            missing=[j for j in jobs if not (directory/'episodes'/('%08d.json.gz'%j['episode'])).exists()]
            for result in wave(pool,missing):
                assert not result['transitions'] and not result['labels']
            for job in jobs:
                with gzip.open(directory/'episodes'/('%08d.json.gz'%job['episode']),'rt') as f:
                    item=json.load(f)
                assert item['metric']['map']['index']==job['map_index']
                assert item['metric']['map']['split']==job['split']
                records.append(item['metric'])
            plan.setdefault('evaluation_progress',{})[name]=len(records)
            plan['updated']=now();write(root/'pipeline.json',plan)
            print(json.dumps(dict(stage='development',group=name,maps=len(records),total=count)),flush=True)
        write(directory/'metrics.json',records)
    assert before=={name:digest(root/name/'latest.pth') for name in learners}
    report=analyze(root,point,engineering_only=plan['preflight'])
    write(root/'latest_comparison.json',report)
    plan.update(last_review_episode=point,comparison_gate_passed=report['gate_passed'],
                stage='pilot_review_complete',status='pilot_complete',large_scale_started=False,stage2_started=False)
    write(root/'pipeline.json',plan)
    return report
''' +s[b:]
replace('updates_per_completed_episode=8))','updates_per_completed_episode=20))')
replace("'alpha','entropy','microbatch','max_nodes')", "'alpha','entropy','microbatch','max_nodes','imitation_loss','imitation_updates','success_episodes')")
replace("while not stop[0]:\n            if (root/'STOP').exists():break", "while not stop[0]:\n            memory_guard(root)\n            if min(completed.values())>=plan['pilot_episodes']:break\n            if (root/'STOP').exists():break")
replace("count=next_wave_size(completed,name,plan['workers'])", "count=min(next_wave_size(completed,name,plan['rollout_wave']),plan['pilot_episodes']-completed[name])")
replace("count=next_wave_size(hypothetical,next_name,plan['workers'])", "count=min(next_wave_size(hypothetical,next_name,plan['rollout_wave']),plan['pilot_episodes']-hypothetical[next_name])")
replace("finished_probe=plan['preflight'] and min(hypothetical.values())>=4", "finished_probe=min(hypothetical.values())>=plan['pilot_episodes']")
replace("enable_updates=plan.get('paired_learning_enabled',False))", "enable_updates=plan.get('paired_learning_enabled',False),updates=2 if plan['preflight'] else 20)")
replace("'last_worker_peak_rss_mib'", "'last_worker_peak_rss_mib'") if False else None
replace("if plan['preflight'] and point>=4:\n                evaluate_pair(root,plan,pool,learners,configs,completed);break", "if point>=plan['pilot_episodes']:break")
replace("plan.update(status='preflight_complete' if plan['preflight'] else 'paused_for_easy_review',\n                    stage='easy',updated=now(),stage2_started=False)", "if min(completed.values())>=plan['pilot_episodes'] and not stop[0] and not (root/'STOP').exists():\n            evaluate_pair(root,plan,pool,learners,configs,completed)\n        else:\n            plan.update(status='pilot_paused',stage='easy',updated=now(),stage2_started=False)")
replace("p.add_argument('--microbatch',type=int,default=32)", "p.add_argument('--microbatch',type=int,default=16)\n    p.add_argument('--train-maps',type=int,default=100)\n    p.add_argument('--pilot-episodes',type=int,default=100)\n    p.add_argument('--rollout-wave',type=int,default=1)\n    p.add_argument('--bootstrap-prepared',action='store_true',help=argparse.SUPPRESS)")
p.write_text(s)
