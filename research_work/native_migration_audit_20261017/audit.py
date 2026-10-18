"""Offline migration audit: source/weight/replay checks, no rollouts or updates."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/tmp/ir2-migration-audit-mpl'
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

WORK=Path(__file__).resolve().parent
REPO=WORK.parents[1]
FORMAL=REPO/'experiments/native_easy_20261015_seed101'
OLDROOT=REPO/'experiments/independent_seeds_20261013'
OLD=OLDROOT/'source_audit_resume_20261014'
NEW=FORMAL/'source'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value):(WORK/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def method(source,file,cls,name):
    tree=ast.parse((source/file).read_text())
    nodes=tree.body if cls is None else next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls).body
    return next(x for x in nodes if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)


def fixture(source):
    sys.path.insert(0,str(source))
    import numpy as np
    from finite_contact.config import Config
    from finite_contact.environment import Environment
    cfg=Config();cfg.map_size=256;cfg.nodes=8;cfg.grid_points=2;cfg.neighbors=4
    cfg.bytes_per_second=64.;cfg.estimated_contact_seconds=3.
    cfg.native_graph=False;cfg.dynamic_padding=False
    env=Environment.__new__(Environment);env.config=cfg
    coords=np.array([[20,20],[80,20],[140,20]])
    edges={tuple(coords[0]):{tuple(coords[1]):60.},tuple(coords[1]):{tuple(coords[0]):60.,tuple(coords[2]):60.},tuple(coords[2]):{tuple(coords[1]):60.}}
    env.graph_data=[(coords,edges,np.array([11,22,33]),np.zeros(3))]*2
    env.truth=np.full((256,256),255,np.uint8)
    env.beliefs=[np.full((256,256),127,np.uint8) for _ in range(2)]
    env.positions=coords[[0,2]].astype(float);env.time=0.;env.wait=np.zeros(2)
    dtype=[('index','<i4'),('value','u1')]
    cell_sets=[np.array([(20*256+20+i,255) for i in range(4)],dtype=dtype),
               np.array([(20*256+80+i,255) for i in range(8)],dtype=dtype)]
    packets=[SimpleNamespace(seq=i,created=0.,size=size,offset=0,
             cells=lambda data=data:data) for i,(data,size) in enumerate(zip(cell_sets,(72,96)))]
    for data in cell_sets:env.beliefs[0].ravel()[data['index']]=data['value']
    env.radio=SimpleNamespace(peer_pose={0:np.array([140.,20.,0.,0.]),1:np.array([20.,20.,0.,0.])},
        peer_pose_time={0:0.,1:0.},known_peer={(0,1):np.full((256,256),127,np.uint8),(1,0):np.full((256,256),127,np.uint8)},
        pending=lambda sender:packets,active={(0,1):None,(1,0):None},
        queues={(0,1):packets,(1,0):packets},tick_index=1)
    env.frontiers=lambda robot:np.array([[20,30],[120,20]])
    env.feature_provider=lambda robot,ids:np.tile(np.arange(8,dtype=float)/100,(len(ids),1))
    env.scorer=lambda features,context:np.array([.001,.06])
    env._rank_cache={}
    output={}
    variants={'B0_fifo':('B0','fifo',True),'B_learned':('B','learned',True),
              'A_fifo':('A','fifo',True),'AB_learned':('AB','learned',True),'AB_uncoupled':('AB','learned',False)}
    for name,(variant,content,coupled) in variants.items():
        cfg.variant,cfg.content,cfg.coupled=variant,content,coupled
        env._rank_cache={}
        observation=env.observation(0)
        scores=env.packet_scores(0,packets)
        output[name]=dict(scores=scores.tolist(),rank=env.rank_packet(0,packets),
            expected_delivery=env.expected_delivery(0),features=observation[0][:3].tolist(),
            candidates=observation[1].tolist(),mask=observation[4].tolist())
    assert output['B0_fifo']['rank']==output['A_fifo']['rank']==0
    assert output['AB_learned']['rank']==output['B_learned']['rank']==output['AB_uncoupled']['rank']==1
    assert abs(output['AB_learned']['expected_delivery']-6.)<1e-6
    assert abs(output['AB_uncoupled']['expected_delivery']-4.)<1e-6
    assert output['AB_learned']['features']!=output['AB_uncoupled']['features']
    cfg.variant,cfg.content,cfg.coupled='AB','learned',True
    before=env.observation(0)[0].copy()
    env.beliefs[1].fill(255);env.truth.fill(1);env._rank_cache={}
    assert np.array_equal(before,env.observation(0)[0]),'Unshared peer map/truth leaked into actor features'
    env.wait[0]=cfg.max_wait_seconds+1
    candidates,mask=env.action_inputs(0)
    assert mask[0,0]==1 and np.any(mask[0,1:]==0)
    output['checks']=dict(learned_ranking_not_FIFO=True,coupling_changes_deliverable_gain=True,
        nonshared_truth_and_peer_belief_do_not_change_features=True,wait_is_explicit_and_bounded=True)
    if source==NEW:
        from native_environment import refresh_native_graphs
        cfg.native_graph=True
        env.truth.fill(255)
        for belief in env.beliefs:belief.fill(255)
        env.positions=coords[[1,2]].astype(float)
        env.radio.peer_pose[0][:2]=env.positions[1];env.radio.peer_pose[1][:2]=env.positions[0]
        env.native_previous_frontiers=[None,None]
        env.native_old_lengths=[[[0,0],[0,0]] for _ in range(2)]
        env.native_steps=0;env.map_info={'index':0}
        env.graphs=[SimpleNamespace(route_node=[env.positions[i]],
            update_graph=lambda *args,**kwargs:(True,coords,edges,np.array([11,22,33]),np.array([0,1,2]))) for i in range(2)]
        refresh_native_graphs(env)
        for robot in range(2):
            c,e,u,v=env.graph_data[robot]
            assert np.array_equal(c[0],env.positions[robot])
            assert u[0]==(22 if robot==0 else 33)
            candidates,mask=env.action_inputs(robot)
            assert candidates[0,0]==0
            obs=env.observation(robot)
            for index in candidates[0,mask[0]==0]:assert 0<=index<len(c)
        output['checks']['native_graph_current_node_reorder_preserves_action_indices']=True
    print(json.dumps(output,allow_nan=False))


def checkpoints():
    sys.path.insert(0,str(NEW))
    import numpy as np
    from finite_contact.memory import load_checkpoint,semantic_digest
    from finite_contact.learning import dense_observation
    from native_worker import RolloutPolicy
    import torch
    torch.set_num_threads(1)
    common=load_checkpoint(FORMAL/'common_start.pth')
    warmup=load_checkpoint(FORMAL/'shared_warmup/latest.pth')
    report={'native':{},'generated':{},'common_start_updates':common['learner']['updates'],
        'common_context_matches_warmup_policy':semantic_digest(common['learner']['context_encoder'])==semantic_digest(warmup['learner']['policy']),
        'trained_head_differs_from_pre_head':semantic_digest(common['learner']['content'])!=semantic_digest(warmup['learner']['content'])}
    assert report['common_context_matches_warmup_policy'] and report['trained_head_differs_from_pre_head']
    del warmup
    for label,root,names in [('native',FORMAL,('IR2_FIFO','AB_learned')),
                             ('generated',OLDROOT/'seed_101',('B0_fifo','AB_learned'))]:
        for group in names:
            saved=load_checkpoint(root/group/'latest.pth')
            cfg=saved['config'];state=saved['learner']
            value=dict(episode=saved['episode'],updates=state['updates'],
                variant=cfg['variant'],content=cfg['content'],coupled=cfg['coupled'],
                content_updates=state['content_updates'],imitation_updates=state.get('imitation_updates',0),
                successful_episodes_stored=len(saved.get('success_episodes',[])),
                log_alpha=float(state['log_alpha'].reshape(-1)[0]),
                optimizer_lrs={key:state[key]['param_groups'][0]['lr'] for key in ('policy_optimizer','alpha_optimizer')},
                critic_lr=state['q_optimizers'][0]['param_groups'][0]['lr'],
                content_adam_steps=sorted({int(x['step']) for x in state['content_optimizer']['state'].values()}),
                policy_adam_steps=sorted({int(x['step']) for x in state['policy_optimizer']['state'].values()}),
                replay_size=len(saved.get('replay',[])))
            if label=='native':
                value.update(content_equals_common=semantic_digest(state['content'])==semantic_digest(common['learner']['content']),
                    context_equals_common=semantic_digest(state['context_encoder'])==semantic_digest(common['learner']['context_encoder']),
                    policy_differs_from_common=semantic_digest(state['policy'])!=semantic_digest(common['learner']['policy']))
                assert value['content_equals_common'] and value['context_equals_common'] and value['policy_differs_from_common']
                assert value['policy_adam_steps']==[80632] and value['content_updates']==5000
                assert value['content_adam_steps']==[5000]
            sampled=[]
            replay=saved.get('replay',[])
            for i in np.linspace(0,len(replay)-1,min(512,len(replay))).astype(int):
                obs=dense_observation(replay[i][0]);features=obs[0];valid=obs[3][0]==0
                indices=obs[1][0][obs[4][0]==0]
                sampled.append(dict(any_positive_rendezvous=bool(np.any(features[valid,3]>1e-8)),
                    legal_candidate_positive_rendezvous=bool(np.any(features[indices,3]>1e-8)),
                    rendezvous_max=float(features[valid,3].max())))
            value['stored_replay_rendezvous_diagnostic']=dict(sample_count=len(sampled),
                any_positive_fraction=float(np.mean([x['any_positive_rendezvous'] for x in sampled])),
                legal_candidate_positive_fraction=float(np.mean([x['legal_candidate_positive_rendezvous'] for x in sampled])),
                mean_max=float(np.mean([x['rendezvous_max'] for x in sampled])),
                caveat='Different replay maps/states; this is signal presence, not causal effect or performance')
            if label=='native' and group=='AB_learned':
                from finite_contact.config import Config
                from finite_contact.learning import tensors
                policy=RolloutPolicy(Config.from_checkpoint(cfg),{k:state[k] for k in ('policy','context_encoder','content')})
                differences=[]
                for index in np.linspace(0,len(replay)-1,16).astype(int):
                    obs=dense_observation(replay[index][0]);changed=[a.copy() for a in obs];changed[0][:,3]=0
                    with torch.no_grad():
                        original=policy.policy(*tensors([obs],'cpu')).exp()
                        removed=policy.policy(*tensors([tuple(changed)],'cpu')).exp()
                    differences.append(float(torch.abs(original-removed).max()))
                value['rendezvous_feature_forward_sensitivity']=dict(stored_observations=16,
                    changed_distributions=sum(x>1e-7 for x in differences),mean_max_probability_change=float(np.mean(differences)),
                    maximum_probability_change=float(max(differences)),
                    scope='Offline actor forward only, no sampled action, no environment step and no SAC')
            report[label][group]=value
            del saved,state
    write('checkpoint_review.json',report)


def main():
    plan=read(FORMAL/'pipeline.json')
    assert plan['status']=='paused_for_easy_review' and (FORMAL/'STOP').exists()
    assert Path('/home/wyz/wjc_ws/training_supervisor_native_easy/PAUSE').exists()
    protected=[FORMAL/'pipeline.json',FORMAL/'source_identity.json',FORMAL/'dataset/native_dataset.json',FORMAL/'latest_comparison.json',FORMAL/'common_start.pth']
    protected += [FORMAL/g/name for g in plan['groups'] for name in ('latest.pth','initial.pth')]
    before={str(p):sha(p) for p in protected}
    old_identity=read(OLDROOT/'source_identity_audit_resume_20261014.json')
    for source,identity in ((OLD,old_identity),(NEW,plan['source_identity'])):
        for rel,checksum in identity.items():assert sha(source/rel)==checksum,(source,rel)
    functions={}
    for cls,names,file in [('Environment',('packet_scores','rank_packet','expected_delivery','collect_labels','coverage','frontier_potential'),'finite_contact/environment.py'),
                           ('Learner',('act','contexts','score','update_content'),'finite_contact/learning.py')]:
        for name in names:
            a,b=[method(src,file,cls,name) for src in (OLD,NEW)]
            functions[cls+'.'+name]=dict(identical_ast=ast.dump(a)==ast.dump(b),old_line=a.lineno,native_line=b.lineno)
            assert functions[cls+'.'+name]['identical_ast']
    files={name:dict(identical=(OLD/name).read_bytes()==(NEW/name).read_bytes(),
        generated_sha256=sha(OLD/name),native_sha256=sha(NEW/name)) for name in
        ('model.py','finite_contact/communication.py','finite_contact/memory.py','finite_contact/sensing.py')}
    for value in files.values():assert value['identical']
    fixture_results={}
    for label,source in (('generated',OLD),('native',NEW)):
        run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fixture',str(source)],
                           check=True,text=True,capture_output=True)
        fixture_results[label]=json.loads(run.stdout)
    for variant in ('B0_fifo','B_learned','A_fifo','AB_learned','AB_uncoupled'):
        assert fixture_results['generated'][variant]==fixture_results['native'][variant],variant
    write('functional_fixture_review.json',fixture_results)
    checkpoints()
    after={str(p):sha(p) for p in protected}
    assert before==after
    for rel,checksum in plan['source_identity'].items():assert sha(NEW/rel)==checksum
    root_driver=(REPO/'driver.py').read_text();frozen_driver=(NEW/'driver.py').read_text()
    native_update=method(NEW,'native_learning.py','NativeLearner','update')
    native_commit=method(NEW,'native_driver.py',None,'commit_episode')
    entry=dict(root_driver_has_finite_contact_dispatch='--finite-contact' in root_driver,
        frozen_driver_has_finite_contact_dispatch='--finite-contact' in frozen_driver,
        root_native_driver_exists=(REPO/'native_driver.py').exists(),
        root_driver_starts_ray_at_import=any(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call)
            and isinstance(x.value.func,ast.Attribute) and isinstance(x.value.func.value,ast.Name)
            and x.value.func.value.id=='ray' and x.value.func.attr=='init' for x in ast.parse(root_driver).body),
        actually_trained_driver=str(NEW/'driver.py'),argument='--finite-contact',
        root_driver_imported_or_executed_in_this_audit=False)
    report=dict(status='passed',scope='Read-only migration/code/weight checks, not a proof of equal learning dynamics or effect',
        checked_generated_source_files=len(old_identity),checked_native_source_files=len(plan['source_identity']),
        core_functions=functions,identical_files=files,entry=entry,
        native_update_references_success_imitation='success_imitation' in ast.unparse(native_update) if hasattr(ast,'unparse') else 'success_imitation' in (NEW/'native_learning.py').read_text(),
        native_commit_remembers_success=any(isinstance(n,ast.Attribute) and n.attr=='remember_success' for n in ast.walk(native_commit)),
        unchanged_protected_files=before,source_and_model_files_unchanged=True,
        new_training_rollouts=0,new_development_rollouts=0,new_test_rollouts=0,new_sac_updates=0,
        native_actor_forward_probes=32,fixture_environment_steps=0,training_remains_paused=True)
    write('audit_summary.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture':fixture(Path(sys.argv[2]).resolve())
    else:main()
