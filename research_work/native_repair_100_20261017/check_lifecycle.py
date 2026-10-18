"""Exercise root entry, graceful STOP, frozen-source resume, and finite budget."""
from pathlib import Path
import subprocess,json,time,hashlib,os
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
PYTHON='/home/wyz/anaconda3/envs/ir2_train/bin/python'
ROOT=REPO/'experiments/native_repair_100_preflight_v2_20261017'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    assert not ROOT.exists()
    log=(WORK/'lifecycle.log').open('w')
    cmd=[PYTHON,str(REPO/'driver.py'),'--finite-contact','--output',str(ROOT)]
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    proc=subprocess.Popen(cmd+['--dataset',str(REPO/'experiments/native_easy_20261015_seed101/dataset/native_dataset.json'),
        '--preflight','--num-meta-agent','2'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    stopped=False;deadline=time.monotonic()+600
    while proc.poll() is None:
        if time.monotonic()>deadline:raise RuntimeError('Lifecycle check timed out')
        if not stopped and (ROOT/'pipeline.json').exists() and read(ROOT/'pipeline.json').get('warmup_episodes',0)>=1:
            (ROOT/'STOP').touch();stopped=True
        time.sleep(.05)
    assert stopped and proc.returncode!=0
    assert read(ROOT/'pipeline.json')['status']=='interrupted_or_failed'
    frozen=read(ROOT/'source_identity.json')
    (ROOT/'STOP').unlink()
    resumed=subprocess.run(cmd+['--resume'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    assert resumed.returncode==0
    plan=read(ROOT/'pipeline.json');assert plan['status']=='pilot_complete'
    assert plan['completed_episodes']=={'IR2_FIFO':4,'AB_learned':4}
    assert plan['sac_updates']=={'IR2_FIFO':12,'AB_learned':12}
    assert plan['warmup_sac_updates']==4 and not plan['large_scale_started'] and not plan['stage2_started']
    protected={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'IR2_FIFO/latest.pth',ROOT/'AB_learned/latest.pth']}
    result=subprocess.run(cmd+['--resume'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0 and protected=={p:sha(ROOT/p) for p in protected}
    assert frozen==read(ROOT/'source_identity.json')
    checks=read(ROOT/'analysis/comparison.json')['engineering_checks']
    assert all(v for k,v in checks.items() if not k.endswith('successful_imitation_active'))
    report=dict(stop_checkpoint_resume=True,finite_budget=True,completed_resume_no_new_sac=True,
        frozen_source_identity=True,geometry_radio_pass=True,engineering_only=True,root=str(ROOT))
    (WORK/'lifecycle_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
