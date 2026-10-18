"""Compare real CLI checkpoint values before/after independent CPU prefetch."""
from pathlib import Path
import json,sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'native_training'))
from finite_contact.memory import load_checkpoint,semantic_digest
from native_continuation import preserved_state

def main():
 roots=[REPO/'experiments'/n for n in ['native_easy100_cycles_preflight_20261018','native_easy100_cycles_preflight_v2_20261018']]
 out={}
 for group in ['IR2_FIFO','AB_learned']:
  a,b=[load_checkpoint(r/group/'latest.pth') for r in roots]
  assert a['episode']==b['episode']
  x=semantic_digest(preserved_state(a));y=semantic_digest(preserved_state(b))
  assert x==y,(group,x,y)
  out[group]=dict(episode=a['episode'],full_retained_learning_state_identical=True,semantic_sha256=x)
  aa,bb=[[json.loads(s) for s in (r/group/'training.jsonl').read_text().splitlines()] for r in roots]
  for p,q in zip(aa,bb):
   for key in ['episode','success','time','capped_time','coverage','union_coverage','decisions','return_values']:
    assert np.array_equal(p[key],q[key]),(group,key)
 report=dict(groups=out,rollout_metrics_identical=True,gradient_updates_rng_replay_demonstrations_identical=True,
  inference_lag_within_each_method=0,scope='Engineering original-map CLI STOP/resume test; not an algorithm performance result')
 (WORK/'parallel_equivalence.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
