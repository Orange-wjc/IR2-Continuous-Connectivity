"""Acceptance must count real attempts, persist through resume, and cover10 maps."""
from pathlib import Path
import sys,tempfile,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baseline_support import Progress,write_json


class Writer:
    def add_scalar(self,*a,**k):pass
    def flush(self):pass


class Review:
    def maybe_record(self,*a):return None


def check(path,failures=3,invalid=False,missing=False,reward_anomaly=False):
    path.mkdir(parents=True)
    maps=['map_{}'.format(i) for i in range(10)]
    attempts=[dict(episode=i,map=maps[i%(9 if missing else 10)],valid=True,success=i>=failures)
              for i in range(299)]
    if invalid:attempts[0]['valid']=False
    saved=dict(recent=[a['success'] for a in attempts[-100:]],visits={m:0 for m in maps},
               per_map={m:[] for m in maps},acceptance_recent=attempts)
    p=Progress(path,{'maps':[{'name':m} for m in maps]},Writer(),
               resume={'completed_episodes':299,'baseline_progress':saved},review_enabled=True)
    p.review=Review()
    assert p.snapshot()['acceptance_recent']==attempts
    p.record({'episode_number':299},True,{'success_rate':1.,'explored_rate':1.,
              'contact_positive_empty_waits':int(reward_anomaly)},10000,307)
    return p


def main():
    with tempfile.TemporaryDirectory(prefix='ir2-acceptance-check-') as directory:
        path=Path(directory)
        good=check(path/'297')
        assert good.acceptance['successes']==297 and good.acceptance['success_percent']==99.
        assert len(good.acceptance['attempts'])==300
        assert min(good.acceptance['per_map_visits'].values())==30
        assert check(path/'296',failures=4).acceptance is None
        assert check(path/'invalid',invalid=True).acceptance is None
        assert check(path/'missing',missing=True).acceptance is None
        assert check(path/'reward_anomaly',reward_anomaly=True).acceptance is None
        evidence=json.loads((path/'297/acceptance.json').read_text())
        assert evidence['successes']==297 and evidence['invalid_attempts']==0
    result=dict(passed=True,exact_297_of300_accepted=True,below_threshold_rejected=True,
                invalid_attempts_not_silently_removed=True,missing_map_rejected=True,
                window_restored_from_checkpoint=True,raw_attempt_evidence_saved=True)
    write_json(ROOT/'baseline_checks/acceptance_verified.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
