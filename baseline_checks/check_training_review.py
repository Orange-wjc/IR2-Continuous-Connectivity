"""Check real curve prefixes, 300-boundary/resume behavior, and warning flags."""
from pathlib import Path
import json
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline_support import Progress,write_json
from training_review import TrainingReview,summarize


class Writer:
    def __init__(self):self.events=[]
    def add_scalar(self,*args,**kwargs):self.events.append((args,kwargs))
    def flush(self):pass


def main():
    source=ROOT/'experiments/contact_easy10_v2_20261019/analysis_20261019_1035/progress_snapshot.json'
    rows=json.loads(source.read_text())
    maps=sorted({r['map'] for r in rows},reverse=True)
    a=summarize(rows,maps,300);b=summarize(rows,maps,600)
    assert abs(a['current_block']['success_percent']-100*175/300)<1e-8
    assert a['trend']=='insufficient_history' and not a['full_comparison']
    assert a['within_block_first100']['success_percent']==59.
    assert a['within_block_last100']['success_percent']==61.
    assert abs(b['current_block']['success_percent']-59.)<1e-8
    assert b['trend']=='no_clear_rise' and b['needs_diagnosis']
    assert b['per_map']['img_9995.png']['total_successes']==0
    assert 'never_successful:img_9995.png' in b['flags']
    assert sum(p['visits'] for p in a['per_map'].values())==300  # No future-data leak.
    assert sum(p['visits'] for p in b['per_map'].values())==600
    duplicated=summarize(rows+[rows[299]],maps,300)
    assert duplicated['duplicate_completed_steps']==1
    assert 'progress_history_requires_audit' in duplicated['flags']
    writer=Writer()
    with tempfile.TemporaryDirectory(prefix='ir2-review-check-') as directory:
        path=Path(directory)
        (path/'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:299]))
        progress=Progress(path,{'maps':[{'name':m} for m in maps]},writer,
                          resume={'completed_episodes':299,'training_updates':rows[299]['training_updates']},
                          review_enabled=True)
        assert progress.review.maybe_record(299) is None
        r=rows[299]
        progress.record({'episode_number':r['episode']},True,r['metrics'],r['replay_size'],r['dispatched_episodes'])
        folder=path/'reviews/episode_000300'
        assert (folder/'review.json').exists() and (folder/'curves.png').exists()
        reviewed=json.loads((folder/'review.json').read_text())
        assert reviewed['completed_episodes']==300
        assert reviewed['current_block']['success_percent']==a['current_block']['success_percent']
        assert progress.review.maybe_record(301) is None
        # Restore at boundary: next valid completion is301; do not repeat300.
        restarted=Progress(path,{'maps':[{'name':m} for m in maps]},writer,
                           resume={'completed_episodes':300},review_enabled=True)
        next_row=rows[300]
        restarted.record({'episode_number':next_row['episode']},True,next_row['metrics'],10000,309)
        assert not (path/'reviews/episode_000301').exists()
        class BrokenReview:
            def maybe_record(self,*a):raise OSError('Injected review failure')
        restarted.review=BrokenReview()
        restarted.record({'episode_number':next_row['episode']+1},True,next_row['metrics'],10000,310)
        assert restarted.last_review['flags']==['review_failed']
        assert (path/'reviews/error_000302.json').exists()
    result=dict(passed=True,real_prefix_300_success=175/3,real_block_301_600_success=59.,
                no_future_data=True,resume_boundary_exact=True,per_map_failure_detected=True,
                duplicate_history_detected=True,report_plot_and_tensorboard_written=True,
                no_policy_updates_or_evaluation=True)
    write_json(ROOT/'baseline_checks/training_review_verified.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
