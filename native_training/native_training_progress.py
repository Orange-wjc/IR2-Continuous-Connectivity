"""Training-only rolling/cycle scalars; never evaluates or stops the learner."""
from collections import deque
import json
from pathlib import Path
import numpy as np

LOSS_KEYS=('policy_loss','q1_loss','q2_loss','alpha','entropy','microbatch','max_nodes',
           'imitation_loss','imitation_updates','success_episodes')


class TrainingProgress:
    def __init__(self,directory,writer,cfg,checkpoint_episode):
        self.directory=Path(directory);self.writer=writer;self.maps=cfg.train_maps
        self.window=deque(maxlen=100);self.episode=0
        self.map_windows={i:deque(maxlen=10) for i in range(self.maps)} if self.maps==10 else {}
        self.map_visits={i:0 for i in self.map_windows}
        marker=self.directory/'tensorboard_history_seeded.json'
        backfill=not marker.exists()
        log=self.directory/'training.jsonl'
        if log.exists():
            with log.open() as handle:
                for line in handle:
                    if not line.strip():continue
                    row=json.loads(line)
                    assert row['episode']<=checkpoint_episode,'Uncommitted rows must be archived before plotting'
                    self._record(row,emit=backfill)
        assert self.episode==checkpoint_episode,(self.episode,checkpoint_episode)
        if backfill:
            updates=self.directory/'updates.jsonl'
            if updates.exists():
                with updates.open() as handle:
                    for line in handle:
                        if not line.strip():continue
                        row=json.loads(line)
                        for key in LOSS_KEYS:
                            if key in row:writer.add_scalar('losses/'+key,row[key],row['updates'])
            marker.write_text(json.dumps(dict(last_inherited_episode=self.episode,
                training_only=True,automatic_evaluation=False,stop_on_success_goal=False),indent=2)+'\n')
        elif self.window:self._emit()
        writer.flush()

    def record(self,row):
        assert row['episode']==self.episode+1,'Training episode order must remain continuous'
        self._record(row,emit=True)

    def _record(self,row,emit):
        assert row['episode']==self.episode+1
        if self.map_windows:
            index=row['map']['index']
            assert index==(row['episode']-1)%self.maps
            self.map_windows[index].append(row);self.map_visits[index]+=1
        self.window.append(row);self.episode=row['episode']
        if emit:self._emit()

    def summary(self):
        rows=list(self.window)
        n=self.episode%self.maps or self.maps
        current=rows[-n:]
        result=dict(completed_cycles=self.episode//self.maps,
            active_cycle=(self.episode-1)//self.maps+1 if self.episode else 1,
            maps_in_cycle=n if self.episode else 0,
            rolling_window_maps=len(rows),
            success_rate_100_percent=100*float(np.mean([x['success'] for x in rows])) if rows else 0.,
            success_rate_20_percent=100*float(np.mean([x['success'] for x in rows[-20:]])) if rows else 0.,
            current_cycle_success_percent=100*float(np.mean([x['success'] for x in current])) if current else 0.)
        if self.map_windows:
            result['per_map']={str(i):dict(visits=self.map_visits[i],window_visits=len(a),
                success_rate_last10_percent=100*float(np.mean([r['success'] for r in a])) if a else None,
                latest_success=bool(a[-1]['success']) if a else None,
                latest_stop_reason=a[-1]['stop_reason'] if a else None) for i,a in self.map_windows.items()}
        return result

    def _emit(self):
        rows=list(self.window);s=self.summary();w=self.writer;ep=self.episode
        for key in ['success_rate_100_percent','success_rate_20_percent','current_cycle_success_percent','completed_cycles']:
            w.add_scalar('train/'+key,s[key],ep)
        w.add_scalar('train/target_success_rate_percent',99.,ep)
        if self.map_windows:
            w.add_scalar('train/success_rate_10_percent',100*float(np.mean([x['success'] for x in rows[-10:]])),ep)
            latest=rows[-1];index=latest['map']['index'];history=list(self.map_windows[index])
            label='map_%02d'%index;visit=self.map_visits[index]
            w.add_scalar('maps/'+label+'/success_rate_last10_percent',s['per_map'][str(index)]['success_rate_last10_percent'],visit)
            w.add_scalar('maps/'+label+'/success',float(latest['success']),visit)
            w.add_scalar('maps/'+label+'/capped_time',latest['capped_time'],visit)
            w.add_scalar('maps/'+label+'/min_local_coverage',min(latest['coverage']),visit)
            w.add_scalar('maps/'+label+'/mean_capped_time_last10',float(np.mean([r['capped_time'] for r in history])),visit)
        w.add_scalar('train/mean_capped_time_100',float(np.mean([x['capped_time'] for x in rows])),ep)
        w.add_scalar('train/mean_union_coverage_100',float(np.mean([x['union_coverage'] for x in rows])),ep)
        w.add_scalar('train/mean_min_local_coverage_100',float(np.mean([min(x['coverage']) for x in rows])),ep)
        w.add_scalar('train/mean_return_100',float(np.mean([np.mean(x['return_values']) for x in rows])),ep)
        w.add_scalar('train/mean_decisions_100',float(np.mean([x['decisions'] for x in rows])),ep)
        if ep%self.maps==0:
            w.add_scalar('cycle/success_rate_percent',s['current_cycle_success_percent'],ep//self.maps)
            w.add_scalar('cycle/mean_capped_time',float(np.mean([x['capped_time'] for x in rows[-self.maps:]])),ep//self.maps)
            w.flush()
