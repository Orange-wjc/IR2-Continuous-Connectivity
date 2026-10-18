"""Read-only 100->500 endpoint regression review; no policy or gradients."""
from pathlib import Path
import json,collections
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
OLD=REPO/'experiments/native_easy_repair_100_20261017_seed101'
NEW=REPO/'experiments/native_easy_500_20261017_seed101'
def read(p):return json.loads(p.read_text())
def main():
 old=read(OLD/'analysis/paired_maps.json');new=read(NEW/'analysis/paired_maps.json')
 assert [p['map'] for p in old]==[p['map'] for p in new]
 out={'same_100_development_maps':True,'new_policy_rollouts':0,'gradient_updates':0,'groups':{}}
 for key in ['fifo','ab']:
  lost=[(a,b) for a,b in zip(old,new) if a[key]['success'] and not b[key]['success']]
  gained=[(a,b) for a,b in zip(old,new) if not a[key]['success'] and b[key]['success']]
  failures=[p for p in new if not p[key]['success']]
  rows=[]
  for a,b in lost:
   x=b[key]
   rows.append(dict(map=b['map'],previous_time=a[key]['time'],current_time=x['time'],
    decisions=x['decisions'],stop_reason=x['stop_reason'],union_coverage=x['union_coverage'],
    min_local_coverage=min(x['coverage']),total_distance=x['total_distance'],wait_seconds=x['wait_seconds']))
  out['groups'][key]=dict(success_100=sum(p[key]['success'] for p in old),success_500=sum(p[key]['success'] for p in new),
   lost_success_count=len(lost),gained_success_count=len(gained),lost_success_maps=rows,
   gained_success_maps=[p['map'] for a,p in gained],
   failure_stop_reasons=dict(collections.Counter(p[key]['stop_reason'] for p in failures)),
   failures_union_at_least_99=sum(p[key]['union_coverage']>=.99 for p in failures))
 out['interpretation']='Observed endpoint regressions are retained, not removed. Decision-limit failures do not by themselves identify a code bug or prove convergence. Engineering/paired budget checks at 500 passed; continue both groups without tuning rewards or filtering hard maps.'
 (WORK/'failure_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 lines=['# 原easy 100→500开发失败案例核对（只读）','',
  '依据已有固定100图开发轨迹，无新环境执行或梯度更新。不剔除退步地图，不按开发成绩改变训练图选择。','']
 for key,g in out['groups'].items():
  lines += [f"## {key}", '',f"成功数{g['success_100']}→{g['success_500']}；从成功变失败{g['lost_success_count']}图，从失败变成功{g['gained_success_count']}图。失败停止原因：{g['failure_stop_reasons']}。联合覆盖≥99%但任务未成功：{g['failures_union_at_least_99']}图。", '',
   '|成功转失败地图|500图端点停止原因|决策数|联合覆盖|较弱本地覆盖|', '|---|---|---:|---:|---:|']
  for row in g['lost_success_maps']:
   lines.append(f"|{row['map']}|{row['stop_reason']}|{row['decisions']}|{row['union_coverage']:.4f}|{row['min_local_coverage']:.4f}|")
  lines += ['']
 lines += ['500图工程预算、继承、冻结、几何/通信检查已通过，未见需要先改算法或重置基线的证据。以上为终点现象分类，不能确定失败因果。下一轮保留同一规则、相同预算和全部开发地图继续验证；不能将基线退步全部归因为训练不足或把AB相对差距扩大当成AB自身都进步。']
 (WORK/'FAILURE_REVIEW.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:{x:v[x] for x in ['success_100','success_500','lost_success_count','gained_success_count','failure_stop_reasons','failures_union_at_least_99']} for k,v in out['groups'].items()},ensure_ascii=False))
if __name__=='__main__':main()
