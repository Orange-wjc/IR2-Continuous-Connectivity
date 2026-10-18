#!/usr/bin/env python3
"""Read-only native easy stage progress; no Torch or training imports."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args()
repo=Path(__file__).resolve().parents[2]
root=a.run or repo/'experiments/native_easy_20261015_seed101'
plan=json.loads((root/'pipeline.json').read_text())
print('状态:',plan['status'],'阶段:',plan['stage'],'更新时间:',plan.get('updated',plan.get('created')))
print('原easy地图:',plan['original_training_file_counts']['easy'],'开发:',plan['development_maps']['easy'],
      '采样进程:',plan['workers'],'有效批量:',plan['effective_batch'],'微批量上限:',plan['microbatch'])
print('开发集评估:', '每%d回合自动执行'%plan['review_interval']
      if plan.get('automatic_development_evaluation',True) else '已关闭自动评估，仅按用户要求执行')
print('共享热身:',plan.get('warmup_episodes',0),'/32回合，SAC:',plan.get('warmup_sac_updates',0))
print('消息头:',plan.get('head_updates',0),'/5000；采样地图:',plan.get('head_collection_maps',0),'/16')
for name in plan['groups']:
    path=root/name/'status.json'
    if path.exists():
        s=json.loads(path.read_text());print(name,'回合:',s['episodes'],'SAC:',s['sac_updates'],'状态:',s['status'])
    elif (root/name/'initial.pth').exists():print(name,'已初始化，等待本组首波采样提交')
    else:print(name,'等待共享初始化完成')
print('复杂阶段已启动:',plan.get('stage2_started',False),'测试策略执行:',plan['test_policy_evaluations'])
review=root/'latest_comparison.json'
if review.exists():
    d=json.loads(review.read_text());print('最近开发对比:',json.dumps(d,ensure_ascii=False))
