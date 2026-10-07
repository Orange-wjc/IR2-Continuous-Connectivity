"""Generate the Chinese report from saved evaluation records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_pretrained import summarize
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'reproduction_results/pretrained_stage2_20261005'
rows = [json.loads(x) for x in (OUT/'episodes.jsonl').read_text().splitlines() if x]
summary = summarize(rows)
manifest = json.loads((OUT/'manifest.json').read_text())
complete = all(summary[k]['attempted']==100 for k in summary)
lines = ['# IR2 官方 Stage 2 预训练模型评估报告', '',
         '## 完成状态', '',
         f'已记录 {len(rows)}/300 次测试。'+('三类地图各 100 次完整评估已完成。' if complete else '全量评估仍在运行，以下为当前累计结果。'), '',
         '本次使用发布的预训练权重，没有重新训练。范围为 Python 仿真评估；未执行课程学习、消融重新训练、Gazebo 基线对比或实机实验。', '',
         '## 配置与来源', '',
         '- 官方源码仓库：https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration',
         f'- 本地及官方仓库版本：`{manifest["git_head"]}`。启动前用 `git ls-remote` 核实官方 HEAD。',
         '- 权重：`model/stage2/checkpoint.pth`；Git 对象哈希 `3e99f032d497909507785aae06f6c092748b8e1c` 与 `upstream/master` 一致。',
         f'- 权重 SHA256：`{manifest["checkpoint_sha256"]}`；checkpoint episode={manifest["checkpoint_episode"]}。',
         f'- PyTorch：{manifest["torch"]}；GPU：{manifest["gpu"]}；conda 环境 BLMRE_py38。',
         '- 算法沿用官方 TestWorker、PolicyNet、图构建及通信实现；将 Ray 外层调度替换为独立进程池，不改变环境内机器人顺序或策略。',
         '- 原始算法文件和训练/测试参数文件未修改。评估包装器按地图类型在进程内设置步数、坐标归一化和图合并阈值。',
         '- 默认 signal-strength 通信，PRUNE_GLOBAL_GRAPH=False；greedy=True；关闭逐步图像/GIF。',
         '- Corridor、Hybrid 各 4 个机器人、196 步；Complex 为 5 个机器人、384 步。',
         '- 每类按原代码文件名逆序选前 100 张；seed=20261005+map_index；地图清单见 manifest.json。',
         '- dataset_audit.json 显示每类所选文件哈希均唯一，与训练集未发现完全相同的文件；此检查不保证没有几何相似地图。', '',
         '## 结果', '',
         '|地图|尝试数|执行失败|完成探索|成功率|平均步数 ± 标准差|最大单机器人路程（推算米）± 标准差|',
         '|---|---:|---:|---:|---:|---:|---:|']
for kind,s in summary.items():
    st=s['steps_execution_ok'];d=s['distance_execution_ok_coordinate_units']
    steps=f'{st["mean"]:.1f} ± {st["std_population"]:.1f}' if st else '—'
    dist=f'{d["mean"]*.25:.1f} ± {d["std_population"]*.25:.1f}' if d else '—'
    rate=f'{s["success_percent_all_attempts"]:.1f}%' if s['attempted'] else '—'
    lines.append(f'|{kind}|{s["attempted"]}|{s["execution_failures"]}|{s["successful"]}|{rate}|{steps}|{dist}|')
lines += ['', '成功要求所有机器人个人地图的自由空间覆盖率均不小于 99%。所有固定测试尝试都进入成功率分母；不跳过执行失败、不替换地图。步数和路程的主表均值采用执行完成样本，包含达到步数上限但未完成探索的 episode；若有执行失败，这两项不包含中断 episode。summary.json 另列仅成功 episode 的统计。标准差为总体标准差（ddof=0）。', '',
          '距离单位：CSV 保留原代码坐标单位。根据论文 Corridor 160×120m 对应图片 640×480px，Hybrid 125×125m 对应 500×500px，Complex 250×250m 对应 1000×1000px，推算比例为 0.25m/px。该比例用于结果展示，不修改传感器或信号强度模型内部参数；其注释含 res=1m/px，说明公开仿真代码的物理标定仍有解释差异。', '',
          '## 与论文 Table II 的对照', '',
          '|地图|论文 Stage 2 成功率|本次成功率|论文步数|本次步数|论文 D(m)|本次推算 D(m)|',
          '|---|---:|---:|---:|---:|---:|---:|']
for kind,s in summary.items():
    p=s['paper_stage2_reference'];st=s['steps_execution_ok'];d=s['distance_execution_ok_coordinate_units']
    rate=f'{s["success_percent_all_attempts"]:.1f}%' if s['attempted'] else '—'
    steps=f'{st["mean"]:.1f}' if st else '—';distance=f'{d["mean"]*.25:.1f}' if d else '—'
    lines.append(f'|{kind}|{p["success_percent"]}%|{rate}|{p["steps"]}|{steps}|{p["distance"]}|{distance}|')
lines += ['', '此表用于对照公开模型在公开默认配置下的表现，不代表论文数值已精确复现。本次未能确认测试地图清单、随机种子和统计口径与论文 Table II 完全一致；这里明确记录本次采用的口径。Table II 的通信细节也无法仅凭默认配置确证一致，不能将差异直接归因于模型效果。', '',
          '## 复跑命令', '', '从源码目录运行：', '', '```bash',
          'conda activate BLMRE_py38',
          'MPLCONFIGDIR=/home/robot/.cache/matplotlib MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -u evaluate_pretrained.py --count 100 --workers 8',
          'MPLCONFIGDIR=/home/robot/.cache/matplotlib MPLBACKEND=Agg python -B report_reproduction.py',
          '```', '',
          '同一输出目录会复用已保存的 episode；重新独立运行请用 --output 指定新目录。恢复时应保持 seed、权重和配置一致。', '',
          '## 文件', '', '- episodes.csv / episodes.jsonl：每次测试原始记录，包含执行错误及各机器人覆盖率。',
          '- summary.json：按地图汇总，含全部执行完成样本和仅成功样本统计。',
          '- manifest.json：权重、配置、版本、硬件和地图清单。',
          '- dataset_audit.json：文件重复与训练集交叉检查。',
          '- comparison.png：公开配置评估与论文参考值对照。', '- hybrid_example.png / hybrid_example.json：完整示例轨迹与各机器人最终地图。', '- validation.json：300 次记录的完整性、成功判定、地图尺寸及权重一致性检查。',
          '- <地图类型>_<索引>.log：原始 worker 输出及异常。']
(OUT/'复现报告.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,3,figsize=(13,4))
kinds=list(summary);x=np.arange(3)
for ax,label,key,paper_key,scale in [(axes[0],'Success (%)','success_percent_all_attempts','success_percent',1),
                                     (axes[1],'Steps','steps_execution_ok','steps',1),
                                     (axes[2],'Max robot distance (inferred m)','distance_execution_ok_coordinate_units','distance',.25)]:
    actual=[];paper=[]
    for kind in kinds:
        value=summary[kind][key]
        actual.append((value['mean'] if isinstance(value,dict) else value)*scale if value is not None else 0)
        paper.append(summary[kind]['paper_stage2_reference'][paper_key])
    ax.bar(x-.18,paper,.36,label='Paper Table II')
    ax.bar(x+.18,actual,.36,label='Released model / default config')
    ax.set_xticks(x);ax.set_xticklabels([k.capitalize() for k in kinds]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle(f'IR2 pretrained Stage 2 evaluation ({len(rows)}/300 attempts)')
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
print(OUT/'复现报告.md')
