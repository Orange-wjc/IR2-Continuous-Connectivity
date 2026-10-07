"""Render one complete pretrained Hybrid episode without altering decisions."""
import json
from pathlib import Path
import random
import numpy as np
import torch
import test_parameter
from model import PolicyNet
from test_multi_robot_worker import TestWorker
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seed=20261005
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
torch.set_num_threads(1)
policy=PolicyNet(6,128).to('cuda')
policy.load_state_dict(torch.load('model/stage2/checkpoint.pth',map_location='cpu',weights_only=True)['policy_model'])
policy.eval()
worker=TestWorker(0,4,policy,0,device='cuda',greedy=True,save_image=False)
paths=[[r.robot_position.copy()] for r in worker.robot_list]
select=worker.select_node
def track(observations,robot_id):
    result=select(observations,robot_id)
    paths[robot_id].append(result[0].copy())
    return result
worker.select_node=track
with torch.no_grad(): ok=worker.work(0)
out=Path('reproduction_results/pretrained_stage2_20261005')
fig,axes=plt.subplots(1,5,figsize=(18,4))
colors=['tab:red','tab:green','tab:blue','tab:orange']
axes[0].imshow(worker.env.ground_truth,cmap='gray',vmin=1,vmax=255)
for i,path in enumerate(paths):
    a=np.asarray(path)
    axes[0].plot(a[:,0],a[:,1],color=colors[i],linewidth=1,label=f'Robot {i+1}')
    axes[0].scatter(a[-1,0],a[-1,1],color=colors[i],s=20)
    axes[i+1].imshow(worker.env.all_robot_belief[i][i],cmap='gray',vmin=1,vmax=255)
    axes[i+1].plot(a[:,0],a[:,1],color=colors[i],linewidth=1)
    axes[i+1].set_title(f'Robot {i+1}: {worker.env.evaluate_exploration_rate(i)*100:.2f}%')
axes[0].set_title('Ground truth / team trajectories');axes[0].legend(fontsize=7)
for ax in axes: ax.set_axis_off()
fig.suptitle(f'Official IR2 Stage 2 / Hybrid {worker.env.file_path} / {worker.perf_metrics.get("travel_steps")} steps / success={worker.perf_metrics.get("success_rate")}')
fig.tight_layout();fig.savefig(out/'hybrid_example.png',dpi=170);plt.close(fig)
(out/'hybrid_example.json').write_text(json.dumps({'seed':seed,'execution_ok':ok,'metrics':worker.perf_metrics,'paths':[np.asarray(p).tolist() for p in paths]},indent=2,default=lambda x:x.item()))
print(out/'hybrid_example.png')
