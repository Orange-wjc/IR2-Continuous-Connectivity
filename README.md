# IR2 Multi-Robot RL Exploration

新对话接手请先阅读[实验计划与新对话交接](finite_contact/EXPERIMENT_HANDOFF.md)，其中包含目标、当前阶段、恢复命令和后续验收计划。

本实验分支新增了[两机器人、简单地图的有限通信窗口实验入口](finite_contact/README.md)。研究实验请在 conda `ir2_train` 中运行 `finite_contact_train.py`；下文原版训练说明仍对应上游 `driver.py`。

已完成从随机权重开始的两机单图学习与真实续训验收；一键启动、预算、模型与限制见[单图训练指南](finite_contact/RUN_SINGLE_MAP.md)。该轮结论限于一张训练图，后续多图开发结果见下文。

后续新增100张多样简单训练图、20张开发验证图和20张留出测试图，使用 `finite_contact_multimap.py` 开展多图移动学习；地图检查、预算和阶段结果见[多图实验记录](finite_contact/MULTIMAP_EXPERIMENT.md)。

后续通信阶段使用全训练集采样和按地图均衡的离线头训练，固定移动权重比较 FIFO、规则与学习选包，见[多图通信实验](finite_contact/MULTIMAP_CONTENT_EXPERIMENT.md)。三组等预算移动适应计划与结果见[联合适应实验](finite_contact/MULTIMAP_JOINT_EXPERIMENT.md)。

该阶段已完成：100 图采样、5000 次通信头更新、46 项行为检查；开发成功数为 FIFO 18/20、规则与学习均 19/20。学习的固定状态选包收益通过检查，但实际平均探索时间尚未超过规则，该通信头阶段尚未超过规则；后续等预算移动适应已完成，见下段。

前一轮单适应种子的三组各完整轮训100张、新增2000次SAC，固定终点在同一20张开发图上，FIFO／规则／学习成功18／19／20张，平均176.503／150.472／125.480秒；学习比等预算规则快16.61%。50项行为检查和360条轨迹物理审计通过，测试图未执行策略。该结果仍限于单训练种子和简单地图开发集，详见[联合适应记录](finite_contact/MULTIMAP_JOINT_EXPERIMENT.md)。

适应种子复现已完成：FIFO 成功18／19／19，规则19／18／20，学习20／17／19。三轮平均探索时间分别171.970／155.520／136.205秒；学习更快但完成率波动更大，当前尚未稳定全面超过规则。54项行为检查、1080条主结果轨迹物理审计通过，测试集保留；完整结果见[三种子复现记录](finite_contact/MULTISEED_EXPERIMENT.md)。

前一轮收尾抽样试验已完成：seed=2六组各追加同样2000次SAC，原pooled的FIFO／规则／学习均成功18/20，候选map_tail为19／20／20；学习平均120.694秒，比同为20/20的规则快17.57%。该阶段训练监测中FIFO退步，随后补做的另两种子复现见下段，默认抽样保持pooled。60项行为检查、900条轨迹物理审计通过，测试策略未执行。预算、全部正负结果与模型见[收尾稳定性试验](finite_contact/STABILITY_MULTIMAP_EXPERIMENT.md)。

收尾抽样的固定条件复现已完成：新增seed=1、3共12组，每组保留2000次真实SAC；与先行seed=2合并，候选学习的三个适应种子分别完成20／20／20张（每轮20张）。但规则在确认种子1／3分别由20/20降到19/20、18/20，种子1学习同为20/20却慢7.545秒，候选没有一致优势，默认仍pooled。64项行为检查和2700条主轨迹物理审计通过；OOM后以两并发完整恢复，额外计算单列，测试策略执行0。全部结果、模型与范围见[收尾抽样复现](finite_contact/STABILITY_REPLICATION_EXPERIMENT.md)。

当前已启动1000张简单地图训练：保留原100张、新增900张，1040张训练／开发／测试地图质量审计通过。两机器人，三方法各新增20000次真实SAC，使用无损压缩成功经验和单worker后台队列，先训练学习方案；用户已执行bash run_thousand_maps_foreground.sh恢复前台训练，按学习→规则→FIFO依次运行；接手时先检查实时状态，避免重复启动。通信头沿用原100图版本且冻结，当前尚无新终点结果。计划与进度见[1000图实验](finite_contact/THOUSAND_MAP_EXPERIMENT.md)。

[IROS 2024] Public code and model for [IR2: Implicit Rendezvous for Robotic Exploration Teams under Sparse Intermittent Connectivity](https://arxiv.org/abs/2409.04730).

Public Website: https://ir2-explore.github.io/

We present IR2, a deep reinforcement learning approach to information sharing for multi-robot exporation under communication constraints. Leveraging attention-based neural networks and hierarchical graph formulation, robots can effectively balance the longer-term trade-offs between disconnecting for solo exploration and reconnecting for information sharing. 

We show that our model trained in this python simulation can generalize well to complex large-scale environments, such as those in Gazebo simulation and real world experiments. We hope that our work will be a useful learning-based benchmark for future research. 


## GIF Demo

This demonstration showcases 4 robots exploring in an unknown `Complex` map under line-of-sight signal strength communication constraints. The top gif illustrates the global map and robot positions assuming no communication constraints. Conversely, the bottom 4 gifs illustrates the individual robots' map and position beliefs subjected to communication constraints. 

If this GIF is taking too long to load, you may view the demonstration [here](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/master/gifs/combined_gif_demo.gif).
<p align="center">
  <img src="gifs/combined_gif_demo.gif" style="width: 750px; height: auto;" />
</p>





## Instructions


#### Dependencies
This repository was tested using the following dependencies. Newer version of these packages may work as well. 
* `python == 3.8`
* `pytorch == 1.10.0`
* `ray == 1.10.0`
* `scikit-image == 0.19.3`
* `scikit-learn == 1.2.1`
* `scipy == 1.10.0`
* `matplotlib == 3.6.3`
* `tensorboard == 2.8.0`


#### Training
1. Set training parameters in `parameters.py`.
2. Run python `driver.py`.


#### Evaluation
1. Set inference parameters in `test_parameters.py`.
2. Run `test_driver.py`.



## Code Structure

* `parameter.py` Training parameters.
* `driver.py` Driver of training program, maintain & update the global network.
* `runner.py` Wrapper of the local network.
* `multi_robot_worker.py` Interact with environment and collect episode experience.
* `model.py` Define attention-based network.
* `env.py` Autonomous exploration environment.
* `graph_generator.py` Generate and update the collision-free graph.
* `graph.py` Graph definition and utilities.
* `node.py` Initialize and update nodes in the coliision-free graph.
* `sensor.py` Simulate the sensor model of Lidar.
* `robot.py` Acts as a replay buffer.
* `ss_realistic_model.py` Realistic signal strength communication model.
* `/model` Trained model.
* `/DungeonMaps` Maps of training environments.



## References
If you intend to use our work in your research, please cite the following publication:
```bibtex
@inproceedings{tan2024ir,
  title={Ir 2: Implicit rendezvous for robotic exploration teams under sparse intermittent connectivity},
  author={Tan, Derek Ming Siang and Ma, Yixiao and Liang, Jingsong and Chng, Yi Cheng and Cao, Yuhong and Sartoretti, Guillaume},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={13245--13252},
  year={2024},
  organization={IEEE}
}
```


## Authors
[Derek Ming Siang Tan](https://github.com/derektan95)\
[Yixiao Ma](https://yixiao-m.github.io)\
[Jingsong Liang](https://github.com/JingsongLiang)\
[Yi Cheng Chng](https://github.com/chngyicheng)\
[Yuhong Cao](https://github.com/caoyuhong001)\
[Guillaume Sartoretti](https://github.com/gsartoretti)
