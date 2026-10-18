# IR2 Multi-Robot RL Exploration

当前模块接入分支：`research/ir2-modular-contact`。基于已保存的原IR2十图训练器增加独立有限通信模块，默认仍关闭；启动改进组与工程验证说明见 [MODULAR_CONTACT.md](MODULAR_CONTACT.md)。原版运行2已由用户暂停，模型及日志保留，不自动启动正式训练。下方原版说明保留作历史参考。

本地原版10图基线分支：`baseline/ir2-easy10`（基于main）。保持原版IR²模型、环境和SAC规则，用10张原easy地图循环训练；默认单GPU/4个Ray worker。前台启动仍为 `python -u driver.py`，准备/检查、完整命令、曲线和中断保存见 [BASELINE_EASY10.md](BASELINE_EASY10.md)。正式训练等待用户手动启动；短工程预检不等于学习效果。下文为原作者说明。

2026-10-18：首次运行在18个有效回合/120次更新保存后因主机RAM保护退出。已修复零剩余邻居预算造成的31候选越界，并无损压缩回放掩码；原batch128、回放10000和SAC更新公式保持。现有模型续训用 `IR2_RESUME=1 python driver.py`，仍由用户前台启动。修复及等价/内存验证详见上方基线说明。

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
