# IR2 Multi-Robot RL Exploration

本实验分支新增了[两机器人、简单地图的有限通信窗口实验入口](finite_contact/README.md)。研究实验请在 conda `ir2_train` 中运行 `finite_contact_train.py`；下文原版训练说明仍对应上游 `driver.py`。

已完成从随机权重开始的两机单图学习与真实续训验收；一键启动、预算、模型与限制见[单图训练指南](finite_contact/RUN_SINGLE_MAP.md)。当前结论限于这一训练图，尚未证明换图泛化。

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
