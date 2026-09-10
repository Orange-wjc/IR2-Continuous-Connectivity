#######################################################################
# Name: robot.py
# Acts as a replay buffer.
#######################################################################

import torch
from copy import deepcopy

# First 15 fields retain the SAC tensor layout. Additional fields stay aligned
# in replay: [episode, team_step, robot_id], then raw reward/termination parts.
REPLAY_FIELD_COUNT = 17


class Robot:
    def __init__(self, robot_id, position, plot=False):
        self.robot_id = robot_id
        self.plot = plot
        self.travel_dist = 0
        self.robot_position = position
        self.observations = None
        
        self.episode_buffer = []
        for i in range(REPLAY_FIELD_COUNT):
            self.episode_buffer.append([])

        if self.plot:
            # initialize the route
            self.xPoints = [self.robot_position[0]]
            self.yPoints = [self.robot_position[1]]

    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[0] += deepcopy(node_inputs).to('cpu')
        self.episode_buffer[1] += deepcopy(edge_inputs).to('cpu')
        self.episode_buffer[2] += deepcopy(current_index).to('cpu')
        self.episode_buffer[3] += deepcopy(node_padding_mask).to('cpu')
        self.episode_buffer[4] += deepcopy(edge_padding_mask).to('cpu')
        self.episode_buffer[5] += deepcopy(edge_mask).to('cpu')

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.detach().cpu().unsqueeze(0).unsqueeze(0)

    def save_robot_position(self):
        self.xPoints.append(self.robot_position[0])
        self.yPoints.append(self.robot_position[1])

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += deepcopy(torch.FloatTensor([[[reward]]])).to('cpu')
        self.episode_buffer[8] += deepcopy(torch.tensor([[[(int(done))]]])).to('cpu')

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[9] += deepcopy(node_inputs).to('cpu')
        self.episode_buffer[10] += deepcopy(edge_inputs).to('cpu')
        self.episode_buffer[11] += deepcopy(current_index).to('cpu')
        self.episode_buffer[12] += deepcopy(node_padding_mask).to('cpu')
        self.episode_buffer[13] += deepcopy(edge_padding_mask).to('cpu')
        self.episode_buffer[14] += deepcopy(edge_mask).to('cpu')

    def save_transition(self, observations, action, reward, terminal,
                        next_observations, episode, team_step, reward_parts):
        self.save_observations(observations)
        self.save_action(action)
        self.save_reward_done(reward, terminal)
        self.save_next_observations(next_observations)
        self.episode_buffer[15].append(torch.tensor(
            [episode, team_step, self.robot_id], dtype=torch.long))
        # frontier, node utility, rendezvous, travel cost, team reward,
        # communication, exploration progress, stagnation, completion, timeout.
        self.episode_buffer[16].append(torch.tensor(reward_parts, dtype=torch.float32))
