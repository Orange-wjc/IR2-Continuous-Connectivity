#######################################################################
# Name: multi_robot_worker.py
# Interact with environment and collect episode experience.
#######################################################################

from parameter import *
import copy
import os
import imageio
import numpy as np
import torch
from env import Env
from robot import Robot, REPLAY_FIELD_COUNT
from team_rollout import action_candidates, run_team_episode


class Worker:
    def __init__(self, meta_agent_id, n_agent, policy_net, global_step, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.n_agent = n_agent
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, n_agent=self.n_agent, k_size=self.k_size, plot=save_image)
        self.local_policy_net = policy_net

        # Distribute starting positions (NOTE: Every's belief is different)
        self.robot_list = []
        self.all_robot_positions = []
        for i in range(self.n_agent):
            iter = min(copy.deepcopy(i), len(self.env.all_node_coords[i])-1 )    # In case idx out of bounds
            robot_position = self.env.all_node_coords[i][iter]     
            robot = Robot(robot_id=i, position=robot_position, plot=save_image)
            self.robot_list.append(robot)
            self.all_robot_positions.append(robot_position)

        self.perf_metrics = dict()
        self.episode_buffer = []
        for i in range(REPLAY_FIELD_COUNT):
            self.episode_buffer.append([])

        self.max_node_coords = 0


    def run_episode(self, curr_episode):
        return run_team_episode(
            self, curr_episode, MAX_EPS_STEPS, GIFS_DIR, collect_experience=True)


    def get_observations(self, robot_position, robot_id, eps, step, plot=True):
        """ Get robot's observation of environment (neural network inputs) """
    
        # Rendezvous Utility Layer (Gen first because it involves A* path planning, and potentially regen graph if no path found)
        map_delta_unnormalized, rendezvous_utility_inputs, success = self.env.generate_rendezvous_utility_layer(robot_id, eps)
        if not success:
            return [], False

        self.env.all_robot_map_belief_area_diff[robot_id] = map_delta_unnormalized
        self.env.all_rendezvous_utility_inputs[robot_id] = rendezvous_utility_inputs

        node_coords = self.env.all_node_coords[robot_id]
        node_utility = self.env.all_node_utility[robot_id]
        guidepost = self.env.all_guidepost[robot_id]

        current_node_index = self.env.find_index_from_coords(robot_position, robot_id)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        n_nodes = node_coords.shape[0]

        node_coords = node_coords * NODE_COORDS_SCALING_FACTOR 
        node_utility = node_utility * NODE_UTILITY_SCALING_FACTOR 

        node_utility_inputs = node_utility.reshape((n_nodes, 1))

        # Augment with all agents' positions
        occupied_node = np.zeros((n_nodes, 1))
        all_robot_positions_belief = self.env.all_robot_positions_belief[robot_id]

        for i, position in enumerate(all_robot_positions_belief):
            if position is not None:    # 'None' when outdated position belief that have been verified to no longer be there
                index = self.env.find_index_from_coords(position, robot_id)
                if index == current_index.item():
                    occupied_node[index] = -1
                else:
                    occupied_node[index] = 1

        # Collate final augmented node_coords inputs
        node_input_parts = [node_coords, node_utility_inputs, rendezvous_utility_inputs,
                            guidepost, occupied_node]
        if USE_CONNECTIVITY_FEATURES:
            connectivity_inputs = self.env.get_connectivity_node_features(
                robot_id, self.env.all_node_coords[robot_id])
            node_input_parts.append(connectivity_inputs)
        node_inputs = np.concatenate(node_input_parts, axis=1)
        assert node_inputs.shape[1] == INPUT_DIM
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, INPUT_DIM)

        if node_coords.shape[0] >= self.node_padding_size:
            print(RED, "[Eps {} | Robot {} | Step {}] node_coords.shape[0] >= self.node_padding_size ({} >= {}). Skipping eps.".format(eps, robot_id+1, step, node_coords.shape[0], self.node_padding_size))
            return [], False
        
        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # Order wrt self.node_coords indices
        edge_inputs = []
        for coord in self.env.all_node_coords[robot_id]:
            node_edges = self.env.all_graph[robot_id][tuple(coord)].values()
            node_edges = [int(self.env.find_index_from_coords(np.array(edge.to_node), robot_id)) for edge in node_edges]
            edge_inputs.append(node_edges)

        bias_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(bias_matrix).float().unsqueeze(0).to(self.device)

        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)

        if current_index >= len(edge_inputs):
            print(RED, "[Eps {} | Robot {} | Step {}] current_index > len(edge_inputs) ({} >= {}). Skipping eps.".format(eps, robot_id+1, step, current_index, len(edge_inputs)))
            return [], False
        current_node_index = current_index.item()
        edge_inputs, edge_padding_mask, valid_edges = action_candidates(
            edge_inputs[current_node_index], current_node_index,
            self.env.all_node_coords[robot_id], self.k_size, self.device)
        if plot:
            self.env.all_curr_vertices[robot_id] = [
                self.env.all_node_coords[robot_id][index] for index in valid_edges]

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations, True   # success


    def select_node(self, observations, robot_id):
        """ Forward pass through policy to get next position to go """
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = edge_inputs[0, 0, action_index.item()]    
        next_position = self.env.all_node_coords[robot_id][next_node_index]

        return next_position, action_index
    

    def work(self, currEpisode):
        """ Interacts with the environment """
        success = self.run_episode(currEpisode)
        return success


    def calculate_edge_mask(self, edge_inputs):
        """ Generates 2D graph connectivity matrix """
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for node_index, neighbors in enumerate(edge_inputs):
            bias_matrix[node_index, neighbors] = 0
        return bias_matrix


    def make_gif(self, path, n, robot_id):
        """ Generate a gif given list of images """
        with imageio.get_writer('{}/eps{}_robot{}_explored_rate_{:.4g}.gif'.format(path, n, robot_id+1, self.env.all_explored_rate[robot_id]), mode='I',
                                duration=0.5) as writer:
            for frame in self.env.all_frame_files[robot_id]:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.all_frame_files[robot_id][:-1]:
            os.remove(filename)


    def make_gif_ground_truth(self, path, n):
        """ Generate a gif given list of images (Combined, no communication constraints) """
        with imageio.get_writer('{}/eps{}_merged_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_rate), mode='I',
                                duration=0.5) as writer:
            for frame in self.env.merged_frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.merged_frame_files[:-1]:
            os.remove(filename)
