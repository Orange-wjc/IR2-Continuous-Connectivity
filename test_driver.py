##############################################################################
# Name: test_driver.py
# [Inference] Driver of training program, maintain & update the global network.
###############################################################################

from test_parameter import *
import ray
import numpy as np
import os
import torch
import csv
import pandas as pd
import random
from model import PolicyNet
from test_multi_robot_worker import TestWorker
from datetime import datetime

CSV_FIELDNAMES = [
    'run', 'eps', 'map_file', 'split', 'model_path', 'checkpoint_file', 'checkpoint_episode',
    'status', 'error', 'test_seed', 'num_robots', 'max_dist', 'steps', 'explored', 'success',
    'connectivity_rate', 'disconnect_count', 'mean_disconnect_duration',
    'max_disconnect_duration', 'mean_reconnect_time', 'largest_component_ratio',
    'mean_component_count', 'weakest_tree_rssi', 'team_bottleneck_rssi', 'stay_rate', 'time_limit_reached'
]

def run_test(run_index):
    """Evaluate each configured map exactly once, including failed executions."""
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')
    csv_file_name = 'data_{}_run_{}.csv'.format(current_datetime, run_index)
    csv_file_path = os.path.join(log_path, csv_file_name)
    os.makedirs(log_path, exist_ok=True)
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES).writeheader()

    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    checkpoint_input_dim = checkpoint.get('input_dim', INPUT_DIM)
    if checkpoint_input_dim != INPUT_DIM:
        raise ValueError(
            'Checkpoint input_dim {} does not match configured input_dim {}. '
            'Set IR2_EVAL_CONNECTIVITY_FEATURE_DIM to 10 for v3.5-A or 5 for v3.4.'
            .format(checkpoint_input_dim, INPUT_DIM))
    model_path = os.path.abspath(MODEL_PATH)
    checkpoint_file = os.path.basename(MODEL_PATH)
    checkpoint_episode = checkpoint.get('episode', '')
    print('|Model path:', model_path)
    print('|Checkpoint episode:', checkpoint_episode)
    run_gifs_dir = os.path.join(
        GIFS_DIR, os.path.splitext(checkpoint_file)[0], 'run_{}'.format(run_index))
    global_network.load_state_dict(checkpoint['policy_model'])
    weights = global_network.state_dict()
    meta_agents = [Runner.remote(i) for i in range(
        min(NUM_META_AGENT, len(EVALUATION_EPISODE_INDICES)))]
    curr_test = 0
    dist_history, failed_episodes = [], []
    job_list = []
    for meta_agent in meta_agents:
        episode_number = EVALUATION_EPISODE_INDICES[curr_test]
        job_list.append(meta_agent.job.remote(
            weights, episode_number, run_index, run_gifs_dir))
        curr_test += 1

    try:
        while job_list:
            done_ids, job_list = ray.wait(job_list)
            success, metrics, info = ray.get(done_ids[0])
            row = {
                'run': run_index, 'eps': info['episode_number'],
                'map_file': MAP_FILE_NAMES[info['episode_number']],
                'split': EVALUATION_SPLIT,
                'model_path': model_path,
                'checkpoint_file': checkpoint_file,
                'checkpoint_episode': checkpoint_episode,
                'status': 'ok' if success else 'error',
                'error': metrics.get('error', ''),
                'test_seed': info['test_seed'], 'num_robots': info['n_agent'],
            }
            if success:
                dist_history.append(metrics['travel_dist'])
                row.update({
                    'max_dist': metrics['travel_dist'], 'steps': metrics['travel_steps'],
                    'explored': metrics['explored_rate'], 'success': metrics['success_rate'],
                })
                for name in CSV_FIELDNAMES:
                    if name in metrics:
                        row[name] = metrics[name]
            else:
                failed_episodes.append(info['episode_number'])
            with open(csv_file_path, mode='a', newline='') as csv_file:
                csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES).writerow(row)
            if curr_test < len(EVALUATION_EPISODE_INDICES):
                episode_number = EVALUATION_EPISODE_INDICES[curr_test]
                job_list.append(meta_agents[info['id']].job.remote(
                    weights, episode_number, run_index, run_gifs_dir))
                curr_test += 1

        df = pd.read_csv(csv_file_path).sort_values(by='eps')
        df.to_csv(csv_file_path, index=False)
        print('|#Configured maps:', len(EVALUATION_EPISODE_INDICES))
        print('|#Execution failures:', failed_episodes)
        if failed_episodes:
            print('Evaluation incomplete: resolve the recorded errors; no replacement maps were used.')
        else:
            print('|#Average (Max) length:', np.mean(dist_history))
            print('|#Average explored:', df['explored'].mean())
            print('|#Average connectivity:', df['connectivity_rate'].mean())
    finally:
        for meta_agent in meta_agents:
            ray.kill(meta_agent)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT if USE_GPU else 0)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number, run_index, gifs_dir):
        """ Execute simulation episode and gather experience tuples & metrics """
        test_seed = TEST_RANDOM_SEED + run_index * NUM_TEST + episode_number
        random.seed(test_seed)
        np.random.seed(test_seed)
        torch.manual_seed(test_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(test_seed)
        n_agent = np.random.randint(NUM_ROBOTS_MIN, NUM_ROBOTS_MAX+1, 1)[0]     
        try:
            worker = TestWorker(self.meta_agent_id, n_agent, self.local_network, episode_number,
                                device=self.device, save_image=SAVE_GIFS, greedy=True,
                                gifs_dir=gifs_dir)
            success = worker.work(episode_number)
            perf_metrics = worker.perf_metrics
            if not success:
                perf_metrics['error'] = 'Graph/observation construction failed; see worker log'
        except Exception as error:
            success = False
            perf_metrics = {'error': '{}: {}'.format(type(error).__name__, error)}
        return success, perf_metrics, n_agent, test_seed

    def job(self, weights, episode_number, run_index, gifs_dir):
        """ Executes simulation episode """
        print(GREEN, "starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id), NC)
        
        # Set the local weights to the global weight values from the master network
        self.set_weights(weights)

        success, metrics, n_agent, test_seed = self.do_job(
            episode_number, run_index, gifs_dir)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
            "run_index": run_index,
            "n_agent": n_agent,
            "test_seed": test_seed
        }

        return success, metrics, info


if __name__ == '__main__':
    ray.init()
    print("Welcome to IR2-MARL Exploration Inference Sim!")
    print('|Run indices:', EVALUATION_RUN_INDICES)
    print('|Map files:', [MAP_FILE_NAMES[index] for index in EVALUATION_EPISODE_INDICES])
    print('|Save GIFs:', SAVE_GIFS)
    for i in EVALUATION_RUN_INDICES:
        run_test(i)
