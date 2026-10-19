#######################################################################
# Name: runner.py
# Wrapper of the local network.
#######################################################################

from parameter import *
import torch
import ray
import numpy as np
from model import PolicyNet, QNet
from multi_robot_worker import Worker
from baseline_support import worker_output
from contact_module.adapter import ContactAdapter


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        torch.set_num_threads(1)
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM, allow_wait=CONTACT_CONFIG.enabled)
        self.local_q_net = QNet(INPUT_DIM, EMBEDDING_DIM, allow_wait=CONTACT_CONFIG.enabled)
        self.auxiliary = None
        self.contact_labels = []
        self.contact_diagnostics = []
        self.local_network.to(self.device)
        self.local_q_net.to(self.device)

    def get_weights(self):
        return self.local_network.state_dict()

    def set_policy_net_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def set_q_net_weights(self, weights1):
        self.local_q_net.load_state_dict(weights1)

    def do_job(self, episode_number):
        """ Execute simulation episode and gather experience tuples & metrics """
        save_img = SAVE_TRAINING_GIFS and episode_number % SAVE_IMG_GAP == 0
        n_agent = np.random.randint(NUM_ROBOTS_MIN, NUM_ROBOTS_MAX+1, 1)[0]    
        contact = (ContactAdapter(CONTACT_CONFIG, self.local_network, self.auxiliary, episode_number)
                   if CONTACT_CONFIG.enabled else None)
        worker = Worker(self.meta_agent_id, n_agent, self.local_network, self.local_q_net, episode_number, device=self.device, save_image=save_img, greedy=False, contact=contact)
        succeess = worker.work(episode_number)
        self.contact_labels = contact.labels if contact is not None else []
        self.contact_diagnostics = contact.empty_wait_details if contact is not None else []

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        return succeess, job_results, perf_metrics

    def job(self, weights_set, episode_number):
        """ Executes simulation episode """
        with worker_output(BASELINE_DIR, self.meta_agent_id):
            print("\n", GREEN, "starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id), NC)

            # Set the local weights to the global weight values from the master network
            self.set_policy_net_weights(weights_set[0])
            self.set_q_net_weights(weights_set[1])
            self.auxiliary = weights_set[2] if len(weights_set) > 2 else None

            success, job_results, metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }
        if CONTACT_CONFIG.learned:
            info['contact_labels'] = self.contact_labels
        if self.contact_diagnostics:
            info['contact_diagnostics'] = self.contact_diagnostics

        return success, job_results, metrics, info


### Wraps around Runner class to define class as a Ray object ### 
@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):        
        super().__init__(meta_agent_id)


if __name__=='__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    print(out[1])
