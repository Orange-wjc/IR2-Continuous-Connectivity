##############################################################################
# Name: driver.py
# Driver of training program, maintain & update the global network.
##############################################################################

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ir2-original-mpl')
os.environ.setdefault('RAY_DEDUP_LOGS', '0')

from parameter import *
import parameter as config
import argparse
import fcntl
from pathlib import Path
import signal
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import ray
import os
import numpy as np
import random
import socket
from torch.utils.tensorboard import SummaryWriter
from model import PolicyNet, QNet
from runner import RLRunner
from datetime import datetime
from baseline_support import prepare_maps, record_setup, Progress, write_json, memory_available_gib
from baseline_replay import stack_masks
from contact_module.learning import ContentTrainer


def writeToTensorBoard(writer, tensorboardData, curr_episode):
    """ Log data to tensorboard """

    tensorboardData = np.array(tensorboardData)
    tensorboardData = list(np.nanmean(tensorboardData, axis=0))
    reward, value, policyLoss, qValueLoss, entropy, policyGradNorm, qValueGradNorm, log_alpha, alphaLoss, travel_dist, success_rate, explored_rate, connectivity_rate, agents_connected_percentage = tensorboardData

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alphaLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=qValueLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policyGradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=qValueGradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Connectivity Rate', scalar_value=connectivity_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Agents Connected [%]', scalar_value=agents_connected_percentage, global_step=curr_episode)

    
def main():
    parser = argparse.ArgumentParser(description='Original IR2 training on a fixed ten-map subset')
    parser.add_argument('--check-config', action='store_true', help='Verify map links/configuration without starting Ray or training')
    args = parser.parse_args()
    Path(BASELINE_DIR).mkdir(parents=True, exist_ok=True)
    run_lock = (Path(BASELINE_DIR) / '.training.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = prepare_maps(config)
    if not args.check_config and Path(MODEL_PATH).exists() and not LOAD_MODEL:
        raise ValueError('Checkpoint already exists; use IR2_RESUME=1 or a new IR2_RUN_NAME')
    if LOAD_MODEL and not args.check_config:
        # Reject a wrong-method resume before touching setup/logs or starting Ray.
        metadata = torch.load(MODEL_PATH, map_location='cpu')
        CONTACT_CONFIG.validate_checkpoint(metadata)
        del metadata
    setup = record_setup(config, manifest, check_only=args.check_config)
    if args.check_config:
        print(__import__('json').dumps(setup, ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('The original global learner requires a visible CUDA GPU')
    if memory_available_gib() < 6:
        raise MemoryError('Need at least 6 GiB available host RAM before starting')
    torch.set_num_threads(2)
    ray.init(address='local', num_cpus=NUM_META_AGENT + 2, num_gpus=NUM_GPU,
             include_dashboard=False, object_store_memory=1024**3,
             _temp_dir='/tmp/ir2-original-ray')
    print('Welcome to IR2-MARL Exploration Training Sim!', flush=True)
    print('IR2 [{}] | maps={} | robots={}-{} | agents={} | batch={} | LR={}'.format(
        CONTACT_CONFIG.method, len(manifest['maps']), NUM_ROBOTS_MIN, NUM_ROBOTS_MAX, NUM_META_AGENT, BATCH_SIZE, LR), flush=True)
    log_dir = os.path.join(TRAIN_DIR, datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir, flush_secs=10)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(GIFS_DIR, exist_ok=True)
    setup.update(training_started=True, tensorboard_log_dir=log_dir)
    write_json(Path(BASELINE_DIR) / 'setup.json', setup)
    stopping = [False]
    def request_stop(*_):
        stopping[0] = True
        print('Stop requested; finish the current optimizer block and save the checkpoint.', flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    ### Defining model & params ###
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # Special handling for log alpha
    checkpoint = None
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(MODEL_PATH)
        CONTACT_CONFIG.validate_checkpoint(checkpoint)
        log_alpha = checkpoint['log_alpha'] if CONTINUE_LOG_ALPHA else torch.FloatTensor([INITIAL_LOG_ALPHA]).to(device) 
    else:
        log_alpha = torch.FloatTensor([INITIAL_LOG_ALPHA]).to(device)
    log_alpha.requires_grad = True

    # Init key networks & params
    global_policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM, allow_wait=CONTACT_CONFIG.enabled).to(device)
    global_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM, allow_wait=CONTACT_CONFIG.enabled).to(device)
    global_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM, allow_wait=CONTACT_CONFIG.enabled).to(device)

    global_target_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM, allow_wait=CONTACT_CONFIG.enabled).to(device)
    global_target_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM, allow_wait=CONTACT_CONFIG.enabled).to(device)
    
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4) 

    policy_lr_decay = optim.lr_scheduler.StepLR(global_policy_optimizer, step_size=DECAY_STEP, gamma=0.96)
    q_net1_lr_decay = optim.lr_scheduler.StepLR(global_q_net1_optimizer,step_size=DECAY_STEP, gamma=0.96)
    q_net2_lr_decay = optim.lr_scheduler.StepLR(global_q_net2_optimizer,step_size=DECAY_STEP, gamma=0.96)
    log_alpha_lr_decay = optim.lr_scheduler.StepLR(log_alpha_optimizer, step_size=DECAY_STEP, gamma=0.96)   
    
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    ### Load models from checkpts ###
    if LOAD_MODEL:
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        policy_lr_decay.load_state_dict(checkpoint['policy_lr_decay'])
        q_net1_lr_decay.load_state_dict(checkpoint['q_net1_lr_decay'])
        q_net2_lr_decay.load_state_dict(checkpoint['q_net2_lr_decay'])
        log_alpha_lr_decay.load_state_dict(checkpoint['log_alpha_lr_decay'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to: ", curr_episode)
        print("log_alpha: ", log_alpha)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()
    progress = Progress(BASELINE_DIR, manifest, writer, checkpoint)
    auxiliary = (ContentTrainer(global_policy_net, CONTACT_CONFIG,
                 (checkpoint or {}).get('contact_learning')) if CONTACT_CONFIG.learned else None)

    def save_checkpoint():
        state = {"policy_model": global_policy_net.state_dict(),
                 "q_net1_model": global_q_net1.state_dict(),
                 "q_net2_model": global_q_net2.state_dict(),
                 "log_alpha": log_alpha,
                 "policy_optimizer": global_policy_optimizer.state_dict(),
                 "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                 "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                 "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                 "episode": curr_episode,
                 "policy_lr_decay": policy_lr_decay.state_dict(),
                 "q_net1_lr_decay": q_net1_lr_decay.state_dict(),
                 "q_net2_lr_decay": q_net2_lr_decay.state_dict(),
                 "log_alpha_lr_decay": log_alpha_lr_decay.state_dict(),
                 "completed_episodes": progress.completed,
                 "invalid_episodes": progress.invalid,
                 "training_updates": progress.updates,
                 "baseline_progress": progress.snapshot(),
                 "contact_config": CONTACT_CONFIG.to_dict()}
        if auxiliary is not None:
            state['contact_learning'] = auxiliary.state_dict()
        temp = Path(MODEL_PATH).with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(MODEL_PATH)
        writer.flush()
        print('Saved model: {} | completed={} | SAC updates={}'.format(
            MODEL_PATH, progress.completed, progress.updates), flush=True)
    
    ### Launch meta agents ###
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    ### Get initial weigths ###
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        q_net1_weights = global_q_net1.to(local_device).state_dict()
        global_policy_net.to(device)
        global_q_net1.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
        q_net1_weights = global_q_net1.to(local_device).state_dict()
    weights_set.append(policy_weights)
    weights_set.append(q_net1_weights)
    if auxiliary is not None:
        weights_set.append(auxiliary.weights())

    ### Allow for batch training parallization across GPU ###
    dp_policy = nn.DataParallel(global_policy_net)              # Policy Net
    dp_q_net1 = nn.DataParallel(global_q_net1)                  # Q Net 1 - Get min of the two (min overestimation)
    dp_q_net2 = nn.DataParallel(global_q_net2)                  # Q Net 2 - Get min of the two (min overestimation)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)    # Q-target Net 1 - Train Q-net 1
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)    # Q-target Net 2 - Train Q-net 2

    ### Launch the first job on each runner ###
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))
    
    metric_name = ['travel_dist', 'success_rate', 'explored_rate', 'connectivity_rate', 'agents_connected_percentage']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    experience_buffer = []
    for i in range(15):     # 15dims of inputs
        experience_buffer.append([])
    
    try:
        while not stopping[0]:
            if memory_available_gib() < 4:
                raise MemoryError('Available host RAM below 4 GiB; stopping this training run')
            
            ### Get done jobs ###
            done_id, job_list = ray.wait(job_list, timeout=1)
            if not done_id:
                continue
            done_jobs = ray.get(done_id)
            
            success = True
            for job in done_jobs:
                success, job_results, metrics, info = job

                if not success:
                    break

                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]

                for n in metric_name:
                    perf_metrics[n].append(metrics[n])


            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))
            
            if not success:
                progress.record(info, False, {}, len(experience_buffer[0]), curr_episode)
                continue

            if auxiliary is not None:
                content_loss = auxiliary.update(info.get('contact_labels', []))
                if content_loss is not None:
                    writer.add_scalar('Contact/Content Loss', content_loss, progress.completed + 1)
                writer.add_scalar('Contact/Content Updates', auxiliary.updates, progress.completed + 1)
            
            ### Start training when replay buffer is filled ###
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training", flush=True)

                ### Keep buffer size to max REPLAY_SIZE ###
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                ### Train with 8 batches of rollouts of length=BATCH_SIZE ###
                for j in range(8):  
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):     # size 15
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    ### Generate input rollouts ###
                    node_inputs_batch = torch.stack(rollouts[0])
                    edge_inputs_batch = torch.stack(rollouts[1])
                    current_inputs_batch = torch.stack(rollouts[2])
                    node_padding_mask_batch = torch.stack(rollouts[3])
                    edge_padding_mask_batch = torch.stack(rollouts[4])
                    edge_mask_batch = stack_masks(rollouts[5])
                    action_batch = torch.stack(rollouts[6])
                    reward_batch = torch.stack(rollouts[7])
                    done_batch = torch.stack(rollouts[8])
                    next_node_inputs_batch = torch.stack(rollouts[9])
                    next_edge_inputs_batch = torch.stack(rollouts[10])
                    next_current_inputs_batch = torch.stack(rollouts[11])
                    next_node_padding_mask_batch = torch.stack(rollouts[12])
                    next_edge_padding_mask_batch = torch.stack(rollouts[13])
                    next_edge_mask_batch = stack_masks(rollouts[14])

                    ### Set tensors to global GPU config ###
                    if device != local_device:
                        node_inputs_batch = node_inputs_batch.to(device)
                        edge_inputs_batch = edge_inputs_batch.to(device)
                        current_inputs_batch = current_inputs_batch.to(device)
                        action_batch = action_batch.to(device)
                        reward_batch = reward_batch.to(device)
                        node_padding_mask_batch = node_padding_mask_batch.to(device)
                        edge_mask_batch = edge_mask_batch.to(device)
                        edge_padding_mask_batch = edge_padding_mask_batch.to(device)

                        next_node_inputs_batch = next_node_inputs_batch.to(device)
                        next_edge_inputs_batch = next_edge_inputs_batch.to(device)
                        next_current_inputs_batch = next_current_inputs_batch.to(device)
                        next_node_padding_mask_batch = next_node_padding_mask_batch.to(device)
                        done_batch = done_batch.to(device)
                        next_edge_mask_batch = next_edge_mask_batch.to(device)
                        next_edge_padding_mask_batch = next_edge_padding_mask_batch.to(device)

                    ### Obtain losses via SAC (Soft Actor-Critic) ###
                    with torch.no_grad():
                        q_values1, _ = dp_q_net1(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                        q_values2, _ = dp_q_net2(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                        q_values = torch.min(q_values1, q_values2)

                    ### Formulated in SAC paper: https://arxiv.org/pdf/1801.01290.pdf ###
                    logp = dp_policy(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                    policy_loss = torch.sum((logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())), dim=1).mean()

                    with torch.no_grad():
                        next_logp = dp_policy(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch)
                        next_q_values1, _ = dp_target_q_net1(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch)
                        next_q_values2, _ = dp_target_q_net2(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime_batch = torch.sum(next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)), dim=1).unsqueeze(1)
                        target_q_batch = reward_batch + GAMMA * (1 - done_batch) * value_prime_batch

                    q_values1, _ = dp_q_net1(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                    q_values2, _ = dp_q_net2(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                    q1 = torch.gather(q_values1, 1, action_batch)
                    q2 = torch.gather(q_values2, 1, action_batch)
                    mse_loss = nn.MSELoss()
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

                    ### Train all networks via backpropogation ###
                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=5, norm_type=2)
                    global_policy_optimizer.step()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=2000, norm_type=2)
                    global_q_net1_optimizer.step()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=2000, norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()
                    progress.updates += 1

                    target_q_update_counter += 1
                    #print("target q update counter", target_q_update_counter % 1024)

                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), value_prime_batch.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(), alpha_loss.item(), *perf_data]
                training_data.append(data)

            progress.record(info, True, metrics, len(experience_buffer[0]), curr_episode)

            ### Log training stats to tensorboard ###
            if len(training_data) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            ### Get the updated global weights ###
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                q_net1_weights = global_q_net1.to(local_device).state_dict()
                global_policy_net.to(device)
                global_q_net1.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
                q_net1_weights = global_q_net1.to(local_device).state_dict()
            weights_set.append(policy_weights)
            weights_set.append(q_net1_weights)
            if auxiliary is not None:
                weights_set.append(auxiliary.weights())

            
             ### Hard Q updates to get target Q-networks every 1024 steps ###
            if target_q_update_counter > 64:
                print("update target q net", flush=True)
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            if curr_episode % 32 == 0:
                save_checkpoint()
                    
    
    finally:
        error = sys.exc_info()[1]
        save_checkpoint()
        state = ('stopped_memory_guard' if isinstance(error, MemoryError) else
                 'stopped_by_request' if stopping[0] else 'interrupted_or_failed')
        write_json(Path(BASELINE_DIR) / 'run_state.json', dict(status=state,
            completed_episodes=progress.completed, invalid_episodes=progress.invalid,
            training_updates=progress.updates, checkpoint=MODEL_PATH,
            error=repr(error) if error else None, time=datetime.now().astimezone().isoformat()))
        for a in meta_agents:
            ray.kill(a)
        writer.close()
        ray.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        if ray.is_initialized():
            ray.shutdown()
