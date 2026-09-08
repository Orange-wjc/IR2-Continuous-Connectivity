##############################################################################
# Name: driver.py
# Driver of training program, maintain & update the global network.
##############################################################################

from parameter import *
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

ray.init()
print("Welcome to IR2-MARL Exploration Training Sim!")

log_dir = os.path.join(TRAIN_DIR, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()) 
writer = SummaryWriter(log_dir)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(GIFS_DIR):
    os.makedirs(GIFS_DIR)


def writeToTensorBoard(writer, tensorboardData, curr_episode):
    """ Log data to tensorboard """

    tensorboardData = np.array(tensorboardData)
    tensorboardData = list(np.nanmean(tensorboardData, axis=0))
    (reward, value, policyLoss, qValueLoss, entropy, policyGradNorm,
     qValueGradNorm, log_alpha, alphaLoss, travel_dist, success_rate,
     explored_rate, connectivity_rate, agents_connected_percentage,
     communication_reward, exploration_progress_reward, disconnect_count,
     mean_disconnect_duration, max_disconnect_duration, mean_reconnect_time,
     largest_component_ratio, mean_component_count,
     team_bottleneck_rssi) = tensorboardData

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
    writer.add_scalar(tag='Perf/Communication Reward', scalar_value=communication_reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Exploration Progress Reward', scalar_value=exploration_progress_reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Disconnect Count', scalar_value=disconnect_count, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Mean Disconnect Duration', scalar_value=mean_disconnect_duration, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Max Disconnect Duration', scalar_value=max_disconnect_duration, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Mean Reconnect Time', scalar_value=mean_reconnect_time, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Largest Component Ratio', scalar_value=largest_component_ratio, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Mean Component Count', scalar_value=mean_component_count, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Team Bottleneck RSSI', scalar_value=team_bottleneck_rssi, global_step=curr_episode)


def get_cpu_state_dict(model):
    """Copy model weights to CPU without moving the training model off its device."""
    return {name: value.detach().to(device='cpu', copy=True) for name, value in model.state_dict().items()}

    
def main():

    ### Defining model & params ###
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    if LOAD_MODEL and LOAD_POLICY_ONLY:
        raise ValueError('LOAD_MODEL and LOAD_POLICY_ONLY cannot both be True')

    checkpoint = None
    policy_checkpoint = None

    # Full resume restores every training state; v3 transfers only the policy.
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        log_alpha = checkpoint['log_alpha'] if CONTINUE_LOG_ALPHA else torch.FloatTensor([INITIAL_LOG_ALPHA]).to(device) 
    else:
        log_alpha = torch.FloatTensor([INITIAL_LOG_ALPHA]).to(device)
        if LOAD_POLICY_ONLY:
            print('Loading pretrained policy only from: ', POLICY_PRETRAINED_PATH)
            policy_checkpoint = torch.load(POLICY_PRETRAINED_PATH, map_location='cpu')
    log_alpha.requires_grad = True

    # Init key networks & params
    global_policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    global_target_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    
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
    gradient_update_count = 0
    policy_warmup_updates = (
        POLICY_TRANSFER_CRITIC_WARMUP_UPDATES if LOAD_POLICY_ONLY else 0)
    initial_policy_source = POLICY_PRETRAINED_PATH if LOAD_POLICY_ONLY else None
    initial_policy_episode = None

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
        gradient_update_count = checkpoint.get('gradient_update_count', 0)
        policy_warmup_updates = checkpoint.get('policy_warmup_updates', 0)
        initial_policy_source = checkpoint.get('initial_policy_source')
        initial_policy_episode = checkpoint.get('initial_policy_episode')

        print("curr_episode set to: ", curr_episode)
        print("log_alpha: ", log_alpha)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])
    elif LOAD_POLICY_ONLY:
        source_input_dim = policy_checkpoint.get('input_dim', INPUT_DIM)
        if source_input_dim != INPUT_DIM:
            raise ValueError(
                'Pretrained policy input_dim {} does not match {}'.format(
                    source_input_dim, INPUT_DIM))
        global_policy_net.load_state_dict(policy_checkpoint['policy_model'])
        initial_policy_episode = policy_checkpoint.get('episode')
        print('Loaded policy from episode: ', initial_policy_episode or 'unknown')
        print('Critics, optimizers, alpha, and episode counter are newly initialized.')
        del policy_checkpoint

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()
    
    ### Launch meta agents ###
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    ### Get initial actor weights ###
    policy_weights = get_cpu_state_dict(global_policy_net)

    ### Allow for batch training parallization across GPU ###
    if USE_GPU_GLOBAL and torch.cuda.device_count() > 1:
        dp_policy = nn.DataParallel(global_policy_net)              # Policy Net
        dp_q_net1 = nn.DataParallel(global_q_net1)                  # Q Net 1 - Get min of the two (min overestimation)
        dp_q_net2 = nn.DataParallel(global_q_net2)                  # Q Net 2 - Get min of the two (min overestimation)
        dp_target_q_net1 = nn.DataParallel(global_target_q_net1)    # Q-target Net 1 - Train Q-net 1
        dp_target_q_net2 = nn.DataParallel(global_target_q_net2)    # Q-target Net 2 - Train Q-net 2
    else:
        dp_policy = global_policy_net
        dp_q_net1 = global_q_net1
        dp_q_net2 = global_q_net2
        dp_target_q_net1 = global_target_q_net1
        dp_target_q_net2 = global_target_q_net2

    ### Launch the first job on each runner ###
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(policy_weights, curr_episode))
    
    metric_name = ['travel_dist', 'success_rate', 'explored_rate', 'connectivity_rate',
                   'agents_connected_percentage', 'mean_communication_reward',
                   'mean_exploration_progress_reward', 'disconnect_count',
                   'mean_disconnect_duration', 'max_disconnect_duration',
                   'mean_reconnect_time', 'largest_component_ratio',
                   'mean_component_count', 'team_bottleneck_rssi']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    experience_buffer = []
    for i in range(15):     # 15dims of inputs
        experience_buffer.append([])
    
    try:
        while True:
            
            ### Get done jobs ###
            done_id, job_list = ray.wait(job_list)
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
            job_list.append(meta_agents[info['id']].job.remote(policy_weights, curr_episode))
            
            if not success:
                continue
            
            ### Start training when replay buffer is filled ###
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

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
                    edge_mask_batch = torch.stack(rollouts[5])
                    action_batch = torch.stack(rollouts[6])
                    reward_batch = torch.stack(rollouts[7])
                    done_batch = torch.stack(rollouts[8])
                    next_node_inputs_batch = torch.stack(rollouts[9])
                    next_edge_inputs_batch = torch.stack(rollouts[10])
                    next_current_inputs_batch = torch.stack(rollouts[11])
                    next_node_padding_mask_batch = torch.stack(rollouts[12])
                    next_edge_padding_mask_batch = torch.stack(rollouts[13])
                    next_edge_mask_batch = torch.stack(rollouts[14])

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

                    gradient_update_count += 1
                    update_policy = gradient_update_count > policy_warmup_updates

                    ### Obtain target values via SAC (Soft Actor-Critic) ###
                    with torch.no_grad():
                        next_logp = dp_policy(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch)
                        next_q_values1, _ = dp_target_q_net1(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch)
                        next_q_values2, _ = dp_target_q_net2(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime_batch = torch.sum(next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)), dim=1).unsqueeze(1)
                        target_q_batch = reward_batch + GAMMA * (1 - done_batch) * value_prime_batch

                        policy_q_values1, _ = dp_q_net1(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                        policy_q_values2, _ = dp_q_net2(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                        policy_q_values = torch.min(policy_q_values1, policy_q_values2)

                    ### Formulated in SAC paper: https://arxiv.org/pdf/1801.01290.pdf ###
                    if update_policy:
                        logp = dp_policy(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                    else:
                        with torch.no_grad():
                            logp = dp_policy(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                    policy_loss = torch.sum((logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - policy_q_values)), dim=1).mean()
                    del (next_logp, next_q_values1, next_q_values2,
                         next_q_values, policy_q_values1, policy_q_values2)
                    entropy = (logp.detach() * logp.detach().exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy + entropy_target)).mean()
                    policy_loss_value = policy_loss.item()
                    entropy_value = entropy.mean().item()
                    alpha_loss_value = alpha_loss.item()

                    ### Train all networks via backpropogation ###
                    if update_policy:
                        global_policy_optimizer.zero_grad()
                        policy_loss.backward()
                        policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=5, norm_type=2)
                        global_policy_optimizer.step()
                    else:
                        policy_grad_norm = torch.tensor(0.0, device=device)

                    if update_policy:
                        log_alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        log_alpha_optimizer.step()
                    elif gradient_update_count == policy_warmup_updates:
                        print('Critic warmup complete; enabling policy and alpha updates.')

                    del logp, policy_loss, policy_q_values

                    # Update critics sequentially so their attention graphs never coexist.
                    mse_loss = nn.MSELoss()
                    q_values1, _ = dp_q_net1(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                    q1 = torch.gather(q_values1, 1, action_batch)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()
                    q1_loss_value = q1_loss.item()
                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=2000, norm_type=2)
                    global_q_net1_optimizer.step()
                    del q_values1, q1, q1_loss

                    q_values2, _ = dp_q_net2(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                    q2 = torch.gather(q_values2, 1, action_batch)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()
                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=2000, norm_type=2)
                    global_q_net2_optimizer.step()
                    del q_values2, q2, q2_loss, target_q_batch

                    target_q_update_counter += 1
                    #print("target q update counter", target_q_update_counter % 1024)

                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), value_prime_batch.mean().item(), policy_loss_value, q1_loss_value,
                        entropy_value, policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(), alpha_loss_value, *perf_data]
                training_data.append(data)

                ### Get the updated actor weights ###
                policy_weights = get_cpu_state_dict(global_policy_net)

            ### Log training stats to tensorboard ###
            if len(training_data) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

             ### Hard Q updates to get target Q-networks every 1024 steps ###
            if target_q_update_counter > 64:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
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
                                "input_dim": INPUT_DIM,
                                "connectivity_feature_dim": CONNECTIVITY_FEATURE_DIM,
                                "use_connectivity_features": USE_CONNECTIVITY_FEATURES,
                                "reward_version": "balanced_v3",
                                "initial_policy_source": initial_policy_source,
                                "initial_policy_episode": initial_policy_episode,
                                "gradient_update_count": gradient_update_count,
                                "policy_warmup_updates": policy_warmup_updates,
                                "reward_config": {
                                    "weak_signal": WEAK_SIGNAL_PENALTY_WEIGHT,
                                    "component_deficit": COMPONENT_DEFICIT_PENALTY_WEIGHT,
                                    "disconnect_duration": DISCONNECT_DURATION_PENALTY_WEIGHT,
                                    "new_disconnect": NEW_DISCONNECT_PENALTY_WEIGHT,
                                    "reconnect": RECONNECT_REWARD_WEIGHT,
                                    "duration_saturation_steps": DISCONNECT_DURATION_SATURATION_STEPS,
                                    "disconnect_grace_steps": DISCONNECT_GRACE_STEPS,
                                    "team_exploration_progress": TEAM_EXPLORATION_PROGRESS_WEIGHT,
                                },
                        }
                path_checkpoint = "./" + MODEL_PATH
                torch.save(checkpoint, path_checkpoint)
                if curr_episode % ARCHIVE_CHECKPOINT_EVERY == 0:
                    archive_path = os.path.join(
                        MODEL_DIR, 'checkpoint_{}.pth'.format(curr_episode))
                    torch.save(checkpoint, archive_path)
                    print('Archived model to: ', archive_path)
                print('Saved model', end='\n')
                    
    
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
