"""Shared action construction and synchronous rollout for training and inference."""

import os

import numpy as np
import torch


def action_candidates(neighbors, current_index, node_coords, k_size, device):
    """Keep stay first; padding validity is independent of node IDs."""
    if k_size < 1:
        raise ValueError('K_SIZE must leave room for the stay action')
    moves = list(dict.fromkeys(index for index in neighbors if index != current_index))
    if len(moves) > k_size - 1:
        moves.sort(key=lambda index: (
            np.linalg.norm(node_coords[index] - node_coords[current_index]), index))
    valid_edges = [current_index] + moves[:k_size - 1]
    padded_edges = valid_edges + [current_index] * (k_size - len(valid_edges))
    edges = torch.tensor(padded_edges, dtype=torch.long, device=device).view(1, 1, -1)
    mask = torch.ones((1, 1, k_size), dtype=torch.long, device=device)
    mask[:, :, :len(valid_edges)] = 0
    return edges, mask, valid_edges


def _observations(worker, episode, step):
    observations = []
    for robot_id, robot in enumerate(worker.robot_list):
        observation, success = worker.get_observations(
            robot.robot_position, robot_id, episode, step, plot=worker.save_image)
        if not success:
            return None
        observations.append(observation)
    return observations


def _plot_team(worker, gifs_dir, step):
    routes = []
    for robot_id, robot in enumerate(worker.robot_list):
        robot.save_robot_position()
        route = [robot.xPoints, robot.yPoints]
        routes.append(route)
        path = os.path.join(gifs_dir, 'robot_{}'.format(robot_id + 1))
        os.makedirs(path, exist_ok=True)
        worker.env.plot_env(worker.global_step, path, step,
                            robot.travel_dist, [route], robot_id)
    path = os.path.join(gifs_dir, 'merged')
    os.makedirs(path, exist_ok=True)
    worker.env.plot_env_ground_truth(
        worker.global_step, path, step,
        max(robot.travel_dist for robot in worker.robot_list), routes)


def run_team_episode(worker, episode, max_steps, gifs_dir, collect_experience):
    """Commit transitions only after the entire team has moved and communicated."""
    env = worker.env
    if not env.initialize_team(worker.all_robot_positions, worker.global_step):
        return False
    observations = _observations(worker, episode, 0)
    if observations is None:
        return False

    completed = env.check_done()
    steps = 0
    stay_count = 0
    for step in range(max_steps):
        if completed:
            break
        # All actions use the same time boundary. No environment writes in this loop.
        choices = [worker.select_node(observation, robot_id)
                   for robot_id, observation in enumerate(observations)]
        next_positions = [position.copy() for position, _ in choices]
        distances = [float(np.linalg.norm(position - robot.robot_position))
                     for position, robot in zip(next_positions, worker.robot_list)]
        success, individual_rewards, team_reward, completed = env.step_team(
            next_positions, worker.global_step, step + 1, distances)
        if not success:
            return False

        for robot, position, distance in zip(worker.robot_list, next_positions, distances):
            robot.robot_position = position
            robot.travel_dist += distance
        worker.all_robot_positions = next_positions
        next_observations = _observations(worker, episode, step + 1)
        if next_observations is None:
            return False

        steps = step + 1
        timed_out = steps == max_steps and not completed
        # The configured horizon is a task deadline, not a collector interruption.
        terminal = completed or timed_out
        stay_count += sum(distance == 0 for distance in distances)
        if collect_experience:
            for robot_id, robot in enumerate(worker.robot_list):
                reward_parts = list(env.individual_reward_components[robot_id]) + [
                    team_reward, env.communication_reward, env.exploration_progress_reward,
                    env.no_exploration_progress_penalty, float(completed), float(timed_out)]
                robot.save_transition(
                    observations[robot_id], choices[robot_id][1],
                    individual_rewards[robot_id] + team_reward, terminal,
                    next_observations[robot_id], episode, step, reward_parts)

        # Reuse these exact tensors for the next decision; don't rebuild the graphs.
        observations = next_observations
        if worker.save_image:
            _plot_team(worker, gifs_dir, step)
        if terminal:
            break

    if collect_experience:
        for robot in worker.robot_list:
            for field, values in zip(worker.episode_buffer, robot.episode_buffer):
                field.extend(values)

    worker.perf_metrics.update(env.get_connectivity_metrics())
    worker.perf_metrics.update({
        'travel_dist': max(robot.travel_dist for robot in worker.robot_list),
        'explored_rate': env.explored_rate,
        'success_rate': completed,
        'travel_steps': steps,
        'stay_rate': stay_count / max(steps * worker.n_agent, 1),
        'time_limit_reached': steps == max_steps and not completed,
    })
    if worker.save_image and steps:
        for robot_id in range(worker.n_agent):
            worker.make_gif(os.path.join(gifs_dir, 'robot_{}'.format(robot_id + 1)),
                            episode, robot_id)
        worker.make_gif_ground_truth(os.path.join(gifs_dir, 'merged'), episode)
    worker.max_node_coords = max(worker.max_node_coords,
                                 max(len(coords) for coords in env.all_node_coords))
    print('[Eps {} Completed] Steps: {}, Node Coords: {}, Max Dist: {:.2f}'.format(
        episode, steps, worker.max_node_coords, worker.perf_metrics['travel_dist']))
    return True
