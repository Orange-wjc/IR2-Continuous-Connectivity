"""Behavioral checks for finite communication and the two-robot learning interface.

Run: /home/wyz/anaconda3/envs/ir2_train/bin/python -m unittest discover -s tests -p test_finite_contact.py
"""
import unittest
from unittest.mock import patch
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from finite_contact.config import Config
from finite_contact.communication import Radio, MAP
from finite_contact.maps import simple_map, manifest
from finite_contact.environment import Environment
from finite_contact.learning import Learner, tensors, bellman_target
from model import SingleHeadAttention


class SuccessLearningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_completed_training_tasks_enter_success_memory(self):
        from finite_contact_train import rollout, evaluate
        cfg = Config(completion=.2, batch_size=2, warmup=2, success_imitation_weight=1.)
        learner = Learner(cfg)
        result, _ = rollout(learner, cfg, 'train', 0, 'movement')
        self.assertTrue(result['success'])
        self.assertEqual(len(learner.success_episodes), 1)
        self.assertTrue(all(x[5] and not x[6] for x in learner.success_episodes[0]['transitions'][-2:]))
        evaluate(learner, cfg, 1, split='validation')
        self.assertEqual(len(learner.success_episodes), 1, 'Validation leaked into demonstrations')
        failed_cfg = Config(horizon_seconds=1., success_imitation_weight=1.)
        failed = Learner(failed_cfg)
        result, _ = rollout(failed, failed_cfg, 'train', 0, 'movement')
        self.assertTrue(result['truncated'])
        self.assertEqual(failed.success_episodes, [])
        with self.assertRaises(ValueError):
            failed.remember_success(list(failed.replay), 1.)
        interrupted_cfg = Config(batch_size=2, warmup=2, success_imitation_weight=1.)
        interrupted = Learner(interrupted_cfg)
        result, _ = rollout(interrupted, interrupted_cfg, 'train', 0, 'movement', max_updates=1)
        self.assertTrue(result['interrupted'])
        self.assertEqual(interrupted.success_episodes, [])

    def test_local_coverage_progress_is_rewarded_after_union_is_complete(self):
        legacy = Config(communication_range=1., gamma=1., coverage_reward='union')
        aligned = Config(**{**legacy.to_dict(), 'coverage_reward': 'local_mean'})
        envs = [Environment(cfg) for cfg in (legacy, aligned)]
        for env in envs:
            env.beliefs[1] = env.truth.copy()
            env.refresh_graphs()
        obs = envs[0].observation(0)
        valid = np.flatnonzero(obs[4][0] == 0)
        action = max(valid, key=lambda a: np.linalg.norm(
            envs[0].graph_data[0][0][obs[1][0, a]] - envs[0].positions[0]))
        before = envs[0].coverage()[0].mean()
        for env in envs:
            env.step([int(action), 0])
        after = envs[0].coverage()[0].mean()
        self.assertGreater(after, before)
        self.assertEqual(envs[0].coverage()[1], 1.)
        np.testing.assert_allclose(envs[0].reward_components['exploration'], [0., 0.])
        np.testing.assert_allclose(envs[1].reward_components['exploration'], [(after - before) * 20.] * 2)
        for robot in range(2):
            np.testing.assert_array_equal(envs[0].beliefs[robot], envs[1].beliefs[robot])
            for a, b in zip(envs[0].observation(robot), envs[1].observation(robot)):
                np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(envs[0].positions, envs[1].positions)
        self.assertEqual(envs[0].radio.sent_bytes, 0)
        self.assertEqual(envs[1].radio.sent_bytes, 0)
        self.assertEqual(Config.from_checkpoint({}).coverage_reward, 'union')

    def test_imitation_learns_real_success_actions_and_retains_fast_episodes(self):
        from finite_contact_train import rollout
        cfg = Config(completion=.2, batch_size=2, warmup=2, success_imitation_weight=1.)
        learner = Learner(cfg)
        rollout(learner, cfg, 'train', 0, 'movement')
        demo = learner.success_episodes[0]['transitions']
        for duration in (30., 10., 20., 40.):
            learner.remember_success(demo, duration)
        self.assertEqual([x['duration'] for x in learner.success_episodes],
                         sorted([learner.success_episodes[0]['duration'], 10., 20.]))
        inputs = tensors([x[0] for x in demo], learner.device)
        actions = torch.tensor([x[1] for x in demo]).view(-1, 1)
        with torch.no_grad():
            learner.log_alpha.fill_(-8.)
            for q in learner.critics:
                q.q_values_layer.weight.zero_()
                q.q_values_layer.bias.zero_()
            before = learner.policy(*inputs).gather(1, actions).mean().item()
        result = learner.update()
        with torch.no_grad():
            after = learner.policy(*inputs).gather(1, actions).mean().item()
        self.assertGreater(after, before)
        self.assertGreater(result['imitation_loss'], 0.)
        self.assertEqual(result['imitation_updates'], 1)
        self.assertEqual(Config.from_checkpoint({}).success_imitation_weight, 0.)

    def test_full_checkpoint_restores_success_memory_and_exact_next_update(self):
        from finite_contact_train import rollout, checkpoint
        cfg = Config(completion=.2, batch_size=2, warmup=2, success_imitation_weight=1.)
        learner = Learner(cfg)
        rollout(learner, cfg, 'train', 0, 'movement')
        learner.update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pth'
            checkpoint(path, learner, cfg, 'movement', 1, None, 0.)
            saved = torch.load(path, map_location='cpu')
        restored = Learner(cfg)
        restored.load(saved['learner'], optimizers=True)
        restored.replay.extend(saved['replay'])
        restored.success_episodes = saved['success_episodes']
        self.assertEqual(learner.imitation_updates, restored.imitation_updates)
        random.setstate(saved['python_rng'])
        first = learner.update()
        random.setstate(saved['python_rng'])
        second = restored.update()
        self.assertEqual(first, second)
        for key, value in learner.policy.state_dict().items():
            torch.testing.assert_close(value, restored.policy.state_dict()[key], rtol=0, atol=0)

    def test_failed_or_incomplete_evaluation_cannot_replace_success_checkpoint(self):
        from finite_contact_train import successful_rank
        successful = {'success': True, 'time': 197., 'coverage_deficit': .2}
        failed = {'success': False, 'time': 256., 'coverage_deficit': .09}
        self.assertEqual(successful_rank([successful], 1), (197., .2))
        self.assertIsNone(successful_rank([failed], 1))
        self.assertIsNone(successful_rank([successful], 2))
        self.assertIsNone(successful_rank([successful, failed], 2))
        self.assertIsNone(successful_rank([{**successful, 'interrupted': True}], 1))

    def test_cli_resume_keeps_saved_content_mode_without_repeating_flag(self):
        from finite_contact_train import checkpoint, main
        cfg = Config(content='fifo', batch_size=2, warmup=2, completion=.2)
        for stage, limit in [('finetune', '--max-updates'), ('content', '--max-content-updates')]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'latest.pth'
                checkpoint(path, Learner(cfg), cfg, stage, 0, None, 0.)
                argv = ['finite_contact_train.py', '--stage', stage, '--resume', str(path),
                        limit, '1', '--episodes', '1', '--max-hours', '.02', '--device', 'cpu']
                with patch('sys.argv', argv), patch('finite_contact_train.provenance'):
                    main()
                saved = torch.load(path, map_location='cpu')
                self.assertEqual(saved['config'], cfg.to_dict())
                counter = 'updates' if stage == 'finetune' else 'content_updates'
                self.assertEqual(saved['learner'][counter], 1)

    def test_actor_rate_fork_preserves_full_state_and_changes_only_actor_step(self):
        from finite_contact_train import checkpoint, rollout
        from finite_contact_fork import fork_checkpoint
        cfg = Config(completion=.2, batch_size=2, warmup=2, success_imitation_weight=1.)
        learner = Learner(cfg)
        rollout(learner, cfg, 'train', 0, 'finetune')
        learner.update()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pth'
            checkpoint(source, learner, cfg, 'finetune', 1, None, 10.)
            with patch('finite_contact_fork.provenance'):
                fork = fork_checkpoint(source, Path(directory) / 'fork', 1e-5)
            old_saved = torch.load(source, map_location='cpu')
            new_saved = torch.load(fork, map_location='cpu')
        self.assertEqual(new_saved['config'], {**cfg.to_dict(), 'policy_learning_rate': 1e-5})
        self.assertEqual(old_saved['python_rng'], new_saved['python_rng'])
        torch.testing.assert_close(old_saved['torch_rng'], new_saved['torch_rng'], rtol=0, atol=0)
        self.assertEqual(len(old_saved['replay']), len(new_saved['replay']))
        self.assertEqual([e['duration'] for e in old_saved['success_episodes']],
                         [e['duration'] for e in new_saved['success_episodes']])
        pair = []
        losses = []
        changes = []
        for saved in (old_saved, new_saved):
            model = Learner(Config.from_checkpoint(saved['config']))
            model.load(saved['learner'], optimizers=True)
            model.replay.extend(saved['replay'])
            model.success_episodes = saved['success_episodes']
            before = {k: v.clone() for k, v in model.policy.state_dict().items()}
            random.setstate(saved['python_rng'])
            losses.append(model.update())
            changes.append(sum((v - before[k]).double().square().sum().item()
                               for k, v in model.policy.state_dict().items()) ** .5)
            pair.append(model)
        self.assertEqual(losses[0], losses[1])
        self.assertGreater(changes[1], 0.)
        self.assertAlmostEqual(changes[1] / changes[0], .1, delta=.003)
        for key in ('critics', 'targets', 'context_encoder', 'content'):
            left, right = pair[0].state()[key], pair[1].state()[key]
            for a, b in zip(left if isinstance(left, list) else [left], right if isinstance(right, list) else [right]):
                for name in a:
                    torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        torch.testing.assert_close(pair[0].log_alpha, pair[1].log_alpha, rtol=0, atol=0)
        self.assertEqual(Config.from_checkpoint({'learning_rate': .0003}).policy_learning_rate, .0003)
        self.assertEqual(Learner(Config(learning_rate=.0003)).policy_optimizer.param_groups[0]['lr'], .0003)

    def test_cli_resumes_fork_with_saved_actor_rate_and_real_updates(self):
        from finite_contact_train import checkpoint, rollout, main
        from finite_contact_fork import fork_checkpoint
        cfg = Config(completion=.2, batch_size=2, warmup=2, success_imitation_weight=1.)
        learner = Learner(cfg)
        rollout(learner, cfg, 'train', 0, 'finetune')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pth'
            output = Path(directory) / 'fork'
            checkpoint(source, learner, cfg, 'finetune', 1, None, 0.)
            with patch('finite_contact_fork.provenance'):
                fork = fork_checkpoint(source, output, 1e-5)
            argv = ['finite_contact_train.py', '--stage', 'finetune', '--resume', str(fork),
                    '--max-updates', str(learner.updates + 1), '--episodes', '2', '--max-hours', '.02', '--device', 'cpu']
            with patch('sys.argv', argv), patch('finite_contact_train.provenance'):
                main()
            saved = torch.load(output / 'latest.pth', map_location='cpu')
            self.assertEqual(saved['learner']['updates'], learner.updates + 1)
            self.assertEqual(saved['learner']['policy_optimizer']['param_groups'][0]['lr'], 1e-5)
            self.assertEqual(saved['learner']['q_optimizers'][0]['param_groups'][0]['lr'], cfg.learning_rate)
            self.assertTrue(saved['success_episodes'])


class PointerChecks(unittest.TestCase):
    def test_old_checkpoint_mode_is_preserved_and_new_runs_default_centered(self):
        self.assertEqual(Config().pointer_mode, "centered")
        old = Config.from_checkpoint({"seed": 3})
        self.assertEqual(old.pointer_mode, "legacy")
        self.assertEqual(old.time_penalty, .02)
        self.assertEqual(Config.from_checkpoint(Config().to_dict()).pointer_mode, "centered")

    def test_centering_removes_common_offset_without_padding_and_restores_gradient(self):
        legacy = SingleHeadAttention(2).double()
        centered = SingleHeadAttention(2, center_logits=True).double()
        with torch.no_grad():
            legacy.w_query.copy_(torch.eye(2))
            legacy.w_key.copy_(torch.eye(2))
        centered.load_state_dict(legacy.state_dict())
        q = torch.tensor([[[1., 0.]]], dtype=torch.float64, requires_grad=True)
        k = torch.tensor([[[28.3, 0.], [29.7, 0.], [10000., 0.]]], dtype=torch.float64)
        mask = torch.tensor([[[0, 0, 1]]])
        original_mask = mask.clone()
        before = legacy(q, k, mask).exp()
        torch.testing.assert_close(before[0, 0, :2], torch.tensor([.5, .5], dtype=torch.float64))
        logp = centered(q, k, mask)
        self.assertGreater(float(logp[0, 0, 1].exp()), .99)
        self.assertEqual(float(logp[0, 0, 2].exp()), 0.)
        shifted = k.clone()
        shifted[..., 0] += 1000.
        torch.testing.assert_close(logp, centered(q, shifted, mask), atol=1e-9, rtol=1e-9)
        (-logp[0, 0, 0]).backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad.abs().sum()), .01)
        self.assertTrue(torch.equal(mask, original_mask))


class DeadlineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_temperature_rate_changes_only_temperature_on_a_paired_first_update(self):
        cfg = Config(batch_size=2, warmup=2)
        fast_cfg = Config(batch_size=2, warmup=2, alpha_learning_rate=.005)
        slow, fast = Learner(cfg), Learner(fast_cfg)
        fast.load(slow.state())
        env = Environment(cfg)
        obs = [env.observation(i) for i in range(2)]
        reward, duration, terminated, truncated = env.step([0, 0])
        batch = [(obs[i], 0, float(reward[i]), duration, env.observation(i),
                  float(terminated), truncated) for i in range(2)]
        slow.replay.extend(batch)
        fast.replay.extend(batch)
        random.seed(17)
        slow.update()
        random.seed(17)
        fast.update()
        for key in ('policy', 'critics', 'targets'):
            a, b = slow.state()[key], fast.state()[key]
            for left, right in zip(a if isinstance(a, list) else [a], b if isinstance(b, list) else [b]):
                for name in left:
                    torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        self.assertLess(float(fast.log_alpha), float(slow.log_alpha))
        slow_change = -2. - float(slow.log_alpha)
        fast_change = -2. - float(fast.log_alpha)
        self.assertAlmostEqual(fast_change / slow_change, 50., delta=.2)
        self.assertEqual(Config.from_checkpoint({'learning_rate': .0003}).alpha_learning_rate, .0003)

    def test_deadline_has_no_future_reward_but_unfinished_action_bootstraps(self):
        cfg = Config(gamma=.5, dt=1.)
        rewards = torch.tensor([[-.02], [20.], [-.04]])
        durations = torch.tensor([[1.], [1.], [2.]])
        terminated = torch.tensor([[0.], [1.], [0.]])
        truncated = torch.tensor([[1.], [0.], [0.]])
        future = torch.full((3, 1), 100.)
        actual = bellman_target(cfg, rewards, durations, terminated, truncated, future)
        torch.testing.assert_close(actual, torch.tensor([[-.02], [20.], [24.96]]))
        cfg.horizon_target = "bootstrap"
        historical = bellman_target(cfg, rewards, durations, terminated, truncated, future)
        torch.testing.assert_close(historical, torch.tensor([[49.98], [20.], [24.96]]))
        self.assertEqual(Config.from_checkpoint({}).horizon_target, "bootstrap")
        self.assertEqual(Config.from_checkpoint(Config().to_dict()).horizon_target, "terminal")

    def test_actual_deadline_is_stored_and_used_by_learning(self):
        from finite_contact_train import rollout
        cfg = Config(horizon_seconds=1., batch_size=2, warmup=2)
        learner = Learner(cfg)
        result, _ = rollout(learner, cfg, "train", 0, "movement")
        self.assertTrue(result["truncated"])
        self.assertFalse(result["interrupted"])
        self.assertEqual(len(learner.replay), 2)
        self.assertTrue(all(not x[5] and x[6] for x in learner.replay))
        self.assertEqual(result["deadline_fraction"], 1.)


class RadioChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(bytes_per_second=100., pose_period=100.)
        self.maps = [np.full((8, 8), 255, np.uint8), np.full((8, 8), 127, np.uint8)]
        self.radio = Radio((8, 8), self.cfg)
        self.radio.enqueue_updates(0, self.maps[0], 0.)
        self.positions = np.array([[1., 1.], [2., 2.]])
        self.velocities = np.zeros((2, 2))

    def serve(self, time, duration=1., connected=True):
        return self.radio.service(self.positions, self.velocities, time, duration,
                                  connected, self.maps)

    def test_disconnection_does_not_deliver_map_or_live_pose(self):
        self.serve(0., duration=100., connected=False)
        self.assertTrue(np.all(self.maps[1] == 127))
        self.assertIsNone(self.radio.peer_pose[1])
        self.assertEqual(self.radio.sent_bytes, 0)
        self.positions[0] = [6, 6]
        self.serve(100., connected=False)
        self.assertIsNone(self.radio.peer_pose[1])

    def test_capacity_partial_packet_and_resume(self):
        self.serve(0., 2.)
        sent_by_zero = sum(e["bytes"] for e in self.radio.events if e["sender"] == 0)
        self.assertEqual(sent_by_zero, 200)
        self.assertTrue(np.all(self.maps[1] == 127), "A partial map packet became visible")
        progress = self.radio.active[(0, 1)].offset
        self.serve(2., 1., connected=False)
        self.assertEqual(self.radio.active[(0, 1)].offset, progress)
        gain = self.serve(3., 1.)
        self.assertGreater(gain[0], 0)
        self.assertEqual(int(np.count_nonzero(self.maps[1] != 127)), int(gain[0]))
        self.assertTrue(np.all(self.radio.known_peer[(0, 1)] == 127), "Sender knew an unreceived ACK")
        self.serve(4., 1.)
        self.assertGreater(np.count_nonzero(self.radio.known_peer[(0, 1)] != 127), 0)

    def test_retransmission_not_rewarded_twice(self):
        for time in range(6):
            self.serve(float(time))
        self.assertTrue(np.all(self.maps[1] == 255))
        novel_before = self.radio.unique_cells
        # Force a genuine retransmission of an identical serialized update.
        other = Radio((8, 8), self.cfg)
        other.enqueue_updates(0, self.maps[0], 6.)
        packets = [self.radio.packet(MAP, 0, 6., p.cells().tobytes(), len(p.cells()))
                   for p in other.queues[(0, 1)]]
        # Model lost confirmation: the receiver has the data but the sender cannot know this.
        self.radio.known_peer[(0, 1)][:] = 127
        self.radio.queues[(0, 1)].extend(packets)
        for time in range(6, 12):
            gain = self.serve(float(time))
            self.assertEqual(gain[0], 0)
        self.assertEqual(self.radio.unique_cells, novel_before)
        self.assertGreater(self.radio.duplicate_cells, 0)

    def test_common_protocol_skips_proven_known_packets(self):
        self.radio.known_peer[(0, 1)][:] = 255
        self.serve(0.)
        self.assertFalse(self.radio.queues[(0, 1)])
        self.assertEqual(self.radio.completed_maps, 0)

    def test_serialized_packet_size(self):
        self.assertTrue(all(p.size <= self.cfg.max_packet_bytes
                            for p in self.radio.queues[(0, 1)]))
        self.assertEqual(sum(len(p.cells()) for p in self.radio.queues[(0, 1)]), 64)


class EnvironmentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frontier_potential_is_local_and_uses_reachable_graph_paths(self):
        env = Environment(Config(frontier_shaping_scale=2.))
        coords = np.array([[0, 0], [20, 0], [40, 0]])
        edges = {(0, 0): {(20, 0): 20}, (20, 0): {(40, 0): 20}}
        env.graph_data[0] = (coords, edges, np.array([0, 0, 1]), np.zeros(3))
        expected = -2. * 40 / 256
        self.assertEqual(env.frontier_potential(0), expected)
        env.truth[:] = 1
        env.beliefs[1][:] = 255
        env.positions[1][:] = 200
        self.assertEqual(env.frontier_potential(0), expected)
        edges[(20, 0)] = {}
        self.assertEqual(env.frontier_potential(0), 0.)
        with self.assertRaises(ValueError):
            Config(frontier_shaping_scale=2., horizon_target='bootstrap').validate()

    def test_shaping_telescopes_with_physical_durations_at_success_and_deadline(self):
        for completion in (.9, .99):
            base = Environment(Config(horizon_seconds=13.1, completion=completion))
            shaped = Environment(Config(horizon_seconds=13.1, completion=completion, frontier_shaping_scale=2.))
            for env in (base, shaped):
                for belief in env.beliefs:
                    belief[:] = env.truth
                    belief[192:240, 127:158] = 127
                env.refresh_graphs()
            initial = np.array([shaped.frontier_potential(i) for i in range(2)])
            self.assertTrue(np.all(initial < 0))
            difference = np.zeros(2)
            while True:
                actions = [int(np.flatnonzero(base.observation(i)[4][0, 1:] == 0)[0] + 1) for i in range(2)]
                elapsed = base.time
                r0, d0, term0, trunc0 = base.step(actions)
                r1, d1, term1, trunc1 = shaped.step(actions)
                self.assertEqual((d0, term0, trunc0), (d1, term1, trunc1))
                np.testing.assert_array_equal(base.positions, shaped.positions)
                difference += base.config.gamma ** elapsed * (r1 - r0)
                if term0 or trunc0:
                    self.assertEqual(term0, completion == .9)
                    np.testing.assert_allclose(difference, -initial, atol=1e-10)
                    break
            np.testing.assert_allclose(sum(shaped.reward_components.values()),
                                       sum(base.reward_components.values()) + shaped.reward_components['frontier_shaping'])

    def test_fifo_fast_scores_equal_original_features(self):
        env = Environment(Config(variant="A"))
        known = env.radio.known_peer[(0, 1)]
        known.ravel()[::3] = 255
        packets = env.radio.pending(0)
        features, _ = env.packet_features(0, packets)
        old = np.array([len(p.cells()) * features[i, 4] for i, p in enumerate(packets)])
        np.testing.assert_array_equal(env.packet_scores(0, packets, "fifo"), old)
        self.assertEqual(env.packet_scores(0, [], "fifo").size, 0)

    def test_content_labels_cover_novel_queue_tail(self):
        env = Environment(Config(label_sampling="spread"))
        packets = env.radio.pending(0)
        self.assertGreater(len(packets), 64)
        env.feature_provider = lambda sender, indices: np.zeros((len(indices), 64), np.float32)
        env.beliefs[1][:] = 255
        env.beliefs[1].ravel()[packets[-1].cells()["index"]] = 127
        spread = env.collect_labels(0)
        self.assertEqual(len(spread), 64)
        self.assertTrue(any(y > 0 for _, _, y in spread))
        env.config.label_sampling = "prefix"
        self.assertTrue(all(y == 0 for _, _, y in env.collect_labels(0)))

    def test_content_prediction_bound_uses_only_sender_evidence(self):
        env = Environment(Config(content_score_bound="sender"),
                          scorer=lambda features, context: np.full(len(features), 100.))
        env.feature_provider = lambda sender, indices: np.zeros((len(indices), 64), np.float32)
        packets = env.radio.pending(0)[:4]
        known = env.radio.known_peer[(0, 1)].ravel()
        known[packets[0].cells()["index"]] = 255
        before = env.packet_scores(0, packets, "learned")
        self.assertEqual(before[0], 0.)
        for p, score in zip(packets, before):
            self.assertLessEqual(score, np.count_nonzero(known[p.cells()["index"]] == 127))
        env.beliefs[1][:] = 255
        env.positions[1][:] = 220
        env.truth[:] = 1
        np.testing.assert_array_equal(env.packet_scores(0, packets, "learned"), before)
        self.assertEqual(Config.from_checkpoint({}).content_score_bound, "none")

    def test_delivery_estimate_finishes_active_packet_first_without_mutation(self):
        cfg = Config(bytes_per_second=100., estimated_contact_seconds=1., delivery_queue_order="service")
        env = Environment(cfg)
        packets = [p for p in env.radio.pending(0) if p.size > 100]
        active, queued = packets[:2]
        active.offset = active.size - 1
        env.radio.active[(0, 1)] = active
        env.radio.queues[(0, 1)] = [queued]
        expected = env.expected_delivery(0)
        self.assertEqual(expected, len(active.cells()))
        self.assertEqual(active.offset, active.size - 1)
        self.assertEqual(env.radio.queues[(0, 1)], [queued])
        cfg.delivery_queue_order = "legacy"
        self.assertEqual(env.expected_delivery(0), 0.)
        self.assertEqual(Config.from_checkpoint({}).delivery_queue_order, "legacy")

    def test_content_budget_is_exact_and_movement_is_frozen(self):
        from finite_contact_train import rollout
        cfg = Config(batch_size=2, warmup=2, horizon_seconds=32., updates_per_decision=4)
        learner = Learner(cfg)
        policy = {k: v.clone() for k, v in learner.policy.state_dict().items()}
        result, _ = rollout(learner, cfg, "train", 0, "content", max_content_updates=3)
        self.assertEqual(learner.content_updates, 3)
        self.assertEqual(learner.updates, 0)
        self.assertEqual(result["interruption_reason"], "content_update_budget")
        for key, value in policy.items():
            self.assertTrue(torch.equal(value, learner.policy.state_dict()[key]))

    def test_map_splits_do_not_duplicate_geometry(self):
        entries = manifest(Config(train_maps=10))
        self.assertEqual(len(entries), 20)
        self.assertEqual(len({entry["sha256"] for entry in entries}), 20)
        for entry in entries:
            truth, starts, _ = simple_map(entry["split"], entry["index"])
            self.assertTrue(np.all(truth[starts[:, 1].astype(int), starts[:, 0].astype(int)] == 255))

    def test_actor_does_not_read_remote_truth(self):
        env = Environment(Config(variant="A"))
        before = env.observation(0)
        env.positions[1] = [220., 220.]
        env.beliefs[1][:] = 255
        env.truth[:] = 1  # simulator truth is not an input to a current observation
        after = env.observation(0)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)

    def test_wait_uses_time_without_movement_reward(self):
        env = Environment(Config(horizon_seconds=4., communication_range=1.))
        positions = env.positions.copy()
        rewards, duration, terminated, truncated = env.step([0, 0])
        self.assertEqual(duration, 1.)
        self.assertEqual(env.time, 1.)
        np.testing.assert_array_equal(env.positions, positions)
        self.assertEqual(env.radio.sent_bytes, 0)
        self.assertTrue(np.all(rewards < 0))
        self.assertFalse(terminated or truncated)

    def test_wait_mask_and_real_sac_update(self):
        cfg = Config(batch_size=2, warmup=2, horizon_seconds=4.)
        env, learner = Environment(cfg), Learner(cfg)
        obs = [env.observation(i) for i in range(2)]
        inputs = tensors(obs, learner.device)
        mask_before = inputs[4].clone()
        logp = learner.policy(*inputs)
        self.assertTrue(torch.all(logp[:, 0].exp() > 0))
        learner.critics[0](*inputs)
        self.assertTrue(torch.equal(mask_before, inputs[4]), "Network mutated the replay mask")
        reward, duration, terminated, truncated = env.step([0, 0])
        for robot in range(2):
            learner.replay.append((obs[robot], 0, reward[robot], duration,
                                   env.observation(robot), float(terminated), truncated))
        before = learner.policy.pointer.w_query.detach().clone()
        loss = learner.update()
        self.assertEqual(loss["updates"], 1)
        self.assertTrue(all(np.isfinite(v) for v in loss.values()))
        self.assertFalse(torch.equal(before, learner.policy.pointer.w_query))

    def test_supervised_head_and_coupled_observation(self):
        cfg = Config(variant="AB", content="learned", batch_size=2)
        env, learner = Environment(cfg), Learner(cfg)
        context = learner.contexts(env)
        env.scorer = learner.score
        env.feature_provider = lambda sender, indices: context[sender][indices]
        samples = env.collect_labels(0)
        self.assertGreater(len(samples), 1)
        learner.content_replay.extend(samples)
        self.assertTrue(np.isfinite(learner.update_content()["content_loss"]))
        before = env.observation(0)
        env.beliefs[1][:] = 127
        after = env.observation(0)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
        rewards, duration, _, _ = env.step([0, 0])
        self.assertTrue(np.all(np.isfinite(rewards)))
        self.assertGreater(duration, 0)

    def test_training_map_evaluation_preserves_rng_and_does_not_train(self):
        from finite_contact_train import evaluate
        cfg = Config(horizon_seconds=2.)
        learner = Learner(cfg)
        py_before = random.getstate()
        np_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        result = evaluate(learner, cfg, 1, split="train", policy="random", repeats=2)
        self.assertEqual(len(result), 2)
        self.assertEqual([r["repeat"] for r in result], [0, 1])
        self.assertTrue(all(r["map"]["split"] == "train" for r in result))
        self.assertTrue(all(r["evaluation_policy"] == "random" for r in result))
        self.assertEqual(learner.updates, 0)
        self.assertEqual(len(learner.replay), 0)
        self.assertEqual(random.getstate(), py_before)
        self.assertEqual(np.random.get_state()[0], np_before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_before[1])
        self.assertEqual(np.random.get_state()[2:], np_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_exact_update_budget_and_requested_stop(self):
        from finite_contact_train import rollout
        cfg = Config(horizon_seconds=64., batch_size=2, warmup=2, pointer_mode="centered")
        learner = Learner(cfg)
        result, env = rollout(learner, cfg, "train", 0, "movement", max_updates=3)
        self.assertEqual(learner.updates, 3)
        self.assertEqual(result["interruption_reason"], "update_budget")
        self.assertTrue(result["interrupted"])
        self.assertIsNone(result["coverage_deficit"])
        np.testing.assert_allclose(sum(env.reward_components.values()), result["return"])
        stopped, _ = rollout(learner, cfg, "train", 0, "movement", stop_requested=lambda: True)
        self.assertEqual(stopped["decisions"], 0)
        self.assertEqual(stopped["interruption_reason"], "requested_stop")
        self.assertEqual(learner.updates, 3)

    def test_wait_diagnostics_and_time_cost(self):
        cfg = Config(horizon_seconds=4., communication_range=1., time_penalty=.07)
        env = Environment(cfg)
        reward, duration, _, _ = env.step([0, 0])
        np.testing.assert_allclose(reward, [-.07, -.07])
        np.testing.assert_allclose(env.wait_disconnected_seconds, [duration, duration])
        np.testing.assert_allclose(env.wait_without_new_map_seconds, [duration, duration])
        np.testing.assert_allclose(sum(env.reward_components.values()), reward)


if __name__ == "__main__":
    unittest.main()
