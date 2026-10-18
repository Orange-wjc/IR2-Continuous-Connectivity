#!/usr/bin/env python3
"""Bounded training/evaluation for the two-robot simple-map feasibility study."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ir2-finite-contact-matplotlib")
os.environ.setdefault("PIP_NO_CACHE_DIR", "1")
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from finite_contact.config import Config
from finite_contact.environment import Environment
from finite_contact.learning import Learner
from finite_contact.maps import manifest, training_map_index

ROOT = Path(__file__).resolve().parent


def append_json(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def successful_rank(results, expected_count):
    """A failed or incomplete evaluation can never replace a usable checkpoint."""
    if len(results) != expected_count or any(not x["success"] or x.get("interrupted", False) for x in results):
        return None
    return (float(np.mean([x["time"] for x in results])),
            float(np.mean([x["coverage_deficit"] for x in results])))


def successful_history(path, expected_count):
    if not path.exists():
        return None
    groups = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        groups.setdefault((row["episode"], row.get("phase", "periodic"), row["learner_updates"]), []).append(row)
    ranks = [successful_rank(group, expected_count) for group in groups.values()]
    return min((rank for rank in ranks if rank is not None), default=None)


def bind_context(learner, env):
    if env.config.content != "learned":
        return
    contexts = learner.contexts(env)
    env.feature_provider = lambda sender, indices: contexts[sender][indices]


def rollout(learner, config, split, index, stage, deadline=None, smoke=False,
            evaluation_policy="greedy", max_updates=None, stop_requested=None,
            max_content_updates=None, episode_update_limit=None):
    if episode_update_limit is not None and (episode_update_limit < 0 or stage not in {"movement", "finetune"}):
        raise ValueError("Per-episode update limits apply to movement/finetune and must be nonnegative")
    env = Environment(config, split, index, scorer=learner.score)
    losses = {}
    decisions = 0
    initial_updates = learner.updates
    returns = np.zeros(2)
    episode_transitions = []
    terminated = truncated = interrupted = False
    interruption_reason = None
    while not (terminated or truncated):
        if stop_requested is not None and stop_requested():
            interruption_reason = "requested_stop"
        elif deadline is not None and time.monotonic() >= deadline:
            interruption_reason = "wall_budget"
        elif max_updates is not None and learner.updates >= max_updates:
            interruption_reason = "update_budget"
        elif max_content_updates is not None and learner.content_updates >= max_content_updates:
            interruption_reason = "content_update_budget"
        if interruption_reason is not None:
            interrupted = True
            break
        if stage == "content":
            contexts = learner.contexts(env)
            env.feature_provider = lambda sender, indices: contexts[sender][indices]
        else:
            bind_context(learner, env)
        observations = [env.observation(i) for i in range(2)]
        if stage == "content":
            for robot in range(2):
                learner.content_replay.extend(env.collect_labels(robot))
        actions = learner.act(observations,
                              greedy=stage == "content" or (stage == "evaluate" and evaluation_policy == "greedy"),
                              random_action=(stage == "evaluate" and evaluation_policy == "random") or
                              (stage == "movement" and learner.updates == 0
                               and len(learner.replay) < config.warmup))
        reward, duration, terminated, truncated = env.step(actions)
        returns += reward
        decisions += 1
        if stage in {"movement", "finetune"}:
            bind_context(learner, env)
            next_obs = [env.observation(i) for i in range(2)]
            if config.compact_replay:
                from finite_contact.learning import compact_observation
                observations = [compact_observation(o) for o in observations]
                next_obs = [compact_observation(o) for o in next_obs]
            if getattr(learner, 'lossless_observations', False):
                from finite_contact.memory import pack_observation
                observations = [pack_observation(o) for o in observations]
                next_obs = [pack_observation(o) for o in next_obs]
            for robot in range(2):
                transition = (observations[robot], actions[robot], float(reward[robot]),
                              duration, next_obs[robot], float(terminated), bool(truncated))
                learner.replay.append(transition)
                if config.success_imitation_weight > 0:
                    episode_transitions.append(transition)
            for _ in range(config.updates_per_decision):
                if episode_update_limit is not None and learner.updates - initial_updates >= episode_update_limit:
                    break
                if max_updates is not None and learner.updates >= max_updates:
                    break
                if not smoke or learner.updates - initial_updates < 2:
                    losses.update(learner.update())
        elif stage == "content":
            for _ in range(config.updates_per_decision):
                if max_content_updates is not None and learner.content_updates >= max_content_updates:
                    break
                losses.update(learner.update_content())
    if terminated and not interrupted and stage in {"movement", "finetune"} and config.success_imitation_weight > 0:
        learner.remember_success(episode_transitions, env.time, env.map_info["layout_id"])
    metrics = env.metrics(terminated)
    metrics.update({"map": env.map_info, "decisions": decisions,
                    "truncated": truncated, "interrupted": interrupted,
                    "interruption_reason": interruption_reason, "learner_updates": learner.updates,
                    "return": returns.tolist(), **losses})
    if interrupted:
        # This episode has no valid fixed-horizon AUC; do not rank it as a normal evaluation.
        metrics["coverage_deficit"] = None
    return metrics, env


def evaluate(learner, config, count, split="validation", deadline=None, directory=None,
             policy="greedy", repeats=1, stop_requested=None):
    # Validation must not perturb future training's action/noise streams.
    py_state, np_state, th_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    results = []
    try:
        for trial in range(count * repeats):
            i, repeat = trial % count, trial // count
            if deadline is not None and time.monotonic() >= deadline:
                break
            metrics, env = rollout(learner, config, split, i, "evaluate", deadline,
                                   evaluation_policy=policy, stop_requested=stop_requested)
            if metrics["interrupted"]:
                break
            metrics.update({"evaluation_policy": policy, "repeat": repeat})
            results.append(metrics)
            if directory is not None:
                np.savez_compressed(directory / f"evaluation_map_{i}_repeat_{repeat}.npz",
                                    truth=env.truth, beliefs=np.stack(env.beliefs),
                                    positions=env.positions)
                append_json(directory / "evaluation_trajectories.jsonl",
                            {"map": env.map_info, "repeat": repeat, "trajectory": env.trajectory})
                for event in env.radio.events:
                    append_json(directory / "evaluation_communication.jsonl", {"map_index": i, "repeat": repeat, **event})
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(th_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return results


def checkpoint(path, learner, config, stage, episode, best, elapsed, full=True):
    state = {"format": "ir2-finite-contact-v1", "config": config.to_dict(), "stage": stage,
             "episode": episode, "best_validation": best, "training_wall_seconds": elapsed,
             "learner": learner.state()}
    if full:
        state.update({"replay": list(learner.replay), "content_replay": list(learner.content_replay),
                      "success_episodes": learner.success_episodes,
                      "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
    temporary = path.with_suffix(".tmp")
    if getattr(learner, 'lossless_observations', False):
        state['observation_storage'] = 'lossless-zlib-v1'
    torch.save(state, temporary, pickle_protocol=4 if getattr(learner, 'lossless_observations', False) else 2)
    temporary.replace(path)


def provenance(directory, config):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()
    versions = {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "numpy": np.__version__, "git_commit": git("rev-parse", "HEAD"),
                "git_branch": git("branch", "--show-current"),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (directory / "environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    (directory / "maps.json").write_text(json.dumps(manifest(config), indent=2), encoding="utf-8")
    (directory / "git_diff.patch").write_text(git("diff"), encoding="utf-8")
    (directory / "dependencies.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    snapshot = directory / "source"
    snapshot.mkdir(exist_ok=True)
    for name in ("finite_contact_train.py", "finite_contact_diagnose.py", "finite_contact_report.py",
                 "finite_contact_content_check.py", "finite_contact_verify.py", "finite_contact_single_map.py",
                 "finite_contact_fork.py",
                 "finite_contact_dataset.py", "finite_contact_multimap.py",
                 "finite_contact_multimap_report.py",
                 "finite_contact_multimap_content.py",
                 "finite_contact_multimap_joint.py", "finite_contact_multiseed.py",
                 "finite_contact_stability.py",
                 "model.py", "env.py", "sensor.py", "parameter.py",
                 "graph_generator.py", "graph.py", "node.py", "ss_realistic_model.py"):
        shutil.copy2(ROOT / name, snapshot / name)
    shutil.copytree(ROOT / "finite_contact", snapshot / "finite_contact", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["movement", "content", "finetune", "evaluate"], default="movement")
    parser.add_argument("--variant", choices=["B0", "A", "B", "AB"])
    parser.add_argument("--content", choices=["fifo", "rule", "learned"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-hours", type=float, default=2.)
    parser.add_argument("--max-updates", type=int, help="Absolute movement SAC update ceiling; same ceiling gives matched fresh-run budgets")
    parser.add_argument("--max-content-updates", type=int, help="Absolute supervised head update ceiling, content stage only")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-maps", type=int)
    parser.add_argument("--eval-train", action="store_true", help="Also evaluate the greedy policy on training maps before/throughout training")
    parser.add_argument("--eval-train-maps", type=int,
                        help="Fixed prefix of training maps for development monitoring; default all")
    parser.add_argument("--final-eval", action="store_true", help="Evaluate final weights before saving final.pth, within wall budget")
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--evaluation-policy", choices=["greedy", "sampled", "random"], default="greedy")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-maps", type=int)
    parser.add_argument("--validation-maps", type=int)
    parser.add_argument("--test-maps", type=int)
    parser.add_argument("--map-family", choices=["legacy", "simple_rooms_v1"])
    parser.add_argument("--map-seed", type=int)
    parser.add_argument("--map-order", choices=["cyclic", "shuffled"])
    parser.add_argument("--sensor-implementation", choices=["legacy", "vectorized"])
    parser.add_argument("--success-episode-limit", type=int)
    parser.add_argument("--success-memory-mode", choices=["fastest", "per_map"])
    parser.add_argument("--success-sampling", choices=["pooled", "map_tail"])
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--horizon", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--replay-size", type=int)
    parser.add_argument("--pointer-mode", choices=["legacy", "centered"])
    parser.add_argument("--time-penalty", type=float)
    parser.add_argument("--alpha-learning-rate", type=float,
                        help="Temperature optimizer only; actor/critic learning rates stay unchanged")
    parser.add_argument("--label-sampling", choices=["prefix", "spread"])
    parser.add_argument("--content-score-bound", choices=["none", "sender"])
    parser.add_argument("--delivery-queue-order", choices=["legacy", "service"])
    parser.add_argument("--horizon-target", choices=["terminal", "bootstrap"],
                        help="Finite task deadline ends returns; bootstrap reproduces historical targets")
    parser.add_argument("--frontier-shaping-scale", type=float,
                        help="Local graph potential shaping; zero disables, requires terminal deadlines")
    parser.add_argument("--success-imitation-weight", type=float,
                        help="Auxiliary actor learning from its own completed training episodes; zero disables")
    parser.add_argument("--coverage-reward", choices=["union", "local_mean"],
                        help="Team union or mean per-robot coverage progress; task success is unchanged")
    parser.add_argument("--policy-learning-rate", type=float,
                        help="Actor rate only; Q networks retain the common learning rate")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", type=Path, help="Continue same stage, restoring optimizers, replay and RNG")
    parser.add_argument("--initialize-from", type=Path, help="Start a new stage/run from finite-contact weights")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--smoke", action="store_true", help="One short episode and two SAC updates, not a convergence test")
    parser.add_argument("--uncoupled", action="store_true")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != Path("/home/wyz/anaconda3/envs/ir2_train").resolve():
        parser.error("Run with /home/wyz/anaconda3/envs/ir2_train/bin/python")
    if args.resume and args.initialize_from:
        parser.error("Choose resume or initialization, not both")
    if args.episodes < 1 or args.max_hours <= 0 or args.eval_every < 1:
        parser.error("Episode count, wall-clock budget and evaluation interval must be positive")
    if args.eval_repeats < 1:
        parser.error("Evaluation repeats must be positive")
    if args.max_updates is not None and (args.max_updates < 1 or args.stage not in {"movement", "finetune"}):
        parser.error("A positive max-updates ceiling applies to movement/finetune only")
    if args.max_content_updates is not None and (args.max_content_updates < 1 or args.stage != "content"):
        parser.error("A positive max-content-updates ceiling applies to content only")
    source = args.resume or args.initialize_from
    if args.stage == "evaluate" and source is None:
        parser.error("Evaluation needs --initialize-from a checkpoint")
    if args.resume and args.run_dir and args.run_dir.resolve() != args.resume.resolve().parent:
        parser.error("Resume in the original run directory; initialize a new run to change directories")
    saved = torch.load(source, map_location="cpu") if source else None
    if saved and saved.get("format") != "ir2-finite-contact-v1":
        parser.error("Use a finite-contact checkpoint; legacy IR2 weights have different input dimensions")
    if args.resume and "replay" not in saved:
        parser.error("Resume needs latest.pth with replay/RNG; best.pth is for initialization/evaluation")
    cfg = Config.from_checkpoint(saved["config"]) if saved else Config()
    for argument, field in (("variant", "variant"), ("content", "content"), ("seed", "seed"),
                            ("train_maps", "train_maps"), ("validation_maps", "validation_maps"),
                            ("test_maps", "test_maps"), ("map_family", "map_family"),
                            ("map_seed", "map_seed"), ("map_order", "map_order"),
                            ("sensor_implementation", "sensor_implementation"),
                            ("success_episode_limit", "success_episode_limit"),
                            ("success_memory_mode", "success_memory_mode"),
                            ("success_sampling", "success_sampling"),
                            ("bandwidth", "bytes_per_second"),
                            ("horizon", "horizon_seconds"), ("batch_size", "batch_size"), ("warmup", "warmup"),
                            ("replay_size", "replay_size"),
                            ("pointer_mode", "pointer_mode"), ("time_penalty", "time_penalty"),
                            ("alpha_learning_rate", "alpha_learning_rate"),
                            ("label_sampling", "label_sampling"), ("content_score_bound", "content_score_bound"),
                            ("delivery_queue_order", "delivery_queue_order"), ("horizon_target", "horizon_target"),
                            ("frontier_shaping_scale", "frontier_shaping_scale"),
                            ("coverage_reward", "coverage_reward"),
                            ("policy_learning_rate", "policy_learning_rate"),
                            ("success_imitation_weight", "success_imitation_weight")):
        value = getattr(args, argument)
        if value is not None:
            setattr(cfg, field, value)
    if args.uncoupled:
        cfg.coupled = False
    if args.stage == "content":
        if saved is None:
            parser.error("Content training needs --initialize-from a movement checkpoint")
        if args.content is None and not args.resume:
            cfg.content = "rule"  # fixed exploration + rule transmissions generate supervised samples
    elif args.stage == "finetune" and args.content is None and not args.resume:
        cfg.content = "learned"
    if args.smoke:
        cfg.horizon_seconds = 8.
        cfg.batch_size = cfg.warmup = 2
        cfg.train_maps = 1
        args.episodes = 1
    cfg.validate()
    train_eval_count = args.eval_train_maps or cfg.train_maps
    if args.eval_train_maps is not None and (not args.eval_train or not 1 <= args.eval_train_maps <= cfg.train_maps):
        parser.error("--eval-train-maps requires --eval-train and a count within the training split")
    if args.resume and (Config.from_checkpoint(saved["config"]).to_dict() != cfg.to_dict() or saved["stage"] != args.stage):
        parser.error("Resume requires identical config and stage; use --initialize-from for a new experiment")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently substitute CPU for an explicitly requested GPU run")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    learner = Learner(cfg, device)
    if saved:
        learner.load(saved["learner"], optimizers=bool(args.resume))
    if args.stage == "content" and not args.resume:
        learner.context_encoder = copy.deepcopy(learner.policy).requires_grad_(False).eval()
    if cfg.content == "learned" and learner.content_updates == 0:
        parser.error("Learned selection requires a checkpoint with trained content scores")
    if args.stage != "content" and cfg.variant in {"B", "AB"} and cfg.content != "learned":
        parser.error("Variants B/AB require learned content; train movement with B0/A first")
    start_episode, best, previous_wall = 0, None, 0.
    if args.resume:
        start_episode = saved["episode"]
        best, previous_wall = saved["best_validation"], saved["training_wall_seconds"]
        learner.replay.extend(saved["replay"])
        learner.content_replay.extend(saved["content_replay"])
        learner.success_episodes = saved.get("success_episodes", [])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if args.run_dir:
        directory = args.run_dir.resolve()
    elif args.resume:
        directory = args.resume.resolve().parent
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        directory = ROOT / "experiments" / f"finite_{cfg.variant}_{args.stage}_seed{cfg.seed}_{stamp}"
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        parser.error("Run directory is nonempty; use a new directory to preserve existing results")
    directory.mkdir(parents=True, exist_ok=True)
    stopping = [False]

    def request_stop(signum, frame):
        stopping[0] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # Creating this file also requests a stop at the next complete decision/update.
    should_stop = lambda: stopping[0] or (directory / "STOP").exists()
    if not args.resume:
        provenance(directory, cfg)
        source_snapshot = directory / "source"
    else:
        resume_record = directory / ("resume_" + str(time.time_ns()))
        resume_record.mkdir()
        provenance(resume_record, cfg)
        source_snapshot = resume_record / "source"
    invocation = {"stage": args.stage, "arguments": vars(args).copy(), "device": device,
                  "source_checkpoint": str(source.resolve()) if source else None,
                  "source_snapshot": str(source_snapshot),
                  "sampling": "sequential two-robot episodes; no Ray workers"}
    invocation["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in invocation["arguments"].items()}
    append_json(directory / "invocations.jsonl", invocation)
    print(json.dumps({"run_dir": str(directory), "device": device, "config": cfg.to_dict()}, ensure_ascii=False), flush=True)
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    if args.stage == "evaluate":
        if saved is None:
            parser.error("Evaluation needs --initialize-from a checkpoint")
        limit = {"train": cfg.train_maps, "validation": cfg.validation_maps, "test": cfg.test_maps}[args.split]
        count = args.eval_maps or limit
        if not 1 <= count <= limit:
            parser.error("Invalid evaluation map count")
        results = evaluate(learner, cfg, count, args.split, deadline, directory,
                           args.evaluation_policy, args.eval_repeats, should_stop)
        for record in results:
            append_json(directory / "evaluation.jsonl", record)
        print(json.dumps({"evaluated_rollouts": len(results), "requested_rollouts": count * args.eval_repeats,
                          "complete": len(results) == count * args.eval_repeats}), flush=True)
        return
    writer = SummaryWriter(str(directory / "tensorboard"))
    learner.on_update = lambda record: append_json(directory / "updates.jsonl", record)
    learner.on_content_update = lambda record: append_json(directory / "content_updates.jsonl", record)
    completed = start_episode
    best_train = None
    training_evaluation_path = directory / "train_policy.jsonl"
    best_train_success = successful_history(training_evaluation_path, train_eval_count) if args.resume else None
    best_validation_success = successful_history(directory / "validation.jsonl", args.eval_maps or cfg.validation_maps) if args.resume else None
    if args.resume and training_evaluation_path.exists():
        records = [json.loads(line) for line in training_evaluation_path.read_text().splitlines()]
        scores = [r["coverage_deficit"] for r in records if r["coverage_deficit"] is not None]
        best_train = min(scores) if scores else None

    def evaluate_training(episode, phase="periodic"):
        nonlocal best_train, best_train_success
        results = evaluate(learner, cfg, train_eval_count, "train", deadline, stop_requested=should_stop)
        for record in results:
            append_json(training_evaluation_path, {"episode": episode, "phase": phase, **record})
            print(json.dumps({"event": "train_policy", "episode": episode, **record}), flush=True)
        if len(results) == train_eval_count:
            rank = successful_rank(results, train_eval_count)
            if rank is not None and (best_train_success is None or rank < best_train_success):
                best_train_success = rank
                checkpoint(directory / "best_train_success.pth", learner, cfg, args.stage, episode, best,
                           previous_wall + time.monotonic() - started, full=False)
            score = float(np.mean([x["coverage_deficit"] for x in results]))
            writer.add_scalar("train_policy/coverage_deficit", score, episode)
            if best_train is None or score < best_train:
                best_train = score
                checkpoint(directory / "best_train.pth", learner, cfg, args.stage, episode, best,
                           previous_wall + time.monotonic() - started, full=False)

    def evaluate_validation(episode, phase="periodic"):
        nonlocal best, best_validation_success
        count = args.eval_maps or cfg.validation_maps
        if not 1 <= count <= cfg.validation_maps:
            raise ValueError("Invalid validation map count")
        results = evaluate(learner, cfg, count, deadline=deadline, stop_requested=should_stop)
        for record in results:
            append_json(directory / "validation.jsonl", {"episode": episode, "phase": phase, **record})
        if len(results) == count:
            rank = successful_rank(results, count)
            if rank is not None and (best_validation_success is None or rank < best_validation_success):
                best_validation_success = rank
                checkpoint(directory / "best_validation_success.pth", learner, cfg, args.stage, episode, best,
                           previous_wall + time.monotonic() - started, full=False)
            score = float(np.mean([x["coverage_deficit"] for x in results]))
            writer.add_scalar("validation/coverage_deficit", score, episode)
            if best is None or score < best:
                best = score
                checkpoint(directory / "best.pth", learner, cfg, args.stage, episode, best,
                           previous_wall + time.monotonic() - started, full=False)

    try:
        if args.eval_train and not args.resume:
            checkpoint(directory / "initial.pth", learner, cfg, args.stage, 0, best, 0., full=False)
            evaluate_training(0, "initial")
        for episode in range(start_episode + 1, args.episodes + 1):
            if (should_stop() or time.monotonic() >= deadline
                    or (args.max_updates is not None and learner.updates >= args.max_updates)
                    or (args.max_content_updates is not None and learner.content_updates >= args.max_content_updates)):
                break
            metrics, env = rollout(learner, cfg, "train", training_map_index(cfg, episode),
                                   args.stage, deadline, args.smoke, max_updates=args.max_updates,
                                   stop_requested=should_stop, max_content_updates=args.max_content_updates)
            metrics.update({"episode": episode, "wall_seconds": previous_wall + time.monotonic() - started})
            append_json(directory / "training.jsonl", metrics)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, episode)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            if metrics["interrupted"]:
                append_json(directory / "interruptions.jsonl", {"episode": episode,
                            "learner_updates": learner.updates, "reason": metrics["interruption_reason"],
                            "content_updates": learner.content_updates,
                            "simulation_time": env.time})
                break
            completed = episode
            append_json(directory / "trajectories.jsonl", {"episode": episode, "trajectory": env.trajectory})
            for record in env.radio.events:
                append_json(directory / "communication.jsonl", {"episode": episode, **record})
            if episode % args.eval_every == 0:
                if args.eval_train:
                    evaluate_training(episode)
                evaluate_validation(episode)
                checkpoint(directory / "latest.pth", learner, cfg, args.stage, completed, best,
                           previous_wall + time.monotonic() - started)
        if args.final_eval and not should_stop() and time.monotonic() < deadline:
            if args.eval_train:
                evaluate_training(completed, "final")
            evaluate_validation(completed, "final")
        checkpoint(directory / "final.pth", learner, cfg, args.stage, completed, best,
                   previous_wall + time.monotonic() - started, full=False)
    finally:
        checkpoint(directory / "latest.pth", learner, cfg, args.stage, completed, best,
                   previous_wall + time.monotonic() - started)
        writer.close()
    print(json.dumps({"completed_episodes": completed, "updates": learner.updates,
                      "content_updates": learner.content_updates,
                      "smoke_only": args.smoke, "checkpoint": str(directory / "latest.pth")}), flush=True)


if __name__ == "__main__":
    main()
