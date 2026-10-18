#!/usr/bin/env python3
"""Summarize recorded single-map experiments without choosing favorable episodes."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ir2-finite-contact-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def trajectory_figure(directory, output):
    trajectories = rows(directory / "evaluation_trajectories.jsonl")
    path = directory / "evaluation_map_0_repeat_0.npz"
    if not trajectories or not path.exists():
        return
    maps = np.load(path)
    trajectory = np.array([x["positions"] for x in trajectories[0]["trajectory"]])
    result = rows(directory / "evaluation.jsonl")[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(maps["truth"], cmap="gray", vmin=1, vmax=255)
    for robot, color in enumerate(["tab:blue", "tab:orange"]):
        axes[0].plot(trajectory[:, robot, 0], trajectory[:, robot, 1], color=color, lw=1.1, label=f"Robot {robot+1}")
        axes[0].scatter(*trajectory[0, robot], c=color, marker="o", s=20)
        axes[0].scatter(*trajectory[-1, robot], c=color, marker="x", s=35)
        axes[robot + 1].imshow(maps["beliefs"][robot], cmap="gray", vmin=1, vmax=255)
        axes[robot + 1].set_title(f"Robot {robot+1} local map: {result['coverage'][robot]:.2%}")
    policy = result.get("evaluation_policy", "greedy").title()
    axes[0].set_title(f"{policy} trajectory (repeat 0), {result['time']:.1f} s")
    axes[0].legend(loc="lower left", fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 255)
        ax.set_ylim(255, 0)
        ax.set_aspect("equal")
    fig.suptitle(directory.name, fontsize=10)
    fig.tight_layout()
    fig.savefig(output / (directory.name + "_trajectory.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--controls", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x-axis", choices=["episodes", "updates"], default="episodes")
    parser.add_argument("--labels", nargs="+", help="Optional distinct labels in the same order as --runs")
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.runs):
        parser.error("Provide one label per run")
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    diagnostic_fig, diagnostic_axes = plt.subplots(2, 2, figsize=(12, 8))
    has_diagnostics = False
    summary = {}
    def xvalue(record):
        return record.get("learner_updates", record.get("updates", 0)) if args.x_axis == "updates" else record["episode"]
    for run_index, run in enumerate(args.runs):
        config = json.loads((run / "config.json").read_text())
        variant = config["variant"]
        label = args.labels[run_index] if args.labels else variant
        color = (plt.get_cmap("tab10")(run_index % 10) if args.labels else
                 {"B0": "tab:blue", "A": "tab:orange", "B": "tab:green", "AB": "tab:red"}[variant])
        train = [r for r in rows(run / "training.jsonl") if not r["interrupted"]]
        policy = rows(run / "train_policy.jsonl")
        validation = rows(run / "validation.jsonl")
        update_records = rows(run / "updates.jsonl")
        episodes = [xvalue(r) for r in train]
        axes[0, 0].plot(episodes, [r["union_coverage"] for r in train], label=label, color=color, alpha=.65)
        axes[0, 1].plot(episodes, [min(r["coverage"]) for r in train], label=label, color=color, alpha=.65)
        axes[1, 0].plot([xvalue(r) for r in policy], [min(r["coverage"]) for r in policy], "o-", label=label + " train", color=color)
        axes[1, 1].plot([xvalue(r) for r in policy], [r["coverage_deficit"] for r in policy], "o-", label=label + " train", color=color)
        if validation:
            groups = {}
            for record in validation:
                groups.setdefault((xvalue(record), record.get("phase", "periodic")), []).append(record)
            axes[1, 0].plot([key[0] for key in groups], [np.mean([min(r["coverage"]) for r in group]) for group in groups.values()], "x--", label=label + " val mean", color=color)
            axes[1, 1].plot([key[0] for key in groups], [np.mean([r["coverage_deficit"] for r in group]) for group in groups.values()], "x--", label=label + " val mean", color=color)
        if update_records:
            has_diagnostics = True
            xs = [r["updates"] for r in update_records]
            diagnostic_axes[0, 0].plot(xs, [r["pointer_clip_input_abs_max"] for r in update_records], label=label + " clip input", color=color)
            diagnostic_axes[0, 0].plot(xs, [r["pointer_raw_abs_max"] for r in update_records], linestyle=":", label=label + " raw", color=color, alpha=.6)
            diagnostic_axes[0, 1].plot(xs, [r["pointer_saturation_fraction"] for r in update_records], label=label, color=color)
            diagnostic_axes[1, 0].plot(xs, [max(r["policy_grad_norm"], 1e-12) for r in update_records], label=label, color=color)
            diagnostic_axes[1, 1].plot(xs, [r["wait_probability"] for r in update_records], label=label, color=color)
        summary[str(run)] = {
            "variant": variant, "label": label, "config": config, "completed_episodes": len(train),
            "updates": update_records[-1]["updates"] if update_records else (train[-1].get("updates", 0) if train else 0),
            "last_completed_episode_updates": train[-1].get("updates", 0) if train else 0,
            "numerics": {"max_saturation_fraction": max(r["pointer_saturation_fraction"] for r in update_records),
                         "zero_gradient_updates": sum(r["policy_grad_norm"] == 0 for r in update_records),
                         "last_update": update_records[-1]} if update_records else None,
            "interruptions": rows(run / "interruptions.jsonl"),
            "training_successes": sum(r["success"] for r in train),
            "training_union_completed": sum(r["union_completion_time"] is not None for r in train),
            "training_tail10": {"mean_union_coverage": float(np.mean([r["union_coverage"] for r in train[-10:]])) if train else None,
                                "mean_min_local_coverage": float(np.mean([min(r["coverage"]) for r in train[-10:]])) if train else None},
            "train_policy_evaluations": policy, "validation_evaluations": validation,
            "stop_record": json.loads((run / "early_stop_reason.json").read_text()) if (run / "early_stop_reason.json").exists() else None,
        }
    titles = ["Training: union coverage (sampled actions)", "Training: minimum local coverage (sampled)",
              "Greedy: minimum local coverage (train / validation mean)", "Greedy: coverage deficit (lower better)"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("SAC updates" if args.x_axis == "updates" else "Episode")
        ax.grid(alpha=.25)
        ax.legend()
        ax.set_ylim(0, 1.02)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.axhline(.99, color="black", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(args.output / "learning_curves.png", dpi=160)
    fig.savefig(args.output / "learning_curves.pdf")
    plt.close(fig)
    if has_diagnostics:
        titles = ["Pointer magnitude: raw / actual clip input", "Fraction with |clip input| > 5",
                  "Actor gradient norm (before clipping)", "Mean WAIT probability in replay minibatch"]
        for ax, title in zip(diagnostic_axes.flat, titles):
            ax.set_title(title)
            ax.set_xlabel("SAC updates")
            ax.grid(alpha=.25)
            ax.legend()
        diagnostic_axes[1, 0].set_yscale("log")
        diagnostic_fig.tight_layout()
        diagnostic_fig.savefig(args.output / "numerical_diagnostics.png", dpi=160)
    plt.close(diagnostic_fig)
    controls = {}
    for directory in args.controls:
        result = rows(directory / "evaluation.jsonl")
        if not result and (directory / "metrics.json").exists():
            result = [json.loads((directory / "metrics.json").read_text())]
        if result:
            controls[str(directory)] = {
                "rollouts": len(result), "successes": sum(r["success"] for r in result),
                "mean_union_coverage": float(np.mean([r["union_coverage"] for r in result])),
                "mean_min_local_coverage": float(np.mean([min(r["coverage"]) for r in result])),
                "mean_coverage_deficit": float(np.mean([r["coverage_deficit"] for r in result])),
                "results": result,
            }
            trajectory_figure(directory, args.output)
    (args.output / "summary.json").write_text(json.dumps({"runs": summary, "controls": controls}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": len(summary), "controls": len(controls)}))


if __name__ == "__main__":
    main()
