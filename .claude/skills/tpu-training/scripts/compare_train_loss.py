#!/usr/bin/env python3
"""Compare TensorBoard train-loss scalars at matching global steps."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass(frozen=True)
class Scalar:
    step: int
    value: float
    wall_time: float


def load_scalars(logdir: Path, tag: str) -> dict[int, Scalar]:
    if not logdir.is_dir():
        raise FileNotFoundError(f"TensorBoard log directory does not exist: {logdir}")

    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    if tag not in scalar_tags:
        raise KeyError(f"scalar tag {tag!r} not found in {logdir}")

    latest: dict[int, Scalar] = {}
    for event in accumulator.Scalars(tag):
        scalar = Scalar(int(event.step), float(event.value), float(event.wall_time))
        old = latest.get(scalar.step)
        if old is None or scalar.wall_time >= old.wall_time:
            latest[scalar.step] = scalar
    return latest


def common_window_means(
    experiment: dict[int, Scalar],
    baseline: dict[int, Scalar],
    center: int,
    radius: int,
) -> tuple[float, float, int]:
    # A restart can omit one upload step. Compare only identical steps so neither
    # window gets an unmatched batch.
    common_steps = sorted(
        step
        for step in experiment.keys() & baseline.keys()
        if center - radius <= step <= center + radius
    )
    if not common_steps:
        return math.nan, math.nan, 0
    experiment_mean = sum(experiment[step].value for step in common_steps) / len(
        common_steps
    )
    baseline_mean = sum(baseline[step].value for step in common_steps) / len(common_steps)
    return experiment_mean, baseline_mean, len(common_steps)


def relative_percent(gap: float, baseline: float) -> float:
    return 100.0 * gap / baseline if baseline else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare learning/loss for two TensorBoard runs at identical steps."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", default="experiment")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--tag", default="learning/loss")
    parser.add_argument("--interval", type=int, default=200)
    parser.add_argument("--step", type=int, action="append")
    parser.add_argument("--after-step", type=int, default=0)
    parser.add_argument("--window-radius", type=int, default=25)
    parser.add_argument("--latest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.window_radius < 0:
        raise ValueError("--window-radius must be non-negative")

    experiment = load_scalars(args.experiment_dir, args.tag)
    baseline = load_scalars(args.baseline_dir, args.tag)
    common_max = min(max(experiment), max(baseline))

    if args.step:
        steps = sorted(set(args.step))
    else:
        first = ((args.after_step // args.interval) + 1) * args.interval
        steps = list(range(first, common_max + 1, args.interval))
    if args.latest_only and steps:
        steps = steps[-1:]

    missing = [step for step in steps if step not in experiment or step not in baseline]
    steps = [step for step in steps if step in experiment and step in baseline]

    print(
        f"tag={args.tag} experiment_range={min(experiment)}..{max(experiment)} "
        f"baseline_range={min(baseline)}..{max(baseline)}"
    )
    for step in steps:
        experiment_value = experiment[step].value
        baseline_value = baseline[step].value
        gap = experiment_value - baseline_value
        experiment_mean, baseline_mean, common_n = common_window_means(
            experiment, baseline, step, args.window_radius
        )
        mean_gap = experiment_mean - baseline_mean
        print(
            f"LOSS_GAP step={step} "
            f"{args.experiment_name}={experiment_value:.6f} "
            f"{args.baseline_name}={baseline_value:.6f} "
            f"gap={gap:+.6f} rel={relative_percent(gap, baseline_value):+.2f}% "
            f"window=+/-{args.window_radius} "
            f"{args.experiment_name}_mean={experiment_mean:.6f} "
            f"{args.baseline_name}_mean={baseline_mean:.6f} "
            f"mean_gap={mean_gap:+.6f} "
            f"mean_rel={relative_percent(mean_gap, baseline_mean):+.2f}% "
            f"common_samples={common_n}"
        )
    if missing:
        print("MISSING_STEPS " + ",".join(str(step) for step in missing))


if __name__ == "__main__":
    main()
