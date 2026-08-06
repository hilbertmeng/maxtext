#!/usr/bin/env python3
"""Export a completed TensorBoard loss series as compact STEP LOSS rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="learning/loss")
    parser.add_argument("--sample-period", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_period <= 0:
        raise ValueError("--sample-period must be positive")
    event_files = sorted(args.tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event files in {args.tensorboard_dir}")

    latest: dict[int, tuple[float, float]] = {}
    for event_file in event_files:
        for raw in RawEventFileLoader(str(event_file)).Load():
            event = Event.FromString(raw)
            step = int(event.step)
            if step % args.sample_period:
                continue
            for value in event.summary.value:
                if value.tag != args.tag or not value.HasField("simple_value"):
                    continue
                old = latest.get(step)
                if old is None or event.wall_time >= old[1]:
                    latest[step] = (float(value.simple_value), float(event.wall_time))

    if not latest:
        raise ValueError(f"no {args.tag!r} values found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{step} {latest[step][0]:.9g}\n" for step in sorted(latest))
    )
    print(f"{args.output} ({len(latest)} points, {min(latest)}..{max(latest)})")


if __name__ == "__main__":
    main()
