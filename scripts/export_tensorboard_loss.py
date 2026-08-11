#!/usr/bin/env python3
"""Export the latest TensorBoard scalar value at each step as ``STEP VALUE``."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("logdir", type=Path)
  parser.add_argument("--output", "-o", type=Path, required=True)
  parser.add_argument("--tag", default="learning/loss")
  args = parser.parse_args()

  latest: dict[int, tuple[float, float]] = {}
  event_files = sorted(args.logdir.rglob("events.out.tfevents.*"))
  if not event_files:
    parser.error(f"no TensorBoard event files under {args.logdir}")

  for event_file in event_files:
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    if args.tag not in accumulator.Tags().get("scalars", []):
      continue
    for event in accumulator.Scalars(args.tag):
      previous = latest.get(event.step)
      if previous is None or event.wall_time >= previous[0]:
        latest[event.step] = (event.wall_time, event.value)

  if not latest:
    parser.error(f"scalar tag {args.tag!r} not found under {args.logdir}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text("".join(f"{step} {latest[step][1]:.9g}\n" for step in sorted(latest)))
  print(f"exported {len(latest)} points ({min(latest)}..{max(latest)}) to {args.output}")


if __name__ == "__main__":
  main()
