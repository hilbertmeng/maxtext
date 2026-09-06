"""Compare same-step raw-gradient norm budgets from existing local TB events."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import re
import struct

from tensorboard.compat.proto import event_pb2


def read_scalars(directory, steps):
    helper_path = Path(__file__).resolve().parents[2] / (
        ".claude/skills/tpu-training/scripts/report_bam_read_health.py")
    spec = importlib.util.spec_from_file_location("read_health", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = defaultdict(dict)
    paths = sorted(Path(directory).glob("events.out.tfevents.*"))
    if not paths:
        raise FileNotFoundError(directory)
    for path in paths:
        with path.open("rb") as handle:
            while header := handle.read(12):
                if len(header) != 12:
                    break
                length = struct.unpack("<Q", header[:8])[0]
                data, footer = handle.read(length), handle.read(4)
                if len(data) != length or len(footer) != 4:
                    break
                for payload, recorded in ((header[:8], header[8:]), (data, footer)):
                    crc = helper._masked_crc32c(payload)
                    if crc is not None and crc != struct.unpack("<I", recorded)[0]:
                        raise ValueError(f"CRC mismatch: {path}")
                event = event_pb2.Event.FromString(data)
                if event.step > max(steps):
                    break
                if event.step not in steps:
                    continue
                for item in event.summary.value:
                    if item.tag.startswith(("raw_grads/", "total_params/")) or item.tag == "learning/raw_grad_norm":
                        value = helper._scalar_value(item)
                        if value is not None:
                            result[event.step][item.tag] = value
    return result


def compare(run, base, steps, by_layer=False):
    output = []
    for step in sorted(steps):
        a, b = run[step], base[step]
        if not a or a.keys() != b.keys() or "learning/raw_grad_norm" not in a:
            raise ValueError(f"Missing or mismatched gradient tags at step {step}")
        groups = defaultdict(lambda: [0.0, 0.0])
        for tag in a:
            if not tag.startswith("raw_grads/"):
                continue
            group = tag.removeprefix("raw_grads/")
            if not by_layer:
                group = re.sub(r"layers_\d+", "layers_*", group)
            groups[group][0] += a[tag] ** 2
            groups[group][1] += b[tag] ** 2
        totals = [sum(v[i] for v in groups.values()) for i in (0, 1)]
        budget = [{"parameter": name, "run_squared_norm": v[0],
                   "base_squared_norm": v[1], "base_minus_run": v[1] - v[0]}
                  for name, v in groups.items()]
        output.append({"step": step, "run_raw_grad": a["learning/raw_grad_norm"],
                       "base_raw_grad": b["learning/raw_grad_norm"],
                       "leaf_squared_norm_totals_run_base": totals,
                       "parameter_groups": sorted(budget, key=lambda v: abs(v["base_minus_run"]), reverse=True)})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_events")
    parser.add_argument("base_events")
    parser.add_argument("--steps", default="100,200,400,600")
    parser.add_argument("--by-layer", action="store_true")
    parser.add_argument("--parameter-norm", help="Compare a recorded parameter leaf norm instead, e.g. gw_b0")
    args = parser.parse_args()
    steps = {int(s) for s in args.steps.split(",")}
    run, base = read_scalars(args.run_events, steps), read_scalars(args.base_events, steps)
    if args.parameter_norm:
        result = []
        for step in sorted(steps):
            keys = sorted(k for k in run[step] if k.startswith("total_params/")
                          and k.endswith("/" + args.parameter_norm))
            if not keys or any(k not in base[step] for k in keys):
                raise ValueError(f"Missing parameter norms at step {step}")
            result.append({"step": step, "parameter_norms": [
                {"parameter": k, "run": run[step][k], "base": base[step][k]} for k in keys]})
    else:
        result = compare(run, base, steps, args.by_layer)
    print(json.dumps({"run_events": args.run_events, "base_events": args.base_events,
                      "results": result}, indent=2))
