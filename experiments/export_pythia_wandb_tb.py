#!/usr/bin/env python3
"""Export a Pythia training-log chain from public W&B (eleutherai/pythia) to one TensorBoard run.

Writes the same scalar tags as the existing /data0/xd/tensorboard_logs/Pythia_*_2022 exports:
loss, learning/loss, lr, learning/current_learning_rate, validation/loss, test/loss.
Steps are the original optimizer steps; later restarts win per metric on overlap; no smoothing.

Usage:
  python3 experiments/export_pythia_wandb_tb.py --name Pythia_12B_2023 \
      --runs ktjvz2on,13dd1daa,wt8xor8f,1pj5zple,2jny2vqa,2iqez2mg,2prrq2c0,kg5ni1dl \
      --logdir /data0/xd/tensorboard_logs --note "..."
"""
import argparse
import json
import os
import time
import urllib.request

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

API = "https://api.wandb.ai/graphql"
ENTITY, PROJECT = "eleutherai", "pythia"
TOTAL_STEPS = 143000
METRICS = {  # W&B key -> TensorBoard tags
    "train/lm_loss": ["loss", "learning/loss"],
    "train/learning_rate": ["lr", "learning/current_learning_rate"],
    "validation/lm_loss": ["validation/loss"],
    "test/lm_loss": ["test/loss"],
}


def gql(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            out = json.load(urllib.request.urlopen(req, timeout=300))
            if "errors" in out:
                raise RuntimeError(out["errors"])
            return out["data"]
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def as_dict(x):
    return json.loads(x) if isinstance(x, str) else (x or {})


def run_info(rid):
    run = gql("""query($p:String!,$e:String!,$r:String!){ project(name:$p, entityName:$e){
                 run(name:$r){ group createdAt config summaryMetrics } } }""",
              {"p": PROJECT, "e": ENTITY, "r": rid})["project"]["run"]
    run["config"] = as_dict(run["config"])
    run["summaryMetrics"] = as_dict(run["summaryMetrics"])
    return run


def scan(rid, chunk=5000):
    """All logged values of METRICS keys, {key: {step: (value, timestamp)}}."""
    out = {k: {} for k in METRICS}
    for lo in range(0, TOTAL_STEPS + 1, chunk):
        data = gql("""query($p:String!,$e:String!,$r:String!,$a:Int64,$b:Int64){ project(name:$p, entityName:$e){
                      run(name:$r){ history(minStep:$a, maxStep:$b, samples:%d) } } }""" % (chunk + 1),
                   {"p": PROJECT, "e": ENTITY, "r": rid, "a": lo, "b": lo + chunk})
        for line in data["project"]["run"]["history"]:
            row = json.loads(line)
            for k in METRICS:
                if row.get(k) is not None:
                    out[k][row["_step"]] = (row[k], row.get("_timestamp"))
    return out


def scalar_event(tag, step, value, wall_time):
    summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))])
    return event_pb2.Event(wall_time=float(wall_time), step=int(step), summary=summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated W&B run ids in step order")
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--note", required=True)
    args = ap.parse_args()
    out_dir = os.path.join(args.logdir, args.name)
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} exists; refusing to overwrite")

    merged = {k: {} for k in METRICS}
    sources = []
    for rid in args.runs.split(","):
        info = run_info(rid)
        data = scan(rid)
        steps = sorted(data["train/lm_loss"])
        for k in METRICS:
            merged[k].update(data[k])  # later restart wins
        sources.append(dict(url=f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{rid}", group=info["group"],
                            createdAt=info["createdAt"], config=info["config"],
                            summary=info["summaryMetrics"],
                            counts={k: len(v) for k, v in data.items()},
                            step_range=[steps[0], steps[-1]] if steps else None))
        print(rid, info["group"], sources[-1]["step_range"], sources[-1]["counts"], flush=True)

    os.makedirs(out_dir)
    writer = EventFileWriter(out_dir)
    events = []
    for k, tags in METRICS.items():
        for step, (value, ts) in merged[k].items():
            for tag in tags:
                events.append((step, tag, value, ts or 0.0))
    for step, tag, value, ts in sorted(events, key=lambda e: (e[0], e[1])):
        writer.add_event(scalar_event(tag, step, value, ts))
    writer.close()

    train = merged["train/lm_loss"]
    missing = [s for s in range(1, TOTAL_STEPS + 1) if s not in train]
    gaps, start = [], None
    for s in missing + [None]:
        if start is None:
            start = prev = s
        elif s is not None and s == prev + 1:
            prev = s
        else:
            gaps.append([start, prev])
            start = prev = s
    gaps = [g for g in gaps if g[0] is not None]
    meta = {
        "run": args.name,
        "metrics": {k: dict(count=len(v), first_step=min(v) if v else None, last_step=max(v) if v else None)
                    for k, v in merged.items()},
        "tags": METRICS,
        "event_file": sorted(os.listdir(out_dir))[0],
        "step_policy": "Completed optimizer updates: official logs retain original steps 1..143000. "
                       "No resampling or smoothing. Does not assert same-batch identity across public and local runs.",
        "source": dict(sources=sources,
                       merge_policy="Later restart wins at overlapping steps, per metric; no smoothing or interpolation.",
                       missing_train_steps=len(missing), missing_train_ranges=gaps,
                       export="W&B GraphQL history scanned in 5000-step windows at full resolution.",
                       version_note=args.note),
        "updated_utc": time.time(),
    }
    json.dump(meta, open(os.path.join(out_dir, "source.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "source"}, indent=1))
    print("missing_train_ranges", gaps)


if __name__ == "__main__":
    main()
