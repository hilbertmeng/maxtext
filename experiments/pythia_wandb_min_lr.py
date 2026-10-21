#!/usr/bin/env python3
"""Compare Pythia v0 (cosine to 0) and v1 (cosine to 0.1x) training loss from public W&B logs.

Reads the anonymous-readable project eleutherai/pythia via the W&B GraphQL API (no login),
verifies each run's architecture/schedule config, downloads complete per-step `train/lm_loss`
for the selected checkpoint chains, and writes per-bin paired differences.

Usage:
  python3 -I experiments/pythia_wandb_min_lr.py \
      --cache /data0/xd/bam_diagnostics/pythia-wandb-min-lr \
      --out experiments/pythia_wandb_min_lr_20261020.json
"""
import argparse
import json
import math
import os
import statistics
import time
import urllib.request

API = "https://api.wandb.ai/graphql"
ENTITY, PROJECT = "eleutherai", "pythia"
TOTAL_STEPS = 143000

# Run chains whose logging rank records train/lm_loss, in step order.  Later segments override
# earlier ones on overlapping steps (resumes restart from the last checkpoint).
CHAINS = {
    # 12B, v0: 36 layers x 5120, min_lr 0.  The separate "13B Decay" chain under
    # .../pythia/13B-decay is 32 x 4096 (6.9B shape) and is excluded.
    "v0_12b": ["3n4tq9zd", "22k7ocbw"],
    # 12B, v1: min_lr 1.2e-5; latest branch after the rerun from ~73k.
    "v1_12b": ["ktjvz2on", "13dd1daa", "wt8xor8f", "1pj5zple", "2jny2vqa", "2iqez2mg",
               "2prrq2c0", "kg5ni1dl"],
    # 2.8B control: both versions use min_lr 1.6e-5.  v1 lacks public logs for 99355-136000.
    "v0_2.8b": ["1c9jr6vp", "12vuw5ef"],
    "v1_2.8b": ["2l7ymzlr", "4x2lfblu", "2j0vfxrj"],
}
EXPECTED = {
    "v0_12b": dict(num_layers=36, hidden_size=5120, min_lr=0.0),
    "v1_12b": dict(num_layers=36, hidden_size=5120, min_lr=1.2e-5),
    "v0_2.8b": dict(num_layers=32, hidden_size=2560, min_lr=1.6e-5),
    "v1_2.8b": dict(num_layers=32, hidden_size=2560, min_lr=1.6e-5),
}
PAIRS = [("v0_12b", "v1_12b"), ("v0_2.8b", "v1_2.8b")]


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


def run_config(rid):
    data = gql("""query($p:String!,$e:String!,$r:String!){ project(name:$p, entityName:$e){
                  run(name:$r){ group config summaryMetrics } } }""",
               {"p": PROJECT, "e": ENTITY, "r": rid})
    run = data["project"]["run"]
    cfg = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in as_dict(run["config"]).items()}
    keep = ["num_layers", "hidden_size", "min_lr", "weight_decay", "train_iters", "lr_decay_iters",
            "lr_decay_style", "seq_length", "train_data_paths", "load"]
    cfg = {k: cfg.get(k) for k in keep}
    cfg["peak_lr"] = (as_dict(run["config"]).get("optimizer", {}).get("value") or {}).get("params", {}).get("lr")
    return run["group"], cfg


def full_history(rid, cache_dir, chunk=2000):
    path = os.path.join(cache_dir, f"{rid}.json")
    if os.path.exists(path):
        return json.load(open(path))
    rows, step = {}, 0
    while step <= TOTAL_STEPS:
        data = gql("""query($p:String!,$e:String!,$r:String!,$a:Int64,$b:Int64){ project(name:$p, entityName:$e){
                      run(name:$r){ history(minStep:$a, maxStep:$b, samples:5000) } } }""",
                   {"p": PROJECT, "e": ENTITY, "r": rid, "a": step, "b": step + chunk})
        for line in data["project"]["run"]["history"]:
            r = json.loads(line)
            if r.get("train/lm_loss") is not None:
                rows[str(r["_step"])] = [r["train/lm_loss"], r.get("train/learning_rate")]
        step += chunk
    json.dump(rows, open(path, "w"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    runs, losses, lrs = {}, {}, {}
    for tag, rids in CHAINS.items():
        merged = {}
        for rid in rids:
            group, cfg = run_config(rid)
            for k, v in EXPECTED[tag].items():
                assert math.isclose(cfg[k], v, rel_tol=0, abs_tol=1e-12), (tag, rid, k, cfg[k])
            rows = full_history(rid, args.cache)
            steps = sorted(int(s) for s in rows)
            runs.setdefault(tag, []).append(dict(run=rid, group=group, config=cfg,
                                                 first_step=steps[0], last_step=steps[-1], logged=len(steps)))
            merged.update({int(s): v for s, v in rows.items()})
        losses[tag] = {s: v[0] for s, v in merged.items()}
        lrs[tag] = {s: v[1] for s, v in merged.items()}

    bins = []
    for b0 in range(0, TOTAL_STEPS, args.bin):
        lo, hi = b0 + 1, min(b0 + args.bin, TOTAL_STEPS)
        row = dict(start=lo, end=hi)
        for a, b in PAIRS:
            common = [s for s in range(lo, hi + 1) if s in losses[a] and s in losses[b]]
            if len(common) < 100:
                continue
            d = [losses[b][s] - losses[a][s] for s in common]
            row[f"{b}-{a}"] = dict(mean=statistics.mean(d), sem=statistics.stdev(d) / math.sqrt(len(d)),
                                    n=len(common), mean_a=statistics.mean(losses[a][s] for s in common),
                                    mean_b=statistics.mean(losses[b][s] for s in common))
        bins.append(row)

    final = {}
    for a, b in PAIRS:
        for lo, hi in [(136001, 143000), (142001, 143000)]:
            common = [s for s in range(lo, hi + 1) if s in losses[a] and s in losses[b]]
            d = [losses[b][s] - losses[a][s] for s in common]
            final[f"{b}-{a}[{lo},{hi}]"] = dict(mean=statistics.mean(d),
                                                sem=statistics.stdev(d) / math.sqrt(len(d)), n=len(d))
    end_lr = {tag: lrs[tag].get(TOTAL_STEPS) for tag in CHAINS}

    json.dump(dict(source=f"wandb {ENTITY}/{PROJECT} (anonymous GraphQL)", metric="train/lm_loss (nats)",
                   difference="v1 - v0 on the same step (same data batch); positive favors v0",
                   bin_steps=args.bin, runs=runs, end_learning_rate=end_lr, final_windows=final, bins=bins),
              open(args.out, "w"), indent=1)
    for k, v in final.items():
        print(f"{k}: {v['mean']:+.5f} +- {v['sem']:.5f} (n={v['n']})")


if __name__ == "__main__":
    main()
