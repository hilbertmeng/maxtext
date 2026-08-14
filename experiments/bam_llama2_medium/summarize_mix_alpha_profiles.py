#!/usr/bin/env python3
"""Print one compact row per local BAM mix-alpha step-10 trace."""

import argparse
import glob
import os

from analyze_bam_xplane import summarize


def mean(values):
  values = list(values)
  return sum(values) / len(values)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("root")
  args = parser.parse_args()
  traces = sorted(glob.glob(
      os.path.join(args.root, "**", "step_10", "**", "*.trace.json.gz"),
      recursive=True))
  print("run\tstep_ms\tmix_ms\tmix_gb\tmix_fetch_ms\tqk_ms\tsoftmax_ms\tav_ms\tfetch_ms")
  for trace in traces:
    run = next(part for part in trace.split(os.sep) if part.startswith("Mix"))
    devices = list(summarize(trace).values())

    def metric(bucket, index=0):
      return mean(values.get(bucket, [0.0, 0.0, 0.0])[index] for _, values in devices)

    print(
        f"{run}\t{mean(step for step, _ in devices):.3f}\t"
        f"{metric('mix_alpha'):.3f}\t{metric('mix_alpha', 2):.3f}\t"
        f"{metric('mix_fetch'):.3f}\t{metric('mha_qk_logits'):.3f}\t"
        f"{metric('mha_softmax'):.3f}\t{metric('mha_av'):.3f}\t"
        f"{metric('fetch_m'):.3f}")


if __name__ == "__main__":
  main()
