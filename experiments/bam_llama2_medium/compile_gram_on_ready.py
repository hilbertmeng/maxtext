#!/usr/bin/env python3
"""tpu-ag: reuse an owned idle screen TPU, using the authoritative AOT preparer.

This adapter changes allocation only: compiler, artifact key, validation and manifest
are those of prepare_train_aot.py. The caller retains TPU lifecycle ownership.
"""
import argparse
import fcntl
import importlib.util
from pathlib import Path
import re
import sys


def main():
  p = argparse.ArgumentParser(__doc__)
  p.add_argument('--tpu', required=True)
  p.add_argument('--zone', required=True)
  p.add_argument('--commit', required=True)
  p.add_argument('--size', choices=('Medium', 'XL'), required=True)
  args = p.parse_args()
  if not args.tpu.startswith('xd-') or not re.fullmatch('[0-9a-f]{40}', args.commit):
    p.error('require an owned TPU and full immutable runtime hash')
  root = Path('/home/lishengping/xd/projects')
  spec = importlib.util.spec_from_file_location('aot', root / 'prepare_train_aot.py')
  aot = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(aot)
  sys.stdout.reconfigure(line_buffering=True)
  aot.verify_remote_commit(args.commit)
  aot.gcloud_ssh(args.tpu, args.zone,
                "! pgrep -f '[M]axText/(train|train_compile).py'", timeout=120)
  topology, steps = ('v5p-16', 13500) if args.size == 'Medium' else ('v5p-32', 50000)
  for variant in ('MulOutput', 'MulMix'):
    options = argparse.Namespace(
        exp=f'Bam{args.size}IndependentLLFGram{variant}', commit=args.commit,
        target_topology=topology, steps=steps, installer='install_xd_maxtext_jax081.sh',
        gcs_root=aot.DEFAULT_GCS_ROOT, output=None, zones=[args.zone], force=False,
        compile_timeout=7200)
    job = aot.Preparer(options)
    candidate = {'name': args.tpu, 'zone': args.zone, 'lifecycle_owner': 'gram-screen'}
    job.candidates = [candidate]
    job.state['candidates'] = [candidate]
    job.state_dir.mkdir(parents=True, exist_ok=True)
    with job.lock_path.open('a') as lock:
      fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
      if not job.reusable():
        try:
          job.compile_on(candidate)
          manifest = job.expected_manifest() | {
              'created_utc': aot.now(), 'compiler_tpu': args.tpu,
              'compiler_zone': args.zone, 'artifact_stat': aot.artifact_stat(job.output)}
          aot.upload_json(manifest, job.manifest_uri)
          if aot.load_manifest(job.manifest_uri) != manifest or not job.reusable():
            raise RuntimeError('AOT manifest verification failed')
        except Exception as exc:
          job.save('failed', error=str(exc))
          raise
      job.save('ready', retained_compiler=True)
      print(f'AOT_READY artifact={job.output} manifest={job.manifest_uri}')


if __name__ == '__main__':
  main()
