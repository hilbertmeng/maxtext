#!/usr/bin/env bash
# Standalone installer entrypoint: benchmark public Git checkout, without a training environment.
set -euo pipefail
python3 - "${1:?region}" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

REPOSITORY = 'https://github.com/hilbertmeng/maxtext.git'
COMMIT = '03f0a0f94b95da58fe44668aefffab298c3535d6'
root = Path(tempfile.mkdtemp(prefix='xd-git-source-probe-'))
env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_CONFIG_GLOBAL='/dev/null',
           GIT_CONFIG_SYSTEM='/dev/null')
env.pop('GIT_CONFIG_COUNT', None)
records = []

def command(argv, cwd=None):
    start = time.monotonic()
    try:
        result = subprocess.run(argv, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        return dict(seconds=time.monotonic()-start, exit_code=result.returncode,
                    stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except subprocess.TimeoutExpired:
        return dict(seconds=time.monotonic()-start, exit_code=124,
                    stdout='', stderr='command timed out after 180s')

def git(args, cwd):
    return command(['git', '-c', 'credential.helper=', '-c', 'http.lowSpeedLimit=1024',
                    '-c', 'http.lowSpeedTime=30', *args], cwd)

def emit(record):
    records.append(record)
    (root/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(record), flush=True)

emit(dict(kind='metadata', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          hostname=platform.node(), root=str(root), repository=REPOSITORY, commit=COMMIT,
          git_version=command(['git', '--version']), authentication='anonymous HTTPS',
          proxy_present=any(env.get(k) for k in ('HTTPS_PROXY','https_proxy','ALL_PROXY','all_proxy'))))

successful = []
for trial in range(1, 6):
    checkout = root/f'cold-{trial}'
    checkout.mkdir()
    start = time.monotonic()
    stages = {}
    for label, args in [('init', ['init', '-q']),
                        ('remote', ['remote', 'add', 'origin', REPOSITORY]),
                        ('fetch', ['fetch', '--depth=1', '--no-tags', 'origin', COMMIT]),
                        ('checkout', ['checkout', '-q', '--detach', COMMIT])]:
        stages[label] = git(args, checkout)
        if stages[label]['exit_code']:
            break
    ok = len(stages) == 4 and all(s['exit_code'] == 0 for s in stages.values())
    verification = {}
    if ok:
        verification['head'] = git(['rev-parse', 'HEAD'], checkout)['stdout']
        verification['tree'] = git(['rev-parse', 'HEAD^{tree}'], checkout)['stdout']
        verification['status'] = git(['status', '--porcelain'], checkout)['stdout']
        verification['fsck'] = git(['fsck', '--no-dangling'], checkout)
        verification['files_sha256'] = {
            name: hashlib.sha256((checkout/name).read_bytes()).hexdigest()
            for name in ('MaxText/train_compile.py', 'MaxText/train.py',
                         'MaxText/layers/attentions.py', 'MaxText/exp.py')}
        ok = (verification['head'] == COMMIT and not verification['status']
              and verification['fsck']['exit_code'] == 0)
        if ok:
            successful.append(checkout)
    emit(dict(kind='cold', trial=trial, ok=ok, seconds=time.monotonic()-start,
              stages=stages, verification=verification))

if successful:
    for trial in range(1, 6):
        start = time.monotonic()
        fetch = git(['fetch', '--depth=1', '--no-tags', 'origin', COMMIT], successful[0])
        checkout = git(['checkout', '-q', '--detach', COMMIT], successful[0])
        emit(dict(kind='warm', trial=trial, seconds=time.monotonic()-start,
                  ok=fetch['exit_code'] == checkout['exit_code'] == 0,
                  fetch=fetch, checkout=checkout))

trials = [r for r in records if r['kind'] in ('cold', 'warm')]
emit(dict(kind='summary', completed=len(trials), passed=sum(r['ok'] for r in trials),
          results=str(root/'results.json'), note='Retain individual failures; no hidden retries.'))
print('GIT_SOURCE_PROBE_DONE results='+str(root/'results.json'), flush=True)
destination = ('gs://newproject-1-llm_base_models_us-central1/log/diagnostics/'
               'git-source-probe-0906/'+platform.node()+'/results.json')
upload = command(['gsutil', '-q', 'cp', str(root/'results.json'), destination])
print(json.dumps(dict(kind='artifact', uri=destination, **upload)), flush=True)
raise SystemExit(0 if len(trials) == 10 and all(r['ok'] for r in trials) else 1)
PY
