#!/usr/bin/env bash
set -euo pipefail
MIX_ROOT=$(cd "$(dirname "$0")/.." && pwd)
MIX_ARTIFACT=/data0/xd/bam_diagnostics/mediumprop-h-mlp-input-mix
mkdir -p "$MIX_ARTIFACT"
exec 9>"$MIX_ARTIFACT/cpu-gate.lock"
flock 9
python3 - "$MIX_ROOT" "$MIX_ARTIFACT" <<'PY'
import hashlib,json,pathlib,subprocess,sys,datetime
root,artifact=map(pathlib.Path,sys.argv[1:]);paths=subprocess.check_output(['git','ls-files','MaxText'],cwd=root,text=True).splitlines()
digest=hashlib.sha256()
for path in paths:
 p=root/path
 if p.is_file() and (p.suffix in ['.py','.yml','.yaml']):digest.update(path.encode());digest.update(p.read_bytes())
fingerprint=digest.hexdigest();receipt=artifact/'cpu-pass.json'
if receipt.exists() and json.loads(receipt.read_text()).get('fingerprint')==fingerprint:
 print('H_MEMORY_INPUT_MIX_CPU_REUSED',fingerprint,flush=True);sys.exit(0)
targeted=artifact/'targeted-cpu-pass.json'
if not targeted.exists() or json.loads(targeted.read_text()).get('fingerprint')!=fingerprint:
 subprocess.run(['python3','/home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py',str(root),'--test-file','MaxText/tests/bam_h_memory_input_mix_test.py','--jobs','4'],check=True)
else:print('Reusing8 successful targeted checks on identical source',flush=True)
subprocess.run(['bash',str(root/'.agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh'),str(root)],check=True)
subprocess.run(['python3','/home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py',str(root),'--test-file','MaxText/tests/summary_writer_test.py','--jobs','1'],check=True)
receipt.write_text(json.dumps(dict(fingerprint=fingerprint,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),checks='8 targeted +47 BAM +5 summary',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
print('H_MEMORY_INPUT_MIX_CPU_OK',fingerprint,flush=True)
PY
