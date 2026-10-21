#!/usr/bin/env python3
"""tpu-ag: transfer the completed raw pair's owned idle VM to aligned AOT/profile.

Preserves four same-VM arms while avoiding a second target allocation. No formal
trainer is used. Run only after the first controller has exited; this process owns
cleanup of the one transferred TPU. All model source stays sealed at a380095b.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, '/home/lishengping/xd/projects')
import prepare_train_aot as aot

ROOT=aot.ROOT
COMMIT='a380095bbd13b9b85f4373eced426020b6824820'
TPU='xd-v5p-16-qknorm-rematdiag-20261021-ew4b'
ZONE='europe-west4-b'
FULL='BamMediumPropMShiftReadG050StandardQKNormAligned128FullProfile'
MINIMAL='BamMediumPropMShiftReadG050StandardQKNormAligned128MinimalProfile'
AOT='gs://newproject-1-llm_base_models_us-central1/log/diagnostics/qknorm_aligned128/a380095/v5p-16/s13500'


def main():
    state_path=ROOT/'logs/20261021-remat16-aligned128-continuation.json'
    state=dict(tpu=TPU,zone=ZONE,commit=COMMIT,status='validating')
    def save(status):
        state['status']=status;state_path.write_text(json.dumps(state,indent=2)+'\n')
    raw=json.loads((ROOT/'logs/20261021-remat16-state.json').read_text())
    assert raw['status'] in ('complete','complete_transfer_alignment')
    assert len(raw['completed_arms'])==2 and raw['winner']==dict(name=TPU,zone=ZONE)
    # Refuse to overlap a trainer or compiler, including either TPU worker.
    guard="if pgrep -f '[M]axText/(train|train_compile).py|[c]ompile_qknorm_remat_hlo.py' >/dev/null; then exit 1; fi"
    aot.run([*aot.gcloud_base(),'compute','tpus','tpu-vm','ssh',TPU,'--internal-ip',
             f'--zone={ZONE}',f'--project={aot.PROJECT}','--worker=all',f'--command={guard}'],timeout=300)
    save('idle_verified')
    try:
        runner=Path(__file__).with_name('compile_qknorm_remat_hlo.py')
        state['compiler_runner_sha256']=aot.sha256(runner)
        aot.gcloud_scp(runner,TPU,ZONE,'/tmp/compile_qknorm_aligned128.py')
        command=(
            'cd /home/lishengping/xd/projects/maxtext && git diff --quiet && git diff --cached --quiet && '
            f'git -c credential.helper= fetch --no-tags https://github.com/hilbertmeng/maxtext.git {COMMIT} && '
            f'git checkout --detach {COMMIT} && '
            'env HARDWARE=tpu JAX_PLATFORMS=cpu MAXTEXT_SKIP_AOT_ANALYSIS=1 '
            '/home/lishengping/miniconda3/bin/python /tmp/compile_qknorm_aligned128.py '
            f'--aligned --steps 13500 --prefix {AOT}/hlo --primary-output {AOT}/{FULL}.pickle '
            '> /tmp/qknorm-aligned128-compile.log 2>&1 && '
            f'gsutil -q cp /tmp/qknorm-aligned128-compile.log {AOT}/compile.log && '
            'tail -4 /tmp/qknorm-aligned128-compile.log')
        save('compiling_on_idle_worker0_cpu')
        print(aot.gcloud_ssh(TPU,ZONE,command,timeout=7200,monitor_compiler=True),flush=True)
        for exp in (FULL,MINIMAL):aot.run(['gsutil','stat',f'{AOT}/{exp}.pickle'])
        save('aot_ready')
        dataset=aot.run(['python3',str(ROOT/'dataset_paths.py'),'--zone',ZONE,
                         '--variant=truepile4096','--exp',FULL,'--commit',COMMIT]).stdout.strip()
        command=['env',f'AOT_ROOT={AOT}','PROFILE_STEPS=13500','PROFILE_DONE_STEP=99',
                 'PROFILE_SKIP=10','PROFILE_PERIOD=1000','PROFILE_DURATION=5','PROFILE_TRACE_COUNT=1',
                 f'DATASET_PATH={dataset}',str(ROOT/'run_profile_matrix.sh'),TPU,ZONE,COMMIT,
                 'qknorm-remat16-align128',FULL,MINIMAL]
        save('profiling_aligned_pair')
        log=ROOT/'logs/20261021-remat16-aligned128-matrix.log'
        state['matrix_log']=str(log);save('profiling_aligned_pair')
        with log.open('w') as output:
            subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True)
        save('complete');print('ALIGNED_PROFILE_PAIR_OK',flush=True)
    finally:
        result=aot.run([str(ROOT/'delete_tpu_xd.sh'),TPU,ZONE,aot.PROJECT],check=False,timeout=900)
        state['cleanup_returncode']=result.returncode;state['cleanup_tail']=result.stdout[-2000:]
        save(state['status']);print(result.stdout,flush=True)
        if result.returncode:raise RuntimeError('Transferred profile TPU cleanup failed')


if __name__=='__main__':main()
