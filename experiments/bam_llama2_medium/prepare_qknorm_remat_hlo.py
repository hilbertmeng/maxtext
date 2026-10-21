#!/usr/bin/env python3
"""tpu-ag controller: use the standard AOT preparer's ownership/racing/cleanup.

Arguments are the same as prepare_train_aot.py. Stage this script and its sibling
compile_qknorm_remat_hlo.py together; all model code comes from the sealed commit.
"""
from pathlib import Path
import shlex
import sys

sys.path.insert(0, '/home/lishengping/xd/projects')
import prepare_train_aot as aot


class HloPreparer(aot.Preparer):
    def compile_on(self, candidate):
        name, zone = candidate['name'], candidate['zone']
        runner = Path(__file__).with_name('compile_qknorm_remat_hlo.py')
        remote = '/tmp/compile_qknorm_remat_hlo.py'
        aot.gcloud_scp(runner, name, zone, remote)
        repo = '/home/lishengping/xd/projects/maxtext'
        prefix = self.output.rsplit('/', 1)[0] + '/hlo'
        command = (
            f'cd {shlex.quote(repo)} && '
            f'test "$(git rev-parse HEAD)" = {shlex.quote(self.args.commit)} && '
            'git diff --quiet && git diff --cached --quiet && '
            'env HARDWARE=tpu JAX_PLATFORMS=cpu MAXTEXT_SKIP_AOT_ANALYSIS=1 '
            '/home/lishengping/miniconda3/bin/python '
            f'{remote} --prefix {shlex.quote(prefix)} '
            f'--primary-output {shlex.quote(self.output)} --steps {self.args.steps}')
        self.save('compiling', compiler_tpu=name, compiler_zone=zone,
                  diagnostic_runner_sha256=aot.sha256(runner), diagnostic_prefix=prefix)
        output = aot.gcloud_ssh(name, zone, command, timeout=self.args.compile_timeout,
                               monitor_compiler=True)
        print(output, flush=True)
        if 'HLO_MATRIX_OK' not in output:
            raise RuntimeError('HLO matrix did not complete')


if __name__ == '__main__':
    HloPreparer(aot.parse_args()).prepare()
