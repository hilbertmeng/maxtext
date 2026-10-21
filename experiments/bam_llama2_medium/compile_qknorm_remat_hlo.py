#!/usr/bin/env python3
"""Run on an owned compiler VM: same source/steps/batch, full/minimal on v5p-8/16.

CPU backend + virtual target topology; never initializes the compiler's TPU backend.
HLO and optional compiler analyses travel directly from the VM to GCS.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--primary-output', required=True)
    parser.add_argument('--steps', type=int, default=13500)
    parser.add_argument('--aligned', action='store_true', help='Only the aligned full/minimal v5p-16 pair')
    args = parser.parse_args()
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    sys.path.insert(0, str(Path.cwd() / 'MaxText'))
    import train_compile
    import jax

    original_save = train_compile.save_compiled
    manifest = []
    with tempfile.TemporaryDirectory(prefix='qknorm-remat-hlo-') as tmp:
        root = Path(tmp)
        arms = (
                ('full', 'BamMediumPropMShiftReadG050StandardQKNormTruePile'),
                ('minimal', 'BamMediumPropMShiftReadG050StandardQKNormMinimalProfile'))
        if args.aligned:
            arms = tuple((policy, f'BamMediumPropMShiftReadG050StandardQKNormAligned128{policy.title()}Profile')
                         for policy in ('full', 'minimal'))
        for topology in (('v5p-16',) if args.aligned else ('v5p-16', 'v5p-8')):
            for policy, exp in arms:
                label = f'{topology}-{policy}'
                arm = root / label
                arm.mkdir()
                (arm / 'output' / f'HLO_{label}').mkdir(parents=True)

                def save(compiled, path, arm=arm):
                    # as_text inspects the optimized executable, not pre-partitioning StableHLO.
                    text = compiled.as_text()
                    if not text:
                        raise RuntimeError('Compiled HLO is empty')
                    with gzip.open(arm / 'optimized-hlo.txt.gz', 'wt') as output:
                        output.write(text)
                    analysis = {}
                    for name in ('memory_analysis', 'cost_analysis'):
                        try:
                            value = getattr(compiled, name)()
                            analysis[name] = value if isinstance(value, (dict, list)) else str(value)
                        except Exception as exc:  # unavailable chip reports must be explicit
                            analysis[name + '_error'] = repr(exc)
                    (arm / 'analysis.json').write_text(json.dumps(analysis, indent=2, default=str)+'\n')
                    original_save(compiled, path)

                train_compile.save_compiled = save
                executable = arm / 'trainstep.pickle'
                train_compile.main([
                    'train_compile.py', 'MaxText/configs/base.yml', f'exp_class={exp}',
                    f'run_name=HLO_{label}', f'base_output_directory={arm}/output',
                    f'compile_topology={topology}', 'compile_topology_num_slices=1',
                    f'steps={args.steps}', f'compiled_trainstep_file={executable}',
                    'upload_all_profiler_results=False'])
                destination = f'{args.prefix.rstrip("/")}/{label}'
                subprocess.run(['gsutil', '-m', 'cp', *map(str, arm.glob('*.gz')),
                                str(arm / 'analysis.json'), str(executable), destination+'/'], check=True)
                if label == 'v5p-16-full':
                    subprocess.run(['gsutil', '-q', 'cp', str(executable), args.primary_output], check=True)
                if args.aligned:
                    alias = args.primary_output.rsplit('/', 1)[0] + f'/{exp}.pickle'
                    subprocess.run(['gsutil', '-q', 'cp', str(executable), alias], check=True)
                manifest.append(dict(topology=topology, policy=policy, exp=exp,
                                     steps=args.steps, prefix=destination))
                print('HLO_ARM_OK', label, flush=True)
                jax.clear_caches()
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        subprocess.run(['gsutil', '-q', 'cp', str(root/'manifest.json'), args.prefix.rstrip('/')+'/'], check=True)
    print('AOT_COMPILE_OK HLO_MATRIX_OK', flush=True)


if __name__ == '__main__':
    main()
