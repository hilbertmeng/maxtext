"""Apply experiment-owned TPU flags before importing/initializing JAX.

Both train.py and train_compile.py use this entry point so a sealed experiment
cannot accidentally compile with one scoped VMEM budget and run with another.
"""
import os
import re
import sys


def configure_from_argv(argv=None):
  argv=sys.argv[1:] if argv is None else argv
  options=dict(arg.split('=',1) for arg in argv if '=' in arg)
  name=options.get('exp_class')
  if not name:return
  import exp
  experiment=getattr(exp,name)
  budget=int(options.get('rmt_scoped_vmem_limit_kib',getattr(experiment,'rmt_scoped_vmem_limit_kib',0)))
  if not budget:return
  if budget<0:raise ValueError('Scoped VMEM budget must be positive')
  flag='--xla_tpu_scoped_vmem_limit_kib'
  value=os.environ.get('LIBTPU_INIT_ARGS','')
  existing=re.findall(re.escape(flag)+r'(?:=|\s+)(\d+)',value)
  if existing and any(int(x)!=budget for x in existing):
    raise ValueError(f'{name} requires scoped VMEM {budget} KiB, conflicting with LIBTPU_INIT_ARGS')
  if not existing:os.environ['LIBTPU_INIT_ARGS']=(value+f' {flag}={budget}').strip()
