import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import concurrent.futures,json,sys,time
from pathlib import Path
import numpy as np
from analyze_write_geometry import analyze_layer
root=Path(sys.argv[1]);n=json.loads((root/'metadata.json').read_text())['n'];tasks=[(str(root),l,n) for l in [3,12]]
t=time.perf_counter();serial=[analyze_layer(t) for t in tasks];serial_seconds=time.perf_counter()-t
arrays=[]
for l in [3,12]:
 with np.load(root/f'analysis/layer_{l:02d}.npz') as f:arrays.append({k:f[k] for k in f.files})
t=time.perf_counter()
with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:parallel=list(pool.map(analyze_layer,tasks))
parallel_seconds=time.perf_counter()-t
for l,expected in zip([3,12],arrays):
 with np.load(root/f'analysis/layer_{l:02d}.npz') as f:
  for k,v in expected.items():np.testing.assert_equal(f[k],v)
out=dict(cpus=os.cpu_count(),serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,speedup=serial_seconds/parallel_seconds,numerically_identical=True)
(root/'cpu_benchmark.json').write_text(json.dumps(out,indent=2));print(out,flush=True)
