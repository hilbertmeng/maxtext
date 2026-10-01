"""Read-only sparse checkpoint Adam-slot summaries, no JAX/model execution."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import tensorstore as ts
from orbax.checkpoint._src.serialization import tensorstore_utils


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('checkpoint')
  parser.add_argument('output')
  parser.add_argument('--epsilon',type=float,default=1e-8)
  parser.add_argument('--max-elements',type=int,default=4000000)
  args=parser.parse_args()
  raw=ts.KvStore.open(args.checkpoint.rstrip('/')+'/').result().read('_METADATA').result().value
  metadata=json.loads(raw)
  def open_array(name):
    kv=tensorstore_utils.build_kvstore_tspec(args.checkpoint,name,use_ocdbt=metadata['use_ocdbt'])
    kv.pop('cache_pool',None)
    return ts.open({'driver':'zarr3' if metadata['use_zarr3'] else 'zarr','kvstore':kv},open=True).result()
  paths=[ast.literal_eval(k) for k in metadata['tree_metadata'] if ast.literal_eval(k)[:2]==('opt_state','nu')]
  def inspect(path):
    name='.'.join(path);v=open_array(name);shape=v.shape
    scanned='layers' in path
    elements=np.prod(shape)//(shape[1] if scanned else 1)
    if elements>args.max_elements:
      return {'path':name,'shape':shape,'skipped':'per-layer size exceeds explicit read budget'}
    m=open_array('.'.join(('opt_state','mu')+path[2:]))
    rows=[]
    for layer in sorted(set([0,shape[1]//2,shape[1]-1])) if scanned else [None]:
      index=[slice(None)]*len(shape)
      if layer is not None:index[1]=layer
      vf=v[tuple(index)].read();mf=m[tuple(index)].read()
      variance=np.asarray(vf.result());moment=np.asarray(mf.result())
      std=np.sqrt(variance);direction=moment/(std+args.epsilon)
      rows.append({'layer':layer,'sqrt_variance_percentiles':np.percentile(std,[0,1,50,99,100]).tolist(),
        'epsilon_dominated_fraction':float(np.mean(std<args.epsilon)),
        'epsilon_attenuation_mean':float(np.mean(std/(std+args.epsilon))),
        'adam_direction_rms':float(np.sqrt(np.mean(direction**2))),
        'finite':bool(np.isfinite(direction).all())})
    return {'path':name,'shape':shape,'rows':rows}
  with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inspect,paths))
  count=int(open_array('opt_state.count').read().result())
  result={'checkpoint':args.checkpoint,'count':count,'epsilon':args.epsilon,'parameters':rows,
          'note':'adam_pax already-corrected moments; first/middle/last layers; no updates or checkpoint writes'}
  Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
  print('OPTIMIZER_HEALTH_COMPLETE',len(rows),sum('skipped' in r for r in rows),flush=True)


if __name__=='__main__':main()
