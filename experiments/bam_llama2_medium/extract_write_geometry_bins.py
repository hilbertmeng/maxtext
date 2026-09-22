"""Freeze real-valued plot coordinates on the original 32 discovery sequences."""
import json,sys
from pathlib import Path
import numpy as np
from analyze_write_geometry import transform,FN
rawroot=Path(sys.argv[1]);out=Path(sys.argv[2]);meta=json.loads((rawroot/'metadata.json').read_text());ix={s:i for i,s in enumerate(meta['fields'])};valid=np.load(rawroot/'valid.npy');rows=[]
for l,h,j in [(18,15,0),(15,12,2),(4,12,7),(18,4,7),(15,12,13),(16,12,13)]:
 a=[]
 for i in range(32):
  r=np.load(rawroot/f'sample_{i:03d}.npy',mmap_mode='r')[l,:,h];mask=valid[i]&(r[:,ix['read_o_gate']]>=.1)&(r[:,ix['write_gate']]>=.1)&(r[:,ix['gram_33']]>1e-20);f,_=transform(r[mask],meta['fields']);a.extend(f[:,j].tolist())
 z=np.asarray(a);z=z[np.isfinite(z)];rows.append(dict(layer=l,head=h,feature=FN[j],edges=np.quantile(z,np.linspace(0,1,9)).tolist(),centers=np.quantile(z,(np.arange(8)+.5)/8).tolist()))
(out/'example_bin_values.json').write_text(json.dumps(rows,indent=2))
