"""Validate jointly removed row heads on a separate half of the fixed cohort."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from summarize_row_heads import stats

def run(root):
 folder=root/'heads/groups';reference=np.load(root/'heads/reference.npz')
 selected=json.loads((root/'heads/group_scenarios.json').read_text())
 scenarios=json.loads((folder/'scenarios.json').read_text());assert scenarios==selected
 metadata=json.loads((folder/'metadata.json').read_text())
 assert metadata['scenarios_sha256']==hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
 files=sorted(folder.glob('groups_[0-9][0-9][0-9].npz'));assert len(files)==64
 matrix=[];manifest={};err=0.
 for index,file in enumerate(files):
  assert file.stem==f'groups_{index:03d}'
  with np.load(file) as a:
   assert str(a['sequence_hash'])==str(reference['sequence_hash'][index])
   value=a['loss'].ravel().astype(float);assert value.shape==(23,)
   np.testing.assert_allclose(value[:5],reference['loss'][index],atol=1e-6,rtol=0)
   err=max(err,float(np.max(np.abs(value[:5]-reference['loss'][index]))))
   assert np.isfinite(value).all();matrix.append(value-reference['loss'][index,0])
  manifest[file.name]=dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
 matrix=np.stack(matrix)
 result=dict(model=metadata['model'],checkpoint=metadata['checkpoint'],selection='First32 individual head mean per layer/path, lowest/highest signed scores; ties by head index',evaluation='Last32 sequences held out of head selection, same 64-sequence cohort',scenarios={})
 for j,s in enumerate(scenarios):
  result['scenarios'][s['name']]=dict(all64=stats(matrix[:,j]),first32=stats(matrix[:32,j]),heldout32=stats(matrix[32:,j]),removed_head_layer_pairs=len(s.get('remove',[])))
 result['top_minus_bottom']={}
 for path in 'QKVO':
  for count in (4,8):
   a=next(j for j,s in enumerate(scenarios) if s['name']==f'{path}_top{count}');b=next(j for j,s in enumerate(scenarios) if s['name']==f'{path}_bottom{count}')
   result['top_minus_bottom'][f'{path}_{count}']=stats((matrix[:,a]-matrix[:,b])[32:])
 (root/'heads/group_summary.json').write_text(json.dumps(result,indent=2))
 (folder/'verification.json').write_text(json.dumps(dict(max_control_error=err,files=manifest),indent=2))
 np.savez_compressed(root/'heads/paired_group_gaps.npz',gap=matrix,names=[s['name'] for s in scenarios],sequence_hash=reference['sequence_hash'])
 print('HEAD_GROUPS_VERIFIED',metadata['model'],matrix.shape,'control_error',err)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
