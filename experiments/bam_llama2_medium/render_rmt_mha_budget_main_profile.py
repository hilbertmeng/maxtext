"""Render the fixed D1200/MLP4118 MHABudget profile with audited theory."""
import argparse
import json
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifact_root',type=Path)
r=parser.parse_args().artifact_root
a=json.loads((r/'original-main-profile.json').read_text());q=1200**2
params={'SwiGLU MLP':3*1200*4118,'Independent QK18 projection + RoPE':2*1200*16*18,'Static QKV M read':3*48*16,'Static MLP M read':48*16,'Static attention + MLP M writes':2*48*16,'Dynamic QK':1200*(4*32+2*16*4+2*16)+4*32+2*16,'Dynamic VO C8':1200*(16*8+2*16)+32*8+2*16,'Dynamic MLP C8 read':1200*(16*8+16)+32*8+16,'Dynamic attention M write':1200*256+256*16*48+1200*16+16*48+16,'Dynamic MLP M write':1200*256+256*16*48+1200*16+16*48+16,'Vector pre-RMSNorm':2*1200}
assert sum(params.values())==17281712
assert sum(params.values())*18+2*50432*1200+2*48*16+48*75==432112752
f={'SwiGLU MLP':3*4118/1200,'Independent QK18 projection + RoPE':2*1200*16*18/q,'Static QKV M read':3*48*16*75/q,'Static MLP M read':48*16*75/q,'Static attention + MLP M writes':2*48*16*75/q,'Dynamic QK':(1200*(4*32+2*16*4+2*16)+4*32*75+2*16*4*75+4*4*32+2*16*(4*4+4))/q,'Dynamic VO C8':(1200*(16*8+2*16)+32*8*75+16*8*75)/q,'Dynamic MLP C8 read':(1200*(16*8+16)+32*8*75+16*8*75)/q,'Dynamic attention M write':(1200*256+256*16*48+1200*16+16*48*75)/q,'Dynamic MLP M write':(1200*256+256*16*48+1200*16+16*48*75)/q,'C256 QK logits':(4096+256)*.5/1200,'C256 AV':(4096+256)*.5/1200}
subparams={'write address down':2*1200*256,'write address up':2*256*16*48,'write gate projection':2*1200*16,'write transforms / other':2*(16*48+16),'QK basis / mix / gate projections':1200*(4*32+2*16*4+2*16),'QK Gram / RMS / gating / other':4*32+2*16}
subf={k:v/q for k,v in subparams.items() if 'projection' in k or k in ('write address down','write address up')};subf.update({'dynamic outer write':2*16*48*75/q,'QK basis M contraction':4*32*75/q,'QK rank-to-head expansion':2*16*4*75/q,'QK Gram / RMS / gating / other':(4*4*32+2*16*(4*4+4))/q})
rows=a['rows'];rows['Dynamic attention + MLP M writes']=[sum(rows[k][i] for k in ('Dynamic attention M write','Dynamic MLP M write')) for i in range(3)];params['Dynamic attention + MLP M writes']=params['Dynamic attention M write']+params['Dynamic MLP M write'];f['Dynamic attention + MLP M writes']=f['Dynamic attention M write']+f['Dynamic MLP M write'];totalp=sum(v for k,v in params.items() if k not in ('Dynamic attention M write','Dynamic MLP M write'));totalf=sum(v for k,v in f.items() if k not in ('Dynamic attention M write','Dynamic MLP M write'))
order=['Scan / residual / LM head / optimizer / other','SwiGLU MLP','Independent QK18 projection + RoPE','C256 QK logits','C256 softmax','C256 AV','Static QKV M read','Static MLP M read','Static attention + MLP M writes','Dynamic QK','subset: QK basis / mix / gate projections','subset: QK basis M contraction','subset: QK rank-to-head expansion','subset: QK Gram / RMS / gating / other','Dynamic VO C8','Dynamic MLP C8 read','Dynamic attention + MLP M writes','subset: write address down','subset: write address up','subset: write gate projection','subset: dynamic outer write','subset: write transforms / other','Vector pre-RMSNorm','RMT health statistics','Complete step','subset: all copy kernels']
out=['| Part | Parameters W_Q/layer | Forward theory W_Q | ms | Step share | XPlane TF | GB |','|---|---:|---:|---:|---:|---:|---:|']
for k in order:
 subset=k.startswith('subset: ');name=k.removeprefix('subset: ');label=('↳ '+name+' (subset)') if subset else k
 if k=='Complete step':val=a['attributed_total'];val[0]=a['step_ms'];ps=totalp/q;fs=totalf
 else:
  val=rows[k];ps=(subparams.get(name,0)/q if subset else params.get(k,0)/q);fs=(subf.get(name,0) if subset else f.get(k,0))
 if k=='Scan / residual / LM head / optimizer / other' or name=='all copy kernels':pstr=fstr='—'
 else:pstr=f'{ps:.6f}';fstr=f'{fs:.6f}' if fs else '≈0'
 out.append(f'| {label} | {pstr} | {fstr} | {val[0]:.2f} | {100*val[0]/a["step_ms"]:.2f}% | {val[1]:.4f} | {val[2]:.2f} |')
(r/'original-main-table.md').write_text('\n'.join(out)+'\n');(r/'original-theory.json').write_text(json.dumps({'params':params,'forward_wq':f,'block_params':totalp,'forward_total_wq':totalf},indent=2));print('block WQ params/flops',totalp/q,totalf);print('MHA theory',12+2*(4096+256)/2/1200,'theory overhead %',(totalf/(12+2*(4096+256)/2/1200)-1)*100); print('totalTFGB',a['attributed_total']);print('matrix flow ms',sum(rows[k][0] for k in params if k not in ('SwiGLU MLP','Independent QK18 projection + RoPE','Dynamic attention + MLP M writes'))+rows['RMT health statistics'][0])
