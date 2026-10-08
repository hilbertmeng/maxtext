"""Forward contraction theory, W_Q units (1 W_Q = 2BTD^2), per-layer average over 28 layers.
Projections from the actual parameter audit; M contractions from the forward code at each runtime.
Excludes norms/gates/softmax elementwise. LM head reported separately."""
import json
D,T,N,d,K,V,C,r,L=1920,4096,20,96,96,40,10,4,28
u=D*D; W=9  # MLP-write layers 1,4,...,25
QKc=72
def mlp_avg(block,final): return (9*sum(block)+final)/L*3/D
c256=(T+256)/2*N*d/u          # per QK or AV, actual C256 block pairs
ideal=(T+1)/2*N*d/u
splash512=(T+512)/2*N*d/u     # Splash 512 blocks, diagonal blocks computed full
emb=(D*N*K+D*N+D*400+400*N*V)/u+N*K*V/u
common={
 'standard Q/K projections (RoPE24)': 2*D*N*24/u,
 'O projection': N*d*D/u,
 'VO key W_R + V/O/write gates': (D*N*C+3*D*N)/u,
 'attention write P_loc down+up': (D*400+400*N*V)/u,
 'attention write outer': N*K*V/u,
 'MLP write gate+address down/up (9/28)': W/L*(D*N+D*384+384*N*V)/u,
 'MLP write outer (9/28)': W/L*N*K*V/u,
 'M C10 compression': K*V*C/u,
 'VO C10 read': N*K*C/u,
 'static V/O full-M reads': 2*K*V*N/u,
 'static Q/K reads (72 rows)': 2*QKc*V*N/u,
 'embedding write (1/28)': emb/L,
}
rank4={'LocalQK packed projection (basis/gate/mix)': 360*D/u,
       'LocalQK shared basis M read': r*K*V/u,
       'LocalQK rank->head expansion q,k': 2*N*r*K/u,
       'LocalQK Gram + HGH': (r*r*V+2*N*(r*r+r))/u}
direct={'LocalQK direct C10 keys + gates': 2*(D*N*C+D*N)/u,
        'LocalQK direct C10 reads q,k': 2*N*K*C/u}
def arm(extra,block,final):
    rows=dict(common); rows.update(extra)
    rows['SwiGLU MLP']=mlp_avg(block,final)
    rows['C256 QK logits']=c256; rows['C256 AV']=c256
    return rows
mha={'Q/K/V/O projections':4.0,'SwiGLU MLP':3*5120/D,'Splash QK logits (512-block)':splash512,'Splash AV (512-block)':splash512}
arms={'MHA':mha,'BAM rank4':arm(rank4,[6294,6106,6294],6294),'BAM DirectC10':arm(direct,[6267,6079,6267],6267)}
lm=50432*D/u/L
out={'definition':__doc__,'lm_head_per_layer':lm,'ideal_causal_attention_each':ideal,'arms':{}}
for k,rows in arms.items():
    tot=sum(rows.values())
    out['arms'][k]={'rows':rows,'blocks_total':tot,'with_lm_head':tot+lm}
base=out['arms']['MHA']
for k in arms:
    a=out['arms'][k]
    a['vs_MHA_blocks_pct']=(a['blocks_total']/base['blocks_total']-1)*100
    a['vs_MHA_with_lm_pct']=(a['with_lm_head']/base['with_lm_head']-1)*100
    # MHA if it also used ideal causal attention (lower bound for its FLOPs)
mha_ideal=4+8+2*ideal+lm
out['MHA_ideal_causal_with_lm']=mha_ideal
for k in ('BAM rank4','BAM DirectC10'):
    out['arms'][k]['vs_MHA_ideal_with_lm_pct']=(out['arms'][k]['with_lm_head']/mha_ideal-1)*100
json.dump(out,open('flops.json','w'),indent=1)
for k,a in out['arms'].items():
    print(f"== {k}: blocks {a['blocks_total']:.5f}  +LM {a['with_lm_head']:.5f}  vsMHA {a['vs_MHA_with_lm_pct']:+.2f}%")
    for n,v in a['rows'].items(): print(f"   {v:9.5f}  {n}")
print('MHA ideal-causal +LM',round(mha_ideal,5),{k:round(out['arms'][k]['vs_MHA_ideal_with_lm_pct'],2) for k in ('BAM rank4','BAM DirectC10')})
