#!/usr/bin/env python3
"""Partition exclusive XPlane op time into additive parts (first matching rule wins); overlapping subsets listed with ↳."""
import json,collections
TH=json.load(open('flops.json'))['arms']
def th(arm,*names): return sum(TH[arm]['rows'][n] for n in names)
RULES=[ # (row, predicate on (tf_op,hlo))
 ('Concat/BAM health statistics', lambda o,h: '_record_concat' in o or 'concat_' in o and 'sow' in o),
 ('Attention core (MHA Splash / BAM C256 QChunk)', lambda o,h: 'attention_op' in o or '_query_chunk_op' in o),
 ('SwiGLU MLP', lambda o,h: '/mlp/' in o),
 ('Standard Q/K(/V) projections', lambda o,h: 'query_projection' in o or 'kv_projection' in o),
 ('QKNorm + RoPE', lambda o,h: 'apply_rotary_embedding' in o or '/qk_norm/' in o),
 ('O projection', lambda o,h: 'out_projection' in o),
 ('LocalQK C10 concat into heads', lambda o,h: '_add_local_qk' in o),
 ('C10 compression', lambda o,h: 'compress_abs_v_cache' in o),
 ('LocalQK direct C10 key/gate/read + static QK', lambda o,h: 'read_local_m_for_qk' in o),
 ('LocalVO key/gates/read/output gating', lambda o,h: '_independent_local_vo' in o),
 ('Static V/O full-M reads', lambda o,h: '_static_column' in o),
 ('Attention write (P_loc, gate, outer)', lambda o,h: 'bam/write_m' in o or '_deferred_write_factors' in o),
 ('MLP write (gate, R384 address, outer)', lambda o,h: 'merge_mlp_write' in o or 'mlp_address_' in o or 'mlp_write_gate' in o),
 ('Embedding write', lambda o,h: 'initial_bam_matrix' in o),
 ('Layer norms', lambda o,h: 'layer_norm' in o),
 ('LM head / loss', lambda o,h: '/lm_head/' in o),
 ('Scan carry / optimizer / unscoped / other', lambda o,h: True),
]
SUB=[ # overlapping subsets
 ('↳ LocalQK direct C10 M contraction (subset)', lambda o,h: 'read_local_m_for_qk' in o and 'read_m_contract' in o),
 ('↳ static Q/K full-M reads (subset)', lambda o,h: 'read_local_m_for_qk' in o and '_static_column' in o),
 ('↳ LocalQK key/gate projections (subset)', lambda o,h: 'read_local_m_for_qk' in o and ('/W_lq_c8/' in o or '/W_lk_c8/' in o or 'read_gate_projection' in o)),
 ('↳ LocalVO C10 M contraction (subset)', lambda o,h: '_independent_local_vo' in o and 'read_m_contract' in o),
 ('↳ attention + MLP M outer writes (subset)', lambda o,h: 'write_outer' in o),
 ('↳ attention core forward scope (subset)', lambda o,h: ('attention_op' in o or '_query_chunk_op' in o) and 'transpose(' not in o),
 ('↳ attention core backward/remat scope (subset)', lambda o,h: ('attention_op' in o or '_query_chunk_op' in o) and 'transpose(' in o),
 ('↳ all copy kernels (cross-cutting subset)', lambda o,h: h.startswith('copy')),
]
THEORY={ # forward W_Q per-layer average: (MHA, BAM)
 'Attention core (MHA Splash / BAM C256 QChunk)': (th('MHA','Splash QK logits (512-block)','Splash AV (512-block)'), th('BAM DirectC10','C256 QK logits','C256 AV')),
 'SwiGLU MLP': (8.0, th('BAM DirectC10','SwiGLU MLP')),
 'Standard Q/K(/V) projections': (3.0, th('BAM DirectC10','standard Q/K projections (RoPE24)')),
 'O projection': (1.0,1.0),
 'LocalQK direct C10 key/gate/read + static QK': (0, th('BAM DirectC10','LocalQK direct C10 keys + gates','LocalQK direct C10 reads q,k','static Q/K reads (72 rows)')),
 'C10 compression': (0, th('BAM DirectC10','M C10 compression')),
 'LocalVO key/gates/read/output gating': (0, (1920*20*10+2*1920*20)/1920**2 + th('BAM DirectC10','VO C10 read')),
 'Static V/O full-M reads': (0, th('BAM DirectC10','static V/O full-M reads')),
 'Attention write (P_loc, gate, outer)': (0, th('BAM DirectC10','attention write P_loc down+up','attention write outer')+1920*20/1920**2),
 'MLP write (gate, R384 address, outer)': (0, th('BAM DirectC10','MLP write gate+address down/up (9/28)','MLP write outer (9/28)')),
 'Embedding write': (0, th('BAM DirectC10','embedding write (1/28)')),
 'LM head / loss': (TH['MHA']['with_lm_head']-TH['MHA']['blocks_total'],)*2,
}
def part(f):
    r=json.load(open(f)); rows=collections.defaultdict(lambda:[0.,0.,0.]); subs=collections.defaultdict(lambda:[0.,0.,0.])
    for x in r['ops']:
        o,h=x['tf_op'],x['hlo']; v=(x['ms'],x['tf'],x['gb'])
        for name,p in RULES:
            if p(o,h):
                for i in range(3): rows[name][i]+=v[i]
                break
        for name,p in SUB:
            if p(o,h):
                for i in range(3): subs[name][i]+=v[i]
    step=sum(r['step_ms'])/len(r['step_ms'])
    return rows,subs,step,r
m,ms,mstep,mr=part('mha-ops.json'); b,bs,bstep,br=part('dc10-ops.json')
out=['| Part | Forward theory W_Q (MHA / DirectC10) | MHA ms | BAM ms | Δ ms | BAM step share | MHA / BAM TF | MHA / BAM GB |','|---|---:|---:|---:|---:|---:|---:|---:|']
def fmt(name,x,y,bold=False):
    t=THEORY.get(name); ts=f'{t[0]:.5f} / {t[1]:.5f}' if t else ('≈0 / ≈0' if 'health' in name or 'norm' in name.lower() or 'RoPE' in name or 'concat' in name else '—')
    s=f'| {name} | {ts} | {x[0]:.2f} | {y[0]:.2f} | {y[0]-x[0]:+.2f} | {100*y[0]/bstep:.2f}% | {x[1]:.4f} / {y[1]:.4f} | {x[2]:.2f} / {y[2]:.2f} |'
    return s.replace(f'| {name} |',f'| **{name}** |') if bold else s
for name,_ in RULES: out.append(fmt(name,m[name],b[name]))
tot=lambda R:[sum(v[i] for v in R.values()) for i in range(3)]
mt,bt=tot(m),tot(b)
out.append(f"| **Complete device step** | **{TH['MHA']['with_lm_head']:.5f} / {TH['BAM DirectC10']['with_lm_head']:.5f}** | **{mstep:.2f}** | **{bstep:.2f}** | **{bstep-mstep:+.2f}** | 100% | **{mt[1]:.4f} / {bt[1]:.4f}** | **{mt[2]:.2f} / {bt[2]:.2f}** |")
for name,_ in SUB: out.append(fmt(name,ms[name],bs[name]))
txt='\n'.join(out); print(txt); open('main-table.md','w').write(txt+'\n')
print('exclusive sum ms MHA/BAM',round(mt[0],2),round(bt[0],2),'coverage',[round(c,4) for c in mr['coverage']],[round(c,4) for c in br['coverage']])
