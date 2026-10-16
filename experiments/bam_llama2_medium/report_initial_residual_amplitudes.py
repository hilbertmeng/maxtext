"""Combine per-sequence initial moments with analytic variance propagation."""
import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np

def gelu_second_moment(q):
    """Gaussian exact-GELU second moment; report has no JAX dependency."""
    return q*(.25+np.arcsin(q/(1+q))/(2*np.pi))+q*q/(np.pi*(1+q)*np.sqrt(1+2*q))


ORDER = ['mha', 'mudd', 'unfixed', 'bam_fix', 'all_original', 'large_comp', 'large_uncomp']
LABELS = dict(mha='MHA', mudd='Mudd', unfixed='BAM unfixed', bam_fix='BAM init fix',
              all_original='BAM all original', large_comp='BAM large E + compensation',
              large_uncomp='BAM large E, no compensation (probe only)')


def rms_aggregate(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def token_concentrations(cohort):
    ids = np.load(cohort)['inputs'][:4, :2048]
    concentrations, shifted, mix_variance = [], [], []
    for row in ids:
        counts = {}; squared = 0; cs = []; ss = []; mv = []
        for t, token in enumerate(row, 1):
            token = int(token); old = counts.get(token, 0)
            squared += 2*old+1; counts[token] = old+1
            cs.append(squared/t**2)
            first = int(row[0])
            # Sum of M-shifted values = sum raw values + .5*first - .5*last.
            adj = 0 if first == token else counts[first]-counts[token]+.5
            ss.append((squared+adj)/t**2)
            mv.append(1. if t == 1 or token == int(row[t-2]) else .5)
        concentrations.append(float(np.mean(cs)))
        shifted.append(float(np.mean(ss)))
        mix_variance.append(float(np.mean(mv)))
    return dict(uniform_prefix=concentrations, mshift_prefix=shifted,
                mshift_token_variance=mix_variance)


def unpack_blocks(x):
    records = x['records']; out = []
    for path, values in records.items():
        if not path.endswith('/initial_amplitude_block'):
            continue
        prefix = path.rsplit('/', 1)[0]
        a = np.asarray(values)
        if a.ndim == 2:
            match = re.search(r'/layers_(\d+)(?:/|$)', prefix)
            indices = [int(match.group(1))]
            a = a[None]
        elif x['resolved']['bam_pair_scan']:
            match = re.search(r'/(?:local|fetch)_(\d+)/', prefix+'/')
            offset = int(match.group(1))
            indices = [i*3+offset for i in range(a.shape[0])]
        else:
            indices = list(range(a.shape[0]))
        for idx, val in zip(indices, a):
            def other(suffix):
                key = prefix+suffix
                v = np.asarray(records[key])
                return v if len(indices) == 1 and v.ndim == 1 else v[indices.index(idx)]
            out.append((idx, val, other('/self_attention/out/initial_amplitude_head'),
                        other('/mlp/wi/initial_amplitude_wi'),
                        other('/mlp/wo/initial_amplitude_gelu')))
    assert sorted(idx for idx, *_ in out) == list(range(18))
    return sorted(out, key=lambda r:r[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory'); args = ap.parse_args(); root = Path(args.directory)
    data = {arm:json.loads((root/(arm+'.json')).read_text()) for arm in ORDER}
    assert len({x['input_ids_sha256'] for x in data.values()}) == 1
    assert len({x['script_sha256'] for x in data.values()}) == 1
    con = token_concentrations(data['mha']['cohort'])
    s = float(np.mean(con['uniform_prefix'])); sm = float(np.mean(con['mshift_prefix']))
    mv = float(np.mean(con['mshift_token_variance']))
    rows = []; priors = []; checks = []
    for arm, x in data.items():
        c = x['resolved']; d = c['emb_dim']; depth = c['num_decoder_layers']
        stdw = .006 if c['init_method'] == '006normal' else math.sqrt(2/(5*d))
        stdo = .006 if c['init_method'] == '006normal' else 2/(depth*math.sqrt(d))
        stde = .006 if c['token_embedding_init_method'] == '006normal' else stdw
        gnorm = stde/math.sqrt(stde*stde+c['normalization_layer_epsilon'])
        emb = rms_aggregate(x['records']['intermediates/token_embedder/initial_amplitude_embedding'])
        gain_o = math.sqrt(d)*stdo
        mseed = None
        if arm in ('mha','mudd'):
            a_prior = gain_o * math.sqrt(d)*stdw*gnorm*math.sqrt(s)
        else:
            stdb = .006 if c['bam_init_method'] == '006normal' else stdw
            qseed = (stde*c['bam_embedding_write_input_scale'])**2*d*stdb**2
            qa = 256*stdb**2*gelu_second_moment(qseed)
            eps = c['bam_write_rms_epsilon'] or c['normalization_layer_epsilon']
            mseed = math.sqrt(16)*.1*math.sqrt(qseed/(qseed+eps))*math.sqrt(qa/(qa+eps))
            qrkey = 256*stdb**2*gelu_second_moment(d*stdb**2*gnorm**2)
            local_o = mseed*math.sqrt(mv)*math.sqrt(32)*.1*.05*math.sqrt(qrkey/(qrkey+1e-4))
            a_prior = gain_o*math.sqrt(mseed*mseed*sm+local_o*local_o)
        for idx, val, head, wi, gelu in unpack_blocks(x):
            if c['mlp_dim_by_layer'] is not None:
                width = c['mlp_dim_by_layer'][idx]
            elif c['mlp_dim_by_block'] is not None:
                width = c['mlp_dim_by_block'][idx%3]
            elif c['dynamic_mlp_dim']:
                width = round(c['mlp_dim']*(idx/(depth-1)+.5)/128)*128
            else:
                width = c['mlp_dim']
            vals = {name:rms_aggregate(val[:,i]) for i,name in enumerate(x['block_fields']) if not name.startswith('cross_')}
            # Cross terms are already signed second moments, not RMS values.
            cross_ra = float(np.mean(val[:,5])); cross_sf = float(np.mean(val[:,6]))
            q = d*stdw**2*vals['mlp_norm_input']**2
            f_theory = math.sqrt(width*stdo**2*gelu_second_moment(q))
            a_projection = gain_o*rms_aggregate(head)
            energy_expected = vals['residual_in']**2+vals['attention']**2+vals['mlp']**2+2*(cross_ra+cross_sf)
            energy_err = abs(energy_expected-vals['residual_out']**2)/max(vals['residual_out']**2,1e-20)
            row = dict(arm=arm, layer=idx+1, mlp_width=width, embedding=emb, **vals,
                       attention_over_embedding=vals['attention']/emb,
                       mlp_over_embedding=vals['mlp']/emb,
                       attention_over_residual=vals['attention']/vals['residual_in'],
                       mlp_over_residual=vals['mlp']/vals['post_attention'],
                       attention_over_mlp=vals['attention']/vals['mlp'],
                       mlp_theory_conditional=f_theory, attention_projection_theory=a_projection,
                       mlp_theory_relative_error=f_theory/vals['mlp']-1,
                       attention_projection_relative_error=a_projection/vals['attention']-1,
                       energy_identity_relative_error=energy_err,
                       residual_attention_cosine=cross_ra/(vals['residual_in']*vals['attention']),
                       postatt_mlp_cosine=cross_sf/(vals['post_attention']*vals['mlp']),
                       measured_wi_rms=rms_aggregate(wi), measured_gelu_rms=rms_aggregate(gelu))
            rows.append(row)
            if idx == 0:
                qnorm = (stde*stde+a_prior*a_prior)/(stde*stde+a_prior*a_prior+c['normalization_layer_epsilon'])
                f_prior = math.sqrt(width*stdo**2*gelu_second_moment(d*stdw**2*qnorm))
                priors.append(dict(arm=arm, embedding_std=stde, seed_m_prior=mseed,
                                   attention_prior=a_prior, attention_measured=vals['attention'],
                                   mlp_prior=f_prior, mlp_measured=vals['mlp']))
        rr = [r for r in rows if r['arm']==arm]
        checks.append(dict(arm=arm, max_energy_identity_relative_error=max(r['energy_identity_relative_error'] for r in rr),
                           median_abs_mlp_theory_error=float(np.median([abs(r['mlp_theory_relative_error']) for r in rr])),
                           max_abs_mlp_theory_error=max(abs(r['mlp_theory_relative_error']) for r in rr),
                           max_abs_projection_theory_error=max(abs(r['attention_projection_relative_error']) for r in rr)))
    assert max(r['energy_identity_relative_error'] for r in rows) < 1e-5
    out=dict(scope=data['mha']['scope'], cohort=data['mha']['cohort'],
             input_ids_sha256=data['mha']['input_ids_sha256'],
             concentrations=con, first_layer_unconditional_priors=priors,
             checks=checks, rows=rows)
    (root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    with (root/'layers.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,9),sharex=True)
    keys=['attention_over_embedding','mlp_over_embedding','attention_over_residual','mlp_over_residual']
    titles=['Attention / embedding','MLP / embedding','Attention / incoming residual','MLP / post-attention residual']
    for arm in ORDER:
        rr=[r for r in rows if r['arm']==arm]
        for ax,key,title in zip(axes.flat,keys,titles):
            ax.plot([r['layer'] for r in rr],[r[key] for r in rr],label=LABELS[arm],
                    linestyle='--' if arm=='large_uncomp' else '-',marker='.',linewidth=1.5)
            ax.set_title(title);ax.set_yscale('log');ax.grid(alpha=.25)
            ax.set_xticks([1,3,6,9,12,15,18])
    for ax in axes[1]:ax.set_xlabel('Layer')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Initial forward: serial Pythia MediumProp, D1200, L18, T2048\n'
                 'RMS over 4 fixed Pile sequences; no training; log y axes')
    fig.tight_layout();fig.savefig(root/'initial-amplitude-ratios.png',dpi=180)
    fig.savefig(root/'initial-amplitude-ratios.pdf');plt.close(fig)
    print(json.dumps(dict(priors=priors,checks=checks),indent=2))


if __name__=='__main__':
    main()
