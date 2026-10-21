"""Host-only residual stream, signed source attribution and address-usage statistics."""
import argparse
import json
from pathlib import Path
import re
import numpy as np

EPS=1e-20


def rms(x):return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
def cosine(a,b):
    a=np.asarray(a,np.float64).reshape(-1);b=np.asarray(b,np.float64).reshape(-1)
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),EPS))
def projected(a,b):return float(np.sum(a.astype(np.float64)*b)/max(np.sum(b.astype(np.float64)**2),EPS))
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-80,80)))
def norm(x,epsilon):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+epsilon)


def usage(energy):
    energy=np.asarray(energy,np.float64);total=energy.sum()
    p=energy/max(total,EPS);n=len(p)
    entropy=-float(np.sum(p*np.log(np.maximum(p,EPS))))
    return dict(energy=energy.tolist(),normalized_energy=p.tolist(),effective_coordinates=float(np.exp(entropy)) if total>EPS else 0.,
        effective_fraction=float(np.exp(entropy)/n) if total>EPS else 0.,
        top1_energy_fraction=float(p.max()),top5_energy_fraction=float(np.sort(p)[-min(5,n):].sum()),
        low_use_fraction=float(np.mean(p<.1/n)))


def matrix_rank(x):
    x=np.asarray(x,np.float64)
    gram=x@np.swapaxes(x,-1,-2) if x.shape[-2]<x.shape[-1] else np.swapaxes(x,-1,-2)@x
    eigen=np.maximum(np.linalg.eigvalsh(gram),0);total=eigen.sum(-1)
    p=eigen/np.maximum(total[...,None],EPS)
    effective=np.where(total>EPS,np.exp(-np.sum(p*np.log(np.maximum(p,EPS)),axis=-1)),0.)
    stable=total/np.maximum(eigen[...,-1],EPS);rank=min(x.shape[-2:])
    return dict(effective_rank=float(effective.mean()),effective_rank_fraction=float((effective/rank).mean()),
        stable_rank=float(stable.mean()),stable_rank_fraction=float((stable/rank).mean()),
        top1_energy_fraction=float(p[...,-1].mean()),top5_energy_fraction=float(p[...,-min(5,rank):].sum(-1).mean()))


def records(arrays,cfg):
    groups={};embedding={};block=int(cfg.get('bam_local_fetch_block_size') or 2)
    tails=cfg.get('bam_final_local_layer_count')
    if tails is None:tails=int(bool(cfg.get('bam_extra_final_local_layer')))
    layers=int(cfg['num_decoder_layers'])
    for path,value in arrays.items():
        parts=path.split('__');metric=parts[-1]
        if not metric.startswith('stream_'):continue
        metric=metric[len('stream_'):]
        if 'embedding_bam_write' in parts:embedding[metric]=value;continue
        if 'layers' in parts:
            local=next((p for p in parts if re.fullmatch(r'(local|fetch)_\d+',p)),None)
            if local:offset=int(local.rsplit('_',1)[1])
            elif 'local' in parts:offset=0
            elif 'fetch' in parts:offset=block-1
            else:raise ValueError(path)
            for i in range(value.shape[0]):groups.setdefault(i*block+offset,{})[metric]=value[i]
        else:
            tail=next((p for p in parts if p.startswith('final_local_layer')),None)
            layer=next((p for p in parts if re.fullmatch(r'layers_\d+',p)),None)
            if tail:index=layers-tails+(int(tail.rsplit('_',1)[1]) if tail!='final_local_layer' else 0)
            elif layer:index=int(layer.split('_')[-1])
            else:continue
            groups.setdefault(index,{})[metric]=value
    if set(groups)!=set(range(layers)):raise ValueError(f'Missing logical layers: {set(range(layers))-set(groups)}')
    return embedding,dict(sorted(groups.items()))


def outer(content,address):return np.einsum('bsnk,bsnv->bskv',content,address,optimize=True)


def writes(record,cfg):
    eps=cfg.get('bam_write_rms_epsilon') or cfg['normalization_layer_epsilon']
    heads=int(cfg['num_query_heads']);k=int(cfg['bam_k'])
    scale=1/np.sqrt(heads) if cfg['bam_sqrt_n_scale'] else 1
    if 'attention_write_content' in record:
        a=outer(record['attention_write_content'],record['attention_write_address'])
    elif 'attention_write_data_raw' in record:
        content=scale*sigmoid(record['attention_write_logits'])[...,None]*norm(record['attention_write_data_raw'],eps)
        a=outer(content,norm(record['attention_write_address_raw'],eps))
    elif 'attention_write_delta_M' in record:a=record['attention_write_delta_M']
    else:raise ValueError('Missing attention write measurement')
    if 'mlp_write_content' in record:
        b=outer(record['mlp_write_content'],record['mlp_write_address'])
    elif 'dense_mlp_write_gate' in record:
        logits=record['dense_mlp_write_gate']+record['mlp_write_bias']
        data=record['mlp_vector'].reshape(record['mlp_vector'].shape[:2]+(heads,k))
        b=outer(scale*sigmoid(logits)[...,None]*norm(data,eps),norm(record['dense_mlp_address_up'],eps))
    else:b=np.zeros_like(a)
    return a,b


def read_statistics(record,cfg):
    heads=int(cfg['num_query_heads']);v=int(cfg['bam_v']);out={}
    mapping={'q':('W_lq_c8','W_lq_gate'),'k':('W_lk_c8','W_lk_gate'),
             'v':('W_R','W_lv_gate'),'o':('W_R','W_R_gate')}
    for arm,(keyname,gatename) in mapping.items():
        raw=record['read_key_'+keyname]
        key=raw.reshape(raw.shape[:2]+(heads,v))
        gate=record['read_gate_'+gatename].reshape(raw.shape[:2]+(heads,))
        logits=record['read_logits_'+gatename].reshape(gate.shape)
        dynamic=float(record['read_key_scale'])*gate[...,None]*norm(key,float(record['read_key_epsilon']))
        static_gate=record['static_gate_general_'+arm+'_static_gate'].reshape(gate.shape)
        static=record['static_'+arm+'_key'].T[None,None]*(static_gate/float(cfg['bam_static_read_gate_init']))[...,None]
        effective=dynamic+static
        probability=sigmoid(logits)
        flat=effective.reshape(-1,v);mean=effective.mean(axis=1,keepdims=True)
        out[arm]=dict(dynamic_rms=rms(dynamic),static_rms=rms(static),combined_rms=rms(effective),
            dynamic_static_cosine=cosine(dynamic,static),gate_mean=float(probability.mean()),
            gate_closed_fraction=float(np.mean(probability<.05)),gate_open_fraction=float(np.mean(probability>.95)),
            token_mean_energy_fraction=float(np.mean(mean**2)/max(np.mean(effective**2),EPS)),
            address_usage=usage(np.mean(flat**2,axis=0)),key_rank=matrix_rank(flat[None]))
    return out


def analyze_sequence(arrays,cfg):
    embedding,layers=records(arrays,cfg);final=layers[max(layers)]
    final_x=final['x_out'];final_m=final['M_out']
    accumulated_x=embedding['embedding_vector'].astype(np.float64)
    accumulated_m=embedding['embedding_M'].astype(np.float64)
    all_a=np.zeros_like(final_m);all_b=np.zeros_like(final_m);all_round=np.zeros_like(final_m)
    ref_m=final['M_in'];prefix_a=np.zeros_like(ref_m);prefix_b=np.zeros_like(ref_m);prefix_round=np.zeros_like(ref_m)
    result={}
    decay=float(cfg['bam_lambda_decay'])
    for i,r in layers.items():
        x=r['x_in'];y=r['x_out'];attention=r['attention_vector'];mlp=r['mlp_vector']
        before=r['M_in'];after=r['M_out'];delta=after-decay*before
        a,b=writes(r,cfg);rounding=delta-a-b
        accumulated_x+=attention+mlp;accumulated_m=decay*accumulated_m+delta
        all_a=decay*all_a+a;all_b=decay*all_b+b;all_round=decay*all_round+rounding
        if i<max(layers):
            prefix_a=decay*prefix_a+a;prefix_b=decay*prefix_b+b;prefix_round=decay*prefix_round+rounding
        row=dict(vector=dict(input_rms=rms(x),output_rms=rms(y),attention_rms=rms(attention),mlp_rms=rms(mlp),
                    attention_to_input=rms(attention)/max(rms(x),EPS),mlp_to_input=rms(mlp)/max(rms(x),EPS),
                    attention_input_cosine=cosine(attention,x),mlp_input_cosine=cosine(mlp,x+attention),
                    closure_relative_rms=rms(y-x-attention-mlp)/max(rms(y),EPS),
                    attention_final_signed_share=projected(attention,final_x),mlp_final_signed_share=projected(mlp,final_x)),
                 matrix=dict(input_rms=rms(before),output_rms=rms(after),delta_rms=rms(delta),
                    delta_to_input=rms(delta)/max(rms(before),EPS),attention_write_rms=rms(a),mlp_write_rms=rms(b),
                    attention_mlp_write_cosine=cosine(a,b),write_history_cosine=cosine(delta,before),
                    write_reconstruction_relative_rms=rms(rounding)/max(rms(a+b),EPS),
                    unchanged_coordinate_fraction=float(np.mean(delta==0)),rank=matrix_rank(after),
                    address_usage=usage(np.mean(after**2,axis=(0,1,2))),
                    attention_final_signed_share=projected(a,final_m),mlp_final_signed_share=projected(b,final_m),
                    rounding_final_signed_share=projected(rounding,final_m)),
                 reads=read_statistics(r,cfg))
        gate_values={name:r[name] for name in ['attention_write_gate','mlp_write_gate','m_shift_gate'] if name in r}
        if 'attention_write_gate' not in gate_values and 'attention_write_logits' in r:
            gate_values['attention_write_gate']=sigmoid(r['attention_write_logits'])
        if 'mlp_write_gate' not in gate_values and 'dense_mlp_write_gate' in r:
            gate_values['mlp_write_gate']=sigmoid(r['dense_mlp_write_gate']+r['mlp_write_bias'])
        for name,z in gate_values.items():
            row[name]=dict(mean=float(z.mean()),closed_fraction=float(np.mean(z<.05)),open_fraction=float(np.mean(z>.95)))
        if 'dense_M_shift_gate' in r:
            z=sigmoid(r['dense_M_shift_gate']+r['m_shift_bias']);row['m_shift_gate']=dict(mean=float(z.mean()),closed_fraction=float(np.mean(z<.05)),open_fraction=float(np.mean(z>.95)))
        # Coordinate energies describe signed address factors, not routing probabilities.
        attention_address=r.get('attention_write_address')
        if attention_address is None and 'attention_write_address_raw' in r:
            epsilon=cfg.get('bam_write_rms_epsilon') or cfg['normalization_layer_epsilon']
            attention_address=norm(r['attention_write_address_raw'],epsilon)
        mlp_address=r.get('mlp_write_address')
        if mlp_address is None and 'dense_mlp_address_up' in r:
            epsilon=cfg.get('bam_write_rms_epsilon') or cfg['normalization_layer_epsilon']
            mlp_address=norm(r['dense_mlp_address_up'],epsilon)
        for source,address in [('attention',attention_address),('mlp',mlp_address)]:
            if address is not None:
                row['matrix'][source+'_address_usage']=usage(np.mean(address**2,axis=tuple(range(address.ndim-1))))
                row['matrix'][source+'_address_rank']=matrix_rank(address.reshape(-1,address.shape[-1])[None])
                content=r.get(source+'_write_content')
                if content is None:
                    epsilon=cfg.get('bam_write_rms_epsilon') or cfg['normalization_layer_epsilon']
                    raw=r['attention_write_data_raw'] if source=='attention' else r['mlp_vector'].reshape(r['mlp_vector'].shape[:2]+(int(cfg['num_query_heads']),int(cfg['bam_k'])))
                    content=norm(raw,epsilon)*gate_values[source+'_write_gate'][...,None]
                active=address*np.sqrt(np.mean(content**2,axis=-1,keepdims=True))
                row['matrix'][source+'_active_address_usage']=usage(np.mean(active**2,axis=tuple(range(active.ndim-1))))
                row['matrix'][source+'_content_rank']=matrix_rank(content)
                row['matrix'][source+'_per_token_address_rank']=matrix_rank(active)
                row['matrix'][source+'_update_rank']=matrix_rank(a if source=='attention' else b)
                row['matrix'][source+'_head_activity']=usage(np.mean(content**2,axis=(0,1,3)))
        result[str(i)]=row
    embedding_decay=decay**len(layers)
    return dict(layers=result,embedding=dict(vector_rms=rms(embedding['embedding_vector']),matrix_rms=rms(embedding['embedding_M']),
        matrix_rank=matrix_rank(embedding['embedding_M']),
        write_gate_mean=float(sigmoid(embedding['dense_W_emb_g']+embedding['embedding_gate_bias']).mean()),
        write_gate_bias_mean=float(embedding['embedding_gate_bias'].mean()),
        matrix_address_usage=usage(np.mean(embedding['embedding_M']**2,axis=(0,1,2))),
        vector_final_signed_share=projected(embedding['embedding_vector'],final_x),
        matrix_final_signed_share=projected(embedding_decay*embedding['embedding_M'],final_m)),
        closure=dict(vector_relative_rms=rms(accumulated_x-final_x)/max(rms(final_x),EPS),
                     matrix_relative_rms=rms(accumulated_m-final_m)/max(rms(final_m),EPS)),
        last_read_carry_source_shares=dict(embedding=projected(decay**(len(layers)-1)*embedding['embedding_M'],ref_m),
            attention=projected(prefix_a,ref_m),mlp=projected(prefix_b,ref_m),rounding=projected(prefix_round,ref_m)),
        final_M_source_shares=dict(embedding=projected(embedding_decay*embedding['embedding_M'],final_m),
            attention=projected(all_a,final_m),mlp=projected(all_b,final_m),rounding=projected(all_round,final_m)))


def scalar_paths(tree,prefix=''):
    if isinstance(tree,dict):
        for key,value in tree.items():yield from scalar_paths(value,prefix+'/'+key)
    elif isinstance(tree,(float,int)):yield prefix,float(tree)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    root=Path(args.directory);meta=json.loads((root/'metadata.json').read_text());rows=[]
    native_verification=root/'native_baseline_verification.json'
    if native_verification.exists():
        verified=json.loads(native_verification.read_text())
        meta['losses']=verified['native_losses']
        for result in meta.get('ablations',{}).values():
            delta=np.asarray(result['losses'])-np.asarray(meta['losses'])
            result.update(paired_delta=delta.tolist(),mean_delta=float(delta.mean()),sem=float(delta.std(ddof=1)/np.sqrt(len(delta))))
    paths=sorted(root.glob('sequence_*.npz'))
    if len(paths)!=meta['sequences']:raise ValueError('Incomplete sequence artifacts')
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            arrays={k:data[k] for k in data.files}
        if not all(np.isfinite(value).all() for value in arrays.values()):raise ValueError('Nonfinite activation in '+str(path))
        _,layers=records(arrays,meta['config'])
        for i in range(1,len(layers)):
            for before,after in [('M_in','M_out'),('x_in','x_out')]:
                if not np.array_equal(layers[i][before],layers[i-1][after]):
                    raise ValueError(f'Layer continuity mismatch: {path.name} layer{i} {before}')
        row=analyze_sequence(arrays,meta['config'])
        rows.append(row);print('ANALYZED',path.name,flush=True)
    paths={}
    for row in rows:
        for key,value in scalar_paths(row):paths.setdefault(key,[]).append(value)
    aggregate={key:dict(mean=float(np.mean(value)),std=float(np.std(value)),min=float(np.min(value)),max=float(np.max(value))) for key,value in paths.items()}
    report=dict(validated_finiteness_and_layer_continuity=True,experiment=meta['experiment'],checkpoint=meta['checkpoint'],length=meta['length'],sequences=len(rows),
        config={k:meta['config'].get(k) for k in ['emb_dim','num_decoder_layers','num_query_heads','bam_k','bam_v','bam_mlp_write_every','bam_write_rms_epsilon','normalization_layer_epsilon','bam_local_full_m_read_bottleneck_dim']},
        sequence_hashes=meta['sequence_hashes'],losses=meta['losses'],ablations=meta.get('ablations'),aggregate=aggregate,per_sequence=rows)
    (root/'stream_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ANALYSIS_COMPLETE',root,len(rows),flush=True)

if __name__=='__main__':main()
