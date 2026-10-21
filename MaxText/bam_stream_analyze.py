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
    x=np.asarray(x,np.float64);gram=np.swapaxes(x,-1,-2)@x
    eigen=np.maximum(np.linalg.eigvalsh(gram),0);total=eigen.sum(-1)
    p=eigen/np.maximum(total[...,None],EPS)
    effective=np.exp(-np.sum(p*np.log(np.maximum(p,EPS)),axis=-1))
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
    result={}
    decay=float(cfg['bam_lambda_decay'])
    for i,r in layers.items():
        x=r['x_in'];y=r['x_out'];attention=r['attention_vector'];mlp=r['mlp_vector']
        before=r['M_in'];after=r['M_out'];delta=after-decay*before
        a,b=writes(r,cfg);rounding=delta-a-b
        accumulated_x+=attention+mlp;accumulated_m=decay*accumulated_m+delta
        all_a=decay*all_a+a;all_b=decay*all_b+b;all_round=decay*all_round+rounding
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
        for name in ['attention_write_gate','mlp_write_gate','m_shift_gate']:
            if name in r:
                z=r[name];row[name]=dict(mean=float(z.mean()),closed_fraction=float(np.mean(z<.05)),open_fraction=float(np.mean(z>.95)))
        if 'dense_M_shift_gate' in r:
            z=sigmoid(r['dense_M_shift_gate']+r['m_shift_bias']);row['m_shift_gate']=dict(mean=float(z.mean()),closed_fraction=float(np.mean(z<.05)),open_fraction=float(np.mean(z>.95)))
        result[str(i)]=row
    embedding_decay=decay**len(layers)
    return dict(layers=result,embedding=dict(vector_rms=rms(embedding['embedding_vector']),matrix_rms=rms(embedding['embedding_M']),
        vector_final_signed_share=projected(embedding['embedding_vector'],final_x),
        matrix_final_signed_share=projected(embedding_decay*embedding['embedding_M'],final_m)),
        closure=dict(vector_relative_rms=rms(accumulated_x-final_x)/max(rms(final_x),EPS),
                     matrix_relative_rms=rms(accumulated_m-final_m)/max(rms(final_m),EPS)),
        final_M_source_shares=dict(embedding=projected(embedding_decay*embedding['embedding_M'],final_m),
            attention=projected(all_a,final_m),mlp=projected(all_b,final_m),rounding=projected(all_round,final_m)))


def scalar_paths(tree,prefix=''):
    if isinstance(tree,dict):
        for key,value in tree.items():yield from scalar_paths(value,prefix+'/'+key)
    elif isinstance(tree,(float,int)):yield prefix,float(tree)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    root=Path(args.directory);meta=json.loads((root/'metadata.json').read_text());rows=[]
    for path in sorted(root.glob('sequence_*.npz')):
        with np.load(path,allow_pickle=False) as data:row=analyze_sequence({k:data[k] for k in data.files},meta['config'])
        rows.append(row);print('ANALYZED',path.name,flush=True)
    paths={}
    for row in rows:
        for key,value in scalar_paths(row):paths.setdefault(key,[]).append(value)
    aggregate={key:dict(mean=float(np.mean(value)),std=float(np.std(value)),min=float(np.min(value)),max=float(np.max(value))) for key,value in paths.items()}
    report=dict(experiment=meta['experiment'],checkpoint=meta['checkpoint'],length=meta['length'],sequences=len(rows),
        config={k:meta['config'].get(k) for k in ['emb_dim','num_decoder_layers','num_query_heads','bam_k','bam_v','bam_mlp_write_every','bam_write_rms_epsilon','normalization_layer_epsilon','bam_local_full_m_read_bottleneck_dim']},
        sequence_hashes=meta['sequence_hashes'],losses=meta['losses'],ablations=meta.get('ablations'),aggregate=aggregate,per_sequence=rows)
    (root/'stream_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ANALYSIS_COMPLETE',root,len(rows),flush=True)

if __name__=='__main__':main()
