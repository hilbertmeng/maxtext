"""Geometric evidence only; GELU feature sharing requires a functional probe."""
import argparse
import json
from pathlib import Path
import numpy as np

def compare(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    qa=np.linalg.qr(a,mode='reduced')[0]
    qb=np.linalg.qr(b,mode='reduced')[0]
    cos=np.linalg.svd(qa.T@qb,compute_uv=False)
    ea=np.sum((qb.T@a)**2)/np.sum(a*a)
    eb=np.sum((qa.T@b)**2)/np.sum(b*b)
    joint=np.concatenate((a/np.linalg.norm(a),b/np.linalg.norm(b)),axis=1)
    eigen=np.linalg.eigvalsh(joint.T@joint).clip(0)
    return {'overlap':float(np.mean(cos*cos)),
       'principal_cos_p10':float(np.quantile(cos,.1)),
       'principal_cos_median':float(np.median(cos)),
       'a_energy_in_b':float(ea),'b_energy_in_a':float(eb),
       'joint_rank256_energy':float(eigen[-a.shape[1]:].sum()/eigen.sum()),
       'joint_rank384_energy':float(eigen[-384:].sum()/eigen.sum()),
       'raw_cosine':float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))),
       'random_overlap_reference':a.shape[1]/a.shape[0]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('arrays');p.add_argument('output');args=p.parse_args()
    arrays=dict(np.load(args.arrays))
    result=[]
    prefix='params/params/decoder/layers/'
    for group,offset in [('local_0',0),('local_1',1),('fetch_2',2)]:
        root=prefix+group+'/block/'
        read=arrays[root+'self_attention/full_m_read_down/kernel']
        attn=arrays[root+'self_attention/P_loc_down/kernel']
        assert read.shape==attn.shape==(1200,6,256)
        for scan in range(6):
            pairs=[('read_attn',read[:,scan],attn[:,scan])]
            if group=='local_1':
                mlp=arrays[root+'mlp_address_down/kernel'][:,scan]
                pairs += [('attn_mlp',attn[:,scan],mlp),('read_mlp',read[:,scan],mlp)]
            for name,a,b in pairs:
                row={'pair':name,'layer':3*scan+offset,
                     'active_write_consumer':3*scan+offset < 17, **compare(a,b)}
                result.append(row)
                print(json.dumps(row),flush=True)
    summary={name:{metric:float(np.mean([r[metric] for r in result if r['pair']==name]))
       for metric in ['overlap','a_energy_in_b','b_energy_in_a','joint_rank256_energy','joint_rank384_energy','raw_cosine']}
       for name in ['read_attn','attn_mlp','read_mlp']}
    active={name:{metric:float(np.mean([r[metric] for r in result
               if r['pair']==name and r['active_write_consumer']]))
       for metric in ['overlap','a_energy_in_b','b_energy_in_a','joint_rank256_energy','joint_rank384_energy','raw_cosine']}
       for name in ['read_attn','attn_mlp','read_mlp']}
    Path(args.output).write_text(json.dumps({'layers':result,'mean_all':summary,'mean_active':active},indent=2)+'\n')
    print('MEAN_ACTIVE',json.dumps(active),flush=True)

if __name__=='__main__':main()
