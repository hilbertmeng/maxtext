import pathlib,json,hashlib,os
os.environ['TF_NUM_INTRAOP_THREADS']='2';os.environ['TF_NUM_INTEROP_THREADS']='2'
import tensorflow as tf,numpy as np
src='gs://newproject-1-common_datasets_europe-west4/pythia_model_test/pile_test/val_with_eos.tfrecord'
rng=np.random.default_rng(20261031);pool=[]
for i,raw in enumerate(tf.compat.v1.io.tf_record_iterator(src)):
 e=tf.train.Example.FromString(raw);a=np.asarray(e.features.feature['input_ids'].int64_list.value,dtype=np.int32)
 a=np.trim_zeros(a,trim='b')
 if len(a):pool.append(a)
 if len(pool)>=1024:break
print('SOURCE_LENGTHS',len(pool),[(int(x),int(n)) for x,n in zip(*np.unique([len(a) for a in pool],return_counts=True))],flush=True)
order=rng.permutation(len(pool));stream=np.concatenate([pool[i] for i in order]);width=4097;n=len(stream)//width
chunks=stream[:n*width].reshape(n,width);sel=rng.choice(n,size=32,replace=False);tokens=chunks[sel]
root=pathlib.Path('/tmp/h-write-balance-20261031');root.mkdir(exist_ok=True)
np.savez_compressed(root/'cohort.npz',tokens=tokens)
m=dict(source=src,seed=20261031,record_shuffle_order=order.tolist(),chunk_indices=sel.tolist(),tokens_shape=list(tokens.shape),sequence_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in tokens],common2048_hashes=[hashlib.sha256(a[:2049].tobytes()).hexdigest() for a in tokens],description='Shuffle1024 validation records, pack contiguous4097-token chunks, randomly select32 before batching; both models receive identical first2048 tokens.')
(root/'cohort.json').write_text(json.dumps(m,indent=2)+'\n');print('COHORT_READY',tokens.shape,flush=True)
