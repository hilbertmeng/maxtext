import hashlib,json,tempfile,unittest,shutil
from pathlib import Path
import numpy as np
from wo_boundary_decomposition import partition_sums
from wo_boundary_review import review_workers,EXPECTED

class ReviewTests(unittest.TestCase):
    def fixture(self,root):
        rng=np.random.default_rng(9);a=rng.uniform(.5,1.5,size=(128,1212,1,1)).astype('float32');o=-.3*a;h=a+o
        g=rng.uniform(.01,.02,size=(128,1212,1)).astype('float32');valid=np.ones((128,1212),bool)
        files={'probe.py':hashlib.sha256(b'version1').hexdigest()};reference=[]
        total=np.einsum('btnh,btd->nhd',h.astype('float64'),g.astype('float64'))
        for worker in range(4):
            p=root/f'worker{worker}';(p/'runner').mkdir(parents=True);(p/'runner/probe.py').write_bytes(b'version1');(p/'exit-code.txt').write_text('0\n')
            rows=np.arange(worker,128,4);part=partition_sums(a[rows],o[rows],h[rows],g[rows],rows,valid[rows])
            shards=[dict(index=f'shard-{worker}-{i}',sha256=hashlib.sha256(f'{worker}-{i}'.encode()).hexdigest())for i in range(4)];reference+=shards
            for step,(loss,raw)in EXPECTED.items():
                d=p/'full'/f'step{step}';d.mkdir(parents=True)
                np.save(d/'actual-wo.npy',np.stack([total]*6,axis=1));np.save(d/'plain-wo.npy',np.stack([total]*6,axis=1))
                report=dict(step=step,complete=True,capture_complete=True,scientific_review_complete=False,
                    native_control={'learning/loss':loss,'learning/raw_grad_norm':raw},plain_loss=loss,plain_raw=raw,instrumented_loss=loss,instrumented_vector_relative_error=0,
                    W_O_leaf_relative_errors=[0]*6,local_rows=rows.tolist(),input_shards=shards)
                (d/'wo-split.json').write_text(json.dumps(report))
                (d/'boundary-manifests.json').write_text(json.dumps([dict(metadata=dict(step=step,layer=i,global_rows=rows.tolist()))for i in range(6)]))
                for layer in range(6):np.savez(d/f'layer-{layer:02d}-partial.npz',**part)
        return files,reference

    def test_full_review_and_bad_input_or_partial_or_worker(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);files,ref=self.fixture(root)
            report=review_workers(root,files,ref);self.assertTrue(report['passed']);self.assertEqual(len(report['rows']),18)
            self.assertLess(report['rows'][0]['normal']['O_projection_fraction'],0)
            bad=[dict(x)for x in ref];bad[0]['sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'input16'):review_workers(root,files,bad)
            p=root/'worker0/full/step35635/layer-00-partial.npz'
            with np.load(p)as f:part={k:f[k]for k in f.files}
            original=part['parts'].copy();part['parts']*=2;np.savez(p,**part)
            with self.assertRaisesRegex(ValueError,'H\^T G'):review_workers(root,files,ref)
            part['parts']=original;np.savez(p,**part);part['global_rows'][0]=1;np.savez(p,**part)
            with self.assertRaisesRegex(ValueError,'row identity'):review_workers(root,files,ref)
            shutil.rmtree(root/'worker3')
            with self.assertRaisesRegex(ValueError,'four workers'):review_workers(root,files,ref)

if __name__=='__main__':unittest.main()
