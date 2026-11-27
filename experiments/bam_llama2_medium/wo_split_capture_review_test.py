import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from wo_boundary_review_test import ReviewTests
from wo_split_capture_review import review_split_workers

class SplitReviewTests(unittest.TestCase):
    def fixture(self,root):
        files,ref=ReviewTests().fixture(root)
        for worker in root.iterdir():
            for step in (35635,35636):
                p=worker/'full'/f'step{step}';r=json.loads((p/'wo-split.json').read_text())
                ctx=dict(primal_wire_layout='producer_bnth_bnht',run='test-run',state=str(step),input=hashlib.sha256(worker.name.encode()).hexdigest(),rng='a'*64)
                r.update(primal_wire_layout='producer_bnth_bnht',primal_execution='value_and_grad',forward_vector_relative_error=0,forward_W_O_leaf_relative_errors=[0]*6,capture_method='split_execution_v1',forward_snapshot_loss=r['plain_loss'],snapshot_context=ctx)
                (p/'wo-split.json').write_text(json.dumps(r))
                np.save(p/'forward-wo.npy',np.load(p/'plain-wo.npy'))
                records=json.loads((p/'boundary-manifests.json').read_text())
                for rec in records:
                    meta=rec['metadata'];identity=dict(shape=[32],dtype='int32',bf16_bits=False,sha256='rows')
                    mask=dict(shape=[32,1212],dtype='bool',bf16_bits=False,sha256='mask')
                    payload=dict(shape=[32,1212,1,1],dtype='bfloat16',bf16_bits=True,sha256='tensor')
                    meta.update(phase='primal',context=ctx,files=dict(AV=payload,localO=payload,H=payload,global_rows=identity,valid=mask))
                    rec['cotangent_metadata']={**meta,'phase':'cotangent','files':dict(G=payload,global_rows=identity,valid=mask)}
                (p/'boundary-manifests.json').write_text(json.dumps(records))
                if step==35635:
                    np.save(p/'native-aot-wo.npy',np.load(p/'plain-wo.npy'))
                    (p/'fidelity-diagnostic.json').write_text(json.dumps(dict(native_moment_consistency=0)))
        return files,ref

    def test_review_requires_matching_split_context_masks_forward_and_native(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);files,ref=self.fixture(root)
            self.assertTrue(review_split_workers(root,files,ref)['passed'])
            p=root/'worker0/full/step35635';rp=p/'wo-split.json';original=rp.read_text();r=json.loads(original);r['forward_snapshot_loss']+=.01;rp.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'forward loss'):review_split_workers(root,files,ref)
            rp.write_text(original);r=json.loads(original);r['primal_wire_layout']='canonical';rp.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'wire-layout'):review_split_workers(root,files,ref)
            rp.write_text(original);mp=p/'boundary-manifests.json';original=mp.read_text();rows=json.loads(original);rows[0]['cotangent_metadata']['context']={**rows[0]['metadata']['context'],'state':'other'};mp.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'context'):review_split_workers(root,files,ref)
            rows=json.loads(original);rows[0]['cotangent_metadata']['files']['valid']['sha256']='different';mp.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'row/mask'):review_split_workers(root,files,ref)
            mp.write_text(original);np.save(p/'forward-wo.npy',np.load(p/'plain-wo.npy')*2)
            with self.assertRaisesRegex(ValueError,'Snapshot actual leaf'):review_split_workers(root,files,ref)
            np.save(p/'forward-wo.npy',np.load(p/'plain-wo.npy'))
            np.save(p/'native-aot-wo.npy',np.load(p/'plain-wo.npy')*2)
            with self.assertRaisesRegex(ValueError,'native per-leaf'):review_split_workers(root,files,ref)

if __name__=='__main__':unittest.main()
