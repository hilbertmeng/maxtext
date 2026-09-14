"""Exact L/F two-group Shapley for joint QK removal; paired type contrasts."""
import argparse,json
from pathlib import Path
import numpy as np
from summarize_row_contribution import stats

def run(root):
 with np.load(root/'qktype/paired_results.npz') as f:
  names=list(f['names']);g=np.asarray(f['gap'],float)
 at=lambda name:g[:,names.index(name)]
 left=at('QK_L_off');right=at('QK_F_off');whole=at('QK_all_off')
 phi_l=(left+whole-right)/2;phi_f=(right+whole-left)/2
 np.testing.assert_allclose(phi_l+phi_f,whole,atol=1e-12,rtol=0)
 result=dict(joint_qk=dict(L_minus_F=stats(left-right),interaction=stats(whole-left-right),shapley_L=stats(phi_l),shapley_F=stats(phi_f)),Q_L_minus_F=stats(at('Q_L_off')-at('Q_F_off')),K_L_minus_F=stats(at('K_L_off')-at('K_F_off')))
 metadata=json.loads((root/'qktype/qktype_000_064_metadata.json').read_text())
 costs={'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow':(1.,.5),
        'BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis':(2.,1.)}
 cost_l,cost_f=costs[metadata['model']]
 result['row_key_kernel_budget']=dict(L_WQ=cost_l,F_WQ=cost_f,
     shapley_L_per_WQ=stats(phi_l/cost_l),shapley_F_per_WQ=stats(phi_f/cost_f),
     paired_L_minus_F_per_WQ=stats(phi_l/cost_l-phi_f/cost_f))
 (root/'qktype/layer_type_allocation.json').write_text(json.dumps(result,indent=2))
 np.savez_compressed(root/'qktype/layer_type_allocation.npz',shapley_L=phi_l,shapley_F=phi_f,L_minus_F=left-right)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(run(a.root),indent=2))
