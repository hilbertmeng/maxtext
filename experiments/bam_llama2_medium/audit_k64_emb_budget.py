import sys,json,importlib.util
sys.path.insert(0,'MaxText')
import exp
spec=importlib.util.spec_from_file_location('audit','experiments/bam_llama2_medium/audit_matched_mlp.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
parent=exp.BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE
out=[]
for d in [1024,1000,1001]:
 name=f'BudgetD{d}'
 setattr(exp,name,type(name,(parent,),dict(model_name=name,base_emb_dim=d,base_mlp_dim=2816,mlp_dim_by_block=None)))
 out.append(mod.audit(name))
out.append(mod.audit('BamMHALlama2MediumC256ScanAotCleanControl'))
step=out[2]['total']-out[1]['total'];target=out[3]['total'];intercept=out[1]['total']-1000*step
print('FORMULA',step,intercept,'target',target,'root',(target-intercept)/step,flush=True)
d=(target-intercept)//step
for v in [d,d+1]:
 name=f'BudgetD{v}';setattr(exp,name,type(name,(parent,),dict(model_name=name,base_emb_dim=v,base_mlp_dim=2816,mlp_dim_by_block=None)));out.append(mod.audit(name))
json.dump(out,open('/tmp/k64-emb-budget-audit.json','w'),indent=2)
for x in out:print('RESULT',x['exp'],x['total'],x['groups'],flush=True)
