"""用小型CPU算例核查累积、裁剪及恢复采样；不产生论文实验指标。"""
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch
import torch
import run
from models import SPECS,lr_for
from revision.dataset import StepBatchSampler


def main():
    """比较一次整批更新和四次累积更新，核查固定日程及恢复位置。"""
    torch.set_num_threads(2);torch.manual_seed(71)
    x=torch.randn(16,3);y=torch.randn(16,2)
    base=torch.nn.Linear(3,2)
    differences=[]
    for clip in [.01,None]:
        a=copy.deepcopy(base);b=copy.deepcopy(base)
        oa=torch.optim.Adam(a.parameters(),lr=1e-3);ob=torch.optim.Adam(b.parameters(),lr=1e-3)
        objective=lambda model,pred,target:torch.nn.functional.mse_loss(pred,target)
        full=iter([{'input':x,'target':y}])
        micro=iter([{'input':x[i:i+4],'target':y[i:i+4]} for i in range(0,16,4)])
        with patch.object(torch.Tensor,'cuda',lambda self,*args,**kwargs:self):
            la,na=run.update(a,oa,objective,full,1,clip)
            lb,nb=run.update(b,ob,objective,micro,4,clip)
        assert abs(la-lb)<1e-6 and abs(na-nb)<1e-6
        delta=max(float((p-q).abs().max()) for p,q in zip(a.parameters(),b.parameters()))
        assert delta<1e-7;differences.append(delta)
    records=[{'task':t} for t in ['rain','low','snow'] for _ in range(3)]
    all_batches=list(StepBatchSampler(records,4,4,5,42))
    resumed=list(StepBatchSampler(records,4,4,5,42,3))
    assert resumed==all_batches[12:] and len(resumed)==8
    schedules={}
    for name,spec in SPECS.items():
        warm=round(40000*spec['warmup_fraction'])
        rates=[lr_for(i,40000,spec,warm) for i in range(40000)]
        assert all(math.isfinite(v) and v>=0 for v in rates)
        schedules[name]={'warmup':warm,'first':rates[0],'maximum':max(rates),'last':rates[-1]}
    result={'status':'passed','purpose':'CPU_control_verification_not_model_experiment','accumulation_parameter_max_difference':differences,'sampler_resume':'exact_match','schedules':schedules,'limitation':'梯度累积等价性仅对本算例成立；MoCE批内路由平衡等依赖物理batch的项不保证与大batch等价。'}
    output=Path(__file__).resolve().parent/'doc/control_checks.json'
    output.parent.mkdir(exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
