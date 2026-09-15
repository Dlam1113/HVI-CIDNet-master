"""CDD-11单个对比模型的探测、测速、训练与最终测试入口。"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
SUPPORT=HERE/'snapshot/main'
sys.path.insert(0,str(SUPPORT if SUPPORT.exists() else Path('/home/Bjj/HVI-CIDNet-clean')))
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import SPECS,build,Objective,optimizer_for,lr_for,digest
from revision.dataset import PairedManifestDataset,StepBatchSampler,assert_no_scene_overlap
from revision.experiment import seed_all
from revision.evaluation import evaluate,save_evaluation

PROJECT=Path('/home/Bjj/HVI-CIDNet-clean')
DATA=PROJECT/'datasets/CDD-11/official'
MANIFEST=PROJECT/'datasets/CDD-11/manifests/600_100_seed20260910'
EXPECTED={'train':'4e68cf29bd587d80c2aa6f8a6b9ad20e32bca1bb3f9133fde684c2a536038040','val':'2f15161b2dc8a46dea0bc8047b0b42e941f7d314ffa26ecd6af2235b3f961e3f','test':'2ffb4abb99fb0b426491869d5351dbe2c686aff6aa8a9907ca931b2773a6cae4'}


def write(path,data):
    """原子写入本次独立输出，防止监测器读到半份JSON。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def source_stamp():
    """固定运行工具与快照文件的来源，用于续训前拒绝静默变更。"""
    paths=[HERE/'run.py',HERE/'models.py']
    manifest=HERE/'snapshot_manifest.json'
    if manifest.exists():
        paths.append(manifest)
        for relative,sha in json.loads(manifest.read_text())['files'].items():
            assert digest(HERE/relative)==sha,'只读源码快照发生变化：'+relative
    if (HERE/'vendor_manifest.json').exists():paths.append(HERE/'vendor_manifest.json')
    return {str(p):digest(p) for p in paths}


def datasets():
    """核实正式清单和内容分组，不重新划分或引入其他训练图像。"""
    for split,sha in EXPECTED.items():
        assert digest(MANIFEST/f'all11_{split}.json')==sha,split+'清单变化'
    train=PairedManifestDataset(DATA,MANIFEST/'all11_train.json',256)
    val=PairedManifestDataset(DATA,MANIFEST/'all11_val.json')
    assert_no_scene_overlap(train,val)
    assert len(train)==6600 and len(val)==1100
    test=json.loads((MANIFEST/'all11_test.json').read_text())
    test_keys={r['target_sha256_rgb'] for r in test['records']}
    assert not test_keys & {r['target_sha256_rgb'] for r in train.records+val.records}
    assert len(test['records'])==2200 and len(test_keys)==200
    return train,val


def rng_state():
    """保存采样之外的随机状态，保留MoCE随机路由的可恢复性。"""
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    """恢复检查点所记录的CPU及GPU随机状态。"""
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def evaluation_rng():
    """用固定种子评估随机路由，结束后恢复训练随机序列。"""
    state=rng_state()
    try:seed_all(42);yield
    finally:restore_rng(state)


def save_checkpoint(path,model,opt,step,best,config,source):
    """原子保存可恢复状态，不覆盖其他实验或源模型权重。"""
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,best_val_psnr=best,config=config,source=source,rng=rng_state()),tmp)
    tmp.replace(path)


def update(model,opt,objective,iterator,accum,clip):
    """每个完整累积组只更新一次：损失除累积次数，裁剪发生在累积之后。"""
    opt.zero_grad(set_to_none=True);total=0.
    for _ in range(accum):
        batch=next(iterator);inp=batch['input'].cuda();target=batch['target'].cuda()
        pred=model(inp);loss=objective(model,pred,target)
        if not torch.isfinite(loss):raise FloatingPointError('训练损失非有限')
        (loss/accum).backward();total+=float(loss.detach())/accum
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip if clip is not None else math.inf,error_if_nonfinite=True)
    opt.step()
    return total,float(norm)


def lpips_network():
    """统一采用原评估脚本的AlexNet LPIPS；训练时将其留在CPU。"""
    import lpips
    return lpips.LPIPS(net='alex').eval().requires_grad_(False)


class TimedLoader:
    """记录每种原尺寸完成整项指标计算的时间，包含推理和三项指标。"""
    def __init__(self,loader,times):
        """保留评估器所需数据集属性，不改变图像或顺序。"""
        self.loader=loader;self.dataset=loader.dataset;self.times=times

    def __iter__(self):
        """在评估器处理完一个batch后记录其端到端耗时。"""
        for batch in self.loader:
            key=str(tuple(reversed(batch['input'].shape[-2:])))
            torch.cuda.synchronize();start=time.perf_counter()
            yield batch
            torch.cuda.synchronize()
            self.times.setdefault(key,[]).append(time.perf_counter()-start)


def validation(model,loader,lpips_model,shape_times=None):
    """完整评估1100对；固定范围、尺寸、SSIM窗口与LPIPS输入规范。"""
    with evaluation_rng():
        lpips_model.cuda();torch.cuda.synchronize();start=time.perf_counter()
        measured=TimedLoader(loader,shape_times) if shape_times is not None else loader
        try:rows,summary=evaluate(model,measured,'cuda',lpips_model,progress=True)
        finally:lpips_model.cpu()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
    assert len(rows)==1100 and len(summary['per_task'])==11 and all(v['pairs']==100 for v in summary['per_task'].values())
    assert all(math.isfinite(r[k]) for r in rows for k in ['psnr','ssim','lpips'])
    return rows,summary,elapsed


def probe(args):
    """只在CPU构造网络、核查依赖与正式划分，不占用当前GPU任务。"""
    torch.set_num_threads(2);torch.manual_seed(42)
    train,val=datasets();model,meta=build(args.model)
    objective=Objective(args.model)
    write(args.output,dict(status='cpu_ready',model=args.model,metadata=meta,train_pairs=len(train),val_pairs=len(val),source=source_stamp(),loss=SPECS[args.model]['loss']))


def bench(args):
    """用正式训练批次测速后完整验证，全部指标标为流程数据而非论文结果。"""
    args.stage='training_profile'
    seed_all(42);torch.set_num_threads(2)
    train,val=datasets();model,meta=build(args.model);model.cuda().train()
    objective=Objective(args.model).cuda();opt=optimizer_for(args.model,model)
    accum=16//args.batch
    assert 16%args.batch==0
    sampler=StepBatchSampler(train.records,args.batch,accum,25,42)
    loader=DataLoader(train,batch_sampler=sampler,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(42))
    iterator=iter(loader);times=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(25):
        torch.cuda.synchronize();t=time.perf_counter()
        loss,norm=update(model,opt,objective,iterator,accum,SPECS[args.model]['clip'])
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        if step>=5:times.append(elapsed)
    peak=torch.cuda.max_memory_allocated()/1024**3
    if peak>21.5:raise RuntimeError('MEMORY_MARGIN:稳定显存余量不足')
    del objective,opt,iterator,loader
    args.stage='native_validation_profile'
    torch.cuda.empty_cache()
    lpips_model=lpips_network();val_loader=DataLoader(val,batch_size=1,num_workers=2)
    shape_times={}
    _,summary,val_seconds=validation(model,val_loader,lpips_model,shape_times)
    # 测试耗时只按已知图像尺寸构成与完整验证测速估算，不提前读取候选测试成绩。
    shape_counts={}
    from PIL import Image
    for split in ['val','test']:
        doc=json.loads((MANIFEST/f'all11_{split}.json').read_text())
        counter=Counter()
        for r in doc['records']:
            with Image.open(DATA/r['input']) as im:counter[str(im.size)]+=1
        shape_counts[split]=dict(counter)
    same_shapes=set(shape_counts['val'])==set(shape_counts['test'])
    assert same_shapes,'测试出现验证未覆盖的原尺寸，阻塞预算推断，需另外核实'
    shape_seconds={key:statistics.mean(values) for key,values in shape_times.items()}
    overhead=max(0,val_seconds-sum(sum(v) for v in shape_times.values()))
    estimate=sum(count*shape_seconds[key] for key,count in shape_counts['test'].items())+2*overhead
    write(args.output,dict(status='profiled',purpose='discarded_pipeline_profile_not_paper_results',model=args.model,batch_size=args.batch,accum_steps=accum,effective_batch=16,crop_size=256,precision='fp32_tf32_disabled',update_mean=statistics.mean(times),update_p90=float(np.percentile(times,90)),update_samples=times,peak_allocated_GiB=peak,val_pairs=summary['pairs'],validation_seconds=val_seconds,test_seconds_estimate=estimate,test_estimate_basis='measured_per_native_shape_weighted_by_test_shape_counts_plus_loader_overhead',shape_counts=shape_counts,validation_mean_seconds_by_shape=shape_seconds,metadata=meta,source=source_stamp()))


def train(args):
    """从零训练或精确恢复本实验，只按完整独立验证集PSNR保存最佳。"""
    cfg=json.loads(args.config.read_text());name=cfg['model'];out=Path(cfg['output'])
    seed_all(42);torch.set_num_threads(2)
    trainset,valset=datasets();model,meta=build(name);model.cuda().train()
    opt=optimizer_for(name,model);objective=Objective(name).cuda();lpips_model=lpips_network()
    source={'runner':source_stamp(),'model':meta,'manifest_sha256':EXPECTED,'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda}
    start=0;best=-math.inf
    if args.resume:
        state=torch.load(out/'last.pt',map_location='cpu')
        assert state['config']==cfg and state['source']==source,'恢复配置或来源已变化，拒绝自动绕过'
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        start=state['step'];best=state['best_val_psnr'];restore_rng(state['rng']);del state
    else:
        out.mkdir(parents=True,exist_ok=False)
        write(out/'config.json',cfg);write(out/'source.json',source)
    val_loader=DataLoader(valset,batch_size=1,num_workers=2)
    sampler=StepBatchSampler(trainset.records,cfg['batch_size'],cfg['accum_steps'],cfg['max_steps'],42,start)
    loader=DataLoader(trainset,batch_sampler=sampler,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(42))
    iterator=iter(loader)
    from tqdm import tqdm
    bar=tqdm(total=cfg['max_steps'],initial=start,desc=name+'训练',mininterval=5,file=sys.stdout)
    run_start=time.monotonic();last_step=start
    try:
        with (out/'train.jsonl').open('a',encoding='utf-8',buffering=1) as log:
            for i in range(start,cfg['max_steps']):
                lr=lr_for(i,cfg['max_steps'],SPECS[name],cfg['warmup_steps'])
                for group in opt.param_groups:group['lr']=lr
                torch.cuda.synchronize();t=time.perf_counter()
                loss,norm=update(model,opt,objective,iterator,cfg['accum_steps'],SPECS[name]['clip'])
                torch.cuda.synchronize();elapsed=time.perf_counter()-t;step=i+1;last_step=step
                record={'step':step,'loss':loss,'lr':lr,'grad_norm_before_clip':norm,'update_seconds':elapsed}
                bar.update(1);bar.set_postfix(loss=f'{loss:.4f}',lr=f'{lr:.2e}',seconds=f'{elapsed:.2f}',refresh=False)
                if step%cfg['eval_every']==0 or step==cfg['max_steps']:
                    rows,summary,seconds=validation(model,val_loader,lpips_model)
                    metrics=summary['groups']['all'];improved=metrics['psnr']>best
                    best=max(best,metrics['psnr'])
                    save_evaluation(out/f'val_{step:07}',rows,summary,{'config':cfg,'source':source,'step':step,'split':'val','purpose':'model_selection','eval_seed':42})
                    if improved:save_checkpoint(out/'best.pt',model,opt,step,best,cfg,source)
                    save_checkpoint(out/'last.pt',model,opt,step,best,cfg,source)
                    record.update(validation_seconds=seconds,validation=metrics,best_val_psnr=best)
                    print('\n完整验证 '+json.dumps(record,ensure_ascii=False),flush=True)
                    write(out/'status.json',dict(phase='training',step=step,best_val_psnr=best,last_validation=metrics,updated=datetime.now().astimezone().isoformat()))
                elif step%1000==0:save_checkpoint(out/'last.pt',model,opt,step,best,cfg,source)
                if step%20==0 or 'validation' in record:log.write(json.dumps(record)+'\n')
                # 超过实验截止即保存退出，不终止其他进程，也不把部分实验当完整结果。
                if time.time()>cfg['deadline_epoch']:
                    save_checkpoint(out/'last.pt',model,opt,step,best,cfg,source)
                    write(out/'incomplete.json',dict(reason='deadline_reached',step=step));return
        write(out/'completed.json',dict(max_steps=cfg['max_steps'],best_val_psnr=best,elapsed_seconds=time.monotonic()-run_start,finished=datetime.now().astimezone().isoformat()))
    except BaseException:
        write(out/'failure.json',dict(step=last_step,last_recoverable_checkpoint=str(out/'last.pt'),time=datetime.now().astimezone().isoformat()))
        raise
    finally:bar.close()


def test(args):
    """训练结束后一次性测试验证最优权重，固定输出目录，禁止依据测试再调参。"""
    cfg=json.loads(args.config.read_text());out=Path(cfg['output'])
    assert (out/'completed.json').exists()
    target=out/'final_test'
    if target.exists():raise FileExistsError('测试目录已存在，禁止自动覆盖或重测挑结果')
    seed_all(42);torch.set_num_threads(2);datasets()
    state=torch.load(out/'best.pt',map_location='cpu')
    model,meta=build(cfg['model']);model.load_state_dict(state['model'],strict=True);model.cuda().eval()
    assert meta==state['source']['model'],'测试模型源码变化'
    assert state['source']['runner']==source_stamp(),'测试工具代码变化'
    chosen={'checkpoint_sha256':digest(out/'best.pt'),'selected_step':state['step'],'selection_metric':'independent_val_macro_PSNR','best_val_psnr':state['best_val_psnr'],'manifest_sha256':EXPECTED['test'],'seed':42,'precision':'fp32_tf32_disabled','protocol':'native_resolution_replicate_pad_multiple8_float_RGB_0_1_no_border_gaussian_SSIM_LPIPS_alex','source':state['source'],'config':cfg}
    write(out/'test_selection.json',chosen)
    ds=PairedManifestDataset(DATA,MANIFEST/'all11_test.json');loader=DataLoader(ds,batch_size=1,num_workers=2)
    metric=lpips_network().cuda();t=time.perf_counter()
    rows,summary=evaluate(model,loader,'cuda',metric,progress=True)
    assert len(rows)==2200 and len(summary['per_task'])==11 and all(x['pairs']==200 for x in summary['per_task'].values())
    assert all(math.isfinite(r[k]) for r in rows for k in ['psnr','ssim','lpips'])
    chosen['test_seconds']=time.perf_counter()-t
    save_evaluation(target,rows,summary,chosen)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    """限制可执行动作，所有训练参数来自不可变的新实验配置。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['probe','bench','train','test'])
    parser.add_argument('--model',choices=list(SPECS));parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--output',type=Path);parser.add_argument('--config',type=Path);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:globals()[args.action](args)
    except BaseException as exc:
        if args.action in ['probe','bench'] and args.output:
            write(args.output,dict(status='failed',model=args.model,batch_size=args.batch,error=repr(exc),stage=getattr(args,'stage',args.action)))
        raise


if __name__=='__main__':main()
