"""补充三个原方法的CDD适配；复用冻结评估入口，不改变正在训练的旧文件。"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import types

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE / 'vendor'))
import run as base
import models as helpers
import torch
from torch import nn
import cv2
import numpy as np

ROOT = HERE / 'snapshot'
ACTIVE = None
BATCH = 16
STEP = 2000
# 补充模型的正式训练预算统一为10,000次更新。
TOTAL_STEPS = 10000
PROFILE = False
ORIGINAL_DATASETS = base.datasets
SPECS = {
    'airnet': dict(optimizer='Adam', lr=1e-3, betas=[.9,.999], weight_decay=0., clip=None,
                   loss='encoder_CE_then_L1_plus_0.1_CE', warmup_fraction=0.,
                   schedule='original_1000_epoch_shape_scaled_to_10000_updates', encoder_steps=1000),
    'snrnet': dict(optimizer='Adam', lr=4e-4, betas=[.9,.99], weight_decay=0., clip=None,
                   loss='original_Charbonnier_sum_eps1e-6_effective_batch_sum', warmup_fraction=0.,
                   schedule='original_multistep_scaled_600000_to_10000', milestones=[833,1667,3333,5000]),
    'zero_dce_pp': dict(optimizer='Adam', lr=1e-4, betas=[.9,.999], weight_decay=1e-4, clip=.1,
                   loss='original_zero_reference_1600TV_1spatial_5color_10exposure_E0.6',
                   warmup_fraction=0., schedule='constant', scale_factor=1),
}


def package(name, path):
    """给原源码提供自身包路径，避免不同仓库同名包互相覆盖。"""
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


class Model(nn.Module):
    """仅适配输入输出接口，网络层、参数及原损失计算保持原方法。"""
    def __init__(self, name):
        """直接从隔离快照构造网络，不加载历史权重或额外数据。"""
        super().__init__()
        self.name = name
        if name == 'airnet':
            package('net', ROOT/'airnet/net')
            module = helpers.load_file(ROOT/'airnet/net/model.py', '_supp_airnet')
            self.network = module.AirNet(types.SimpleNamespace(batch_size=BATCH))
        elif name == 'snrnet':
            package('models', ROOT/'snrnet/models')
            package('models.archs', ROOT/'snrnet/models/archs')
            module = helpers.load_file(ROOT/'snrnet/models/archs/low_light_transformer.py', '_supp_snr')
            self.network = module.low_light_transformer(nf=64,nframes=5,groups=8,
                front_RBs=1,back_RBs=1,center=None,predeblur=True,HR_in=True,w_TSA=True)
        else:
            module = helpers.load_file(ROOT/'zero_dce_pp/model.py', '_supp_zero')
            self.network = module.enhance_net_nopool(scale_factor=1)

    def forward(self, x):
        """评估统一输出RGB；SNR使用原LOLv1的5×5模糊与信噪比掩码。"""
        if self.name == 'airnet':
            return self.network(x, x)
        if self.name == 'zero_dce_pp':
            return self.network(x)[0]
        images = x.detach().cpu().permute(0,2,3,1).numpy()
        nf = np.stack([cv2.blur(im*255.0,(5,5))/255.0 for im in images])
        nf = torch.from_numpy(nf).permute(0,3,1,2).to(x.device)
        dark = .299*x[:,0:1] + .587*x[:,1:2] + .114*x[:,2:3]
        light = .299*nf[:,0:1] + .587*nf[:,1:2] + .114*nf[:,2:3]
        mask = light / ((dark-light).abs()+.0001)
        maximum = mask.reshape(mask.shape[0],-1).max(1)[0].reshape(-1,1,1,1)
        mask = (mask/(maximum+.0001)).clamp(0,1).float()
        return self.network(x, mask)


def stamp():
    """核对补充快照和原冻结入口，检查点绑定两套源码及依赖文件。"""
    result = ORIGINAL_STAMP()
    manifest = HERE/'snapshot_manifest.json'
    for rel, sha in json.loads(manifest.read_text())['files'].items():
        assert helpers.digest(HERE/rel) == sha, '补充源码或依赖变化：'+rel
    for path in [Path(__file__), manifest]:
        result[str(path)] = helpers.digest(path)
    return result


def build(name):
    """保存实际构造、原始文件哈希和已披露适配，不依据默认配置推断历史。"""
    model = Model(name)
    origin = json.loads((HERE/'snapshot_manifest.json').read_text())['origins'][name]
    meta = dict(model=name, parameters=sum(p.numel() for p in model.parameters()),
        origin=origin, initialization='random_seed42', pretrained_weights=None,
        extra_training_data=None, physical_batch=BATCH, spec=SPECS[name],
        adaptation='CDD11_frozen_split_crop256_effective16_FP32_task_uniform')
    if name == 'airnet':
        meta.update(moco_queue_size=BATCH*256, note='原MoCo队列按物理batch构造；微批次动量、BN及队列更新不等同于大batch')
    if name == 'snrnet':
        meta.update(note='原LOLv1单图路径及原sum Charbonnier；原32×128裁剪调整为有效16×256；原尺寸验证测试')
    if name == 'zero_dce_pp':
        meta.update(note='训练不使用GT，原零参考目标用于全部CDD输入；原方法目标为低光增强，未引入监督复原损失')
    return model, meta


class Objective(nn.Module):
    """零参考损失延迟到GPU空闲后的运行阶段构造，CPU核查不占GPU。"""
    def __init__(self, name):
        """只保存方法名称和原标量损失；不访问设备。"""
        super().__init__()
        self.name = name
        if name == 'snrnet':
            selected = helpers.load_selected(ROOT/'snrnet/models/loss.py', ['CharbonnierLoss'])
            self.charbonnier = selected['CharbonnierLoss']()

    def zero_loss(self, x, enhanced, curve):
        """使用原四项零参考损失和权重，GT不进入优化目标。"""
        if not hasattr(self,'color'):
            module = helpers.load_file(ROOT/'zero_dce_pp/Myloss.py','_supp_zero_loss')
            self.color=module.L_color();self.spatial=module.L_spa()
            self.exposure=module.L_exp(16);self.tv=module.L_TV()
        return (1600*self.tv(curve) + self.spatial(enhanced,x).mean()
                + 5*self.color(enhanced).mean() + 10*self.exposure(enhanced,.6).mean())


class TwoViews(base.PairedManifestDataset):
    """AirNet对同一场景退化图独立裁剪两次，保持查询图与目标配对。"""
    def __getitem__(self, key):
        """使用独立且可恢复的增强种子，不将其他任务或场景作为正样本。"""
        item=super().__getitem__(key)
        index, seed = key if isinstance(key,tuple) else (key,key)
        second=super().__getitem__((index,seed ^ 0x9E3779B9))
        item['key_input']=second['input']
        return item


def datasets():
    """正式清单和评估保持不变，只为AirNet训练提供原方法所需第二视图。"""
    train,val=ORIGINAL_DATASETS()
    if ACTIVE=='airnet':
        train=TwoViews(base.DATA,base.MANIFEST/'all11_train.json',256)
    return train,val


def update(model,opt,objective,iterator,accum,clip):
    """完成一次参数更新；保留阶段目标、原sum损失及累积后的梯度裁剪。"""
    opt.zero_grad(set_to_none=True);total=0.
    for _ in range(accum):
        batch=next(iterator);x=batch['input'].cuda()
        if ACTIVE=='airnet':
            key=batch['key_input'].cuda()
            if STEP<SPECS['airnet']['encoder_steps']:
                _,logits,labels,_=model.network.E(x_query=x,x_key=key)
                loss=nn.functional.cross_entropy(logits,labels)
            else:
                prediction,logits,labels=model.network(x_query=x,x_key=key)
                loss=nn.functional.l1_loss(prediction,batch['target'].cuda())+.1*nn.functional.cross_entropy(logits,labels)
        elif ACTIVE=='zero_dce_pp':
            enhanced,curve=model.network(x)
            loss=objective.zero_loss(x,enhanced,curve)
        else:
            # 原Charbonnier是sum；补偿累积除数，得到有效16张的总和，避免随微批量改变尺度。
            loss=objective.charbonnier(model(x),batch['target'].cuda())*accum
        if not torch.isfinite(loss):raise FloatingPointError('补充模型损失非有限')
        (loss/accum).backward();total+=float(loss.detach())/accum
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip if clip is not None else math.inf,error_if_nonfinite=True)
    opt.step()
    return total,float(norm)


def optimizer_for(name,model):
    """保留原Adam和权重衰减；测速时AirNet使用联合阶段起始学习率。"""
    spec=SPECS[name]
    lr=1e-4 if PROFILE and name=='airnet' else spec['lr']
    return torch.optim.Adam(model.parameters(),lr=lr,betas=tuple(spec['betas']),weight_decay=spec['weight_decay'])


def lr_for(index,total,spec,warmup):
    """将原训练日程按比例映射为10k，明确AirNet总预算含1k编码器预训练。"""
    global STEP
    STEP=index
    assert total==TOTAL_STEPS and warmup==0
    if ACTIVE=='zero_dce_pp':return 1e-4
    if ACTIVE=='snrnet':return spec['lr']*.5**sum(index+1>=m for m in spec['milestones'])
    epoch=index//max(1,total//1000)
    previous=max(0,epoch-1)
    return 1e-3*.1**(previous//60) if previous<=100 else 1e-4*.5**((previous-100)//125)


def bench(args):
    """先测完整联合训练及完整验证，再单独测AirNet编码器阶段，均不作论文结果。"""
    global STEP
    STEP=SPECS['airnet']['encoder_steps']
    ORIGINAL_BENCH(args)
    if ACTIVE!='airnet':return
    import gc,time
    gc.collect();torch.cuda.empty_cache();base.seed_all(42)
    train,_=datasets();model,_=build(ACTIVE);model.cuda().train()
    opt=optimizer_for(ACTIVE,model)
    for group in opt.param_groups:group['lr']=1e-3
    objective=Objective(ACTIVE);accum=16//BATCH
    sampler=base.StepBatchSampler(train.records,BATCH,accum,25,42)
    loader=base.DataLoader(train,batch_sampler=sampler,num_workers=2,generator=torch.Generator().manual_seed(42))
    iterator=iter(loader);times=[];STEP=0;torch.cuda.reset_peak_memory_stats()
    for i in range(25):
        torch.cuda.synchronize();start=time.perf_counter()
        update(model,opt,objective,iterator,accum,None)
        torch.cuda.synchronize()
        if i>=5:times.append(time.perf_counter()-start)
    peak=torch.cuda.max_memory_allocated()/1024**3
    if peak>21.5:raise RuntimeError('MEMORY_MARGIN:编码器阶段显存余量不足')
    report=json.loads(args.output.read_text())
    encoder_steps=SPECS['airnet']['encoder_steps']
    report.update(encoder_update_mean=statistics.mean(times),encoder_update_samples=times,
                  encoder_peak_allocated_GiB=peak,encoder_steps=encoder_steps,
                  joint_steps=TOTAL_STEPS-encoder_steps)
    base.write(args.output,report)


ORIGINAL_STAMP=base.source_stamp
ORIGINAL_BENCH=base.bench
base.SPECS=SPECS;base.build=build;base.Objective=Objective
base.source_stamp=stamp;base.datasets=datasets;base.update=update
base.optimizer_for=optimizer_for;base.lr_for=lr_for;base.bench=bench


def main():
    """读取当前独立配置后调用冻结流程，拒绝偏离10k或有效batch16。"""
    global ACTIVE,BATCH,PROFILE
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['probe','bench','train','test'])
    parser.add_argument('--model',choices=list(SPECS));parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--output',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();ACTIVE=args.model;BATCH=args.batch;PROFILE=args.action=='bench'
    if args.config:
        cfg=json.loads(args.config.read_text());ACTIVE=cfg['model'];BATCH=cfg['batch_size']
        assert cfg['max_steps']==TOTAL_STEPS and BATCH*cfg['accum_steps']==16
    assert ACTIVE in SPECS and 16%BATCH==0
    try:getattr(base,args.action)(args)
    except BaseException as error:
        if args.action in ['probe','bench'] and args.output:
            base.write(args.output,dict(status='failed',model=ACTIVE,batch_size=BATCH,
                error=repr(error),stage=getattr(args,'stage',args.action)))
        raise


if __name__=='__main__':main()
