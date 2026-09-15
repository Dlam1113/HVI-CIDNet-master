"""按已核查源码构建对比网络，保留各方法的训练目标。"""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch
from torch import nn

MAIN=Path('/home/Bjj/HVI-CIDNet-clean')
EXTERNAL=Path('/home/Bjj/comparison_models')
SNAPSHOT=Path(__file__).resolve().parent/'snapshot'
if SNAPSHOT.exists():
    MAIN=SNAPSHOT/'main'
    EXTERNAL=SNAPSHOT/'external'

# 优先级在观察任何新测试分数之前固定；最终启动仍由实测预算决定。
SPECS={
 'cidnet':dict(priority=0,root=str(MAIN),file='net/CIDNet.py',cls='CIDNet',kwargs={},loss='legacy_dual_domain',optimizer='Adam',lr=1e-4,betas=[.9,.999],weight_decay=0.,clip=.01,schedule='legacy_two_stage',warmup_fraction=.01875),
 'promptir':dict(priority=1,root=str(EXTERNAL/'PromptIR'),file='net/model.py',cls='PromptIR',kwargs=dict(decoder=True),loss='l1',optimizer='AdamW',lr=2e-4,betas=[.9,.999],weight_decay=.01,clip=None,schedule='warm_cosine',warmup_fraction=.1),
 'histoformer':dict(priority=2,root=str(EXTERNAL/'Histoformer'),file='basicsr/models/archs/histoformer_arch.py',cls='Histoformer',kwargs=dict(inp_channels=3,out_channels=3,dim=36,num_blocks=[4,4,6,8],num_refinement_blocks=4,heads=[1,2,4,8],ffn_expansion_factor=2.667,bias=False,LayerNorm_type='WithBias',dual_pixel_task=False),loss='l1_pearson',optimizer='AdamW',lr=3e-4,betas=[.9,.999],weight_decay=1e-4,clip=.01,schedule='plateau_cosine',warmup_fraction=0.),
 'moce_ir_s':dict(priority=3,root=str(EXTERNAL/'MoCE-IR'),file='src/net/moce_ir.py',cls='MoCEIR',kwargs=dict(inp_channels=3,out_channels=3,dim=32,levels=4,heads=[1,2,4,8],num_blocks=[4,6,6,8],num_dec_blocks=[2,4,4],ffn_expansion_factor=2,num_refinement_blocks=4,LayerNorm_type='WithBias',bias=False,rank=2,num_experts=4,depth_type='constant',stage_depth=[1,1,1],rank_type='spread',topk=1,with_complexity=False,complexity_scale='max'),loss='l1_fft_balance',optimizer='AdamW',lr=2e-4,betas=[.9,.999],weight_decay=.01,clip=None,schedule='warm_cosine',warmup_fraction=.1),
 'nafnet':dict(priority=4,root=str(EXTERNAL/'NAFNet'),file='basicsr/models/archs/NAFNet_arch.py',cls='NAFNetLocal',kwargs=dict(img_channel=3,width=32,middle_blk_num=1,enc_blk_nums=[1,1,1,28],dec_blk_nums=[1,1,1,1],train_size=(1,3,256,256),fast_imp=False),loss='psnr',optimizer='AdamW',lr=1e-3,betas=[.9,.9],weight_decay=0.,clip=.01,schedule='cosine',warmup_fraction=0.),
 'restormer':dict(priority=5,root=str(EXTERNAL/'Restormer'),file='basicsr/models/archs/restormer_arch.py',cls='Restormer',kwargs=dict(inp_channels=3,out_channels=3,dim=48,num_blocks=[4,6,6,8],num_refinement_blocks=4,heads=[1,2,4,8],ffn_expansion_factor=2.66,bias=False,LayerNorm_type='WithBias',dual_pixel_task=False),loss='l1',optimizer='AdamW',lr=3e-4,betas=[.9,.999],weight_decay=1e-4,clip=.01,schedule='plateau_cosine',warmup_fraction=0.),
}


def digest(path):
    """计算文件摘要，绑定实际加载的模型与损失源码。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(path,name):
    """独立导入已知源码，避免多个仓库同名net包互相覆盖。"""
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_selected(path,names):
    """原样编译指定损失定义，避免执行BasicSR全局注册与无关导入。"""
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    namespace={'torch':torch,'nn':nn,'np':np,'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def naf_namespace(root):
    """为NAFNet提供原文件的包路径，跳过自动加载所有其他架构的注册入口。"""
    for name,relative in [('basicsr','basicsr'),('basicsr.models','basicsr/models'),('basicsr.models.archs','basicsr/models/archs')]:
        module=types.ModuleType(name);module.__path__=[str(root/relative)];sys.modules[name]=module


def build(name):
    """构造从零初始化网络并返回来源清单，禁止加载旧对比权重。"""
    spec=SPECS[name]
    root=Path(spec['root'])
    if name=='cidnet':
        from net.CIDNet import CIDNet
        model=CIDNet()
    else:
        if name=='nafnet':naf_namespace(root)
        module=load_file(root/spec['file'],'_cdd_'+name)
        model=getattr(module,spec['cls'])(**spec['kwargs'])
    tracked=[root/spec['file']]
    if name=='cidnet':tracked += [MAIN/'net/HVI_transform.py',MAIN/'net/LCA.py',MAIN/'net/transformer_utils.py']
    if name=='nafnet':tracked += [root/'basicsr/models/archs/arch_util.py',root/'basicsr/models/archs/local_arch.py',root/'basicsr/models/losses/losses.py']
    if name=='histoformer':tracked += [root/'basicsr/models/image_restoration_model.py']
    if name=='moce_ir_s':tracked += [root/'src/train.py',root/'src/options.py',root/'src/utils/loss_utils.py']
    source={str(p):digest(p) for p in tracked}
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    diff=subprocess.run(['git','diff','--',*[str(p.relative_to(root)) for p in tracked]],cwd=root,capture_output=True,text=True)
    metadata={'model':name,'constructor':spec['cls'],'kwargs':spec['kwargs'],'files':source,'git_commit':revision.stdout.strip(),'working_diff':diff.stdout,'pretrained_weights':None,'extra_training_data':None,'initialization':'random_seed42','parameters':sum(p.numel() for p in model.parameters())}
    manifest=Path(__file__).resolve().parent/'snapshot_manifest.json'
    if manifest.exists():metadata['snapshot_origin']=json.loads(manifest.read_text())['origins'][name]
    return model,metadata


class Objective(nn.Module):
    """保留原方法目标；网络内部前向和控制点等计算均不改动。"""
    def __init__(self,name):
        """从已核查原文件读取PSNR、Pearson及FFT定义。"""
        super().__init__();self.name=name;self.kind=SPECS[name]['loss'];self.l1=nn.L1Loss()
        if name=='cidnet':
            from revision.experiment import LegacyRestorationLoss
            self.native=LegacyRestorationLoss()
        elif name=='nafnet':
            self.native=load_selected(EXTERNAL/'NAFNet/basicsr/models/losses/losses.py',['PSNRLoss'])['PSNRLoss']()
        elif name=='histoformer':
            self.pearson=load_selected(EXTERNAL/'Histoformer/basicsr/models/image_restoration_model.py',['pearson_correlation_loss'])['pearson_correlation_loss']
        elif name=='moce_ir_s':
            self.native=load_selected(EXTERNAL/'MoCE-IR/src/utils/loss_utils.py',['FFTLoss'])['FFTLoss'](loss_weight=.1)

    def forward(self,model,pred,target):
        """按原训练公式合成标量损失，不遗漏路由平衡或相关性项。"""
        if self.name=='cidnet':return self.native(model,pred,target)
        if self.name=='nafnet':return self.native(pred,target)
        loss=self.l1(pred,target)
        if self.name=='histoformer':
            pear=(1-self.pearson(None,pred,target))/2
            loss=loss+pear[~pear.isnan()*~pear.isinf()].mean()
        if self.name=='moce_ir_s':loss=loss+self.native(pred,target)+.01*model.total_loss
        return loss


def optimizer_for(name,model):
    """使用已记录的方法专属优化器与超参数。"""
    s=SPECS[name]
    return getattr(torch.optim,s['optimizer'])(model.parameters(),lr=s['lr'],betas=tuple(s['betas']),weight_decay=s['weight_decay'])


def lr_for(step,total,spec,warmup):
    """按本次公开预算缩放日程，明确不声称复现原论文长训练预算。"""
    peak=spec['lr'];schedule=spec['schedule']
    minimum=0. if schedule=='warm_cosine' else (1e-7 if schedule in ['cosine','legacy_two_stage'] else 1e-6)
    if schedule=='legacy_two_stage':
        from revision.schedules import lr_at_step
        return lr_at_step(step,total,warmup,peak,'legacy_two_stage')
    if warmup and step<warmup:return peak*(step+1)/warmup
    if schedule=='plateau_cosine':
        plateau=round(total*92000/300000)
        if step<plateau:return peak
        fraction=(step-plateau)/max(1,total-plateau-1)
    else:fraction=(step-warmup)/max(1,total-warmup-1)
    return minimum+(peak-minimum)*.5*(1+math.cos(math.pi*min(1,max(0,fraction))))
