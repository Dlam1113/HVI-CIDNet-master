"""单GPU自主队列：等待现有任务，先测速全候选，再按预算训练和完整测试。"""
from datetime import datetime,timezone,timedelta
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
WORK=HERE/'experiments'
DOC=HERE/'doc'
PYTHON=Path('/home/Bjj/anaconda3/envs/CIDNet/bin/python')
DEADLINE=datetime(2026,9,20,23,59,0,tzinfo=timezone(timedelta(hours=8))).timestamp()
ORDER=['cidnet','promptir','histoformer','moce_ir_s','nafnet','restormer']
BLOCKED={
 'airnet':'现存scratch入口冻结编码器，未保留原两阶段对比学习；11任务双视图采样及原训练路径尚需核实，禁止以冻结随机编码器替代。',
 'snrnet':'现存适配自定义SNR图及多帧结构，尚未核实与原低光流程一致，属于关键输入协议问题；暂不启动正式CDD训练。',
 'zero_dce_pp':'现存混合/LOL脚本目标不同；零参考曝光目标用于雨雾雪复合任务的具体协议尚未确定，不能擅自改成配对L1并当作原方法。',
}


def write(path,obj):
    """原子保存队列状态，以便Codex心跳和人工读取。"""
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def note(message):
    """每个小阶段向独立doc进展追加状态，不改论文。"""
    stamp=datetime.now().astimezone().isoformat();DOC.mkdir(parents=True,exist_ok=True)
    with (DOC/'progress.md').open('a',encoding='utf-8') as f:f.write(f'\n- {stamp} {message}\n')
    print(stamp,message,flush=True)


def status(phase,**kwargs):
    """更新当前任务，便于故障后确认是否仍有GPU子进程运行。"""
    write(HERE/'queue_status.json',dict(phase=phase,updated=datetime.now().astimezone().isoformat(),**kwargs))


def gpu_jobs():
    """查询GPU计算进程，不杀死或调整其他已有任务。"""
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('GPU状态无法确认：'+result.stderr)
    return [s for s in result.stdout.splitlines() if s.strip()]


def wait_gpu():
    """GPU有其他进程时继续等待；截止后不启动新GPU工作。"""
    announced=False
    while gpu_jobs():
        if not announced:note('等待现有GPU任务自然结束；未停止或重启任何任务。');announced=True
        status('waiting_existing_gpu',processes=gpu_jobs())
        if time.time()>=DEADLINE:return False
        time.sleep(30)
    return time.time()<DEADLINE


def execute(args,log,phase,model,gpu=True):
    """串行运行本队列子进程，保存完整命令、退出码和实测墙钟耗时。"""
    if gpu and not wait_gpu():return 99
    cmd=[str(PYTHON),'-u','-B',str(HERE/'run.py'),*args]
    log.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();note('启动 '+phase+' '+model+'；命令：`'+' '.join(cmd)+'`')
    with log.open('a',encoding='utf-8') as stream:
        p=subprocess.Popen(cmd,cwd=HERE,stdout=stream,stderr=subprocess.STDOUT)
        status(phase,model=model,pid=p.pid,command=cmd,log=str(log))
        code=p.wait()
    elapsed=time.monotonic()-start
    with (DOC/'commands.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(command=cmd,phase=phase,model=model,exit_code=code,seconds=elapsed,finished=datetime.now().astimezone().isoformat()),ensure_ascii=False)+'\n')
    note(f'{phase} {model} 结束，exit={code}，实际耗时{elapsed:.1f}秒。')
    return code


def cost(profile,steps=40000,every=2000):
    """预算包含更新、完整验证和最终测试，并加保守耗时系数。"""
    update=max(profile['update_mean']*1.20,profile['update_p90']*1.1)
    val=profile['validation_seconds']*1.25
    test=profile['test_seconds_estimate']*1.30
    return dict(training_seconds=steps*update,validation_seconds=math.ceil(steps/every)*val,test_seconds=test,startup_seconds=600,total_seconds=steps*update+math.ceil(steps/every)*val+test+600)


def budget(profiles):
    """在所有候选完成测速/记录阻塞后固定启动顺序，不看任何测试分数。"""
    reserve=12*3600
    available=max(0,DEADLINE-time.time()-reserve)
    planning_start=time.time()
    rows=[];selected=[];used=0.
    for name in ORDER:
        p=profiles.get(name)
        if not p or p.get('status')!='profiled':
            rows.append(dict(model=name,status='blocked_profile',reason=p.get('error','未取得完整测速') if p else 'missing'));continue
        estimate=cost(p)
        allowed=used+estimate['total_seconds']<=available
        rows.append(dict(model=name,status='selected' if allowed else 'not_started_budget',profile=p,estimate=estimate))
        if allowed:selected.append(name);used+=estimate['total_seconds']
    for name,reason in BLOCKED.items():rows.append(dict(model=name,status='blocked_protocol',reason=reason,total_seconds=None))
    accumulated=0.
    for row in rows:
        if row['status']=='selected':
            seconds=row['estimate']['total_seconds']
            row['planned_start']=datetime.fromtimestamp(planning_start+accumulated).astimezone().isoformat()
            row['planned_complete_including_test']=datetime.fromtimestamp(planning_start+accumulated+seconds).astimezone().isoformat()
            row['latest_start_preserving_12h_reserve']=datetime.fromtimestamp(DEADLINE-reserve-(used-accumulated)).astimezone().isoformat()
            accumulated+=seconds
    plan=dict(created=datetime.now().astimezone().isoformat(),deadline='2026-09-20T23:59:00+08:00',available_seconds=available,reserve_seconds=reserve,steps_per_model=40000,eval_every=2000,selected=selected,estimated_total_seconds=used,candidates=rows,selection_basis='priority_and_measured_compute_only_before_candidate_test_results',excluded_scope='no_full_model_training_no_R1.4-R1.7_ablation')
    write(HERE/'budget_plan.json',plan)
    lines=['# CDD-11实测预算与取舍','',f'创建：{plan["created"]}；实验截止：{plan["deadline"]}。','',f'预留故障/结果核查：12小时；选中实验保守估计合计{used/3600:.2f}小时。','', '| 候选 | 状态 | 40k更新+完整验证+独立测试（小时） |','|---|---|---|']
    for row in rows:lines.append('|'+row['model']+'|'+row['status']+'|'+(f'{row["estimate"]["total_seconds"]/3600:.2f}' if 'estimate' in row else '无法可靠估计：'+row.get('reason',''))+'|')
    lines+=['','| 模型 | 预计开始 | 预计完成（含测试） | 为全部选中模型及12小时余量保留的最迟开始 |','|---|---|---|---|']
    for row in rows:
        if row['status']=='selected':lines.append('|'+row['model']+'|'+row['planned_start']+'|'+row['planned_complete_including_test']+'|'+row['latest_start_preserving_12h_reserve']+'|')
    lines += ['', '各方法固定40,000次更新、有效batch16、256裁剪，完整验证每2000次，最终测试2200对。方法专属损失/优化器及学习率形状保留，日程按公开预算缩放；不是官方完整长训练协议复现，也不保证所有方法充分收敛。', '预算在任何候选测试成绩产生前确定。训练失败、负结果和未收敛均记录，不择优删模型。', '测试时间依据完整1100对验证实测及2200对测试的原分辨率尺寸构成估算；不提前用测试分数决定配置。完整模型已有测试实耗另作为环境参考。', '未启动的模型限制CDD-11比较覆盖广度，不能宣称覆盖所有原论文比较方法或所有最新方法；本次不完成R1.4—R1.7的额外实验，也不补R1.6组件消融。']
    (DOC/'budget.md').write_text('\n'.join(lines),encoding='utf-8')
    note('全部候选已形成实测预算或明确协议阻塞；固定本轮顺序：'+','.join(selected))
    return plan


def make_config(name,profile,estimate):
    """在训练开始前保存不可变协议，并为最后测试留出时间。"""
    from models import SPECS
    spec=SPECS[name]
    cfg=dict(model=name,seed=42,batch_size=profile['batch_size'],accum_steps=profile['accum_steps'],crop_size=256,effective_batch=16,precision='fp32_tf32_disabled',max_steps=40000,warmup_steps=round(40000*spec['warmup_fraction']),eval_every=2000,output=str(WORK/(name+'_40k_s42')),training_spec=spec,deadline_epoch=DEADLINE-estimate['test_seconds']-1800,selection='validation_macro_PSNR',pretrained_weights=None,extra_training_data=None,estimated=estimate,adaptation='formal_CDD11_600train_100val_200test_scene_split_task_uniform_sampling_fixed256_crop_single_GPU_budget_scaled_schedule',test_policy='one_final_validation_selected_checkpoint_no_test_tuning',profile_file=str(WORK/(name+'_profile.json')))
    path=WORK/(name+'_config.json')
    if path.exists():
        old=json.loads(path.read_text())
        # 原配置不因调度器重启时剩余时间变化而重写。
        return path,old
    write(path,cfg);return path,cfg


def collect():
    """汇总已完成的独立测试结果，包含失败/未启动状态，不选择性删行。"""
    rows=[]
    for name in ORDER:
        out=WORK/(name+'_40k_s42');p=out/'final_test/summary.json'
        if p.exists():
            s=json.loads(p.read_text());rows.append(dict(model=name,status='tested',metrics=s['groups']['all'],per_task=s['per_task'],groups=s['groups'],source=str(p)))
        else:rows.append(dict(model=name,status='not_tested',failure=(out/'failure.json').exists(),source=str(out)))
    write(DOC/'results_inventory.json',rows)
    lines=['# 新CDD-11对比结果（仅已完成独立测试）','','| 模型 | 状态 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |','|---|---|---|---|---|']
    for row in rows:
        m=row.get('metrics',{})
        lines.append('|'+row['model']+'|'+row['status']+'|'+ '|'.join(f'{m[k]:.6f}' if k in m else '待完成' for k in ['psnr','ssim','lpips'])+'|')
    lines+=['','所有状态均保留。此表不改旧论文数值，不代表等预算充分收敛排名。原始逐图结果、配置与选优依据以各实验独立输出为准。']
    (DOC/'results_table.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    """保持一个串行队列；后台可脱离SSH继续，错误留给心跳处理。"""
    HERE.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True);DOC.mkdir(exist_ok=True)
    lock=(HERE/'queue.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('已有队列，拒绝重复启动');return
    note('队列开始；所有GPU任务串行；先完整测速再预算，不自动追加本文完整模型。')
    profiles={}
    for name in ORDER:
        probe_path=WORK/(name+'_probe.json')
        if not probe_path.exists():execute(['probe','--model',name,'--output',str(probe_path)],WORK/(name+'_probe.log'),'cpu_probe',name,gpu=False)
        probe=json.loads(probe_path.read_text()) if probe_path.exists() else {'status':'failed','error':'probe_missing'}
        if probe.get('status')!='cpu_ready':profiles[name]=probe;continue
        result=WORK/(name+'_profile.json')
        if result.exists() and json.loads(result.read_text()).get('status')=='profiled':profiles[name]=json.loads(result.read_text());continue
        for batch in [16,8,4,2,1]:
            attempt=WORK/f'{name}_profile_b{batch}.json'
            if not attempt.exists():execute(['bench','--model',name,'--batch',str(batch),'--output',str(attempt)],WORK/f'{name}_profile_b{batch}.log','profile',name)
            p=json.loads(attempt.read_text()) if attempt.exists() else {'status':'failed','error':'profile_missing'}
            profiles[name]=p
            if p.get('status')=='profiled':write(result,p);break
            error=p.get('error','')
            if p.get('stage')=='native_validation_profile':
                note(name+'原分辨率完整验证失败，缩小训练batch不能解决；保留协议阻塞，不自动改为缩放/切块。');break
            if not ('out of memory' in error.lower() or 'MEMORY_MARGIN' in error):break
            note(name+f' batch{batch}显存不足，仅缩小物理batch并保持有效batch16；未改变结构或裁剪。')
    saved_plan=HERE/'budget_plan.json'
    plan=json.loads(saved_plan.read_text()) if saved_plan.exists() else budget(profiles)
    for name in plan['selected']:
        row=next(x for x in plan['candidates'] if x['model']==name)
        config_path,cfg=make_config(name,profiles[name],row['estimate']);out=Path(cfg['output'])
        if (out/'final_test/summary.json').exists():continue
        if not (out/'completed.json').exists():
            if time.time()+row['estimate']['total_seconds']+3600>DEADLINE:
                note(name+'按当前剩余时间不再满足完整训练+验证+测试预算，不启动。');continue
            if out.exists() and not (out/'last.pt').exists():note(name+'已有不完整输出但无可恢复权重，保留并阻塞，不覆盖。');continue
            cmd=['train','--config',str(config_path)]+(['--resume'] if out.exists() else [])
            code=execute(cmd,WORK/(name+'_train.console.log'),'training',name)
            if code or not (out/'completed.json').exists():note(name+'训练未完整完成，保留失败记录，继续其他独立模型。');collect();continue
        if time.time()+row['estimate']['test_seconds']>DEADLINE:
            note(name+'剩余时间不足完整测试，不启动。');continue
        code=execute(['test','--config',str(config_path)],WORK/(name+'_test.console.log'),'final_test',name)
        if code:note(name+'最终测试失败，保留日志等待核查，不依测试成绩调整模型。')
        collect()
    collect();status('queue_complete',budget_file=str(HERE/'budget_plan.json'));note('当前已批准预算队列处理完成；结果/阻塞均已记录，未另行加训。')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('queue_failed',error=repr(exc));note('队列错误：'+repr(exc));traceback.print_exc();raise
