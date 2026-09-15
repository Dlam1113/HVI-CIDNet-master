"""作者授权的后续20k队列；不修改当前40k训练或被冻结的模型入口。"""
from datetime import datetime
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from orchestrate import cost
from models import SPECS

HERE=Path(__file__).resolve().parent
PHASE=HERE/'phase20k_20260913'
DOC=PHASE/'doc'
PYTHON='/home/Bjj/anaconda3/envs/CIDNet/bin/python'
ORDER=['promptir','histoformer','moce_ir_s','nafnet','restormer']
CID=HERE/'experiments/cidnet_40k_s42'


def write(path,data):
    """原子保存本阶段的新记录，不覆盖历史40k方案。"""
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def note(message):
    """将阶段变化、命令和实际耗时追加至本阶段doc。"""
    DOC.mkdir(parents=True,exist_ok=True)
    line=datetime.now().astimezone().isoformat()+' '+message
    with (DOC/'progress.md').open('a',encoding='utf-8') as stream:stream.write('\n- '+line+'\n')
    print(line,flush=True)


def status(phase,**details):
    """记录队列阶段，供作者手动查看，不重新启用Codex心跳。"""
    write(PHASE/'status.json',dict(phase=phase,updated=datetime.now().astimezone().isoformat(),**details))


def unchanged_runner():
    """核对当前CIDNet记录的冻结入口，拒绝静默改变训练计算。"""
    saved=json.loads((CID/'source.json').read_text())['runner']
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in saved.items()),'冻结训练入口变化'


def prepare():
    """用已有实测生成全部20k配置，只估计耗时，不按截止日期排除模型。"""
    unchanged_runner()
    PHASE.mkdir(exist_ok=False);DOC.mkdir()
    now=time.time()
    cid_profile=json.loads((HERE/'experiments/cidnet_profile.json').read_text())
    cid_status=json.loads((CID/'status.json').read_text())
    remaining=max(0,40000-cid_status['step'])
    cid_estimate=cost(cid_profile,remaining,2000)
    if (CID/'completed.json').exists():
        cid_estimate['total_seconds']=cid_estimate['test_seconds']+600
    if (CID/'final_test/summary.json').exists():cid_estimate['total_seconds']=0
    used=0.;rows=[];selected=[]
    for name in ORDER:
        profile=json.loads((HERE/f'experiments/{name}_profile.json').read_text())
        assert profile['status']=='profiled'
        estimate=cost(profile,20000,2000);spec=SPECS[name]
        row=dict(model=name,status='selected',estimate=estimate)
        row['planned_start']=datetime.fromtimestamp(now+cid_estimate['total_seconds']+used).astimezone().isoformat()
        used+=estimate['total_seconds'];selected.append(name)
        row['planned_complete_including_test']=datetime.fromtimestamp(now+cid_estimate['total_seconds']+used).astimezone().isoformat()
        out=HERE/'experiments'/(name+'_20k_s42')
        assert not out.exists(),'新的20k输出目录已存在，拒绝覆盖'
        cfg=dict(model=name,seed=42,batch_size=profile['batch_size'],accum_steps=profile['accum_steps'],crop_size=256,effective_batch=16,precision='fp32_tf32_disabled',max_steps=20000,warmup_steps=round(20000*spec['warmup_fraction']),eval_every=2000,output=str(out),training_spec=spec,deadline_epoch=1e30,author_disabled_time_limit=True,selection='validation_macro_PSNR',pretrained_weights=None,extra_training_data=None,estimated=estimate,adaptation='author_requested_20k_formal_CDD11_split_fixed256_single_GPU_budget_scaled_schedule',test_policy='one_final_validation_selected_checkpoint_no_test_tuning',profile_file=str(HERE/f'experiments/{name}_profile.json'))
        path=PHASE/(name+'_20k_config.json');write(path,cfg);row['config']=str(path);rows.append(row)
    plan=dict(created=datetime.now().astimezone().isoformat(),max_steps=20000,eval_every=2000,selected=selected,candidates=rows,remaining_cidnet_estimate=cid_estimate,cidnet_last_complete_validation=cid_status['step'],estimated_selected_seconds=used,deadline=None,author_disabled_time_limit=True,unchanged='current_CIDNet_40k_and_full_proposed_model',blocked_protocol=['airnet','snrnet','zero_dce_pp'],heartbeat='remains_paused')
    write(PHASE/'budget_plan.json',plan)
    lines=['# 后续对比模型20k方案：已取消时间限制','', '作者要求全部后续模型改为20000次更新后训练，并取消截止日期筛选；现有CIDNet仍40k。Codex定时审查保持暂停。','',f'全部五项含验证/测试约{used/3600:.2f}小时；另计当前CIDNet剩余约{cid_estimate["total_seconds"]/3600:.2f}小时。这些是含安全系数的估计，不是完成保证，也不是排除模型的条件。','', '| 模型 | batch×累积 | 更新/预热 | 含验证与测试小时 | 状态 | 预计含测试完成 |','|---|---|---|---|---|---|']
    for row in rows:
        cfg=json.loads(Path(row['config']).read_text())
        lines.append(f'|{row["model"]}|{cfg["batch_size"]}×{cfg["accum_steps"]}|20000/{cfg["warmup_steps"]}|{row["estimate"]["total_seconds"]/3600:.2f}|{row["status"]}|{row.get("planned_complete_including_test","未启动")}|')
    lines+=['','每2000步完整验证1100对，最终用验证PSNR最优权重测试2200对。沿用原模型、损失、优化器及调度形状，预热比例不变；PromptIR和MoCE预热2000步。', '20k是作者指定预算，并不证明各模型充分收敛。必须公开本文完整模型总投入80k、CIDNet40k与后续20k的差异，以及MoCE物理batch与路由平衡损失的关系。Histoformer/MoCE/Restormer均重新纳入；旧预算排除记录仅保留为历史。', 'AirNet、SNR-Net、Zero-DCE++不因时间排除，但关键训练协议仍须核实；不能把其错误适配直接接入。已核验五项先运行，不等待这些独立问题。', '服务器队列等当前GPU任务自然结束后接续，无需手动触发；不会恢复Codex定时审查。停止后续接续时仅退出本队列父进程，当前训练应保留。为复用冻结训练入口，新配置以1e30数值禁用其截止判断；模型训练公式没有修改。']
    (DOC/'budget.md').write_text('\n'.join(lines),encoding='utf-8')
    note('所有后续候选已生成独立20k配置；选中顺序：'+','.join(selected))
    print(json.dumps(plan,ensure_ascii=False,indent=2))


def wait_gpu():
    """保持单卡串行，只等待，不停止任何已有GPU进程。"""
    while True:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        if not result.stdout.strip():return True
        status('waiting_existing_gpu',processes=result.stdout.strip())
        time.sleep(30)


def execute(action,config,label):
    """只启动本阶段允许的训练/最终测试，保存真实退出码及耗时。"""
    if not wait_gpu():return 99
    unchanged_runner()
    cmd=[PYTHON,'-u','-B',str(HERE/'run.py'),action,'--config',str(config)]
    log=PHASE/(label+'_'+action+'.console.log')
    start=time.monotonic();note('启动：`'+' '.join(cmd)+'`')
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen(cmd,cwd=HERE,stdout=stream,stderr=subprocess.STDOUT)
        status(action,model=label,pid=child.pid,command=cmd,log=str(log))
        code=child.wait()
    row=dict(command=cmd,exit_code=code,seconds=time.monotonic()-start,finished=datetime.now().astimezone().isoformat())
    with (DOC/'commands.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    note(f'{label} {action}结束，exit={code}，耗时{row["seconds"]:.1f}秒')
    return code


def collect(plan):
    """汇总成功、失败和未启动状态，不按结果筛掉不利模型。"""
    rows=[]
    for candidate in plan['candidates']:
        cfg=json.loads(Path(candidate['config']).read_text());out=Path(cfg['output']);p=out/'final_test/summary.json'
        row=dict(model=cfg['model'],budget_status=candidate['status'],output=str(out),status='not_tested')
        if p.exists():row.update(status='tested',summary=json.loads(p.read_text()))
        elif (out/'failure.json').exists():row['status']='training_failed'
        elif (out/'completed.json').exists():row['status']='trained_test_pending'
        elif out.exists():row['status']='incomplete'
        rows.append(row)
    write(DOC/'results_inventory.json',rows)


def run():
    """等待现有CIDNet自然完成，先最终测试，再按新预算串行训练和测试。"""
    lock=(HERE/'queue.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('已有调度器占用锁，拒绝重复启动')
    (PHASE/'queue.pid').write_text(str(os.getpid()))
    plan=json.loads((PHASE/'budget_plan.json').read_text());unchanged_runner()
    note('20k队列开始，等待原CIDNet自然结束；定时审查仍暂停。')
    if not wait_gpu():return
    if (CID/'completed.json').exists() and not (CID/'final_test').exists():
        execute('test',HERE/'experiments/cidnet_config.json','cidnet_40k')
    elif not (CID/'completed.json').exists():note('CIDNet未正常完成，保留原任务状态，不自动恢复或重训；继续其他独立模型。')
    for row in plan['candidates']:
        if row['status']!='selected':continue
        config=Path(row['config']);cfg=json.loads(config.read_text());out=Path(cfg['output'])
        if (out/'final_test/summary.json').exists():continue
        if not wait_gpu():break
        if not (out/'completed.json').exists():
            if out.exists():note(row['model']+'已有不完整目录，保留并阻塞，不覆盖或盲目恢复');continue
            if execute('train',config,row['model']) or not (out/'completed.json').exists():collect(plan);continue
        execute('test',config,row['model']);collect(plan)
    collect(plan);status('queue_complete');note('本阶段预算队列处理完毕，不另行加训。')


def main():
    """准备和后台执行分开，配置先落盘供审阅，执行只读取冻结配置。"""
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:run()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if PHASE.exists():status('queue_failed',error=repr(error));note('队列异常：'+repr(error))
        traceback.print_exc();raise
