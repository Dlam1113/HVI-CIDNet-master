"""五项队列完成后，依次测速并训练剩余三项；无截止日期，不占用现有GPU。"""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
DOC=HERE/'doc'
PYTHON='/home/Bjj/anaconda3/envs/CIDNet/bin/python'
ORDER=['airnet','snrnet','zero_dce_pp']
# 作者最新授权的补充模型统一训练预算与命名标签。
TOTAL_STEPS=10000
BUDGET_TAG='10k'


def write(path,value):
    """原子更新本队列记录，实验输出不复用历史目录。"""
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def note(message):
    """追加命令、配置和阶段结果，不删除既有进展。"""
    line=datetime.now().astimezone().isoformat()+' '+message
    DOC.mkdir(exist_ok=True)
    with (DOC/'progress.md').open('a',encoding='utf-8') as stream:stream.write('\n- '+line+'\n')
    print(line,flush=True)


def status(phase,**details):
    """记录等待、运行和失败状态，不启动Codex定时审查。"""
    write(HERE/'status.json',dict(phase=phase,updated=datetime.now().astimezone().isoformat(),**details))


def wait_gpu():
    """等待全部已有GPU计算自然结束，不发送停止或重启信号。"""
    while True:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        if not result.stdout.strip():return
        status('waiting_gpu',pids=result.stdout.strip());time.sleep(30)


def execute(model,action,extra,label):
    """执行一个独立子进程并记录退出码，模型异常不会伪装为已完成。"""
    wait_gpu()
    command=[PYTHON,'-u','-B',str(HERE/'supplemental_run.py'),action,'--model',model]+extra
    log=HERE/(label+'.console.log')
    start=time.monotonic();note('启动：`'+' '.join(command)+'`')
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=HERE,stdout=stream,stderr=subprocess.STDOUT)
        status(action,model=model,pid=child.pid,command=command,log=str(log))
        code=child.wait()
    record=dict(command=command,exit_code=code,seconds=time.monotonic()-start,
                finished=datetime.now().astimezone().isoformat())
    with (DOC/'commands.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    note(f'{model} {action}返回{code}，耗时{record["seconds"]:.1f}秒')
    return code,log


def profile(model):
    """优先batch16，仅遇明确显存不足时依次降批量，保留有效batch16。"""
    for batch in [16,8,4,2,1]:
        target=HERE/f'{model}_profile_b{batch}.json'
        if target.exists():raise RuntimeError('已有测速结果，拒绝无判断重复运行：'+str(target))
        code,log=execute(model,'bench',['--batch',str(batch),'--output',str(target)],f'{model}_profile_b{batch}')
        report=json.loads(target.read_text()) if target.exists() else {}
        if code==0 and report.get('status')=='profiled':return report
        error=log.read_text(errors='replace')[-18000:].lower()
        if not any(token in error for token in ['cuda out of memory','memory_margin','outofmemoryerror']):
            note(model+'非显存错误，记录阻塞，不擅自更改方法');return None
        if report.get('stage')=='native_validation_profile':
            note(model+'原尺寸验证显存不足，降低训练batch无法解决；不擅自缩图或分块');return None
    return None


def configure(model,profile):
    """实测后固定10k预算与原方法日程；训练输出独立、验证选优规则固定。"""
    # 导入只提供元数据，不构造模型或访问GPU。
    from supplemental_run import SPECS
    spec=SPECS[model]
    update=max(profile['update_mean']*1.2,profile['update_p90']*1.1)
    train_seconds=TOTAL_STEPS*update
    if model=='airnet':train_seconds=(TOTAL_STEPS-SPECS[model]['encoder_steps'])*update+SPECS[model]['encoder_steps']*profile['encoder_update_mean']*1.2
    estimate=dict(training_seconds=train_seconds,validation_seconds=8*profile['validation_seconds']*1.25,
                  test_seconds=profile['test_seconds_estimate']*1.3,startup_seconds=600)
    estimate['total_seconds']=sum(estimate.values())
    output=BASE/'experiments'/(model+f'_{BUDGET_TAG}_s42')
    assert not output.exists(),'已有训练输出，拒绝覆盖'
    cfg=dict(model=model,seed=42,batch_size=profile['batch_size'],accum_steps=profile['accum_steps'],
        effective_batch=16,crop_size=256,precision='fp32_tf32_disabled',max_steps=TOTAL_STEPS,warmup_steps=0,
        eval_every=2000,output=str(output),training_spec=spec,deadline_epoch=1e30,
        author_disabled_time_limit=True,selection='validation_macro_PSNR',pretrained_weights=None,
        extra_training_data=None,estimated=estimate,adaptation=profile['metadata'],
        test_policy='one_final_validation_selected_checkpoint_no_test_tuning')
    path=HERE/(model+f'_{BUDGET_TAG}_config.json');write(path,cfg)
    note(f'{model}测速通过，10k含验证测试估计{estimate["total_seconds"]/3600:.2f}小时；未设截止筛选')
    return path,cfg


def collect():
    """汇总所有三项状态，包括阻塞和不利结果，不选择性排除。"""
    rows=[]
    for model in ORDER:
        out=BASE/'experiments'/(model+f'_{BUDGET_TAG}_s42');result=out/'final_test/summary.json'
        row=dict(model=model,output=str(out),max_steps=TOTAL_STEPS,status='pending')
        if result.exists():row.update(status='tested',summary=json.loads(result.read_text()))
        elif (out/'completed.json').exists():row['status']='trained_test_pending'
        elif (out/'failure.json').exists():row['status']='training_failed'
        elif out.exists():row['status']='incomplete'
        elif (HERE/(model+'_blocked.json')).exists():row.update(status='blocked',details=json.loads((HERE/(model+'_blocked.json')).read_text()))
        rows.append(row)
    write(DOC/'results_inventory.json',rows)


def main():
    """等待前五项队列正式结束，持共享锁后先实测三项，再串行训练和测试。"""
    own=(HERE/'queue.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'queue.pid').write_text(str(os.getpid()))
    note('补充三项10k接续已启动；等待前五项处理完成，不恢复定时审查。')
    while True:
        previous=json.loads((BASE/'phase20k_20260913/status.json').read_text())
        if previous['phase']=='queue_complete':break
        if previous['phase']=='queue_failed':raise RuntimeError('前序队列异常，需人工核查后接续，避免任务重叠')
        status('waiting_first_five_queue',previous_phase=previous['phase']);time.sleep(30)
    lock=(BASE/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    configs=[]
    for model in ORDER:
        report=profile(model)
        if report is None:
            write(HERE/(model+'_blocked.json'),dict(reason='profile_failed_see_logs',max_steps_requested=TOTAL_STEPS))
            collect();continue
        config,cfg=configure(model,report);configs.append((model,config,cfg));collect()
    write(DOC/'measured_budget.json',dict(models=[cfg for _,_,cfg in configs],
        total_estimated_seconds=sum(cfg['estimated']['total_seconds'] for _,_,cfg in configs),deadline=None))
    for model,config,cfg in configs:
        code,_=execute(model,'train',['--config',str(config)],model+'_train')
        if code==0 and (Path(cfg['output'])/'completed.json').exists():
            execute(model,'test',['--config',str(config)],model+'_test')
        collect()
    collect();status('queue_complete');note('三项已逐一处理，具体完成或阻塞见results_inventory.json；不自动追加预算。')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('queue_failed',error=repr(error));note('异常：'+repr(error));traceback.print_exc();raise
