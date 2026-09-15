"""作者手动选择单个模型；防止与后台调度或其他GPU任务重复运行。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

HERE=Path(__file__).resolve().parent
SUPPLEMENT=HERE/'supplemental20k_20260913'
NORMAL=['promptir','histoformer','moce_ir_s','nafnet','restormer']
EXTRA=['airnet','snrnet','zero_dce_pp']
PYTHON='/home/Bjj/anaconda3/envs/CIDNet/bin/python'
# 作者最新授权的后续模型统一训练预算；PromptIR 仍保留其历史20k配置。
REMAINING_STEPS=10000
BUDGET_TAG='10k'


def note(message):
    """追加手动执行记录，不改写已有自动队列历史。"""
    text=datetime.now().astimezone().isoformat()+' '+message
    (HERE/'doc').mkdir(exist_ok=True)
    with (HERE/f'doc/manual{BUDGET_TAG}_progress.md').open('a',encoding='utf-8') as stream:
        stream.write('\n- '+text+'\n')
    print(text,flush=True)


def check_idle():
    """拒绝自动调度器仍运行或GPU仍被占用的情况，不停止任何进程。"""
    for folder,script in [(HERE/'phase20k_20260913','orchestrate_20k.py'),(SUPPLEMENT,'supplemental_queue.py')]:
        pidfile=folder/'queue.pid'
        if pidfile.exists():
            command=Path('/proc')/pidfile.read_text().strip()/'cmdline'
            if command.exists() and script.encode() in command.read_bytes():
                raise RuntimeError('后台接续仍在运行，请先按说明仅停止调度父进程：'+script)
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    if result.stdout.strip():raise RuntimeError('GPU仍有任务，请等当前测试或训练结束：\n'+result.stdout)


def remaining_config(model):
    """从已冻结的20k配置派生剩余模型的10k配置，并保留原配置作为历史记录。"""
    source=HERE/'phase20k_20260913'/(model+'_20k_config.json')
    target=HERE/'phase20k_20260913'/(model+f'_{BUDGET_TAG}_config.json')
    cfg=json.loads(source.read_text())
    cfg['max_steps']=REMAINING_STEPS
    cfg['warmup_steps']=round(REMAINING_STEPS*cfg['training_spec']['warmup_fraction'])
    cfg['output']=str(HERE/'experiments'/(model+f'_{BUDGET_TAG}_s42'))
    cfg['adaptation']='author_requested_10k_formal_CDD11_split_fixed256_single_GPU_budget_scaled_schedule'
    cfg['source_config']=str(source)
    estimate=cfg.get('estimated')
    if estimate:
        estimate['training_seconds']*=REMAINING_STEPS/20000
        estimate['validation_seconds']*=8/10
        estimate['total_seconds']=sum(estimate[key] for key in (
            'training_seconds','validation_seconds','test_seconds','startup_seconds'))
    serialized=json.dumps(cfg,ensure_ascii=False,indent=2)
    if target.exists() and target.read_text()!=serialized:
        raise RuntimeError('现有10k配置与当前授权不一致，拒绝静默覆盖：'+str(target))
    if not target.exists():
        temporary=target.with_suffix('.tmp')
        temporary.write_text(serialized,encoding='utf-8')
        temporary.replace(target)
        note(model+'已由冻结20k配置派生10k配置：'+str(target))
    return target


def main():
    """每次只处理指定模型；训练完成后按验证最优权重执行一次独立测试。"""
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=NORMAL+EXTRA)
    parser.add_argument('--prepare-only',action='store_true',help='只生成并核对配置，不启动训练或测试')
    args=parser.parse_args();check_idle()
    lock=(HERE/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_idle()
    if args.model in NORMAL:
        config=(HERE/'phase20k_20260913'/'promptir_20k_config.json'
                if args.model=='promptir' else remaining_config(args.model))
        runner=HERE/'run.py'
    else:
        sys.path.insert(0,str(SUPPLEMENT))
        import supplemental_queue as queue
        config=SUPPLEMENT/(args.model+f'_{BUDGET_TAG}_config.json')
        runner=SUPPLEMENT/'supplemental_run.py'
        if not config.exists():
            note(args.model+'尚无正式配置，先测速及完整验证；测速数值不作为论文结果。')
            # 复用已成功完成的完整测速，避免无意义地重复运行。
            candidates=[]
            for batch in [16,8,4,2,1]:
                path=SUPPLEMENT/f'{args.model}_profile_b{batch}.json'
                if path.exists():
                    report=json.loads(path.read_text())
                    if report.get('status')=='profiled':candidates.append(report)
            profile=candidates[0] if candidates else queue.profile(args.model)
            if profile is None:raise RuntimeError('测速未通过；保留错误记录，不修改模型或缩小评估图像。')
            config,_=queue.configure(args.model,profile)
    cfg=json.loads(config.read_text())
    expected_steps=20000 if args.model=='promptir' else REMAINING_STEPS
    assert cfg['max_steps']==expected_steps
    if args.prepare_only:
        note(args.model+'配置核对完成，未启动训练或测试：'+str(config));return
    output=Path(cfg['output'])
    if (output/'final_test/summary.json').exists():
        note(args.model+'已有最终测试结果，本次不重复运行：'+str(output));return
    actions=[]
    if not (output/'completed.json').exists():
        if output.exists():raise RuntimeError('发现未完成训练目录，请先核查断点，不覆盖或从零重跑：'+str(output))
        actions.append('train')
    if (output/'final_test').exists():raise RuntimeError('已有未完成测试目录，需先核查，不能覆盖：'+str(output/'final_test'))
    actions.append('test')
    for action in actions:
        check_idle()
        command=[PYTHON,'-u','-B',str(runner),action,'--config',str(config)]
        note('执行：`'+' '.join(command)+'`')
        code=subprocess.call(command,cwd=runner.parent)
        note(args.model+' '+action+'退出码='+str(code))
        if code:raise SystemExit(code)
    note(args.model+'训练及最终测试完成：'+str(output/'final_test/summary.json'))


if __name__=='__main__':main()
