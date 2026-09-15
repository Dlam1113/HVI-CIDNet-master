"""仅接续作者指定的PromptIR；等待当前测试结束，不开启其他模型或定时审查。"""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
PYTHON='/home/Bjj/anaconda3/envs/CIDNet/bin/python'
RECORD=HERE/'doc/promptir_manual_launch'


def record(phase,**details):
    """原子记录等待与实际启动状态，不能把等待写成已开始训练。"""
    RECORD.mkdir(parents=True,exist_ok=True)
    value=dict(phase=phase,pid=os.getpid(),updated=datetime.now().astimezone().isoformat(),**details)
    tmp=RECORD/'status.tmp';tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(RECORD/'status.json')


def note(text):
    """在doc追加本次单模型接续进展，不改写旧记录。"""
    line=datetime.now().astimezone().isoformat()+' '+text
    with (RECORD/'progress.md').open('a',encoding='utf-8') as stream:stream.write('\n- '+line+'\n')
    print(line,flush=True)


def main():
    """持单次请求锁等待GPU，正常测试完成后只执行PromptIR手动入口。"""
    RECORD.mkdir(parents=True,exist_ok=True)
    lock=(RECORD/'request.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output=HERE/'experiments/promptir_20k_s42'
    if output.exists():raise RuntimeError('PromptIR输出已存在，先核查，不重复启动')
    record('waiting_gpu');note('作者授权启动PromptIR，等待CIDNet最终测试和GPU空闲。')
    while True:
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
        if query.returncode:raise RuntimeError(query.stderr)
        if not query.stdout.strip():break
        record('waiting_gpu',processes=query.stdout.strip());time.sleep(30)
    summary=HERE/'experiments/cidnet_40k_s42/final_test/summary.json'
    if not summary.exists():raise RuntimeError('GPU已空闲但CIDNet最终测试结果不存在，需核查测试退出情况。')
    command=[PYTHON,'-u','-B',str(HERE/'manual_model.py'),'promptir']
    note('启动命令：`'+' '.join(command)+'`')
    child=subprocess.Popen(command,cwd=HERE)
    record('manual_entry_running',child_pid=child.pid,command=command)
    code=child.wait()
    complete=(output/'completed.json').exists() and (output/'final_test/summary.json').exists()
    record('completed' if code==0 and complete else 'failed',exit_code=code,artifacts_complete=complete)
    note(f'PromptIR入口结束，退出码={code}，训练和最终测试产物完整={complete}。不会接续其他模型。')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        record('blocked_or_failed',error=repr(error));raise
