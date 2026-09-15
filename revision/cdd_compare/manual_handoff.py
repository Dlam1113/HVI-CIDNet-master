"""作者手动运行此入口时，仅关闭两个接续父进程，保留训练和测试子进程。"""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import time

HERE=Path(__file__).resolve().parent


def main():
    """核对PID、入口名及工作目录后停止父进程，备份并追加交接记录。"""
    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=HERE/'doc'/('manual_handoff_'+timestamp);backup.mkdir(parents=True,exist_ok=False)
    messages=[]
    for folder,script,cwd in [
        (HERE/'supplemental20k_20260913','supplemental_queue.py',HERE/'supplemental20k_20260913'),
        (HERE/'phase20k_20260913','orchestrate_20k.py',HERE)]:
        pidfile=folder/'queue.pid'
        if not pidfile.exists():continue
        pid=int(pidfile.read_text());proc=Path('/proc')/str(pid)
        try:
            args=proc.joinpath('cmdline').read_bytes().decode().split('\0')
            matches=any(Path(arg).name==script for arg in args if arg)
            matches=matches and (proc/'cwd').resolve()==cwd.resolve()
        except FileNotFoundError:matches=False
        if not matches:
            messages.append(f'{script}原父进程已不在运行；未发送信号。');continue
        state=folder/'status.json'
        if state.exists():shutil.copy2(state,backup/(folder.name+'_status.json'))
        os.kill(pid,signal.SIGTERM)
        messages.append(f'仅向{script}父进程{pid}发送SIGTERM，未向子进程或进程组发送信号。')
        state.write_text(json.dumps(dict(phase='manual_takeover_current_child_preserved',
            previous_pid=pid,updated=datetime.now().astimezone().isoformat(),
            notice='训练或测试子进程继续；执行新模型前须检查GPU空闲。'),ensure_ascii=False,indent=2),encoding='utf-8')
    with (HERE/'doc/manual20k_progress.md').open('a',encoding='utf-8') as stream:
        for text in messages:stream.write('\n- '+timestamp+' '+text+'\n');print(text)
    print('交接完成。若GPU仍有训练或测试，请等待；不要重复启动同一模型。')


if __name__=='__main__':main()
