"""等待本轮已启动的测速子进程自然结束，再恢复修正后的调度器。"""
from pathlib import Path
import json
import os
import shutil
import sys
import time


def main():
    """不发送GPU进程信号；保留旧失败尝试，再在同一正式协议下重新测速。"""
    root=Path(__file__).resolve().parent
    child=1238547
    while (Path('/proc')/str(child)).exists():
        state=(Path('/proc')/str(child)/'stat').read_text().split()[2]
        if state=='Z':break
        time.sleep(15)
    backup=root/'deployment_backups/20260913_shape_budget_attempt';backup.mkdir(exist_ok=False)
    for suffix in ['json','log']:
        p=root/f'experiments/cidnet_profile_b16.{suffix}'
        if p.exists():shutil.move(str(p),str(backup/p.name))
    with (root/'doc/progress.md').open('a') as stream:
        stream.write('\n- 首次CIDNet测速的GPU验证自然结束，横竖图计数不成比例触发预算检查；失败结果与日志保留。改为按尺寸计时并加权估算，仅重做流程测速，不改正式清单或模型。\n')
    (root/'queue.pid').write_text(str(os.getpid()))
    os.execv(sys.executable,[sys.executable,'-u','-B',str(root/'orchestrate.py')])


if __name__=='__main__':main()
