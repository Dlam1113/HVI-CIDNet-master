"""首次部署时备份实际代码及配置；不复制数据或权重、不改原仓库。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
MAIN=Path('/home/Bjj/HVI-CIDNet-clean')
EXTERNAL=Path('/home/Bjj/comparison_models')


def sha(path):
    """记录实际复制的每个文件摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """在独立目录创建带部署时间记录的原文件备份，已存在时拒绝覆盖。"""
    snapshot=HERE/'snapshot';snapshot.mkdir(exist_ok=False)
    files={};origins={}
    definitions={
        'cidnet':(MAIN,'main',['net','data','loss','revision']),
        'promptir':(EXTERNAL/'PromptIR','external/PromptIR',['net','train.py','options.py','README.md']),
        'histoformer':(EXTERNAL/'Histoformer','external/Histoformer',['basicsr','options','README.md']),
        'moce_ir_s':(EXTERNAL/'MoCE-IR','external/MoCE-IR',['src','README.md']),
        'nafnet':(EXTERNAL/'NAFNet','external/NAFNet',['basicsr','options','README.md']),
        'restormer':(EXTERNAL/'Restormer','external/Restormer',['basicsr','Deraining/Options','README.md'])}
    for name,(root,dest,items) in definitions.items():
        paths=[]
        for item in items:
            p=root/item
            if p.is_file():paths.append(p)
            elif p.is_dir():paths.extend(x for x in p.rglob('*') if x.is_file() and x.suffix in ['.py','.yaml','.yml'] and '__pycache__' not in x.parts)
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True).stdout.strip()
        relative=[str(p.relative_to(root)) for p in paths]
        diff=subprocess.run(['git','diff','--',*relative],cwd=root,text=True,capture_output=True).stdout
        origins[name]={'original_root':str(root),'backup_root':str(snapshot/dest),'git_commit':commit,'existing_source_diff':diff,'files':len(paths),'copied_at':datetime.now().astimezone().isoformat()}
        for p in paths:
            target=snapshot/dest/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
            assert sha(p)==sha(target)
            files[str(target.relative_to(HERE))]=sha(target)
    result={'created':datetime.now().astimezone().isoformat(),'policy':'originals_unchanged_independent_source_backup_no_data_no_weights','origins':origins,'files':files}
    (HERE/'snapshot_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(files),'origins':origins},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
