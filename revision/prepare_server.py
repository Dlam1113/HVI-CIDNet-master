"""服务器数据准备流水线：可断点下载，校验后解压，最后生成固定清单。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from revision.cdd11 import download, extract, prepare, sha256_file, write_json


def run(root, train_count, val_count, seed):
    """顺序执行数据准备，持续保存阶段状态；不会启动 GPU 训练。"""
    import fcntl

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root/"manifests"/("%d_%d_seed%d" % (train_count, val_count, seed))
    status = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "train_scenes": train_count, "val_scenes": val_count, "seed": seed}
    with (root/"preparation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for phase, operation in (("download", download), ("extract", extract)):
                status["phase"] = phase
                write_json(root/"preparation_status.json", status)
                operation(root)
            status["phase"] = "audit_and_manifest"
            write_json(root/"preparation_status.json", status)
            if (output/"summary.json").exists():
                summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
                if (summary["counts"]["train_scenes"], summary["counts"]["val_scenes"], summary["seed"]) != (train_count, val_count, seed):
                    raise ValueError("既有清单配置不一致")
                for filename, info in summary["files"].items():
                    if sha256_file(output/filename) != info["sha256"]:
                        raise ValueError("既有清单摘要不一致：" + filename)
            else:
                summary = prepare(root/"official", output, train_count, val_count, seed)
            status.update(phase="complete", counts=summary["counts"], manifest_dir=str(output),
                          finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(root/"preparation_status.json", status)
            print("数据准备完成：" + str(output), flush=True)
        except Exception as exc:
            status.update(failed_phase=status["phase"], phase="failed", error=repr(exc))
            write_json(root/"preparation_status.json", status)
            raise


def main():
    """仅允许在服务器 Linux 环境执行下载和准备任务。"""
    if sys.platform != "linux":
        raise RuntimeError("请在 Ubuntu 服务器执行数据准备。")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=600)
    parser.add_argument("--val-scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    run(args.root, args.train_scenes, args.val_scenes, args.seed)


if __name__ == "__main__":
    main()
