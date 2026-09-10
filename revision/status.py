"""只读采集服务器数据准备进度，供本地进度页面展示。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from revision.cdd11 import ARCHIVES


def snapshot(root):
    """使用已分配磁盘块估算下载进度，避免把稀疏预分配文件当成下载完成。"""
    root = Path(root).resolve()
    state_path = root/"preparation_status.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"phase": "waiting"}
    files = []
    for filename, (size, checksum) in ARCHIVES.items():
        final = root/"downloads"/filename
        verified = (root/(filename+".verified.json")).exists()
        extracted = (root/(filename+".extracted.json")).exists()
        allocated = 0
        modified = None
        if final.exists():
            allocated = min(size, final.stat().st_size)
            modified = final.stat().st_mtime
        elif (root/"downloads"/"http_parts"/filename).is_dir():
            for part in (root/"downloads"/"http_parts"/filename).iterdir():
                if part.is_file() and part.suffix in {".part", ".tmp"}:
                    allocated += part.stat().st_size
                    modified = max(modified or 0, part.stat().st_mtime)
        else:
            for partial in (root/"downloads"/".cache"/"huggingface"/"download").glob("*"+checksum+"*.incomplete"):
                stat = partial.stat()
                allocated += min(size, getattr(stat, "st_blocks", 0)*512)
                modified = stat.st_mtime
        files.append({"name": filename, "label": "测试数据包" if filename == "test.zip" else "训练数据包",
                      "total_bytes": size, "written_bytes": min(size, allocated),
                      "downloaded": final.exists(), "verified": verified, "extracted": extracted,
                      "modified_timestamp": modified})
    log_path = root/"preparation.log"
    log_lines = []
    if log_path.exists():
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size-8192))
            tail = stream.read().decode("utf-8", errors="replace")
        log_lines = [re.sub(r"https?://\S+", "[下载服务地址]", line) for line in tail.splitlines()[-12:]]
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "state": state, "files": files,
            "total_bytes": sum(f["total_bytes"] for f in files),
            "written_bytes": sum(f["written_bytes"] for f in files),
            "free_bytes": shutil.disk_usage(root).free, "root": str(root), "log": log_lines,
            "progress_basis": "HTTP 分段按实际接收字节计数；Xet 旧缓存仅按磁盘块估算"}


def main():
    """向标准输出提供 JSON，既不修改数据也不控制下载进程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.root), ensure_ascii=False))


if __name__ == "__main__":
    main()
