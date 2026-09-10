"""按 HTTP Range 分段下载官方文件；逐段续传并验证响应范围。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import time


def fetch_part(url, directory, index, start, end, attempts=8):
    """按已接收字节续传单段，网络失败不会丢失此前写入的部分。"""
    import requests

    directory = Path(directory)
    part = directory/("%05d.part" % index)
    temporary = directory/("%05d.tmp" % index)
    expected = end-start+1
    if part.exists():
        if part.stat().st_size != expected:
            raise ValueError("分段长度不正确：" + str(part))
        return part
    for attempt in range(attempts):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > expected:
            raise ValueError("临时分段超过预期长度")
        if offset == expected:
            temporary.replace(part)
            return part
        try:
            # 每次续传重新获取下载地址，避免长时间下载时临时签名过期。
            with requests.get(url, params={"download": "true", "segment": str(index),
                                           "offset": str(offset), "attempt": str(attempt)},
                              headers={"Range": "bytes=%d-%d" % (start+offset, end),
                                       "Accept-Encoding": "identity"},
                              timeout=(20, 60), stream=True) as response:
                response.raise_for_status()
                prefix = "bytes %d-%d/" % (start+offset, end)
                if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(prefix):
                    raise ValueError("服务器未返回请求的字节范围")
                with temporary.open("ab") as target:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:
                            if target.tell()+len(chunk) > expected:
                                raise ValueError("响应数据超过分段长度")
                            target.write(chunk)
            if temporary.stat().st_size != expected:
                raise IOError("网络响应提前结束")
            temporary.replace(part)
            return part
        except (requests.RequestException, OSError) as exc:
            if attempt+1 == attempts:
                raise RuntimeError("分段 %d 在 %d 次重试后失败：%s" % (index, attempts, type(exc).__name__)) from None
            time.sleep(min(30, 2**attempt))
    raise RuntimeError("下载分段未完成")


def ranged_download(url, target, size, workers=4, chunk_size=64*1024*1024):
    """并行下载可复用分段，再按编号顺序拼接为原始压缩包。"""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != size:
            raise ValueError("已有压缩包长度不匹配，请检查：" + str(target))
        return target
    directory = target.parent/"http_parts"/target.name
    directory.mkdir(parents=True, exist_ok=True)
    ranges = [(i, start, min(size-1, start+chunk_size-1))
              for i, start in enumerate(range(0, size, chunk_size))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_part, url, directory, i, start, end): i for i, start, end in ranges}
        complete = 0
        for future in as_completed(futures):
            future.result()
            complete += 1
            print("%s：完成分段 %d/%d" % (target.name, complete, len(ranges)), flush=True)
    assembled = target.with_name(target.name+".assembling")
    print("正在组装压缩包：" + target.name, flush=True)
    with assembled.open("wb") as output:
        for index, _, _ in ranges:
            with (directory/("%05d.part" % index)).open("rb") as source:
                shutil.copyfileobj(source, output, length=8*1024*1024)
    if assembled.stat().st_size != size:
        raise ValueError("组装后的长度不匹配")
    assembled.replace(target)
    return target


def cleanup_parts(target):
    """仅在调用方完成整包校验后移除本工具的临时分段，保留官方压缩包。"""
    target = Path(target).resolve()
    directory = target.parent/"http_parts"/target.name
    if directory.is_dir():
        for part in directory.iterdir():
            if part.is_file() and part.suffix in {".part", ".tmp"} and part.stem.isdigit():
                if not part.resolve().is_relative_to(target.parent):
                    raise ValueError("临时分段路径越界")
                part.unlink()
