"""下载、校验和固定划分 CDD-11；数据只写入服务器指定目录。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import stat
import sys
import zipfile

REPO_ID = "gy65896/CDD-11"
REVISION = "52a185b4266f6b1093b6391294ce2c230cc35c6c"
ARCHIVES = {
    "test.zip": (3770337157, "421567fceda89659a6d047fc78d82f77c68703203336d9c795213ba7096646dc"),
    "train.zip": (22403961860, "34cc619a674fa16d13a8b1d38586662d2b5ca235e57de299694968d2d07675bd"),
}
SINGLE = ("low", "haze", "rain", "snow")
DOUBLE = ("low_haze", "low_rain", "low_snow", "haze_rain", "haze_snow")
TRIPLE = ("low_haze_rain", "low_haze_snow")
TASKS = SINGLE + DOUBLE + TRIPLE


def sha256_file(path):
    """分块计算文件摘要，避免把大型压缩包全部读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """通过临时文件原子保存 UTF-8 记录，避免中断留下半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def download(root):
    """在 Linux 服务器下载固定版本原始压缩包，并核对官方大小与 SHA-256。"""
    if sys.platform != "linux":
        raise RuntimeError("按项目约定，数据集下载只能在 Linux 服务器执行。")
    from revision.http_download import ranged_download, cleanup_parts

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    required = 2*sum(size for size, _ in ARCHIVES.values()) + 40 * 1024**3
    existing = sum(p.stat().st_size for p in (root / "downloads").glob("*.zip"))
    if shutil.disk_usage(root).free < required - existing:
        raise RuntimeError("可用磁盘空间不足以同时保留压缩包和解压数据。")
    for filename, (size, checksum) in ARCHIVES.items():
        print("开始下载：" + filename, flush=True)
        archive = ranged_download(
            "https://huggingface.co/datasets/%s/resolve/%s/%s" % (REPO_ID, REVISION, filename),
            root/"downloads"/filename, size)
        if archive.stat().st_size != size or sha256_file(archive) != checksum:
            raise RuntimeError("压缩包大小或 SHA-256 校验失败：" + filename)
        write_json(root / (filename + ".verified.json"), {
            "repo_id": REPO_ID, "revision": REVISION,
            "filename": filename, "bytes": size, "sha256": checksum,
        })
        cleanup_parts(archive)
        print("完整性校验通过：" + filename, flush=True)


def extract_archive(archive, destination):
    """校验所有压缩路径，逐文件读取并检查 CRC；保留原始压缩包。"""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination) or "\\" in entry.filename:
                raise ValueError("压缩包含越界路径：" + entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("压缩包含符号链接：" + entry.filename)
        for index, entry in enumerate(source.infolist()):
            target = destination / entry.filename
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            with source.open(entry) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            temporary.replace(target)
            if index % 500 == 0:
                print("已解压 %d 个条目" % index, flush=True)


def extract(root):
    """仅解压已通过官方 SHA-256 校验的训练和测试包。"""
    root = Path(root).resolve()
    for filename, (size, checksum) in ARCHIVES.items():
        archive = root / "downloads" / filename
        if not archive.exists() or archive.stat().st_size != size or sha256_file(archive) != checksum:
            raise RuntimeError("请先下载并校验：" + filename)
        marker = root / (filename + ".extracted.json")
        if marker.exists():
            print("解压记录已存在，将在清单阶段重新检查配对：" + filename, flush=True)
            continue
        extract_archive(archive, root / "official")
        write_json(marker, {"archive_sha256": checksum, "revision": REVISION})


def image_names(directory):
    """列出一层目录中的图像，忽略系统隐藏文件而不默默跳过缺失目录。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(str(directory))
    return sorted(p.name for p in directory.iterdir()
                  if p.is_file() and not p.name.startswith(".")
                  and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


def find_split(root, split):
    """识别官方包内的 train 或 CDD-11_train 命名，不依赖固定解压层数。"""
    root = Path(root).resolve()
    candidates = [p.parent for p in root.rglob("clear")
                  if p.is_dir() and p.parent.name.lower() in {split, "cdd-11_" + split, "cdd11_" + split}]
    if len(candidates) != 1:
        raise ValueError("无法唯一识别 %s 划分：%s" % (split, candidates))
    return candidates[0]


def inspect_split(directory, tasks=TASKS):
    """检查配对与解码有效性，返回全部原图及摘要，重复内容交由分组规则处理。"""
    from PIL import Image

    directory = Path(directory)
    names = image_names(directory / "clear")
    if not names:
        raise ValueError("清晰图目录为空")
    expected = set(names)
    for task in tasks:
        actual = set(image_names(directory / task))
        if actual != expected:
            raise ValueError("%s 配对不齐：缺失 %d，多余 %d" % (task, len(expected-actual), len(actual-expected)))
    hashes = {}
    for i, name in enumerate(names):
        with Image.open(directory / "clear" / name) as im:
            clear = im.convert("RGB")
            shape = clear.size
            hashes[name] = hashlib.sha256(str(shape).encode() + clear.tobytes()).hexdigest()
        for task in tasks:
            with Image.open(directory / task / name) as im:
                if im.size != shape:
                    raise ValueError("尺寸不匹配：%s/%s" % (task, name))
                im.verify()
        if i % 100 == 0:
            print("%s 已检查 %d/%d 个场景" % (directory.name, i, len(names)), flush=True)
    return names, hashes


def group_clean_content(names, hashes):
    """按解码清晰内容分组，以组内字典序最小文件名作为固定代表，不删除原图。"""
    groups = {}
    for name in sorted(names):
        groups.setdefault(hashes[name], []).append(name)
    representatives, duplicates = [], []
    for checksum, members in sorted(groups.items(), key=lambda item: item[1][0]):
        representative = members[0]
        representatives.append(representative)
        if len(members) > 1:
            duplicates.append({"target_sha256_rgb": checksum,
                               "representative": representative,
                               "members": members, "excluded_names": members[1:]})
    return representatives, duplicates


def select_scenes(names, train_count, val_count, seed):
    """对排序后的唯一内容代表做固定种子抽样，训练、验证与备用集合互斥。"""
    if train_count < 1 or val_count < 1 or train_count + val_count > len(names):
        raise ValueError("训练和验证场景数必须为正且不超过唯一清晰内容组数量。")
    ordered = sorted(names)
    random.Random(seed).shuffle(ordered)
    return (sorted(ordered[:train_count]), sorted(ordered[train_count:train_count+val_count]),
            sorted(ordered[train_count+val_count:]))


def make_records(directory, names, tasks, hashes, root):
    """以相对路径生成跨 Windows/Linux 可复用的成对样本清单。"""
    return [{"scene_id": directory.name + "/" + name,
             "task": task, "input": (directory/task/name).relative_to(root).as_posix(),
             "target": (directory/"clear"/name).relative_to(root).as_posix(),
             "target_sha256_rgb": hashes[name]}
            for name in names for task in tasks]


def prepare(root, output, train_count=600, val_count=100, seed=20260910, expected_counts=(1183, 200)):
    """按唯一清晰内容组划分训练和验证，完整保留官方测试，并保存可复核审计。"""
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("清单目录非空，拒绝覆盖已有实验划分：" + str(output))
    train_dir, test_dir = find_split(root, "train"), find_split(root, "test")
    train_names, train_hashes = inspect_split(train_dir)
    test_names, test_hashes = inspect_split(test_dir)
    if expected_counts and (len(train_names), len(test_names)) != expected_counts:
        raise ValueError("官方场景数与预期不一致：%s" % ((len(train_names), len(test_names)),))
    representatives, duplicate_groups = group_clean_content(train_names, train_hashes)
    _, test_duplicates = group_clean_content(test_names, test_hashes)
    if test_duplicates:
        raise ValueError("官方测试划分存在重复清晰内容，需核实；不会自动删减官方测试集。")
    if set(train_hashes.values()) & set(test_hashes.values()):
        raise ValueError("官方训练与测试出现相同清晰图像内容，停止生成清单。")
    selected, validation, unused = select_scenes(representatives, train_count, val_count, seed)
    excluded = sorted(name for group in duplicate_groups for name in group["excluded_names"])
    selected_hashes = {train_hashes[name] for name in selected}
    validation_hashes = {train_hashes[name] for name in validation}
    if selected_hashes & validation_hashes:
        raise ValueError("训练与验证出现同一清晰内容，停止生成清单。")
    for group in duplicate_groups:
        representative = group["representative"]
        group["representative_split"] = ("train" if representative in selected else
                                         "val" if representative in validation else "unused")
    protocol = {
        "name": "official_train_unique_clear_content_representative_split_v1",
        "content_hash": "SHA-256(str(PIL_RGB_size).encode() + decoded_RGB_bytes)",
        "representative_rule": "lexicographically_smallest_filename_per_identical_clear_content_group",
        "sampling_rule": "sort_representatives_then_random.Random(seed).shuffle; train_then_val_then_unused",
        "selected_scene_tasks": "keep_all_11_degradations_for_each_selected_representative",
        "test_rule": "preserve_all_official_test_scenes_without_sampling",
        "raw_images_modified_or_deleted": False,
    }
    summary = {"repo_id": REPO_ID, "revision": REVISION, "seed": seed,
               "train_scene_ids": selected, "val_scene_ids": validation,
               "unused_train_scene_ids": sorted(unused + excluded),
               "unused_unique_train_scene_ids": unused,
               "excluded_duplicate_train_scene_ids": excluded, "test_scene_ids": test_names,
               "protocol": protocol,
               "audit": {"official_train_scenes": len(train_names),
                         "official_test_scenes": len(test_names),
                         "unique_train_contents": len(representatives),
                         "unique_test_contents": len(set(test_hashes.values())),
                         "duplicate_train_groups": duplicate_groups,
                         "excluded_duplicate_train_files": len(excluded),
                         "representative_scene_ids": representatives,
                         "train_val_content_overlap": 0,
                         "official_train_test_content_overlap": 0},
               "counts": {"train_scenes": len(selected), "val_scenes": len(validation),
                          "test_scenes": len(test_names)}, "files": {}}
    for mode, tasks in (("all11", TASKS), ("single4", SINGLE)):
        for split, directory, names, hashes in (
            ("train", train_dir, selected, train_hashes),
            ("val", train_dir, validation, train_hashes),
            ("test", test_dir, test_names, test_hashes),
        ):
            filename = mode + "_" + split + ".json"
            records = make_records(directory, names, tasks, hashes, root)
            write_json(output / filename, {"schema_version": 1, "mode": mode, "split": split,
                "source_revision": REVISION, "seed": seed, "records": records})
            summary["files"][filename] = {"pairs": len(records), "sha256": sha256_file(output/filename)}
    write_json(output / "summary.json", summary)
    print(json.dumps({"counts": summary["counts"], "files": summary["files"]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    """提供可独立执行的下载、解压和清单生成命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "extract", "prepare"])
    parser.add_argument("--root", type=Path, required=True, help="服务器 CDD-11 数据根目录")
    parser.add_argument("--output", type=Path, help="清单输出目录")
    parser.add_argument("--train-scenes", type=int, default=600)
    parser.add_argument("--val-scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.action == "download":
        download(args.root)
    elif args.action == "extract":
        extract(args.root)
    else:
        if args.output is None:
            parser.error("prepare 需要 --output")
        prepare(args.root / "official", args.output, args.train_scenes, args.val_scenes, args.seed)


if __name__ == "__main__":
    main()
