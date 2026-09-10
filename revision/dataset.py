"""依据已审计清单加载成对图像，并提供可恢复的均衡任务采样。"""

from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler


class PairedManifestDataset(Dataset):
    """训练输入与清晰目标共用一次裁剪、翻转；评估保留原始分辨率。"""

    def __init__(self, root, manifest, crop_size=0):
        """读取清单并验证路径、重复条目、配对存在性和裁剪尺寸。"""
        self.root = Path(root).resolve()
        self.document = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.records = self.document["records"]
        self.crop_size = crop_size
        if crop_size < 0 or (crop_size and crop_size % 8):
            raise ValueError("训练裁剪尺寸需为正的 8 的倍数，评估用 0。")
        if not self.records:
            raise ValueError("样本清单为空")
        keys = [(r["scene_id"], r["task"]) for r in self.records]
        if len(set(keys)) != len(keys):
            raise ValueError("清单包含重复的场景与任务组合")
        for record in self.records:
            for field in ("input", "target"):
                path = (self.root / record[field]).resolve()
                if not path.is_relative_to(self.root) or not path.is_file():
                    raise ValueError("清单路径越界或文件不存在：" + str(path))

    def __len__(self):
        """返回退化图与目标图的配对数量。"""
        return len(self.records)

    def __getitem__(self, key):
        """用样本自带的增强种子，确保改变工作进程数或断点恢复不改变增强。"""
        if isinstance(key, tuple):
            index, augmentation_seed = key
        else:
            index, augmentation_seed = key, key
        rng = random.Random(augmentation_seed)
        record = self.records[index]
        with Image.open(self.root/record["input"]) as im:
            low = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(self.root/record["target"]) as im:
            high = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        if low.shape != high.shape:
            raise ValueError("成对图像尺寸不同：" + record["scene_id"])
        if self.crop_size:
            height, width = low.shape[:2]
            size = self.crop_size
            if min(height, width) < size:
                raise ValueError("训练图像小于裁剪尺寸：" + record["scene_id"])
            top, left = rng.randrange(height-size+1), rng.randrange(width-size+1)
            low, high = low[top:top+size, left:left+size], high[top:top+size, left:left+size]
            if rng.random() < 0.5:
                low, high = low[:, ::-1], high[:, ::-1]
            if rng.random() < 0.5:
                low, high = low[::-1], high[::-1]
        return {"input": torch.from_numpy(low.transpose(2, 0, 1).copy()),
                "target": torch.from_numpy(high.transpose(2, 0, 1).copy()),
                "task": record["task"], "scene_id": record["scene_id"]}


class StepBatchSampler(Sampler):
    """按优化步数采样：先等概率选任务，再选该任务的图像。"""

    def __init__(self, records, batch_size, accum_steps, total_steps, seed, start_step=0):
        """用全局优化步数生成独立种子，使恢复后样本序列与连续训练一致。"""
        if min(batch_size, accum_steps, total_steps) < 1 or not 0 <= start_step <= total_steps:
            raise ValueError("批量、累积次数及总步数无效")
        self.groups = defaultdict(list)
        for index, record in enumerate(records):
            self.groups[record["task"]].append(index)
        self.tasks = sorted(self.groups)
        self.batch_size, self.accum_steps = batch_size, accum_steps
        self.total_steps, self.start_step, self.seed = total_steps, start_step, seed

    def __len__(self):
        """返回剩余微批次数，而非优化器更新次数。"""
        return (self.total_steps-self.start_step) * self.accum_steps

    def __iter__(self):
        """每个样本同时携带固定增强种子，不使用工作进程的全局随机状态。"""
        for step in range(self.start_step, self.total_steps):
            for micro in range(self.accum_steps):
                rng = random.Random("%d:%d:%d" % (self.seed, step, micro))
                yield [(rng.choice(self.groups[rng.choice(self.tasks)]), rng.randrange(2**32))
                       for _ in range(self.batch_size)]


def assert_no_scene_overlap(train, validation):
    """禁止训练与验证清单共享同一清晰内容，也禁止用官方测试集选模型。"""
    if train.document["split"] != "train" or validation.document["split"] != "val":
        raise ValueError("训练入口只接受 train 和 val 清单，不能用 test 清单选择权重。")
    a = {r["target_sha256_rgb"] for r in train.records}
    b = {r["target_sha256_rgb"] for r in validation.records}
    if a & b:
        raise ValueError("训练与验证存在同一清晰场景")
    if {r["task"] for r in train.records} != {r["task"] for r in validation.records}:
        raise ValueError("训练与验证的任务集合不同")
