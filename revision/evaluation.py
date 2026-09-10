"""在原分辨率计算 RGB PSNR、SSIM 及可选 LPIPS，保存逐图与分组结果。"""

from collections import defaultdict
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from revision.cdd11 import SINGLE, DOUBLE, TRIPLE, write_json


def predict(model, image):
    """仅把边缘补齐到 8 的倍数，预测后裁回原尺寸，不改变图像分辨率。"""
    height, width = image.shape[-2:]
    image = F.pad(image, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
    return model(image)[..., :height, :width].clamp(0, 1)


def image_metrics(prediction, target):
    """使用浮点 RGB、范围 0 至 1、无裁边计算 PSNR 和高斯窗口 SSIM。"""
    from skimage.metrics import structural_similarity

    pred = prediction.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float64)
    truth = target.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float64)
    mse = float(np.mean((pred-truth)**2))
    psnr = -10*math.log10(mse) if mse else math.inf
    ssim = structural_similarity(truth, pred, data_range=1.0, channel_axis=2,
                                  gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return {"psnr": psnr, "ssim": float(ssim)}


def aggregate(rows):
    """分别汇总各任务和单重、双重、三重退化，并给出任务等权平均。"""
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    per_task = {}
    metric_names = [name for name in ("psnr", "ssim", "lpips") if name in rows[0]]
    for task, values in sorted(groups.items()):
        per_task[task] = {"pairs": len(values), **{
            name: float(np.mean([row[name] for row in values])) for name in metric_names}}
    summary = {"pairs": len(rows), "per_task": per_task, "groups": {}}
    for group, tasks in (("single", SINGLE), ("double", DOUBLE), ("triple", TRIPLE), ("all", tuple(groups))):
        included = [per_task[task] for task in tasks if task in per_task]
        if included:
            summary["groups"][group] = {name: float(np.mean([value[name] for value in included])) for name in metric_names}
    return summary


def evaluate(model, loader, device, lpips_model=None, progress=True):
    """评估清单内全部样本，不按测试分数筛选图片或改变模型权重。"""
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                output = predict(model, batch["input"].to(device))
                target = batch["target"].to(device)
                if not torch.isfinite(output).all():
                    raise FloatingPointError("评估输出包含非有限数值")
                for i in range(len(output)):
                    row = {"scene_id": batch["scene_id"][i], "task": batch["task"][i],
                           **image_metrics(output[i], target[i])}
                    if lpips_model is not None:
                        row["lpips"] = float(lpips_model(output[i:i+1]*2-1, target[i:i+1]*2-1).item())
                    rows.append(row)
                if progress and (index+1) % 50 == 0:
                    print("已评估 %d 张图像" % len(rows), flush=True)
    finally:
        model.train(was_training)
    return rows, aggregate(rows)


def save_evaluation(output, rows, summary, provenance):
    """保存逐图指标与配置来源；保留场景编号供后续配对统计分析。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/"per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output/"summary.json", {"provenance": provenance, **summary})
