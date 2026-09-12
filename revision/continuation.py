"""续训阶段的来源与配置核对；仅处理元数据，不执行模型计算。"""

import math


def comparable_config(config):
    """忽略运行环境，补齐旧版未记录的可选指标默认值。"""
    result = {key: value for key, value in config.items() if key != "source"}
    result.setdefault("val_lpips", False)
    return result


def continuation_metadata(state, config, path, digest):
    """仅允许新阶段改变预算和调度，保留网络、数据、批量及优化器状态。"""
    previous = comparable_config(state["config"])
    current = comparable_config(config)
    allowed = {"max_steps", "lr", "warmup_steps", "eval_every", "scheduler",
               "val_lpips", "continuation"}
    for key in previous.keys() | current.keys():
        if key not in allowed and previous.get(key) != current.get(key):
            raise ValueError("续训不能改变已验证的设置或数据：" + key)
    step = state["step"]
    if type(step) is not int or not 0 <= step <= previous["max_steps"]:
        raise ValueError("源权重的阶段步数无效")
    if not math.isfinite(state["best_val_psnr"]):
        raise ValueError("源权重缺少有效验证选择分数")
    groups = state["optimizer"]["param_groups"]
    rates = [group["lr"] for group in groups]
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("源优化器学习率无效")
    if len(set(rates)) != 1:
        raise ValueError("当前续训仅支持各参数组使用同一学习率")
    offset = previous.get("continuation", {}).get("parent_lineage_step", 0)
    return {"parent_checkpoint": str(path), "parent_sha256": digest,
            "parent_phase_step": step, "parent_lineage_step": offset + step,
            "parent_best_val_psnr": state["best_val_psnr"],
            "initial_lr": rates[0], "optimizer_policy": "restore_full_Adam_state",
            "parent_config": state["config"]}


def validation_fields(summary):
    """输出真实存在的全任务宏平均指标，缺测项不伪造为零。"""
    return {"val_macro_" + name: value for name, value in summary["groups"]["all"].items()
            if name in {"psnr", "ssim", "lpips"}}


def metric_text(fields):
    """以固定精度显示最近完整验证结果，LPIPS缺测时明确提示。"""
    return " | ".join(name.upper() + "=" + (
        "%.5f" % fields["val_macro_" + name] if "val_macro_" + name in fields else "未测")
        for name in ("psnr", "ssim", "lpips"))
