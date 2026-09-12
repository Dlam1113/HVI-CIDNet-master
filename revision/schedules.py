"""仅用标量计算更新步学习率；旧两段形状按更新次数重参数化。"""

import math


def continuation_lr_at_step(step, total, warmup, peak, initial):
    """从源权重学习率平滑升至小峰值，再单段余弦下降至1e-7。

    这是独立续训阶段的显式策略，不替换已完成实验的两段调度。
    预热至少两次更新，余弦段也至少两次，首尾都覆盖端点。
    """
    if any(type(x) is not int for x in (step, total, warmup)):
        raise ValueError("更新计数必须是整数")
    if not 0 <= step < total or not 2 <= warmup <= total - 2:
        raise ValueError("续训预热和余弦段各需至少两个更新")
    if any(not math.isfinite(x) or x <= 0 for x in (peak, initial)) or peak < 1e-7:
        raise ValueError("续训学习率必须有限且为正，峰值不得小于1e-7")
    if initial > peak:
        raise ValueError("源学习率高于指定峰值，不符合当前平滑升温策略")
    if step < warmup:
        return initial + (peak-initial) * step / (warmup-1)
    return _cosine_between(step-warmup, total-warmup, peak, 1e-7)


def _cosine_between(position, length, start, end):
    """计算含首尾端点的余弦插值，至少两个位置才能覆盖两个端点。"""
    if position == 0:
        return start
    if position == length - 1:
        return end
    fraction = position / (length - 1)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * fraction))


def lr_at_step(step, total, warmup, base, schedule="legacy_two_stage"):
    """返回从零编号的参数更新所用学习率，不执行优化器或模型计算。

    legacy_two_stage 保留旧实验先升至 2e-4、重启到 base、再降至
    1e-7 的形状，但按更新次数重新定义区间，不是逐 epoch 精确复现。
    前 warmup 个更新负责预热；warmup>1 时包含 0 和 base 两端，
    warmup=1 时唯一预热点取 base，warmup=0 时直接进入第一余弦段。
    第一余弦段为 [warmup, total//4-1]，第二段为
    [total//4, total-1]，各自至少两个更新，并且都包含端点。

    single_cosine 与此前 revision.experiment.learning_rate 的公式兼容：
    预热第一个更新用 base/warmup，之后单段下降至 1e-7。
    两种预热约定刻意分开，避免悄悄改变已记录的旧版新增协议。
    """
    for name, value in (("step", step), ("total", total), ("warmup", warmup)):
        if type(value) is not int:
            raise ValueError(name + " 必须是整数")
    if total < 1 or not 0 <= step < total or not 0 <= warmup < total:
        raise ValueError("要求 total>=1、0<=step<total、0<=warmup<total")
    if not isinstance(base, (int, float)) or not math.isfinite(base) or base <= 0:
        raise ValueError("base 必须是有限正数")
    if schedule not in {"legacy_two_stage", "single_cosine"}:
        raise ValueError("未知学习率调度：" + str(schedule))

    if schedule == "single_cosine":
        if step < warmup:
            return base * (step + 1) / max(1, warmup)
        fraction = (step - warmup) / max(1, total - warmup - 1)
        return 1e-7 + 0.5 * (base - 1e-7) * (1 + math.cos(math.pi * fraction))

    split = total // 4
    if split - warmup < 2 or total - split < 2:
        raise ValueError("两段余弦各需至少两个更新：total//4-warmup>=2 且 total-total//4>=2")
    if step < warmup:
        return base if warmup == 1 else base * step / (warmup - 1)
    if step < split:
        return _cosine_between(step - warmup, split - warmup, base, 2e-4)
    return _cosine_between(step - split, total - split, base, 1e-7)
