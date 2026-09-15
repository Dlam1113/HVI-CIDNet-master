import warnings
from tqdm import tqdm
import numpy as np
import torch
import time
try:
    from thop import profile
    HAS_THOP = True
except Exception:
    HAS_THOP = False
from model import SARRefinementModel

# ============================================================
# 1. Confusion Matrix
# ============================================================

def fast_hist(pred, target, num_classes, ignore_index=255):
    """
    计算语义分割混淆矩阵。

    pred:   预测结果，shape = [H, W]，已经 argmax 后的类别图
    target: 标签图，shape = [H, W]
    num_classes: 类别数
    ignore_index: 忽略标签，比如 255
    """
    mask = (target != ignore_index) & (target >= 0) & (target < num_classes)

    pred = pred[mask]
    target = target[mask]

    hist = np.bincount(
        num_classes * target.astype(np.int64) + pred.astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

    return hist


# ============================================================
# 2. Segmentation Metrics
# ============================================================

def compute_seg_metrics(conf_mat):
    """
    根据混淆矩阵计算分割指标。

    输出：
    - iou: 每一类 IoU
    - dice: 每一类 Dice
    - miou_all: 所有类别平均 IoU，包含背景
    - miou_fg: 前景类别平均 IoU，不包含背景
    - mdice_all: 所有类别平均 Dice，包含背景
    - mdice_fg: 前景类别平均 Dice，不包含背景
    """
    eps = 1e-12

    tp = np.diag(conf_mat).astype(np.float64)
    gt = conf_mat.sum(axis=1).astype(np.float64)
    pd = conf_mat.sum(axis=0).astype(np.float64)

    iou = tp / np.maximum(gt + pd - tp, eps)
    dice = (2.0 * tp) / np.maximum(gt + pd, eps)

    metrics = {
        "iou": iou,
        "dice": dice,
        "miou_all": float(iou.mean() * 100.0),
        "miou_fg": float(iou[1:].mean() * 100.0),
        "mdice_all": float(dice.mean() * 100.0),
        "mdice_fg": float(dice[1:].mean() * 100.0),
    }

    return metrics


# ============================================================
# 3. Evaluate One Model
# ============================================================

@torch.no_grad()
def evaluate_model(model, loader, device, num_classes, ignore_index=255):
    """
    测试一个分割模型。

    model: 语义分割模型
    loader: 测试集 DataLoader
    device: cuda 或 cpu
    num_classes: 类别数
    ignore_index: 忽略标签
    """
    model.eval()

    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].numpy()

        logits = model(images)

        # 兼容模型返回 tuple/list 的情况
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        preds = logits.argmax(dim=1).cpu().numpy()

        for pred, target in zip(preds, masks):
            conf_mat += fast_hist(
                pred,
                target,
                num_classes=num_classes,
                ignore_index=ignore_index
            )

    metrics = compute_seg_metrics(conf_mat)
    return metrics


# ============================================================
# 4. Model Complexity
# ============================================================

def count_params_m(model):
    """
    计算模型参数量，单位是 M。
    """
    return sum(p.numel() for p in model.parameters()) / 1e6


def compute_macs_flops(model, input_shape=(1, 3, 200, 200)):
    """
    计算 MACs 和 FLOPs。

    需要安装 thop：
    pip install thop
    """
    if not HAS_THOP:
        return None, None

    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(*input_shape)

    try:
        macs, _ = profile(model_cpu, inputs=(dummy,), verbose=False)
        macs_g = macs / 1e9
        flops_g = 2.0 * macs / 1e9
        return macs_g, flops_g
    except Exception as e:
        warnings.warn(f"THOP failed: {e}")
        return None, None
# ============================================================
# 4.5 Latency and FPS Measurement (Model Only)
# ============================================================

# ============================================================
# 4.5 Latency and FPS Measurement (Model Only) - 优化版
# ============================================================

def compute_latency_fps(model, input_shape=(1, 3, 200, 200), device="cpu", num_runs=300, warmup=50):
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)

    # 🌟 核心优化：如果是 CPU，强行减少测试次数，否则会等很久
    if device == "cpu":
        warmup = min(warmup, 5)   # CPU 预热 5 次就够了
        num_runs = min(num_runs, 30) # CPU 测 30 次足以得出稳定平均值
        print(f"  [!] 检测到 CPU 测试，已自动将次数缩减为 -> 预热:{warmup}次, 测试:{num_runs}次")

    print(f"  -> 正在进行硬件预热 ({warmup} 次)...")
    with torch.no_grad():
        for _ in tqdm(range(warmup), desc="Warm-up", leave=False):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()

    latencies = []
    
    print(f"  -> 正在测量延迟与 FPS ({num_runs} 次)...")
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Testing"):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000.0)  # 转换为毫秒(ms)

    # 计算统计指标
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return p50, p95, fps

# ============================================================
# 5. Example Usage
# ============================================================

def print_metrics(metrics, class_names=None):
    """
    打印测试指标。
    """
    print(f"mIoU_all  : {metrics['miou_all']:.2f}%")
    print(f"mIoU_fg   : {metrics['miou_fg']:.2f}%")
    print(f"mDice_all : {metrics['mdice_all']:.2f}%")
    print(f"mDice_fg  : {metrics['mdice_fg']:.2f}%")

    if class_names is not None:
        print("\nClass-wise IoU:")
        for name, value in zip(class_names, metrics["iou"]):
            print(f"{name}: {value * 100.0:.2f}%")
            
def profile_model_complexity(model, input_shape=(1, 3, 200, 200), device="cpu"):
    """
    对模型进行全面体检，输出 Params, MACs, FLOPs, Latency 和 FPS
    """
    print("=" * 50)
    print("🚀 Model Complexity & Latency Profiler")
    print("=" * 50)
    print(f"Input Shape: {input_shape}, Test Device: {device.upper()}")
    
    # 1. 测参数量
    params_m = count_params_m(model)
    print(f"[+] Params     : {params_m:.3f} M")
    
    # 2. 测算力消耗
    macs_g, flops_g = compute_macs_flops(model, input_shape=input_shape)
    if macs_g is not None:
        print(f"[+] MACs       : {macs_g:.3f} G")
        print(f"[+] FLOPs      : {flops_g:.3f} G")
    else:
        print("[!] THOP module not installed, skipping MACs/FLOPs.")

    # 3. 测延迟与FPS (Model Only)
    p50, p95, fps = compute_latency_fps(model, input_shape=input_shape, device=device)
    print(f"[+] MO P50     : {p50:.2f} ms")
    print(f"[+] MO P95     : {p95:.2f} ms")
    print(f"[+] MO FPS     : {fps:.2f} frames/sec")
    print("=" * 50)

if __name__ == "__main__":

    

    # 2. 实例化你的 SGP-Lite 
    # ⚠️ 这里的参数必须和推理时一模一样，代表最轻量化的形态
    print("正在加载 SGP-Lite 模型...")
    sgp_lite_model = SARRefinementModel(
        in_chans=1, 
        num_classes=1, 
        embed_dim=64, 
        use_lsm=True
    )

    # 3. 设置极其重要的输入尺寸
    # Batch=1, Channel=1 (灰度图), Height=800, Width=800
    test_shape = (1, 1, 256, 256)

    # 4. 边缘设备模拟测试 (CPU)
    # 如果你要投 IoTJ 等强调边缘部署的期刊，CPU 下的延迟非常有说服力
    print("\n" + "*"*20 + " [测试场景 1: CPU 边缘推理] " + "*"*20)
    profile_model_complexity(sgp_lite_model, input_shape=test_shape, device="cpu")

    # 5. 云端/服务器极速测试 (GPU)
    if torch.cuda.is_available():
        # 强制使用空闲 GPU (假设是 GPU 1) 防止被其他任务干扰测速
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
        print("\n" + "*"*20 + " [测试场景 2: GPU 云端加速] " + "*"*20)
        profile_model_complexity(sgp_lite_model, input_shape=test_shape, device="cuda")