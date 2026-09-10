"""返修实验入口：显式清单、按更新步数训练、独立验证和完整测试。"""

import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from revision.cdd11 import sha256_file, write_json
from revision.dataset import PairedManifestDataset, StepBatchSampler, assert_no_scene_overlap
from revision.evaluation import evaluate, predict, save_evaluation


def source_info():
    """记录当前代码提交、工作区状态、解释器和 PyTorch 版本。"""
    root = Path(__file__).resolve().parents[1]
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda}


def build_model(name):
    """复用现有网络；四个名称对应 CIDNet 基线和曲线、RGB 细化消融组合。"""
    if name == "cidnet":
        from net.CIDNet import CIDNet
        return CIDNet()
    from net.DualSpaceCIDNet import DualSpaceCIDNet
    return DualSpaceCIDNet(use_curve=name in {"full", "curve"},
                           use_rgb_refiner=name in {"full", "refiner"},
                           refiner_mid_ch=64, curve_M=11)


def nonfinite_parameters(model):
    """列出含 NaN 或无穷值的参数，避免带着已知数值问题开始长训练。"""
    return [name for name, value in model.named_parameters() if not torch.isfinite(value).all()]


def seed_all(seed):
    """固定随机种子并关闭 cuDNN 自动选算法；具体环境仍随运行记录保存。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LegacyRestorationLoss(torch.nn.Module):
    """沿用 train.py 的 RGB/HVI 双域 L1、SSIM、边缘和 VGG 感知损失。"""

    def __init__(self):
        """损失权重与旧训练默认值一致，设备由外部 .to(device) 设置。"""
        super().__init__()
        from loss.losses import L1Loss, SSIM, EdgeLoss, PerceptualLoss
        self.l1 = L1Loss(loss_weight=1.0)
        self.ssim = SSIM(weight=0.5)
        self.edge = EdgeLoss(loss_weight=50.0)
        self.perceptual = PerceptualLoss(
            {"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},
            perceptual_weight=1.0, criterion="mse")

    def domain_loss(self, output, target):
        """计算一个颜色域的损失，VGG 项系数为 0.01。"""
        return (self.l1(output, target) + self.ssim(output, target) + self.edge(output, target)
                + 0.01*self.perceptual(output, target)[0])

    def forward(self, model, output, target):
        """将最终 RGB 输出重新转为 HVI 后监督，与旧入口保持一致。"""
        return self.domain_loss(output, target) + self.domain_loss(model.HVIT(output), model.HVIT(target))


def learning_rate(step, total, warmup, peak):
    """采用明确的线性预热与单次余弦下降；属于新增实验协议。"""
    if step < warmup:
        return peak*(step+1)/max(1, warmup)
    fraction = (step-warmup)/max(1, total-warmup-1)
    return 1e-7 + 0.5*(peak-1e-7)*(1+math.cos(math.pi*fraction))


def checkpoint(path, model, optimizer, step, best, config):
    """原子保存权重、优化器、随机状态与数据来源，支持准确恢复采样位置。"""
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "best_val_psnr": best, "config": config,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, temporary)
    temporary.replace(path)


def train(args):
    """按固定优化步数联合训练；仅使用 val 清单选择最佳权重。"""
    seed_all(args.seed)
    device = torch.device(args.device)
    train_set = PairedManifestDataset(args.root, args.train_manifest, args.crop_size)
    val_set = PairedManifestDataset(args.root, args.val_manifest)
    assert_no_scene_overlap(train_set, val_set)
    config = {"model": args.model, "seed": args.seed, "batch_size": args.batch_size,
        "accum_steps": args.accum_steps, "crop_size": args.crop_size, "max_steps": args.max_steps,
        "lr": args.lr, "warmup_steps": args.warmup_steps, "eval_every": args.eval_every,
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "val_manifest_sha256": sha256_file(args.val_manifest),
        "sampling": "task_uniform_then_scene_uniform_with_replacement",
        "scheduler": "linear_warmup_then_single_cosine", "gamma_augmentation": False,
        "loss": "legacy_rgb_hvi_l1_1_ssim_0.5_edge_50_vgg_0.01", "source": source_info()}
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best = 0, -math.inf
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        if state["config"] != config:
            raise ValueError("恢复配置或代码、清单来源发生变化，请建立新实验。")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best_val_psnr"]
    invalid = nonfinite_parameters(model)
    if invalid:
        raise FloatingPointError("模型参数含非有限值，暂不能开始正式训练：" + ", ".join(invalid))
    if args.output.exists() and not args.resume:
        raise FileExistsError("输出目录已存在；请指定新目录或显式恢复已有实验。")
    loss_fn = LegacyRestorationLoss().to(device)
    if args.resume:
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/"config.json", config)
    sampler = StepBatchSampler(train_set.records, args.batch_size, args.accum_steps,
                                args.max_steps, args.seed, start)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    with (args.output/"train.jsonl").open("a", encoding="utf-8") as log:
        for micro_index, batch in enumerate(loader):
            step = start + micro_index//args.accum_steps
            lr = learning_rate(step, args.max_steps, args.warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            image, target = batch["input"].to(device), batch["target"].to(device)
            value = loss_fn(model, model(image), target)
            if not torch.isfinite(value):
                raise FloatingPointError("训练损失非有限：step=%d" % step)
            (value/args.accum_steps).backward()
            running += value.item()/args.accum_steps
            if (micro_index+1) % args.accum_steps:
                continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step+1, "loss": running, "lr": lr, "grad_norm": float(norm)}
            running = 0.0
            if (step+1) % args.eval_every == 0 or step+1 == args.max_steps:
                rows, summary = evaluate(model, val_loader, device)
                record["val_macro_psnr"] = summary["groups"]["all"]["psnr"]
                improved = record["val_macro_psnr"] > best
                best = max(best, record["val_macro_psnr"])
                save_evaluation(args.output/("val_step_%07d" % (step+1)), rows, summary, config)
                checkpoint(args.output/"last.pt", model, optimizer, step+1, best, config)
                if improved:
                    checkpoint(args.output/"best.pt", model, optimizer, step+1, best, config)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (step+1) % 20 == 0 or step == start:
                print(json.dumps(record), flush=True)


def test(args):
    """从本入口生成的权重恢复模型，对指定清单执行统一指标评估。"""
    state = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(state["config"]["model"]).to(args.device)
    model.load_state_dict(state["model"], strict=True)
    dataset = PairedManifestDataset(args.root, args.manifest)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers)
    lpips_model = None
    if args.lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()
    rows, summary = evaluate(model, loader, args.device, lpips_model)
    provenance = {"checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": sha256_file(args.manifest), "training": state["config"],
        "source": source_info(), "metric_protocol": "float_RGB_0_1_no_border_gaussian_SSIM",
        "lpips": "alex_normalized_minus1_1" if args.lpips else None,
        "inference": "native_resolution_replicate_pad_multiple8_no_postprocessing"}
    save_evaluation(args.output, rows, summary, provenance)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def smoke(args):
    """用一个样本检查前后向与尺寸；结果明确标记为流程检查而非实验成绩。"""
    seed_all(args.seed)
    dataset = PairedManifestDataset(args.root, args.manifest, args.crop_size)
    sample = dataset[(0, args.seed)]
    model = build_model(args.model).to(args.device)
    image = sample["input"].unsqueeze(0).to(args.device)
    target = sample["target"].unsqueeze(0).to(args.device)
    output = predict(model, image)
    value = (output-target).abs().mean()
    value.backward()
    result = {"purpose": "pipeline_smoke_only_not_research_result", "model": args.model,
              "output_shape": list(output.shape), "l1": value.item(),
              "finite_output": bool(torch.isfinite(output).all()),
              "finite_gradients": all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
              "nonfinite_parameters": nonfinite_parameters(model), "source": source_info()}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["finite_output"] or not result["finite_gradients"]:
        raise FloatingPointError("流程检查出现非有限输出或梯度")


def main():
    """解析独立命令，不导入旧 train.py、eval.py 或 measure.py 的全局参数解析。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "test", "smoke"):
        command = sub.add_parser(action)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--device", default="cpu" if action == "smoke" else "cuda")
        command.add_argument("--workers", type=int, default=2)
        command.add_argument("--cpu-threads", type=int, default=2)
        command.add_argument("--output", type=Path, required=action != "smoke")
        if action in {"train", "smoke"}:
            command.add_argument("--model", choices=["cidnet", "refiner", "curve", "full"], default="full")
            command.add_argument("--crop-size", type=int, default=32 if action == "smoke" else 256)
            command.add_argument("--seed", type=int, default=42)
        if action == "train":
            command.add_argument("--train-manifest", type=Path, required=True)
            command.add_argument("--val-manifest", type=Path, required=True)
            command.add_argument("--max-steps", type=int, required=True)
            command.add_argument("--batch-size", type=int, default=2)
            command.add_argument("--accum-steps", type=int, default=8)
            command.add_argument("--lr", type=float, default=1e-4)
            command.add_argument("--warmup-steps", type=int, default=1000)
            command.add_argument("--eval-every", type=int, default=2000)
            command.add_argument("--resume", type=Path)
        else:
            command.add_argument("--manifest", type=Path, required=True)
        if action == "test":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--lpips", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.workers < 0:
        parser.error("CPU 线程数必须为正，工作进程数不能为负。")
    if args.action == "train" and (args.max_steps < 1 or args.eval_every < 1 or args.warmup_steps < 0
                                  or args.warmup_steps >= args.max_steps):
        parser.error("训练步数、验证间隔或预热步数无效。")
    torch.set_num_threads(args.cpu_threads)
    {"train": train, "test": test, "smoke": smoke}[args.action](args)


if __name__ == "__main__":
    main()
