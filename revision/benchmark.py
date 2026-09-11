"""由用户手动运行的短程测速；不生成正式权重，不启动长训练。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from revision.cdd11 import TASKS, sha256_file, write_json


def parse_args():
    """解析测速参数；热身指计时热身，与正式学习率预热无关。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/CDD-11"))
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("full", "cidnet", "refiner", "curve"), default="full")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--measure-updates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", action="store_true", help="可选旧入口的0.60至1.20随机gamma；默认与revision入口一致关闭")
    parser.add_argument("--anomaly-detection", action="store_true", help="可选逐算子异常追踪；默认关闭，仍严格检查数值")
    parser.add_argument("--single-batch", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.warmup_updates < 1 or args.measure_updates < 10:
        parser.error("batch与热身更新数须为正，正式计时至少10次更新。")
    if args.workers < 0 or args.cpu_threads < 1:
        parser.error("数据进程数不能为负，CPU线程数须为正。")
    args.data_root = args.data_root.resolve()
    args.manifest_dir = (args.manifest_dir or args.data_root/"manifests/600_100_seed20260910").resolve()
    args.output = args.output.resolve()
    return args


def check_manifest_bundle(args):
    """验证正式6600/1100清单及摘要，禁止测速意外使用测试集或缩小验证集。"""
    summary = json.loads((args.manifest_dir/"summary.json").read_text(encoding="utf-8"))
    documents = {}
    for split, scenes in (("train", 600), ("val", 100)):
        filename = "all11_%s.json" % split
        path = args.manifest_dir/filename
        if sha256_file(path) != summary["files"][filename]["sha256"]:
            raise ValueError("清单摘要不匹配：" + filename)
        document = json.loads(path.read_text(encoding="utf-8"))
        records = document["records"]
        if document["split"] != split or len(records) != scenes*len(TASKS):
            raise ValueError("测速要求正式600训练场景与100验证场景的11任务清单。")
        by_task = {task: [r for r in records if r["task"] == task] for task in TASKS}
        scene_sets = [{r["scene_id"] for r in by_task[task]} for task in TASKS]
        if any(len(by_task[task]) != scenes for task in TASKS) or any(s != scene_sets[0] for s in scene_sets):
            raise ValueError("11任务的场景集合或数量不一致。")
        if len({r["target_sha256_rgb"] for r in records}) != scenes:
            raise ValueError("同一清单出现重复清晰内容。")
        documents[split] = document
    if ({r["target_sha256_rgb"] for r in documents["train"]["records"]}
            & {r["target_sha256_rgb"] for r in documents["val"]["records"]}):
        raise ValueError("训练与验证清晰内容重叠。")
    return summary


def check_idle_gpu():
    """避免手动测速与已有GPU计算进程抢占单张3090。"""
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                             "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    if result.stdout.strip():
        raise RuntimeError("GPU已有计算进程；请等待结束后手动重试：\n" + result.stdout.strip())


def run_parent(args):
    """按16、8顺序尝试；只在训练显存不足时回退，子进程退出后彻底释放显存。"""
    check_manifest_bundle(args)
    check_idle_gpu()
    args.output.mkdir(parents=True, exist_ok=False)
    attempts = []
    for batch in args.batch_sizes:
        destination = args.output/("batch_%d" % batch)
        command = [sys.executable, "-u", "-B", "-m", "revision.benchmark",
                   "--data-root", str(args.data_root), "--manifest-dir", str(args.manifest_dir),
                   "--output", str(destination), "--model", args.model, "--single-batch", str(batch),
                   "--warmup-updates", str(args.warmup_updates), "--measure-updates", str(args.measure_updates),
                   "--workers", str(args.workers), "--cpu-threads", str(args.cpu_threads), "--seed", str(args.seed)]
        for flag, enabled in (("--gamma", args.gamma), ("--anomaly-detection", args.anomaly_detection)):
            if enabled:
                command.append(flag)
        print("\n开始短程测速：batch=%d，累积1，FP32；成功后完整验证1100对。" % batch, flush=True)
        code = subprocess.run(command).returncode
        report_path = destination/"report.json"
        result = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"status": "failed", "exit_code": code}
        attempts.append({"batch": batch, "report": str(report_path), "result": result})
        summary = {"purpose": "manual_short_benchmark_only", "attempts": attempts,
                   "selected_batch": batch if code == 0 else None,
                   "status": "complete" if code == 0 else "incomplete"}
        write_json(args.output/"summary.json", summary)
        if code == 0:
            print("测速完成，报告：" + str(args.output/"summary.json"), flush=True)
            return 0
        if code != 20:
            print("检测到非训练显存问题，停止；请检查报告，不能自动改模型或验证方式。", flush=True)
            return code or 1
        print("训练显存不足，结束本进程后尝试下一个batch。", flush=True)
    return 20


def timing_statistics(values):
    """同时给出均值、中位数与慢分位；总预算优先参考均值和慢分位。"""
    import numpy as np
    return {"count": len(values), "mean_seconds": float(np.mean(values)),
            "median_seconds": float(np.median(values)), "p90_seconds": float(np.percentile(values, 90)),
            "min_seconds": min(values), "max_seconds": max(values), "sum_seconds": sum(values)}


def run_child(args):
    """执行有限次真实更新和一次完整验证；临时模型退出即丢弃，不保存研究权重。"""
    import random
    import torch
    from torch.utils.data import DataLoader
    from revision.dataset import PairedManifestDataset, StepBatchSampler, assert_no_scene_overlap
    from revision.evaluation import evaluate, save_evaluation
    from revision.experiment import build_model, seed_all, LegacyRestorationLoss, source_info
    from revision.numerics import NumericalGuard, require_finite_tensor

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    phase = "initialization"
    report = {"purpose": "manual_short_benchmark_only_not_research_result", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "batch_size": args.single_batch,
              "model": args.model, "refiner_mid_ch": 64, "curve_M": 11, "crop_size": 256,
              "use_rgb_refiner": args.model in {"full", "refiner"},
              "use_curve": args.model in {"full", "curve"},
              "accum_steps": 1, "precision": "FP32_TF32_disabled", "optimizer": "Adam",
              "lr": 1e-4, "lr_schedule": "constant_for_timing_only_not_final_training_schedule",
              "weight_decay": 0, "adam_betas": [0.9, 0.999], "adam_eps": 1e-8,
              "gradient_clip_norm": 0.01,
              "loss": "legacy_rgb_hvi_l1_1_ssim_0.5_edge_50_vgg_0.01",
              "gamma_augmentation": args.gamma, "anomaly_detection": args.anomaly_detection,
              "warmup_updates_excluded_from_timing": args.warmup_updates,
              "measured_updates": args.measure_updates, "workers": args.workers,
              "validation_protocol": "all_1100_native_resolution_RGB_PSNR_SSIM_no_LPIPS",
              "source": source_info()}
    write_json(args.output/"report.json", report)
    try:
        check_manifest_bundle(args)
        torch.set_num_threads(args.cpu_threads)
        seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("此命令要求可见且仅有一张CUDA显卡。")
        report["gpu"] = torch.cuda.get_device_name(0)
        cache = Path(torch.hub.get_dir())/"checkpoints/vgg19-dcbb9e9d.pth"
        local_vgg = Path("experiments/pretrained_models/vgg19-dcbb9e9d.pth")
        if not cache.is_file() and not local_vgg.is_file():
            raise FileNotFoundError("VGG损失权重不在本地缓存；请先准备，测速不自动下载：" + str(cache))
        device = torch.device("cuda:0")
        train_set = PairedManifestDataset(args.data_root/"official", args.manifest_dir/"all11_train.json", 256)
        val_set = PairedManifestDataset(args.data_root/"official", args.manifest_dir/"all11_val.json")
        assert_no_scene_overlap(train_set, val_set)
        report["train_manifest_sha256"] = sha256_file(args.manifest_dir/"all11_train.json")
        report["val_manifest_sha256"] = sha256_file(args.manifest_dir/"all11_val.json")
        total = args.warmup_updates + args.measure_updates
        sampler = StepBatchSampler(train_set.records, args.single_batch, 1, total, args.seed)
        loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers)
        model = build_model(args.model).float().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = LegacyRestorationLoss().float().to(device)
        iterator = iter(loader)
        model.train()
        torch.cuda.synchronize()
        report["setup_seconds"] = time.perf_counter() - started
        times, all_times = [], []
        phase = "training"
        with NumericalGuard(model, optimizer) as guard, torch.autograd.set_detect_anomaly(args.anomaly_detection):
            with (args.output/"updates.jsonl").open("w", encoding="utf-8") as log:
                for step in range(total):
                    if step == args.warmup_updates:
                        torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    batch = next(iterator)
                    image, target = batch["input"].to(device), batch["target"].to(device)
                    if args.gamma:
                        image = image ** (random.randint(60, 120)/100.0)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(image)
                    require_finite_tensor(output, "测速原始输出")
                    loss = loss_fn(model, output, target)
                    require_finite_tensor(loss, "测速损失")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-begin
                    all_times.append(seconds)
                    if step >= args.warmup_updates:
                        times.append(seconds)
                    record = {"update": step+1, "seconds": seconds, "loss": float(loss),
                              "gradient_norm_before_clip": float(norm), "timed": step >= args.warmup_updates}
                    log.write(json.dumps(record)+"\n")
                    log.flush()
                    if (step+1) % 10 == 0:
                        print("更新 %d/%d：%.3f 秒，loss %.5f" % (step+1, total, seconds, record["loss"]), flush=True)
                    del output, loss, image, target, batch
            guard.check_state()
            report["update_timing"] = timing_statistics(times)
            report["all_update_seconds_including_warmup"] = sum(all_times)
            report["train_peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
            report["train_peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
            phase = "validation"
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            rows, metrics = evaluate(model, val_loader, device, progress=True)
            torch.cuda.synchronize()
            report["validation_seconds"] = time.perf_counter()-begin
            if len(rows) != 1100 or set(metrics["per_task"]) != set(TASKS):
                raise ValueError("完整验证未覆盖1100对和全部11任务。")
            report["validation_pairs"] = len(rows)
            report["validation_peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
            phase = "saving_report"
            begin = time.perf_counter()
            save_evaluation(args.output/"timing_validation_not_paper_results", rows, metrics,
                            {"purpose": "timing_only_not_paper_results", "source": report["source"]})
            report["validation_report_save_seconds"] = time.perf_counter()-begin
        report.update(status="complete", wall_seconds=time.perf_counter()-started,
                      checkpoint_saved=False, model_weights_discarded=True)
        write_json(args.output/"report.json", report)
        print(json.dumps({k: report[k] for k in ("batch_size", "update_timing", "validation_seconds",
              "train_peak_allocated_gib", "train_peak_reserved_gib")}, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        oom = isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        report.update(status="out_of_memory" if oom else "failed", failed_phase=phase,
                      error=repr(exc), wall_seconds=time.perf_counter()-started)
        write_json(args.output/"report.json", report)
        traceback.print_exc()
        return 20 if oom and phase in {"initialization", "training"} else 1


def main():
    """仅在用户执行命令时启动测速；导入模块和--help不触发训练。"""
    args = parse_args()
    return run_child(args) if args.single_batch is not None else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
