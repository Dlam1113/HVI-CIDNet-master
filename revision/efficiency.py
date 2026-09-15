"""R1.2独立推理效率评测；仅手动run执行模型，plan和--help不导入PyTorch。"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
BUILTINS = ("full", "cidnet", "refiner", "curve")


def utc_now():
    """返回带时区的记录时间，避免Windows和服务器时区混淆。"""
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """原子保存严格JSON；不允许把NaN作为合法实验结果写入。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    """分块计算文件摘要，用于证明测量所用输入及源码版本。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions():
    """只读包元数据，不导入torch，不探测CUDA或执行模型。"""
    versions = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "einops", "fvcore", "iopath", "timm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_info():
    """记录当前提交和工作区状态；无Git时如实保留错误信息。"""
    result = {}
    for label, command in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"])):
        try:
            result[label] = subprocess.check_output(["git"] + command, cwd=ROOT, text=True,
                                                    stderr=subprocess.STDOUT, timeout=15).strip()
        except (OSError, subprocess.SubprocessError) as error:
            result[label] = None
            result[label + "_error"] = str(error)
    return result


def parse_args(argv=None):
    """解析手动计划/运行入口；默认batch1、FP32、TF32关闭且不启用训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "_worker"), help="plan仅打印计划；run手动评测；_worker为内部入口")
    parser.add_argument("--models", nargs="+", help="full/cidnet/refiner/curve/promptir/moce_ir_s")
    parser.add_argument("--spec-file", type=Path, help="多个模型的名称、权重、显式权重键和外部仓库路径JSON")
    parser.add_argument("--checkpoint", type=Path, help="单模型权重；多模型请用--spec-file")
    parser.add_argument("--checkpoint-key", help="单模型权重字典键；root代表纯state_dict；本项目默认model")
    parser.add_argument("--strip-prefix", default="", help="显式移除全部权重键共有前缀，例如net.")
    parser.add_argument("--external-root", type=Path, help="单个外部模型仓库根目录")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=300, help="每个模型/尺寸/输入的重复次数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda:0", choices=("cuda", "cuda:0"))
    parser.add_argument("--images", nargs="+", type=Path, help="可选固定真实RGB图像；各尺寸双三次缩放，仅测效率")
    parser.add_argument("--allow-untrained", action="store_true", help="显式允许随机初始化，结果标注为预备测量")
    parser.add_argument("--output", type=Path, help="run必须指定不存在的新目录，不覆盖旧结果")
    parser.add_argument("--job-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.action == "_worker":
        if not args.job_file or not args.output:
            parser.error("内部子进程需要job-file与output。")
        return args
    if args.spec_file and any((args.models, args.checkpoint, args.external_root, args.checkpoint_key, args.strip_prefix)):
        parser.error("--spec-file不能与单模型配置参数或--models混用。")
    if args.action == "run" and args.output is None:
        parser.error("run必须指定--output新目录。")
    if min(args.sizes) < 32 or any(size % 32 for size in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("输入尺寸须为不重复的32倍数，且不小于32。")
    if args.warmup < 1 or args.runs < 10 or args.cpu_threads < 1:
        parser.error("预热和线程数须为正，重复计时至少10次。")
    return args


def make_plan(args):
    """生成统一协议和任务，不导入模型、不创建结果目录、不检查GPU。"""
    from revision.efficiency_models import MODEL_NAMES

    if args.spec_file:
        payload = json.loads(args.spec_file.read_text(encoding="utf-8-sig"))
        specs = payload["models"]
        base = args.spec_file.resolve().parent
    else:
        names = args.models or ["full"]
        if len(names) != 1 and any((args.checkpoint, args.external_root, args.checkpoint_key, args.strip_prefix)):
            raise ValueError("多个模型的权重需分别在--spec-file中配置。")
        specs = []
        for name in names:
            key = args.checkpoint_key
            if key is None and args.checkpoint and name in BUILTINS:
                key = "model"
            specs.append({"name": name, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                          "checkpoint_key": None if key == "root" else key,
                          "strip_prefix": args.strip_prefix,
                          "external_root": str(args.external_root) if args.external_root else None})
        base = Path.cwd()
    if not isinstance(specs, list) or not specs:
        raise ValueError("模型清单models必须为非空数组。")
    normalized = []
    labels = set()
    for raw in specs:
        spec = dict(raw)
        unknown = set(spec) - {"name", "label", "checkpoint", "checkpoint_key", "strip_prefix", "external_root"}
        if unknown:
            raise ValueError("模型清单存在未知字段：" + str(sorted(unknown)))
        if spec.get("name") not in MODEL_NAMES:
            raise ValueError("不支持的模型：" + str(spec.get("name")))
        label = spec.get("label", spec["name"])
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError("每个模型须有非空且不重复的label/name。")
        labels.add(label)
        spec["label"] = label
        if not spec.get("checkpoint"):
            spec["checkpoint"] = None
            if spec.get("checkpoint_key") is not None or spec.get("strip_prefix"):
                raise ValueError(label + "未指定权重，不能指定权重容器键或前缀。")
        for field in ("checkpoint", "external_root"):
            if spec.get(field):
                candidate = Path(spec[field]).expanduser()
                spec[field] = str((base / candidate).resolve()) if not candidate.is_absolute() else str(candidate.resolve())
        if not spec.get("checkpoint") and not args.allow_untrained:
            raise ValueError(label + "缺少权重；最终测量应提供权重，随机初始化预检须显式--allow-untrained。")
        if spec["name"] not in BUILTINS and not spec.get("external_root"):
            raise ValueError(label + "需要external_root指向具体模型仓库。")
        if spec["name"] in BUILTINS and spec.get("external_root"):
            raise ValueError(label + "直接使用本项目源码，不能指定external_root。")
        normalized.append(spec)
    images = [str(path.expanduser().resolve()) for path in (args.images or [])]
    protocol = {"batch_size": 1, "channels": 3, "precision": "float32", "tf32": False,
                "autocast": False, "grad_mode": "no_grad", "cudnn_benchmark": False,
                "cudnn_deterministic": True, "cpu_threads": args.cpu_threads, "seed": args.seed,
                "sizes": args.sizes, "warmup_per_input": args.warmup, "runs_per_input": args.runs,
                "input": "RGB_images_bicubic_resize" if images else "seeded_uniform_RGB_0_1",
                "images": images, "device": args.device,
                "timing": "synchronized_wall_clock_model_forward_only_includes_internal_preprocessing",
                "memory": "warmed_peak_allocated_includes_model_one_input_output_and_intermediates",
                "reserved_memory": "warmed_allocator_reserved_reported_separately",
                "flops": "operator_audit_multiply_add_2_scalar_and_nonlinear_separate"}
    jobs = [{"spec": spec, "size": size, "protocol": protocol, "allow_untrained": args.allow_untrained}
            for spec in normalized for size in args.sizes]
    return {"purpose": "R1.2_inference_efficiency", "created_at": utc_now(), "protocol": protocol,
            "models": normalized, "jobs": jobs, "packages": package_versions(),
            "output": str(args.output.resolve()) if args.output else None,
            "note": "plan不执行模型；run需GPU空闲。每个模型/尺寸独立进程，所有结果仍需人工核查。"}


def query_compute_processes():
    """只通过nvidia-smi读取进程，无法确认空闲时直接报错。"""
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                             "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True, timeout=15)
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        try:
            rows.append({"pid": int(row[0].strip()), "name": ",".join(row[1:]).strip()})
        except ValueError as error:
            raise RuntimeError("无法可靠解析GPU计算进程：" + str(row)) from error
    return rows


def require_idle_gpu(own_pid=None):
    """拒绝与已有GPU任务并行；不自动终止任何进程。"""
    others = [row for row in query_compute_processes() if row["pid"] != own_pid]
    if others:
        raise RuntimeError("GPU正被其他计算任务使用，停止效率评测：" + json.dumps(others, ensure_ascii=False))


def gpu_snapshot():
    """保存卡号、驱动、温度和时钟快照，供解释运行波动。"""
    fields = "index,uuid,name,driver_version,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True, timeout=15)
    return [dict(zip(fields.split(","), [value.strip() for value in row]))
            for row in csv.reader(result.stdout.splitlines()) if row]


def timing_summary(milliseconds):
    """以图像/秒统计batch1吞吐量，并保留分位数和波动警示。"""
    if not milliseconds or any(not math.isfinite(value) or value <= 0 for value in milliseconds):
        raise ValueError("计时必须是非空、有限且大于零的毫秒数。")
    ordered = sorted(milliseconds)

    def percentile(fraction):
        """线性插值计算分位数，与常见NumPy默认定义一致。"""
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    mean = statistics.mean(milliseconds)
    deviation = statistics.pstdev(milliseconds)
    return {"runs": len(milliseconds), "mean_ms": mean, "p50_ms": percentile(.5),
            "p95_ms": percentile(.95), "std_ms": deviation, "cv": deviation / mean,
            "min_ms": min(milliseconds), "max_ms": max(milliseconds), "images_per_second": 1000. / mean,
            "review_timing_variability": deviation / mean > .1,
            "variability_note": "CV>0.10只是复核提示，不是普适稳定性证明。"}


def prepare_input(protocol, size, input_index):
    """在CPU准备固定的合法RGB输入，传输和读取均不计入推理延迟。"""
    import torch

    if protocol["images"]:
        from PIL import Image
        import numpy as np
        path = Path(protocol["images"][input_index])
        with Image.open(path) as source:
            original_size = list(source.size)
            image = source.convert("RGB").resize((size, size), resample=Image.Resampling.BICUBIC)
            array = np.asarray(image).copy().astype("float32") / 255.
        sample = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
        meta = {"path": str(path), "sha256": file_hash(path), "original_size": original_size,
                "resize": "Pillow_BICUBIC", "purpose": "efficiency_only_not_image_quality_evaluation"}
    else:
        generator = torch.Generator(device="cpu").manual_seed(protocol["seed"])
        sample = torch.rand((1, 3, size, size), generator=generator, dtype=torch.float32)
        meta = {"distribution": "uniform_0_1", "seed": protocol["seed"]}
    meta["tensor_sha256"] = hashlib.sha256(sample.numpy().tobytes()).hexdigest()
    return sample, meta


def check_prediction(output, sample):
    """预检和计时后检查输出，不把额外检查加入延迟，也不添加clamp。"""
    import torch

    if not isinstance(output, torch.Tensor) or output.shape != sample.shape:
        raise ValueError("模型须直接返回与输入同形状的RGB张量，不能静默挑选输出。")
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("模型原始输出包含NaN/Inf，效率结果无效。")


def measure_one_input(model, sample, protocol, device, validate_curves=False):
    """测完整模型调用的墙钟延迟；先预热，再重置峰值，迭代间释放上一输出。"""
    import torch

    with torch.no_grad():
        if validate_curves:
            from revision.numerics import NumericalGuard
            with NumericalGuard(model):
                output = model(sample)
                check_prediction(output, sample)
        else:
            output = model(sample)
            check_prediction(output, sample)
        del output
        for _ in range(protocol["warmup_per_input"]):
            output = model(sample)
            torch.cuda.synchronize(device)
            del output
        gc.collect()
        torch.cuda.synchronize(device)
        baseline = {"allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device)}
        torch.cuda.reset_peak_memory_stats(device)
        raw = []
        for _ in range(protocol["runs_per_input"]):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model(sample)
            torch.cuda.synchronize(device)
            raw.append((time.perf_counter() - started) * 1000.)
            del output
        memory = {"baseline": baseline, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        memory["peak_extra_allocated_bytes"] = memory["peak_allocated_bytes"] - baseline["allocated_bytes"]
        output = model(sample)
        check_prediction(output, sample)
        del output
    return {"timing": timing_summary(raw), "milliseconds": raw, "memory": memory}


def runtime_environment(torch, device):
    """记录实际GPU、精度开关和软件栈，而不只记录命令中的期望值。"""
    props = torch.cuda.get_device_properties(device)
    return {"host": platform.node(), "platform": platform.platform(), "python": sys.version,
            "executable": sys.executable, "packages": package_versions(), "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu_name": props.name,
            "gpu_total_bytes": props.total_memory, "gpu_compute_capability": [props.major, props.minor],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def run_worker(job, output_dir):
    """仅手动run派生的独立进程执行前向；不存在优化器、反传或训练。"""
    # GPU空闲检查必须发生在导入torch和构造模型之前。
    require_idle_gpu()
    import torch
    from revision.efficiency_models import build_model

    protocol = job["protocol"]
    torch.set_num_threads(protocol["cpu_threads"])
    torch.set_num_interop_threads(protocol["cpu_threads"])
    random.seed(protocol["seed"])
    import numpy as np
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("此协议要求单张可见CUDA GPU；不会自动回退CPU或其他精度。")
    device = torch.device(protocol["device"])
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_at": utc_now(), "job": job,
              "environment": runtime_environment(torch, device), "git": git_info(),
              "measurement_source_sha256": {path.name: file_hash(path) for path in
                  (ROOT / "revision").glob("efficiency*.py")},
              "gpu_before": gpu_snapshot(), "inputs": [], "paper_ready": False}
    write_json(output_dir / "report.json", report)
    try:
        model, metadata = build_model(job["spec"])
        require_idle_gpu(own_pid=os.getpid())
        model = model.float().eval().to(device)
        report["model"] = metadata
        report["parameters_total"] = sum(parameter.numel() for parameter in model.parameters())
        report["parameters_trainable"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        number_inputs = len(protocol["images"]) or 1
        with (output_dir / "latencies.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["input_index", "repeat", "milliseconds"])
            writer.writeheader()
            for index in range(number_inputs):
                sample_cpu, input_meta = prepare_input(protocol, job["size"], index)
                sample = sample_cpu.to(device)
                del sample_cpu
                require_idle_gpu(own_pid=os.getpid())
                measured = measure_one_input(model, sample, protocol, device, job["spec"]["name"] in BUILTINS)
                for repeat, milliseconds in enumerate(measured.pop("milliseconds"), 1):
                    writer.writerow({"input_index": index, "repeat": repeat, "milliseconds": milliseconds})
                stream.flush()
                measured["input"] = input_meta
                report["inputs"].append(measured)
                del sample
                write_json(output_dir / "report.json", report)
                print("已测 %s / %d / 输入%d：%.3f ms" % (job["spec"]["label"], job["size"], index + 1,
                                                           measured["timing"]["mean_ms"]), flush=True)
        require_idle_gpu(own_pid=os.getpid())
        report["gpu_after_timing"] = gpu_snapshot()
        with (output_dir / "latencies.csv").open(encoding="utf-8-sig", newline="") as stream:
            report["timing"] = timing_summary([float(row["milliseconds"]) for row in csv.DictReader(stream)])
        report["peak_allocated_mib"] = max(row["memory"]["peak_allocated_bytes"] for row in report["inputs"]) / 2**20
        report["peak_reserved_mib"] = max(row["memory"]["peak_reserved_bytes"] for row in report["inputs"]) / 2**20
        # FLOPs追踪在全部计时和显存测量之后执行，其钩子、缓存和开销不会混入效率数据。
        from revision.efficiency_ops import profile_complexity
        report["complexity_per_input"] = []
        for index in range(number_inputs):
            require_idle_gpu(own_pid=os.getpid())
            sample_cpu, _ = prepare_input(protocol, job["size"], index)
            sample = sample_cpu.to(device)
            del sample_cpu
            with torch.no_grad():
                complexity = profile_complexity(model, sample)
            report["complexity_per_input"].append(complexity)
            del sample
        report["status"] = "measured_pending_review"
        if any(item.get("status") != "counted_approximate" for item in report["complexity_per_input"]):
            report["status"] = "needs_complexity_review"
        if metadata.get("dynamic_routing"):
            report["status"] = "needs_complexity_review"
            report["dynamic_complexity_note"] = ("逐输入只追踪一次实际路径；观察到的计数范围不是理论上下界，"
                "也不与延迟测量的全部路由一一对应。随机eval路由未被关闭。")
        if report["timing"]["review_timing_variability"]:
            report["timing_review_required"] = True
        report["weight_status"] = "checkpoint" if job["spec"].get("checkpoint") else "untrained_preliminary"
        report["finished_at"] = utc_now()
        write_json(output_dir / "report.json", report)
        return 0 if report["status"] == "measured_pending_review" else 2
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), finished_at=utc_now())
        write_json(output_dir / "report.json", report)
        raise


def export_summary(output, plan, reports):
    """汇总同批测量，显式标出失败、随机初始化和部分FLOPs，禁止补造数值。"""
    rows = []
    for job, report in zip(plan["jobs"], reports):
        complexity = report.get("complexity_per_input", [])
        values = [item["gflops_counted"] for item in complexity if item.get("gflops_counted") is not None]
        timing = report.get("timing", {})
        rows.append({"model": job["spec"]["label"], "input": "1x3x%dx%d" % (job["size"], job["size"]),
                     "parameters_M": report.get("parameters_total", 0) / 1e6 if "parameters_total" in report else None,
                     "GFLOPs_counted_observed_min": min(values) if values else None,
                     "GFLOPs_counted_observed_max": max(values) if values else None,
                     "complexity_status": ";".join(item.get("status", "unknown") for item in complexity),
                     "peak_allocated_MiB": report.get("peak_allocated_mib"),
                     "peak_reserved_MiB": report.get("peak_reserved_mib"), "mean_ms": timing.get("mean_ms"),
                     "p50_ms": timing.get("p50_ms"), "p95_ms": timing.get("p95_ms"),
                     "images_per_second": timing.get("images_per_second"),
                     "weight_status": report.get("weight_status", "unknown"), "status": report["status"]})
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["model", "status"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", {"purpose": plan["purpose"], "protocol": plan["protocol"],
        "status": "measured_pending_review" if len(reports) == len(plan["jobs"]) and all(
            report["status"] == "measured_pending_review" for report in reports) else "incomplete_or_needs_review",
        "paper_ready": False, "completed_jobs": len(reports), "planned_jobs": len(plan["jobs"]), "rows": rows})


def run_parent(plan):
    """先查依赖和空闲GPU，再逐任务启动新进程；不覆盖、不自动安装或启动训练。"""
    for name in ("torch", "fvcore", "numpy", "PIL"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError("缺少依赖%s；请先按文档在独立评测环境中手动准备，未执行模型。" % name)
    for spec in plan["models"]:
        if spec.get("checkpoint") and not Path(spec["checkpoint"]).is_file():
            raise FileNotFoundError("权重不存在：" + spec["checkpoint"])
        if spec.get("external_root") and not Path(spec["external_root"]).is_dir():
            raise FileNotFoundError("外部仓库不存在：" + spec["external_root"])
    for path in plan["protocol"]["images"]:
        if not Path(path).is_file():
            raise FileNotFoundError("输入图像不存在：" + path)
    require_idle_gpu()
    if len(gpu_snapshot()) != 1:
        raise RuntimeError("当前协议针对物理单卡服务器；检测到的GPU数量不是1。")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    reports = []
    any_failure = False
    for index, job in enumerate(plan["jobs"]):
        require_idle_gpu()
        job_path = output / ("job_%03d.json" % index)
        job_output = output / ("job_%03d_%s_%d" % (index, job["spec"]["name"], job["size"]))
        write_json(job_path, job)
        command = [sys.executable, "-u", "-B", "-m", "revision.efficiency", "_worker",
                   "--job-file", str(job_path), "--output", str(job_output)]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        print("开始 %d/%d：%s，%dx%d" % (index + 1, len(plan["jobs"]), job["spec"]["label"], job["size"], job["size"]), flush=True)
        with (output / ("job_%03d.console.log" % index)).open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        report_path = job_output / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
            "status": "failed", "exit_code": result.returncode, "error": "子进程初始化失败，见console.log"}
        if result.returncode and report.get("status") == "running":
            report["status"] = "failed"
        reports.append(report)
        export_summary(output, plan, reports)
        any_failure = any_failure or result.returncode != 0
        print("任务状态：%s；记录：%s" % (report["status"], job_output), flush=True)
        if report["status"] == "failed":
            print("本任务失败，停止后续任务；不自动降精度、缩小尺寸或更换模型。", flush=True)
            break
    return 2 if any_failure else 0


def main(argv=None):
    """计划只读打印；只有显式run或其子进程才会执行模型前向。"""
    args = parse_args(argv)
    try:
        if args.action == "_worker":
            return run_worker(json.loads(args.job_file.read_text(encoding="utf-8")), args.output)
        plan = make_plan(args)
        if args.action == "plan":
            print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        return run_parent(plan)
    except Exception as error:
        print("效率评测停止：%s: %s" % (type(error).__name__, error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
