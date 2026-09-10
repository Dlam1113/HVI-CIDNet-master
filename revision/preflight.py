"""使用 CPU 和合成样本检查网络、完整损失及训练测试入口。"""

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from revision.cdd11 import write_json
from revision.experiment import build_model, nonfinite_parameters, LegacyRestorationLoss, source_info, train, test


def make_fixture(root):
    """构造互不相同的合成训练、验证和测试图像，避免访问正式实验数据。"""
    for index, split in enumerate(("train", "val", "test")):
        rng = np.random.default_rng(42+index)
        pixels = rng.integers(20, 200, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root/(split+".png"))
        write_json(root/(split+".json"), {"split": split, "records": [{
            "scene_id": split+"/sample", "task": "low", "input": split+".png",
            "target": split+".png", "target_sha256_rgb": "synthetic_"+split}]})


def run(output):
    """检查真实网络前后向，并用 CIDNet 完成两步 CPU 训练、恢复及评估。"""
    torch.set_num_threads(2)
    report = {"purpose": "synthetic_pipeline_check_not_research_result", "source": source_info(), "models": {}}
    for name in ("cidnet", "refiner", "curve", "full"):
        torch.manual_seed(42)
        model = build_model(name).cpu()
        image = torch.rand(1, 3, 32, 32)
        prediction = model(image)
        (prediction-image).abs().mean().backward()
        report["models"][name] = {"shape": list(prediction.shape),
            "finite_output": bool(torch.isfinite(prediction).all()),
            "finite_gradients": all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
            "nonfinite_parameters": nonfinite_parameters(model)}
        write_json(output, report)
    model = build_model("cidnet").cpu()
    criterion = LegacyRestorationLoss().cpu()
    image = torch.rand(1, 3, 32, 32)
    value = criterion(model, model(image), image)
    value.backward()
    report["legacy_loss_cpu"] = {"value": value.item(), "finite": bool(torch.isfinite(value))}
    if not report["legacy_loss_cpu"]["finite"]:
        raise FloatingPointError("完整损失检查失败")
    del criterion, model
    with tempfile.TemporaryDirectory(prefix="cdd11_preflight_") as tmp:
        root = Path(tmp)
        make_fixture(root)
        args = SimpleNamespace(root=root, train_manifest=root/"train.json", val_manifest=root/"val.json",
            model="cidnet", device="cpu", seed=42, batch_size=1, accum_steps=2,
            crop_size=32, max_steps=2, lr=1e-4, warmup_steps=1, eval_every=1,
            workers=0, output=root/"run", resume=None)
        train(args)
        state = torch.load(root/"run/last.pt", map_location="cpu")
        assert state["step"] == 2
        args.resume = root/"run/last.pt"
        train(args)
        test_args = SimpleNamespace(root=root, checkpoint=root/"run/best.pt", device="cpu",
            manifest=root/"test.json", workers=0, lpips=False, output=root/"evaluation")
        test(test_args)
        report["training_integration"] = {"steps": 2, "resume": "passed", "test": "passed",
                                          "last_step": state["step"]}
    report["status"] = "passed_with_model_parameter_findings" if any(
        value["nonfinite_parameters"] for value in report["models"].values()) else "passed"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(v["finite_output"] and v["finite_gradients"] for v in report["models"].values()):
        raise FloatingPointError("部分模型的输出或梯度检查失败")


def main():
    """保存可复核的预检报告，不对外宣称合成样本指标代表模型性能。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
