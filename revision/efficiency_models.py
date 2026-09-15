"""效率评测专用模型适配器；导入本文件不会导入 torch 或创建任何模型。"""

from collections.abc import Mapping
import hashlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys


MODEL_NAMES = ("full", "cidnet", "refiner", "curve", "promptir", "moce_ir_s")


def _sha256(path):
    """分块计算来源文件和权重的摘要，不加载张量。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(root, relative_files):
    """记录实际源文件摘要与 Git 提交，允许没有 Git 的独立源码目录。"""
    files = {}
    for relative in relative_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError("模型来源文件不存在：" + str(path))
        files[str(path)] = _sha256(path)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain", "--", *relative_files], cwd=root,
                            text=True, capture_output=True, check=False)
    return {"source_root": str(root), "source_files": files,
            "source_git_commit": result.stdout.strip() if result.returncode == 0 else None,
            "source_git_status_for_recorded_files": status.stdout.strip() if status.returncode == 0 else None}


def _require_dependencies(distributions):
    """只读取安装包元数据，不通过试导入创建 CUDA 上下文。"""
    versions = {}
    missing = []
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise ModuleNotFoundError("当前评测解释器缺少模型依赖：" + ", ".join(missing)
                                  + "；本适配器不会自动安装依赖。")
    return {"dependency_distribution_versions": versions,
            "dependency_probe": "distribution_metadata_only_before_model_import"}


def _load_external_module(path, name):
    """从已核查的独立源码文件加载模型，避免外部 net 包覆盖本项目同名包。"""
    module_name = "_efficiency_external_" + name
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError("无法创建模型模块加载器：" + str(path))
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_weights(model, spec):
    """仅按显式容器键和前缀严格加载可信本地权重，不加载优化器或丢弃不匹配键。"""
    import torch

    checkpoint = spec.get("checkpoint")
    key = spec.get("checkpoint_key")
    prefix = spec.get("strip_prefix")
    if checkpoint is None:
        if key is not None or prefix:
            raise ValueError("未提供权重时不能指定 checkpoint_key 或 strip_prefix。")
        return {"checkpoint": None, "checkpoint_sha256": None,
                "random_initialization": True, "strict_checkpoint_load": False,
                "evidence_scope": "随机初始化仅用于方法检查，不作为最终论文效率证据"}
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("权重文件不存在：" + str(path))
    checksum = _sha256(path)
    load_kwargs = {"map_location": "cpu"}
    # 正式训练权重含 NumPy 随机状态，只读取用户指定的可信本地文件。
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    payload = torch.load(path, **load_kwargs)
    if key is not None:
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint_key 必须是非空的单层字典键。")
        if not isinstance(payload, Mapping) or key not in payload:
            raise ValueError("权重不存在显式指定的容器键：" + key)
        state = payload[key]
    else:
        state = payload
    if not isinstance(state, Mapping) or not state:
        raise ValueError("所选权重不是非空 state_dict；请显式指定 checkpoint_key。")
    if any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
           for name, value in state.items()):
        raise ValueError("所选字典含非张量条目；请明确选择 model 或 state_dict 等容器键。")
    if prefix:
        if not isinstance(prefix, str) or any(not name.startswith(prefix) for name in state):
            raise ValueError("strip_prefix 必须匹配全部权重键；禁止默默丢弃其他条目。")
        state = {name[len(prefix):]: value for name, value in state.items()}
        if "" in state:
            raise ValueError("移除前缀后出现空权重键。")
    model.load_state_dict(state, strict=True)
    return {"checkpoint": str(path), "checkpoint_sha256": checksum,
            "checkpoint_key": key, "strip_prefix": prefix or None,
            "random_initialization": False, "strict_checkpoint_load": True,
            "evidence_scope": "已严格加载指定权重；训练来源与评测协议仍由结果报告说明"}


def build_model(spec):
    """按显式规格构建 CPU 模型并返回来源信息；调用方负责设种子、设备、eval 和评测。

    spec 至少包含 name，可附 checkpoint、external_root、checkpoint_key、strip_prefix。
    external_root 必须指 PromptIR 或 MoCE-IR 仓库本身，而不是 comparison_models 父目录。
    checkpoint_key=None 表示纯 state_dict；本项目正式权重需显式填 model，
    Lightning 权重通常需显式填 state_dict 并填 strip_prefix=net.。
    """
    if not isinstance(spec, Mapping) or spec.get("name") not in MODEL_NAMES:
        raise ValueError("未知模型，请从以下名称中选择：" + ", ".join(MODEL_NAMES))
    name = spec["name"]
    metadata = {"name": name, "dynamic_routing": name == "moce_ir_s",
                "stochastic_eval_routing": name == "moce_ir_s",
                "constructor_device": "cpu", "optimizer_state_loaded": False,
                "seed_owner": "caller", "forward_executed_by_adapter": False}
    if name in {"full", "cidnet", "refiner", "curve"}:
        if spec.get("external_root") is not None:
            raise ValueError("本项目四种结构不接受 external_root，避免误用另一份网络代码。")
        root = Path(__file__).resolve().parents[1]
        files = ["net/CIDNet.py", "net/DualSpaceCIDNet.py", "net/HVI_transform.py",
                 "net/LCA.py", "net/transformer_utils.py", "net/NeuralCurve.py"]
        metadata.update(_source_metadata(root, files))
        metadata.update(_require_dependencies(("torch", "einops", "huggingface-hub")))
        kwargs = {"channels": [36, 36, 72, 144], "heads": [1, 2, 4, 8], "norm": False}
        if name == "cidnet":
            from net.CIDNet import CIDNet
            model = CIDNet(**kwargs)
            metadata["architecture_class"] = "net.CIDNet.CIDNet"
        else:
            from net.DualSpaceCIDNet import DualSpaceCIDNet
            kwargs.update(use_rgb_refiner=name in {"full", "refiner"}, refiner_mid_ch=64,
                          use_curve=name in {"full", "curve"}, curve_M=11)
            model = DualSpaceCIDNet(**kwargs)
            metadata["architecture_class"] = "net.DualSpaceCIDNet.DualSpaceCIDNet"
        metadata["architecture_kwargs"] = kwargs
        metadata["architecture_basis"] = "本项目原始网络构造；完整模型明确 refiner_mid_ch=64、curve_M=11"
    else:
        if not spec.get("external_root"):
            raise ValueError(name + " 必须显式提供对应外部仓库 external_root。")
        root = Path(spec["external_root"]).expanduser().resolve()
        if name == "promptir":
            relative = "net/model.py"
            metadata.update(_source_metadata(root, [relative, "train_scratch.py", "test_custom.py"]))
            metadata.update(_require_dependencies(("torch", "einops")))
            module = _load_external_module(root/relative, name)
            kwargs = {"inp_channels": 3, "out_channels": 3, "dim": 48,
                      "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4,
                      "heads": [1, 2, 4, 8], "ffn_expansion_factor": 2.66,
                      "bias": False, "LayerNorm_type": "WithBias", "decoder": True}
            model = module.PromptIR(**kwargs)
            metadata["architecture_class"] = "PromptIR(net/model.py)"
            metadata["architecture_basis"] = "服务器 PromptIR/train_scratch.py:39-45 的显式构造"
        else:
            relative = "src/net/moce_ir.py"
            metadata.update(_source_metadata(root, [relative, "train_scratch.py", "test_custom.py"]))
            metadata.update(_require_dependencies(("torch", "einops", "numpy", "fvcore")))
            module = _load_external_module(root/relative, name)
            kwargs = {"inp_channels": 3, "out_channels": 3, "dim": 32, "levels": 4,
                      "heads": [1, 2, 4, 8], "num_blocks": [4, 6, 6, 8],
                      "num_dec_blocks": [2, 4, 4], "ffn_expansion_factor": 2,
                      "num_refinement_blocks": 4, "LayerNorm_type": "WithBias", "bias": False,
                      "rank": 2, "num_experts": 4, "depth_type": "constant",
                      "stage_depth": [1, 1, 1], "rank_type": "spread", "topk": 1,
                      "with_complexity": False, "complexity_scale": "max",
                      "expert_layer": "FFTAttention"}
            constructor_kwargs = dict(kwargs, expert_layer=module.FFTAttention)
            model = module.MoCEIR(**constructor_kwargs)
            metadata["architecture_class"] = "MoCEIR(src/net/moce_ir.py)"
            metadata["architecture_basis"] = ("服务器 train_scratch.py:get_moce_ir_s_opt 及 test_custom.py:"
                                               "get_model_opts('MoCE_IR_S')；其余参数显式复现已核查构造默认值")
            metadata["routing_note"] = ("此版本eval模式仍在logits上加入randn_like噪声，再按topk=1选择专家；"
                "同一输入重复运行也可能激活不同专家。保留原实现，不移除噪声；一次JIT追踪不代表计时全过程的路径。")
        metadata["architecture_kwargs"] = kwargs
    metadata.update(_load_weights(model, spec))
    return model, metadata
