"""使用纯 Python 假 torch 检验权重适配，不导入真实 torch、不构建真实网络。"""

import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from revision.efficiency_models import _load_weights


class FakeTensor:
    """仅供 isinstance 检查的普通 Python 对象，不含张量计算。"""


class FakeModel:
    """记录加载参数的普通对象，不继承任何深度学习模型。"""

    def __init__(self, fail=False):
        """保存调用记录，并可模拟严格加载不匹配的异常。"""
        self.calls = []
        self.fail = fail

    def load_state_dict(self, state, strict):
        """只记录参数或抛出模拟异常，不读取真实模型权重。"""
        self.calls.append((state, strict))
        if self.fail:
            raise RuntimeError("模拟严格加载的键或形状不匹配")


class EfficiencyModelWeightTests(unittest.TestCase):
    """隔离替换 sys.modules 中的 torch，退出后恢复原有导入状态。"""

    def setUp(self):
        """创建只有哨兵字节的临时文件，用来验证路径与摘要记录。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)/"fake_checkpoint.pth"
        self.content = b"not a real pytorch checkpoint"
        self.path.write_bytes(self.content)
        self.previous_torch = sys.modules.get("torch")

    def tearDown(self):
        """确认假模块未泄漏，再移除测试自己的临时文件。"""
        self.assertIs(sys.modules.get("torch"), self.previous_torch)
        self.temporary.cleanup()

    def invoke(self, payload, spec=None, model=None):
        """提供假 Tensor 类型与假 load 函数，不接触真实 PyTorch。"""
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        calls = []

        def fake_load(path, map_location=None, weights_only=None):
            """返回测试预设字典，记录 CPU 映射等加载参数。"""
            calls.append({"path": path, "map_location": map_location,
                          "weights_only": weights_only})
            return payload

        fake_torch.load = fake_load
        model = model if model is not None else FakeModel()
        spec = {"checkpoint": str(self.path)} if spec is None else spec
        with patch.dict(sys.modules, {"torch": fake_torch}):
            metadata = _load_weights(model, spec)
            self.assertIs(sys.modules["torch"], fake_torch)
        return metadata, model, calls

    def test_bare_state_dict_is_strict_and_records_checksum(self):
        """裸 state_dict 必须严格加载，记录正确摘要并仅映射到 CPU。"""
        state = {"layer.weight": FakeTensor(), "layer.bias": FakeTensor()}
        metadata, model, calls = self.invoke(state)
        self.assertEqual(model.calls, [(state, True)])
        self.assertEqual(metadata["checkpoint_sha256"], hashlib.sha256(self.content).hexdigest())
        self.assertEqual(metadata["checkpoint"], str(self.path.resolve()))
        self.assertFalse(metadata["random_initialization"])
        self.assertTrue(metadata["strict_checkpoint_load"])
        self.assertEqual(calls[0]["map_location"], "cpu")
        self.assertIs(calls[0]["weights_only"], False)

    def test_explicit_container_and_prefix_are_used_without_dropping_keys(self):
        """明确选择容器并统一移除前缀，必须保留选定权重的所有条目。"""
        state = {"net.layer.weight": FakeTensor(), "net.layer.bias": FakeTensor()}
        payload = {"state_dict": state, "optimizer": {"not_loaded": True}}
        metadata, model, _ = self.invoke(payload, {
            "checkpoint": str(self.path), "checkpoint_key": "state_dict", "strip_prefix": "net."})
        loaded, strict = model.calls[0]
        self.assertTrue(strict)
        self.assertEqual(set(loaded), {"layer.weight", "layer.bias"})
        self.assertIs(loaded["layer.weight"], state["net.layer.weight"])
        self.assertEqual(metadata["checkpoint_key"], "state_dict")
        self.assertEqual(metadata["strip_prefix"], "net.")

    def test_missing_container_key_is_rejected(self):
        """找不到指定容器时拒绝加载，不自动回退到其他字典。"""
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "容器键"):
            self.invoke({"state_dict": {"weight": FakeTensor()}}, {
                "checkpoint": str(self.path), "checkpoint_key": "model"}, model)
        self.assertEqual(model.calls, [])

    def test_mixed_prefixes_are_rejected(self):
        """部分键没有指定前缀时拒绝整次加载，禁止过滤掉不匹配键。"""
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "全部权重键"):
            self.invoke({"net.weight": FakeTensor(), "other.bias": FakeTensor()}, {
                "checkpoint": str(self.path), "strip_prefix": "net."}, model)
        self.assertEqual(model.calls, [])

    def test_unselected_training_container_is_rejected(self):
        """未明确选择训练 checkpoint 的模型字段时，非张量内容不能被当作权重。"""
        with self.assertRaisesRegex(ValueError, "非张量"):
            self.invoke({"model": {"weight": FakeTensor()}, "step": 2000})

    def test_strict_mismatch_is_not_swallowed(self):
        """模型报告严格加载不匹配时，异常必须向调用方传播。"""
        model = FakeModel(fail=True)
        with self.assertRaisesRegex(RuntimeError, "严格加载"):
            self.invoke({"weight": FakeTensor()}, model=model)
        self.assertIs(model.calls[0][1], True)

    def test_missing_checkpoint_marks_random_initialization(self):
        """无权重时明确标记随机初始化，不能调用任何权重加载函数。"""
        metadata, model, calls = self.invoke(None, {"checkpoint": None})
        self.assertTrue(metadata["random_initialization"])
        self.assertFalse(metadata["strict_checkpoint_load"])
        self.assertIsNone(metadata["checkpoint_sha256"])
        self.assertEqual(model.calls, [])
        self.assertEqual(calls, [])

    def test_missing_checkpoint_with_container_key_is_rejected(self):
        """无权重却带有容器参数时不能默默忽略配置矛盾。"""
        with self.assertRaisesRegex(ValueError, "未提供权重"):
            self.invoke(None, {"checkpoint": None, "checkpoint_key": "model"})


if __name__ == "__main__":
    unittest.main()
