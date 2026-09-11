"""仅用 CPU 参数和人工张量检查守卫；不调用模型前后向或 optimizer.step。"""

import unittest

import torch
from torch import nn

from net.NeuralCurve import NeuralCurveLayer
from revision.numerics import NumericalGuard, require_finite_tensor, validate_model_numerics


class ParameterFixture(nn.Module):
    """只承载曲线初始化参数的测试容器，禁止执行前向。"""

    def __init__(self, points=11):
        """在 CPU 创建原曲线参数及一个普通参数，不创建训练数据或损失。"""
        super().__init__()
        self.i_curve = NeuralCurveLayer(in_channels=4, M=points, num_curves=1)
        self.extra = nn.Parameter(torch.zeros(1))

    def forward(self, value):
        """任何意外模型执行均使测试立即失败。"""
        raise AssertionError("本测试禁止执行模型前向")


class NumericalPolicyTests(unittest.TestCase):
    """检查精确白名单、优化器约束和只读钩子，不运行神经网络计算。"""

    def setUp(self):
        """为每项测试生成独立 CPU 参数，避免相互影响。"""
        self.model = ParameterFixture()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def test_exact_legacy_pattern_is_reported_without_mutation(self):
        """允许的 -inf 仍保留在原参数里，不能被守卫替换成有限数。"""
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        report = validate_model_numerics(self.model, self.optimizer)
        self.assertEqual(report["allowed"][0]["indices"], [0])
        self.assertEqual(report["allowed"][0]["parameter"], "i_curve.curve_predictor.4.bias")
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(before[name], value))

    def test_other_nonfinite_parameter_is_rejected(self):
        """名字无关的普通参数出现 NaN 时必须失败。"""
        with torch.no_grad():
            self.model.extra.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            validate_model_numerics(self.model)

    def test_wrong_bias_position_or_sign_is_rejected(self):
        """首点以外的无穷或首点的正无穷都不能被白名单掩盖。"""
        for index, value in ((1, -float("inf")), (0, float("inf"))):
            model = ParameterFixture()
            with torch.no_grad():
                model.i_curve.curve_predictor[-2].bias[index] = value
            with self.assertRaises(FloatingPointError):
                validate_model_numerics(model)

    def test_nonzero_first_weight_row_is_rejected(self):
        """固定首点对应的线性权重行必须保持旧实现的零值。"""
        with torch.no_grad():
            self.model.i_curve.curve_predictor[-2].weight[0, 0] = 1
        with self.assertRaises(FloatingPointError):
            validate_model_numerics(self.model)

    def test_other_curve_structure_is_rejected(self):
        """其他控制点数量不能自动获得本次 11 点协议的例外。"""
        with self.assertRaises(FloatingPointError):
            validate_model_numerics(ParameterFixture(points=7))

    def test_fake_curve_name_does_not_grant_exception(self):
        """仅伪造相同参数名称不能获得真实模块的白名单权限。"""
        model = nn.Module()
        model.i_curve = nn.Module()
        model.i_curve.register_parameter("bias", nn.Parameter(torch.tensor([-float("inf")])))
        with self.assertRaises(FloatingPointError):
            validate_model_numerics(model)

    def test_optimizer_policy_is_strict(self):
        """权重衰减、零 eps 和其他优化器均应明确拒绝。"""
        for key, value in (("weight_decay", 0.01), ("eps", 0)):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            optimizer.param_groups[0][key] = value
            with self.assertRaises(ValueError):
                validate_model_numerics(self.model, optimizer)
        with self.assertRaises(ValueError):
            validate_model_numerics(self.model, torch.optim.SGD(self.model.parameters(), lr=1e-4))

    def test_nonfinite_optimizer_state_is_rejected(self):
        """人工模拟损坏的动量状态，不需要执行任何优化步骤。"""
        self.optimizer.state[self.model.extra] = {"exp_avg": torch.tensor([float("nan")])}
        with self.assertRaises(FloatingPointError):
            validate_model_numerics(self.model, self.optimizer)

    def test_nonfinite_buffer_is_rejected(self):
        """参数之外的模型缓冲区也不能藏有非有限值。"""
        self.model.register_buffer("bad", torch.tensor([float("inf")]))
        with self.assertRaises(FloatingPointError):
            validate_model_numerics(self.model)

    def test_manual_logits_and_controls_are_checked(self):
        """直接传人工张量验证钩子逻辑，不触发模型调用。"""
        logits = torch.zeros(2, 11)
        logits[:, 0] = -float("inf")
        NumericalGuard._check_logits(None, (), logits)
        controls = torch.full((2, 11), 0.5)
        controls[:, 0] = 0
        NumericalGuard._check_controls(None, (), controls)
        logits[0, 1] = float("inf")
        with self.assertRaises(FloatingPointError):
            NumericalGuard._check_logits(None, (), logits)
        controls[0, 0] = 0.1
        with self.assertRaises(FloatingPointError):
            NumericalGuard._check_controls(None, (), controls)

    def test_manual_gradients_are_checked(self):
        """人工写入梯度字段检查策略，不执行 backward。"""
        guard = NumericalGuard(self.model, self.optimizer)
        self.model.extra.grad = torch.zeros_like(self.model.extra)
        guard.check_gradients()
        self.model.extra.grad.fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            guard.check_gradients()

    def test_context_hooks_are_removed_after_error(self):
        """上下文异常退出后必须卸载只读钩子，且不调用 forward。"""
        modules = list(self.model.modules())
        before = [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules]
        with self.assertRaises(RuntimeError):
            with NumericalGuard(self.model, self.optimizer):
                raise RuntimeError("人工异常")
        after = [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules]
        self.assertEqual(before, after)

    def test_raw_infinite_output_cannot_be_hidden_by_clamp(self):
        """只检查人工原始输出张量，无穷值须在范围裁剪前失败。"""
        with self.assertRaises(FloatingPointError):
            require_finite_tensor(torch.tensor([float("inf")]), "人工模型输出")


if __name__ == "__main__":
    unittest.main()
