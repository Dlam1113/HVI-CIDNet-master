"""仅用假 JIT 值与整数形状验证计数，不导入 torch，不执行模型。"""

from collections import Counter
import sys
import unittest

from revision.efficiency_ops import (
    _dense_operator, _element_operator, _format_counts, _make_handler,
    adaptive_average_counts, convolution_counts, einsum_counts,
    matmul_counts, reduction_counts, softmax_counts, value_shape,
    snapshot_forward_hooks, remove_new_forward_hooks,
)


class FakeValue:
    """提供计数器需要的 JIT 形状与常量接口。"""

    def __init__(self, shape=None, constant=None, dtype="Float"):
        """保存纯 Python 数据，不创建任何真实张量。"""
        self.shape, self.constant, self.dtype = shape, constant, dtype

    def type(self):
        """假值兼任形状类型对象。"""
        return self

    def sizes(self):
        """返回可为空的静态形状。"""
        return self.shape

    def scalarType(self):
        """返回假张量的标量类型名称。"""
        return self.dtype

    def toIValue(self):
        """返回预设常量。"""
        return self.constant


class EfficiencyOpsTests(unittest.TestCase):
    """检查最容易影响论文计数的广播、分组、分类和失败可见性。"""

    def test_no_torch_import(self):
        """保证测试和模块顶层没有隐式导入训练框架。"""
        self.assertNotIn("torch", sys.modules)

    def test_broadcast_and_vector_matmul(self):
        """批次广播取输出大小，向量点乘采用同一乘加约定。"""
        self.assertEqual(matmul_counts((1, 3, 5), (4, 5, 7), (4, 3, 7)), {"dense_flops": 840})
        self.assertEqual(matmul_counts((5,), (5,), ()), {"dense_flops": 10})

    def test_group_and_transposed_convolution(self):
        """深度卷积不重复乘输入通道；转置卷积依据输入空间计数。"""
        self.assertEqual(convolution_counts([(2, 8, 10, 10), (8, 1, 3, 3)], (2, 8, 8, 8)),
                         {"dense_flops": 18432})
        result = convolution_counts([(1, 4, 5, 5), (4, 3, 2, 2)], (1, 3, 10, 10), True, True)
        self.assertEqual(result, {"dense_flops": 2400, "scalar_arithmetic_flops": 300})

    def test_einsum_attention_and_broadcast(self):
        """标准注意力收缩与广播逐元素乘法不能混用同一倍数。"""
        self.assertEqual(einsum_counts("bhcn,bhdn->bhcd", [(2, 3, 4, 5), (2, 3, 7, 5)]),
                         {"dense_flops": 1680})
        self.assertEqual(einsum_counts("...i,...i->...i", [(2, 3), (1, 3)]),
                         {"scalar_arithmetic_flops": 6})
        with self.assertRaises(ValueError):
            einsum_counts("ab,c->a", [(2, 3), (4,)])

    def test_reduction_and_softmax(self):
        """均值除法纳入基础算术，softmax的指数与比较另列。"""
        self.assertEqual(reduction_counts((2, 3, 5), (2, 3, 1), True), {"scalar_arithmetic_flops": 30})
        self.assertEqual(softmax_counts((2, 3, 5), -1),
                         {"scalar_arithmetic_flops": 84, "nonlinear_element_ops": 30, "comparison_element_ops": 24})
        self.assertEqual(adaptive_average_counts((2, 3, 5, 5), (2, 3, 1, 1)), {"scalar_arithmetic_flops": 150})

    def test_integer_and_nonlinear_are_not_flops(self):
        """整数乘法和atan2不能偷偷增加GFLOPs列。"""
        out = FakeValue((2, 3))
        self.assertEqual(_element_operator("aten::atan2", [out, out], [out]), {"nonlinear_element_ops": 6})
        integer = FakeValue((2, 3), dtype="Long")
        self.assertEqual(_element_operator("aten::mul", [integer, integer], [integer]),
                         {"integer_or_boolean_element_ops": 6})
        self.assertEqual(_element_operator("aten::pow", [out, FakeValue(constant=2)], [out]),
                         {"scalar_arithmetic_flops": 6})

    def test_hvi_max_dim_and_mask(self):
        """HVI沿通道取最大值属于归约，布尔掩码单列为比较。"""
        source, output = FakeValue((2, 3, 4, 5)), FakeValue((2, 4, 5))
        self.assertEqual(_element_operator("aten::max", [source, FakeValue(constant=1), FakeValue(constant=False)], [output]),
                         {"comparison_element_ops": 80})
        mask = FakeValue((2, 4, 5), dtype="Bool")
        self.assertEqual(_element_operator("aten::eq", [output, output], [mask]),
                         {"comparison_element_ops": 40})

    def test_linear_bias_and_scaled_addmm(self):
        """偏置加法和非默认alpha/beta缩放独立于稠密乘加。"""
        out = FakeValue((2, 7))
        self.assertEqual(_dense_operator("aten::linear", [FakeValue((2, 5)), FakeValue((7, 5)), FakeValue((7,))], [out]),
                         {"dense_flops": 140, "scalar_arithmetic_flops": 14})
        result = _dense_operator("aten::addmm", [FakeValue((7,)), FakeValue((2, 5)), FakeValue((5, 7)),
                                                FakeValue(constant=2), FakeValue(constant=3)], [out])
        self.assertEqual(result, {"dense_flops": 140, "scalar_arithmetic_flops": 42})

    def test_unknown_shape_is_reported(self):
        """未知形状进入未支持清单，不能以零计数掩盖。"""
        unresolved, reasons = Counter(), {}
        handler = _make_handler("aten::mul", _element_operator, unresolved, reasons)
        self.assertEqual(handler([], [FakeValue()]), Counter())
        self.assertEqual(unresolved["aten::mul"], 1)
        self.assertTrue(reasons["aten::mul"])
        with self.assertRaises(ValueError):
            value_shape(FakeValue((2, None)))

    def test_categories_remain_separate(self):
        """JSON明细始终区分基础算术和非线性名义操作。"""
        operators, totals = _format_counts(Counter({"dense_flops|aten::mm": 10, "nonlinear_element_ops|aten::sin": 3}))
        self.assertEqual(totals, {"dense_flops": 10, "nonlinear_element_ops": 3})
        self.assertEqual(operators["aten::sin"], {"nonlinear_element_ops": 3})

    def test_failed_trace_removes_only_new_hooks(self):
        """模拟追踪中途报错，确保保留原钩子并清除新增钩子和附属标记。"""
        class FakeModule:
            """仅模拟模块钩子容器，不包含任何模型或张量计算。"""

            def __init__(self):
                """创建原有钩子和新版本 PyTorch 的关联标记容器。"""
                self.original = object()
                self._forward_pre_hooks = {1: self.original}
                self._forward_hooks = {2: self.original}
                self._forward_hooks_with_kwargs = {2: True}
                self._forward_pre_hooks_with_kwargs = {1}
                self._forward_hooks_always_called = {}

            def modules(self):
                """只返回自身，替代真实模型的模块遍历。"""
                return iter((self,))

        model = FakeModule()
        snapshot = snapshot_forward_hooks(model)
        try:
            model._forward_pre_hooks[3] = object()
            model._forward_hooks[4] = object()
            model._forward_pre_hooks_with_kwargs.add(3)
            model._forward_hooks_with_kwargs[4] = True
            model._forward_hooks_always_called[4] = True
            raise ValueError("模拟追踪失败")
        except ValueError:
            pass
        finally:
            remove_new_forward_hooks(snapshot)
        self.assertEqual(model._forward_pre_hooks, {1: model.original})
        self.assertEqual(model._forward_hooks, {2: model.original})
        self.assertEqual(model._forward_hooks_with_kwargs, {2: True})
        self.assertEqual(model._forward_pre_hooks_with_kwargs, {1})
        self.assertEqual(model._forward_hooks_always_called, {})


if __name__ == "__main__":
    unittest.main()
