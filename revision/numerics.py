"""保留旧曲线首控制点行为，同时严格拦截其他非有限参数和运行结果。"""

import math

import torch
from torch import nn


def require_finite_tensor(value, label):
    """检查原始张量，禁止用后续 clamp 或替换数值掩盖 NaN、无穷大。"""
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise FloatingPointError(label + " 包含非有限值或不是张量")


def _legacy_curves(model):
    """仅识别真实的旧版 11 点单曲线层，按对象身份建立精确参数白名单。"""
    from net.NeuralCurve import NeuralCurveLayer

    curves = []
    for name, module in model.named_modules():
        if not isinstance(module, NeuralCurveLayer):
            continue
        if type(module) is not NeuralCurveLayer or module.M != 11 or module.num_curves != 1:
            raise FloatingPointError("不支持的曲线结构，不能沿用首点白名单：" + name)
        predictor = module.curve_predictor
        if (type(predictor) is not nn.Sequential or len(predictor) != 6
                or type(predictor[-2]) is not nn.Linear
                or type(predictor[-1]) is not nn.Sigmoid):
            raise FloatingPointError("曲线末层结构与旧实现不一致：" + name)
        linear = predictor[-2]
        if (linear.in_features != 64 or linear.out_features != 11
                or linear.bias is None or tuple(linear.bias.shape) != (11,)
                or tuple(linear.weight.shape) != (11, 64)):
            raise FloatingPointError("曲线末层维度与旧实现不一致：" + name)
        curves.append((name, module, linear))
    return curves


def _validate_state_value(value, label):
    """递归检查 Adam 状态，空初始状态合法，已存在的数值必须有限。"""
    if isinstance(value, torch.Tensor):
        require_finite_tensor(value, label)
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_state_value(item, label + "." + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_state_value(item, label + "[%d]" % index)
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise FloatingPointError(label + " 包含非有限状态")
    elif value is not None:
        raise TypeError("无法审查的优化器状态类型：" + label)


def _validate_optimizer(model, optimizer):
    """仅接受无权重衰减的 Adam，避免把允许的负无穷偏置加入梯度。"""
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("旧曲线数值兼容守卫仅支持 torch.optim.Adam")
    expected = {id(parameter) for parameter in model.parameters()}
    actual = []
    for group in optimizer.param_groups:
        if group.get("weight_decay", 0) != 0:
            raise ValueError("旧曲线首点为负无穷偏置，Adam weight_decay 必须为 0")
        eps = group.get("eps", 0)
        lr = group.get("lr", -1)
        if not math.isfinite(eps) or eps <= 0 or not math.isfinite(lr) or lr < 0:
            raise ValueError("Adam 的 eps 必须为有限正数，学习率必须为有限非负数")
        if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in group["betas"]):
            raise ValueError("Adam 的 beta 必须在 [0, 1) 内")
        actual.extend(id(parameter) for parameter in group["params"])
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Adam 参数组必须恰好包含模型全部参数，且不能重复")
    for parameter, state in optimizer.state.items():
        if id(parameter) not in expected:
            raise ValueError("Adam 状态含有当前模型之外的参数")
        _validate_state_value(state, "Adam.state")


def validate_model_numerics(model, optimizer=None):
    """允许且仅允许真实旧曲线 bias[0] 为负无穷，并检查其零权重首行。"""
    allowed = {}
    report = []
    for name, module, linear in _legacy_curves(model):
        bias = linear.bias.detach()
        valid = torch.isneginf(bias[0]) & torch.isfinite(bias[1:]).all()
        valid = valid & torch.eq(linear.weight.detach()[0], 0).all()
        if not bool(valid):
            raise FloatingPointError("旧曲线必须仅首偏置为 -inf、首权重行全零：" + name)
        allowed[id(linear.bias)] = name
        report.append({"parameter": (name + "." if name else "") + "curve_predictor.4.bias",
                       "indices": [0], "meaning": "旧实现首控制点固定为0"})
    for name, parameter in model.named_parameters():
        if id(parameter) not in allowed:
            require_finite_tensor(parameter, "参数 " + name)
    for name, value in model.named_buffers():
        require_finite_tensor(value, "缓冲区 " + name)
    if optimizer is not None:
        _validate_optimizer(model, optimizer)
    return {"policy": "legacy_curve11_first_bias_negative_infinity_only", "allowed": report}


class NumericalGuard:
    """用只读钩子观察原计算，不替换参数、张量、梯度或模型返回值。"""

    def __init__(self, model, optimizer=None):
        """保存对象，进入上下文后才安装检查钩子。"""
        self.model = model
        self.optimizer = optimizer
        self.handles = []

    def __enter__(self):
        """审查参数和优化器，并在旧曲线及模型原始输出处安装只读检查。"""
        if self.handles:
            raise RuntimeError("同一个 NumericalGuard 不能重复进入")
        validate_model_numerics(self.model, self.optimizer)
        try:
            for name, module, linear in _legacy_curves(self.model):
                self.handles.append(module.register_forward_pre_hook(self._check_curve_inputs))
                self.handles.append(linear.register_forward_hook(self._check_logits))
                self.handles.append(module.curve_predictor[-1].register_forward_hook(self._check_controls))
            self.handles.append(self.model.register_forward_hook(self._check_model_output))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """无论成功或报错都卸载钩子，恢复模型原有执行方式。"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    @staticmethod
    def _check_curve_inputs(module, inputs):
        """曲线特征和待调整通道都必须有限，防止零权重乘无穷大产生 NaN。"""
        if len(inputs) < 2:
            raise ValueError("旧曲线守卫要求按原接口传入特征和图像通道")
        require_finite_tensor(inputs[0], "曲线输入特征")
        require_finite_tensor(inputs[1], "曲线输入图像通道")

    @staticmethod
    def _check_logits(module, inputs, output):
        """在 Sigmoid 之前核查，仅第零列允许原实现预期的负无穷。"""
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[1] != 11:
            raise FloatingPointError("曲线预测 logits 形状异常")
        valid = torch.isneginf(output[:, 0]).all() & torch.isfinite(output[:, 1:]).all()
        if not bool(valid):
            raise FloatingPointError("曲线 logits 非有限值超出固定首点白名单")

    @staticmethod
    def _check_controls(module, inputs, output):
        """Sigmoid 后全部控制点必须有限、在 [0,1] 内，首点必须仍为零。"""
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[1] != 11:
            raise FloatingPointError("曲线控制点形状异常")
        valid = torch.isfinite(output).all() & (output >= 0).all() & (output <= 1).all()
        valid = valid & torch.eq(output[:, 0], 0).all()
        if not bool(valid):
            raise FloatingPointError("曲线控制点无效或固定首点行为发生变化")

    @staticmethod
    def _check_model_output(module, inputs, output):
        """检查模型原始输出，调用者裁剪图像范围之前就拦截异常值。"""
        require_finite_tensor(output, "模型原始输出")

    def check_gradients(self):
        """按需检查全部已有梯度；训练每步还应保留非有限梯度裁剪报错。"""
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                require_finite_tensor(parameter.grad, "梯度 " + name)
        for name, module, linear in _legacy_curves(self.model):
            for value in (linear.bias.grad, linear.weight.grad):
                if value is not None and not bool(torch.eq(value[0], 0).all()):
                    raise FloatingPointError("固定首控制点的梯度应为零：" + name)

    def check_state(self):
        """在验证、保存或测速结束时复核参数与 Adam 状态，不执行优化更新。"""
        return validate_model_numerics(self.model, self.optimizer)
