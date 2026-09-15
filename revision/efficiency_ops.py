"""按明确口径统计 JIT 算子；非线性操作次数不冒充硬件 FLOPs。"""

from collections import Counter
from math import ceil, floor, prod
import re
import warnings


def value_shape(value):
    """读取静态张量形状；动态形状不能按零成本悄悄处理。"""
    try:
        shape = value.type().sizes()
    except (AttributeError, RuntimeError):
        shape = None
    if shape is None or any(not isinstance(n, int) or n < 0 for n in shape):
        raise ValueError("张量形状未静态确定")
    return tuple(shape)


def value_constant(value):
    """读取 JIT 常量；不能求值的运行时参数返回 None。"""
    try:
        return value.toIValue()
    except (AttributeError, RuntimeError):
        return None


def floating_output(value):
    """区分浮点输出和整数、布尔输出，避免把形状计算计入 FLOPs。"""
    try:
        dtype = value.type().scalarType()
    except (AttributeError, RuntimeError):
        dtype = None
    if dtype is None:
        raise ValueError("输出张量类型未静态确定")
    if dtype.startswith("Complex"):
        raise ValueError("尚未定义复数算术的计数口径")
    return dtype in {"Float", "Double", "Half", "BFloat16"}


def _has_tensor(value):
    """判断可选偏置是否为张量，常量 None 表示没有偏置。"""
    try:
        value_shape(value)
        return True
    except ValueError:
        try:
            kind = value.type().kind()
        except (AttributeError, RuntimeError):
            kind = None
        if kind == "TensorType":
            raise ValueError("可选张量形状尚未确定")
        return False


def matmul_counts(left, right, output):
    """矩阵乘包括广播批次和向量输入，每个乘加统一计为两次 FLOPs。"""
    if not left or not right:
        raise ValueError("矩阵乘输入不能是标量")
    inner = left[-1]
    if inner != (right[0] if len(right) == 1 else right[-2]):
        raise ValueError("矩阵乘收缩维度不匹配")
    return {"dense_flops": 2 * prod(output) * inner}


def convolution_counts(inputs, outputs, transposed=False, has_bias=False):
    """依据实际权重维度处理分组卷积，偏置加法与乘加分别记录。"""
    x, weight = inputs
    if len(x) < 3 or len(weight) != len(x) or len(outputs) != len(x):
        raise ValueError("卷积维度不合法")
    spatial = x[2:] if transposed else outputs[2:]
    result = {"dense_flops": 2 * x[0] * prod(weight) * prod(spatial)}
    if has_bias:
        result["scalar_arithmetic_flops"] = prod(outputs)
    return result


def _einsum_labels(term, rank, ellipsis_rank):
    """将省略号展开为对齐的广播维度标签，不分配任何真实张量。"""
    if term.count("...") > 1:
        raise ValueError("einsum 省略号无效")
    letters = list(term.replace("...", ""))
    local_rank = rank-len(letters)
    if local_rank < 0 or ("..." not in term and local_rank != 0):
        raise ValueError("einsum 维度与方程不一致")
    if "..." not in term:
        return letters
    left, right = term.split("...")
    return list(left) + ["@%d" % i for i in range(ellipsis_rank-local_rank, ellipsis_rank)] + list(right)


def einsum_counts(equation, shapes):
    """统计两个操作数的标准收缩或逐元素乘法，复杂路径明确保留为未支持。"""
    equation = equation.replace(" ", "")
    if "->" not in equation:
        raise ValueError("einsum 需要显式输出方程以审计计数")
    source, target = equation.split("->")
    terms = source.split(",")
    if len(terms) != 2 or len(shapes) != 2:
        raise ValueError("暂不估计超过两个操作数的 einsum 优化路径")
    if any(re.sub(r"[a-zA-Z.]", "", term) for term in terms + [target]):
        raise ValueError("einsum 方程含有不支持的标签")
    ell_rank = max(len(s)-len(t.replace("...", "")) for t, s in zip(terms, shapes))
    labels = [_einsum_labels(t, len(s), ell_rank) for t, s in zip(terms, shapes)]
    if any(len(set(items)) != len(items) for items in labels):
        raise ValueError("einsum 对角线操作尚未纳入计数")
    dimensions = {}
    for names, shape in zip(labels, shapes):
        for name, size in zip(names, shape):
            previous = dimensions.get(name, 1)
            if previous != size and previous != 1 and size != 1:
                raise ValueError("einsum 广播维度不匹配")
            dimensions[name] = max(previous, size)
    output_labels = list(target.replace("...", ""))
    if "..." in target:
        output_labels += ["@%d" % i for i in range(ell_rank)]
    if len(set(output_labels)) != len(output_labels) or any(x not in dimensions for x in output_labels):
        raise ValueError("einsum 输出标签不合法")
    reduced = set(dimensions)-set(output_labels)
    if any(name not in labels[0] or name not in labels[1] for name in reduced):
        raise ValueError("einsum 含有单边预归约，不能假定实际收缩路径")
    count = prod(dimensions.values())
    return {"dense_flops" if reduced else "scalar_arithmetic_flops": (2 if reduced else 1)*count}


def reduction_counts(input_shape, output_shape, mean=False):
    """归约采用每组 K-1 次加法；均值额外计每组一次除法。"""
    before, after = prod(input_shape), prod(output_shape)
    if after == 0 or before < after or before % after:
        raise ValueError("归约形状不能确定有效分组")
    return {"scalar_arithmetic_flops": before-after + (after if mean else 0)}


def softmax_counts(shape, dim):
    """按稳定 softmax 的名义算法计数，指数和比较单列，不能当成硬件指令数。"""
    if not isinstance(dim, int) or not shape or not -len(shape) <= dim < len(shape):
        raise ValueError("softmax 维度不能确定")
    total, width = prod(shape), shape[dim]
    if width == 0:
        raise ValueError("softmax 维度为空")
    groups = total//width
    return {"scalar_arithmetic_flops": 3*total-groups,
            "nonlinear_element_ops": total, "comparison_element_ops": total-groups}


def adaptive_average_counts(input_shape, output_shape):
    """用真实自适应池化区间计算加法和除法，支持重叠池化窗口。"""
    if len(input_shape) != 4 or len(output_shape) != 4 or input_shape[:2] != output_shape[:2]:
        raise ValueError("仅支持四维自适应平均池化")
    spans = []
    for before, after in zip(input_shape[2:], output_shape[2:]):
        if after < 1:
            raise ValueError("池化输出尺寸必须为正")
        spans.append(sum(ceil((i+1)*before/after)-floor(i*before/after) for i in range(after)))
    return {"scalar_arithmetic_flops": prod(input_shape[:2])*prod(spans)}


def _dense_operator(op, inputs, outputs):
    """分派卷积、线性层和矩阵乘，避免直接套用 fvcore 的单乘加单位。"""
    out = value_shape(outputs[0])
    if not floating_output(outputs[0]):
        raise ValueError("稠密运算输出不是支持的浮点类型")
    if op in {"aten::mm", "aten::bmm", "aten::matmul", "aten::mv", "aten::dot"}:
        return matmul_counts(value_shape(inputs[0]), value_shape(inputs[1]), out)
    if op == "aten::linear":
        x, w = value_shape(inputs[0]), value_shape(inputs[1])
        if len(w) != 2 or x[-1] != w[-1]:
            raise ValueError("线性层输入与权重不匹配")
        return {"dense_flops": 2*prod(out)*w[-1],
                "scalar_arithmetic_flops": prod(out) if len(inputs) > 2 and _has_tensor(inputs[2]) else 0}
    if op == "aten::addmm":
        result = matmul_counts(value_shape(inputs[1]), value_shape(inputs[2]), out)
        beta = value_constant(inputs[3]) if len(inputs) > 3 else 1
        alpha = value_constant(inputs[4]) if len(inputs) > 4 else 1
        if not isinstance(beta, (int, float)) or not isinstance(alpha, (int, float)):
            raise ValueError("addmm 缩放系数无法确定")
        result["scalar_arithmetic_flops"] = prod(out)*((beta != 0) + (beta not in (0, 1)) + (alpha != 1))
        return result
    if op == "aten::einsum":
        equation = value_constant(inputs[0])
        if not isinstance(equation, str):
            raise ValueError("einsum 方程不是静态字符串")
        values = list(inputs[1].node().inputs())
        return einsum_counts(equation, [value_shape(item) for item in values])
    transposed = "transpose" in op
    if op in {"aten::_convolution", "aten::convolution"}:
        transposed = value_constant(inputs[6])
        if not isinstance(transposed, bool):
            raise ValueError("卷积转置标记无法确定")
    return convolution_counts([value_shape(inputs[0]), value_shape(inputs[1])], out,
                              transposed=transposed, has_bias=_has_tensor(inputs[2]))


def _element_operator(op, inputs, outputs):
    """计算浮点逐元素算术；比较、选择与非线性另列，不混进 GFLOPs。"""
    out = value_shape(outputs[0])
    count = prod(out)
    name = op.split("::", 1)[1].rstrip("_")
    if name in {"eq", "ne", "gt", "ge", "lt", "le", "relu", "clamp", "clamp_min", "clamp_max"}:
        factor = 2 if name == "clamp" else 1
        return {"comparison_element_ops": factor*count}
    if name in {"where", "masked_fill"}:
        return {"selection_element_ops": count}
    if not floating_output(outputs[0]):
        return {"integer_or_boolean_element_ops": count}
    if name in {"sum", "mean"}:
        return reduction_counts(value_shape(inputs[0]), out, mean=name == "mean")
    if name in {"softmax", "_softmax"}:
        return softmax_counts(value_shape(inputs[0]), value_constant(inputs[1]))
    if name == "adaptive_avg_pool2d":
        return adaptive_average_counts(value_shape(inputs[0]), out)
    if name in {"add", "sub", "rsub", "mul", "div", "true_divide", "neg"}:
        factor = 1
        if name in {"add", "sub", "rsub"} and len(inputs) > 2:
            alpha = value_constant(inputs[2])
            if not isinstance(alpha, (int, float)):
                raise ValueError("逐元素加法的 alpha 无法确定")
            factor += alpha != 1
        return {"scalar_arithmetic_flops": factor*count}
    if name == "pow" and len(inputs) > 1 and value_constant(inputs[1]) == 2:
        return {"scalar_arithmetic_flops": count}
    if name in {"max", "min", "amax", "amin"}:
        before = prod(value_shape(inputs[0]))
        if len(inputs) > 1 and _has_tensor(inputs[1]):
            return {"comparison_element_ops": count}
        if count > before:
            raise ValueError("最大最小值归约形状异常")
        return {"comparison_element_ops": before-count}
    if name in {"norm", "linalg_vector_norm"}:
        order = value_constant(inputs[1]) if len(inputs) > 1 else 2
        if order != 2:
            raise ValueError("仅定义二范数的名义算术口径")
        before = prod(value_shape(inputs[0]))
        return {"scalar_arithmetic_flops": 2*before-count, "nonlinear_element_ops": count}
    return {"nonlinear_element_ops": count}


DENSE_OPERATORS = frozenset({"aten::mm", "aten::bmm", "aten::matmul", "aten::mv", "aten::dot",
    "aten::linear", "aten::addmm", "aten::einsum", "aten::_convolution", "aten::convolution",
    "aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::conv_transpose1d",
    "aten::conv_transpose2d", "aten::conv_transpose3d"})
ELEMENT_OPERATORS = frozenset("aten::"+name for name in (
    "add", "add_", "sub", "sub_", "rsub", "mul", "mul_", "div", "div_", "true_divide", "neg",
    "pow", "sin", "cos", "atan2", "atan", "sqrt", "rsqrt", "exp", "log", "abs", "sigmoid",
    "sigmoid_", "tanh", "tanh_", "gelu", "silu", "prelu", "leaky_relu", "floor", "ceil",
    "remainder", "fmod", "eq", "ne", "gt", "ge", "lt", "le", "relu", "relu_", "clamp",
    "clamp_", "clamp_min", "clamp_max", "where", "masked_fill", "masked_fill_", "sum", "mean",
    "max", "min", "amax", "amin", "softmax", "_softmax", "norm", "linalg_vector_norm",
    "adaptive_avg_pool2d"))


def _make_handler(op, function, unresolved, reasons):
    """把纯形状计数包装成 fvcore 钩子；失败明确进入未支持清单。"""
    def handler(inputs, outputs):
        """仅读取 JIT 值，按分类返回计数，不执行真实张量运算。"""
        try:
            counts = function(op, inputs, outputs)
            return Counter({category+"|"+op: int(value) for category, value in counts.items() if value})
        except (ValueError, TypeError, IndexError, AttributeError, RuntimeError) as exc:
            unresolved[op] += 1
            reasons.setdefault(op, set()).add(str(exc))
            return Counter()
    return handler


def _ignore_handler(op, observed):
    """记录明确排除的内存或形状操作，而不是伪造其浮点代价。"""
    def handler(inputs, outputs):
        """保存算子出现次数，浮点计数保持为空。"""
        observed[op] += 1
        return Counter()
    return handler


def _format_counts(counts):
    """将 fvcore 的分类键还原为可审计、可 JSON 序列化的逐算子明细。"""
    by_operator, totals = {}, Counter()
    for key, value in counts.items():
        category, op = key.split("|", 1)
        by_operator.setdefault(op, {})[category] = int(value)
        totals[category] += int(value)
    return dict(sorted(by_operator.items())), dict(totals)


def snapshot_forward_hooks(model):
    """保存原有前向钩子的编号，便于清理 fvcore 异常退出时遗留的新钩子。"""
    names = ("_forward_pre_hooks", "_forward_hooks", "_forward_pre_hooks_with_kwargs",
             "_forward_hooks_with_kwargs", "_forward_hooks_always_called")
    return [(module, {name: set(getattr(module, name, ())) for name in names})
            for module in model.modules()]


def remove_new_forward_hooks(snapshot):
    """只移除本次新增的前向钩子及标记，原有钩子对象、编号和顺序保持不变。"""
    for module, original in snapshot:
        for name, original_keys in original.items():
            current = getattr(module, name, None)
            if current is None:
                continue
            for key in set(current)-original_keys:
                if isinstance(current, set):
                    current.discard(key)
                else:
                    current.pop(key, None)


def profile_complexity(model, sample):
    """由用户手动调用后才追踪模型；返回有边界说明的计数，绝不标为完整硬件 FLOPs。"""
    result = {"status": "failed", "gflops_counted": None, "by_operator": {},
              "unsupported_ops": {}, "uncalled_modules": [], "conventions": {
                  "dense": "卷积/矩阵乘采用1 MAC=2 FLOPs；偏置加法额外单列。",
                  "reported_gflops": "仅稠密乘加与已支持的浮点基础算术之和除以1e9；不是完整硬件FLOPs。",
                  "nonlinear": "pow/sin/cos/atan2/sqrt/激活等每个输出元素记一次名义操作，单列且不加入GFLOPs；pow(x,2)按一次乘法。",
                  "softmax": "稳定softmax按减最大值、exp、求和、除法的名义算法拆分；不声称反映库实际指令。",
                  "exclusions": "内存读写/搬运/索引/分支/形状与整数运算不折算浮点成本；比较、选择、非线性单列。未注册插值/归一化等不得默认为零。",
                  "tracing": "仅此输入形状和本次执行路径；Python标量转换或控制流可能被冻结，须查看trace_warnings。",
                  "attribution": "直接调用模块其他方法时可能列入uncalled_modules；其中已追踪算子仍计入根模型，不能据此认定整个模块漏算。",
                  "scope": "approximate/counting convention；无论未支持算子是否为空，都不称为完整硬件FLOPs。"}}
    hook_snapshot = []
    caught = []
    try:
        import torch
        import fvcore
        from fvcore.nn.jit_analysis import JitModelAnalysis

        inputs = sample if isinstance(sample, tuple) else (sample,)
        if model.training:
            raise ValueError("请传入独立的 eval 模型；复杂度统计不会替调用者更改训练模式。")
        if not all(isinstance(item, torch.Tensor) for item in inputs):
            raise ValueError("当前只接受一个张量或张量元组输入")
        result["input_shapes"] = [list(item.shape) for item in inputs]
        result["versions"] = {"torch": str(torch.__version__), "fvcore": str(getattr(fvcore, "__version__", "unknown"))}
        hook_snapshot = snapshot_forward_hooks(model)
        analysis = JitModelAnalysis(model, inputs)
        unresolved, reasons, ignored = Counter(), {}, Counter()
        defaults_ignored = sorted(getattr(analysis, "_ignored_ops", ()))
        handlers = {op: _ignore_handler(op, ignored) for op in defaults_ignored}
        handlers.update({op: _make_handler(op, _dense_operator, unresolved, reasons) for op in DENSE_OPERATORS})
        handlers.update({op: _make_handler(op, _element_operator, unresolved, reasons) for op in ELEMENT_OPERATORS})
        analysis.set_op_handle(**handlers)
        analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings("all")
        with warnings.catch_warnings(record=True) as caught, torch.no_grad():
            warnings.simplefilter("always")
            by_operator, totals = _format_counts(analysis.by_operator())
            unsupported = Counter(analysis.unsupported_ops()) + unresolved
            uncalled = sorted(analysis.uncalled_modules())
        arithmetic = totals.get("dense_flops", 0)+totals.get("scalar_arithmetic_flops", 0)
        result.update(status="partial" if unsupported else "counted_approximate",
                      gflops_counted=arithmetic/1e9, arithmetic_flops_counted=arithmetic,
                      category_totals=totals, by_operator=by_operator,
                      unsupported_ops=dict(sorted(unsupported.items())),
                      unsupported_reasons={op: sorted(values) for op, values in sorted(reasons.items())},
                      uncalled_modules=uncalled, explicitly_ignored_ops=dict(sorted(ignored.items())),
                      registered_default_exclusions=defaults_ignored,
                      trace_warnings=sorted({str(item.message) for item in caught}))
    except Exception as exc:
        result["error"] = type(exc).__name__+": "+str(exc)
        result["trace_warnings"] = sorted({str(item.message) for item in caught})
    finally:
        remove_new_forward_hooks(hook_snapshot)
    return result
