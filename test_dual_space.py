# DualSpaceCIDNet 测试脚本
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from net.DualSpaceCIDNet import DualSpaceCIDNet
from net.CIDNet import CIDNet

print("=" * 60)
print("DualSpaceCIDNet 主网络单元测试")
print("=" * 60)

# 测试配置
batch_size = 2
height, width = 256, 256

# 模拟输入
x = torch.randn(batch_size, 3, height, width)

print(f"\n测试配置:")
print(f"  批次大小: {batch_size}")
print(f"  图像尺寸: {height}x{width}")
print("-" * 40)

# 测试1: 基础前向传播
print("\n[测试1] DualSpaceCIDNet (基础前向传播)")
model = DualSpaceCIDNet(
    channels=[36, 36, 72, 144],
    heads=[1, 2, 4, 8],
    fusion_type='learnable',
    cross_space_attn=True
)
out = model(x)
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {out.shape}")
print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
assert out.shape == x.shape, "输出形状错误！"
print("  [OK] 测试通过")

# 测试2: 带中间结果的前向传播
print("\n[测试2] forward_with_intermediates")
results = model.forward_with_intermediates(x)
print(f"  最终输出形状: {results['output'].shape}")
print(f"  RGB分支输出形状: {results['rgb_out'].shape}")
print(f"  HVI分支输出形状: {results['hvi_out'].shape}")
if results['fusion_weights'] is not None:
    print(f"  RGB权重形状: {results['fusion_weights']['rgb'].shape}")
    print(f"  HVI权重形状: {results['fusion_weights']['hvi'].shape}")
print("  [OK] 测试通过")

# 测试3: 无跨空间注意力
print("\n[测试3] DualSpaceCIDNet (无跨空间注意力)")
model_no_cross = DualSpaceCIDNet(
    channels=[36, 36, 72, 144],
    heads=[1, 2, 4, 8],
    cross_space_attn=False
)
out_no_cross = model_no_cross(x)
print(f"  输出形状: {out_no_cross.shape}")
print(f"  参数量: {sum(p.numel() for p in model_no_cross.parameters()):,}")
print("  [OK] 测试通过")

# 测试4: 使用AdaptiveFusion
print("\n[测试4] DualSpaceCIDNet (AdaptiveFusion)")
model_adaptive = DualSpaceCIDNet(
    channels=[36, 36, 72, 144],
    heads=[1, 2, 4, 8],
    fusion_type='adaptive'
)
out_adaptive = model_adaptive(x)
print(f"  输出形状: {out_adaptive.shape}")
print(f"  参数量: {sum(p.numel() for p in model_adaptive.parameters()):,}")
print("  [OK] 测试通过")

# 参数量对比
print("\n" + "-" * 40)
print("参数量对比:")

original_model = CIDNet(channels=[36, 36, 72, 144], heads=[1, 2, 4, 8])
original_params = sum(p.numel() for p in original_model.parameters())
print(f"  原CIDNet:        {original_params:>10,}")

dual_params = sum(p.numel() for p in model.parameters())
print(f"  DualSpaceCIDNet: {dual_params:>10,}")

increase = (dual_params - original_params) / original_params * 100
print(f"  参数增加:        {increase:>9.1f}%")

# GPU测试
if torch.cuda.is_available():
    print("\n[测试5] GPU兼容性测试")
    x_cuda = x.cuda()
    model_cuda = DualSpaceCIDNet().cuda()
    out_cuda = model_cuda(x_cuda)
    print(f"  GPU设备: {out_cuda.device}")
    print(f"  输出形状: {out_cuda.shape}")
    print("  [OK] GPU测试通过")
else:
    print("\n[测试5] GPU不可用，跳过GPU测试")

print("\n" + "=" * 60)
print("所有测试通过！DualSpaceCIDNet主网络可正常使用。")
print("=" * 60)
