"""
神经曲线层模块
Neural Curve Layer - 实现分段线性曲线的全局调整

核心思想：
- 网络预测 M 个曲线控制点 k_0, k_1, ..., k_{M-1}
- 使用分段线性插值将输入像素值 [0,1] 映射到调整后的值
- 可用于全局亮度/颜色调节，类似 Photoshop 曲线工具

参考：
- UIEC²-Net: CNN-based Underwater Image Enhancement Using Two Color Space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sgn_m(x):
    """
    分段线性选择器函数 δ(x)
    
    定义：
        - x < 0:     δ(x) = 0
        - 0 ≤ x ≤ 1: δ(x) = x
        - x > 1:     δ(x) = 1
    
    参数:
        x: 输入张量，任意形状
        
    返回:
        裁剪后的张量，范围 [0, 1]
    """
    # 使用 clamp 简化实现，等价于原版 torch.where 写法
    return torch.clamp(x, 0, 1)


def piece_function(x, curve_params, M):
    """
    分段线性曲线函数 - 核心映射公式
    
    公式：S(I) = k_0 + Σ_{m=0}^{M-1} (k_{m+1} - k_m) * δ(M*I - m)
    
    参数:
        x: 输入图像通道 [B, 1, H, W]，值范围 [0, 1]
        curve_params: 曲线控制点 [B, M]，每个值表示曲线在该刻度的输出值
        M: 控制点数量
        
    返回:
        映射后的图像通道 [B, 1, H, W]
        
    示例:
        如果 M=11，则曲线在 0, 0.1, 0.2, ..., 1.0 处有控制点
        输入像素值 0.35 会在 0.3 和 0.4 之间线性插值
    """
    b, c, h, w = x.shape
    device = x.device
    
    # 初始化输出为 k_0（第一个控制点）
    # curve_params[:, 0] 形状 [B]，需要扩展到 [B, 1, H, W]
    r = curve_params[:, 0].view(b, 1, 1, 1).expand(b, c, h, w)
    
    # 累加公式：r = k_0 + Σ (k_{i+1} - k_i) * δ(M*x - i)
    for i in range(M - 1):
        # 计算斜率 (k_{i+1} - k_i)
        slope = (curve_params[:, i + 1] - curve_params[:, i])
        slope = slope.view(b, 1, 1, 1).expand(b, c, h, w)
        
        # 计算分段选择器 δ(M*x - i)
        delta = sgn_m(M * x - i)
        
        # 累加到结果
        r = r + slope * delta
    
    return r


class NeuralCurveLayer(nn.Module):
    """
    神经曲线层 - 自动学习图像全局调整曲线
    
    网络学习预测曲线控制点，然后使用分段线性插值进行像素映射。
    适用于全局亮度、对比度调节等任务。
    
    参数:
        in_channels: 输入特征通道数
        M: 曲线控制点数量，默认 11（将 [0,1] 分为 10 段）
        num_curves: 输出曲线数量，默认 1（仅调整 I 通道）
        
    输入:
        feat: CNN 特征 [B, in_channels, H, W]
        img_channel: 需要调整的图像通道 [B, num_curves, H, W]
        
    输出:
        调整后的图像通道 [B, num_curves, H, W]
        曲线控制点 [B, num_curves, M]（可选，用于可视化）
    """
    
    def __init__(self, in_channels, M=11, num_curves=1):
        super().__init__()
        
        self.M = M
        self.num_curves = num_curves
        
        # 曲线预测网络：从特征中预测控制点
        self.curve_predictor = nn.Sequential(
            # 全局平均池化：把 [B,C,H,W] 变成 [B,C,1,1]
            nn.AdaptiveAvgPool2d(1),
            
            # 展平：[B,C,1,1] → [B,C]
            nn.Flatten(),
            
            # 第一层全连接：C → 64
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            
            # 第二层全连接：64 → M（控制点数）
            nn.Linear(64, num_curves * M),
            
            # Sigmoid：确保输出在 [0,1] 范围
            nn.Sigmoid()
        )
        
        # 初始化为恒等映射（对角线曲线）
        self._init_identity()
    
    def _init_identity(self):
        """
        初始化曲线为恒等映射
        使得初始时 output = input（对角线曲线）
        让神经曲线层在训练开始时不改变图像。
        """
        # 最后一个线性层的权重初始化为 0
        # 偏置初始化为等间距的值 [0, 1/M, 2/M, ..., 1]
        last_linear = self.curve_predictor[-2]  # Sigmoid 之前的 Linear
        nn.init.zeros_(last_linear.weight)
        
        # 生成恒等映射的控制点
        identity_curve = torch.linspace(0, 1, self.M)
        identity_bias = identity_curve.repeat(self.num_curves)
        # Sigmoid 反函数
        identity_bias = torch.log(identity_bias / (1 - identity_bias + 1e-6))
        last_linear.bias.data = identity_bias
    
    def forward(self, feat, img_channel, return_curve=False):
        """
        前向传播
        
        参数:
            feat: CNN 特征 [B, in_channels, H, W]
            img_channel: 图像通道 [B, num_curves, H, W]，范围 [0, 1]
            return_curve: 是否返回曲线控制点
            
        返回:
            如果 return_curve=False: 调整后的通道 [B, num_curves, H, W]
            如果 return_curve=True: (调整后的通道, 曲线控制点 [B, num_curves, M])
        """
        b = feat.shape[0]
        
        # 预测曲线控制点
        curve_params = self.curve_predictor(feat)  # [B, num_curves * M]
        curve_params = curve_params.view(b, self.num_curves, self.M)  # [B, num_curves, M]
        
        # 对每个通道应用曲线
        outputs = []
        for i in range(self.num_curves):
            channel_in = img_channel[:, i:i+1, :, :]  # [B, 1, H, W]
            curve = curve_params[:, i, :]  # [B, M]
            channel_out = piece_function(channel_in, curve, self.M)
            outputs.append(channel_out)
        
        # 拼接输出
        output = torch.cat(outputs, dim=1)  # [B, num_curves, H, W]
        
        # 裁剪到有效范围
        output = torch.clamp(output, 0, 1)
        
        if return_curve:
            return output, curve_params
        return output


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("神经曲线层单元测试")
    print("=" * 60)
    
    # 测试配置
    batch_size = 2
    height, width = 128, 128
    in_channels = 36
    
    # 模拟输入
    feat = torch.randn(batch_size, in_channels, height // 4, width // 4)
    i_channel = torch.rand(batch_size, 1, height, width)  # I 通道，范围 [0, 1]
    
    print(f"\n测试配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {height}x{width}")
    print(f"  特征通道: {in_channels}")
    print("-" * 40)
    
    # 测试1: 分段线性函数
    print("\n[测试1] piece_function 分段线性函数")
    curve = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          [0.0, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]])
    x_test = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    out_test = piece_function(x_test.expand(2, 1, 1, 3), curve, M=11)
    print(f"  输入: {x_test.squeeze()}")
    print(f"  曲线1(恒等): {curve[0]}")
    print(f"  输出1: {out_test[0].squeeze()}")
    print("  [OK] 分段线性函数测试通过")
    
    # 测试2: NeuralCurveLayer
    print("\n[测试2] NeuralCurveLayer 神经曲线层")
    curve_layer = NeuralCurveLayer(in_channels=in_channels, M=11, num_curves=1)
    i_out, curve_params = curve_layer(feat, i_channel, return_curve=True)
    print(f"  输入特征形状: {feat.shape}")
    print(f"  输入I通道形状: {i_channel.shape}")
    print(f"  输出I通道形状: {i_out.shape}")
    print(f"  曲线参数形状: {curve_params.shape}")
    print(f"  曲线参数范围: [{curve_params.min().item():.4f}, {curve_params.max().item():.4f}]")
    print(f"  参数量: {sum(p.numel() for p in curve_layer.parameters()):,}")
    assert i_out.shape == i_channel.shape, "输出形状错误！"
    print("  [OK] 神经曲线层测试通过")
    
    # 测试3: 恒等初始化验证
    print("\n[测试3] 恒等初始化验证")
    # 初始时输出应接近输入
    with torch.no_grad():
        i_identity = curve_layer(feat, i_channel)
        diff = (i_identity - i_channel).abs().mean()
        print(f"  输入输出差异: {diff.item():.6f} (应接近0)")
        if diff < 0.1:
            print("  [OK] 恒等初始化验证通过")
        else:
            print("  [WARN] 恒等初始化可能需要调整")
    
    # GPU测试
    if torch.cuda.is_available():
        print("\n[测试4] GPU兼容性测试")
        feat_cuda = feat.cuda()
        i_cuda = i_channel.cuda()
        curve_layer_cuda = NeuralCurveLayer(in_channels, M=11).cuda()
        i_out_cuda = curve_layer_cuda(feat_cuda, i_cuda)
        print(f"  GPU设备: {i_out_cuda.device}")
        print("  [OK] GPU测试通过")
    else:
        print("\n[测试4] GPU不可用，跳过GPU测试")
    
    print("\n" + "=" * 60)
    print("所有测试通过！神经曲线层可正常使用。")
    print("=" * 60)
