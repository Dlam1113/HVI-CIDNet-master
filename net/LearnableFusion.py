"""
可学习权重融合模块
Learnable Fusion - 实现RGB和HVI输出的智能加权融合

核心思想：
- 不同图像区域对RGB和HVI的依赖程度不同
- 暗部区域可能更依赖HVI（亮度增强更好）
- 纹理区域可能更依赖RGB（细节保持更好）
- 通过学习每个像素位置的融合权重，实现自适应融合


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableFusion(nn.Module):
    """
    像素级可学习融合模块
    
    学习每个像素位置的最优融合权重，实现RGB和HVI输出的自适应融合
    
    融合公式：
    output = w_rgb * rgb_out + w_hvi * hvi_out
    其中 w_rgb + w_hvi = 1（通过softmax保证）
    
    参数:
        in_channels: RGB/HVI输出的通道数，默认3
        mid_channels: 中间层通道数，默认32
        use_input: 是否使用原始输入作为辅助信息，默认True
    """
    
    def __init__(self, in_channels=3, mid_channels=32, use_input=True):
        super().__init__()
        
        self.use_input = use_input
        
        # 输入通道数：RGB输出 + HVI输出 + (可选)原始输入
        if use_input:
            total_in = in_channels * 3  # rgb_out + hvi_out + input = 9
        else:
            total_in = in_channels * 2  # rgb_out + hvi_out = 6
        
        # 权重预测网络（4层CNN）
        self.weight_net = nn.Sequential(
            # 层1：初始特征提取
            nn.Conv2d(total_in, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 层2：特征处理
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 层3：特征精炼
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 层4：输出2通道权重图
            nn.Conv2d(mid_channels, 2, kernel_size=1, stride=1, padding=0),
        )
        
    def forward(self, rgb_out, hvi_out, x_input=None):
        """
        前向传播
        
        参数:
            rgb_out: RGB分支输出 [B, 3, H, W]
            hvi_out: HVI分支输出 [B, 3, H, W]
            x_input: 原始输入 [B, 3, H, W]（可选）
            
        返回:
            融合后的输出 [B, 3, H, W]
        """
        # 构建权重预测网络的输入
        if self.use_input and x_input is not None:
            weight_input = torch.cat([rgb_out, hvi_out, x_input], dim=1)
        else:
            weight_input = torch.cat([rgb_out, hvi_out], dim=1)
        
        # 预测权重图 [B, 2, H, W]
        weights = self.weight_net(weight_input)
        
        # Softmax归一化，保证权重和为1
        weights = F.softmax(weights, dim=1)
        w_rgb = weights[:, 0:1, :, :]  # [B, 1, H, W]
        w_hvi = weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 加权融合
        output = w_rgb * rgb_out + w_hvi * hvi_out
        
        return output, w_rgb, w_hvi

class AdaptiveFusion(nn.Module):
    """
    自适应融合模块（建设性建议3的实现 - 注意力可视化友好版）
    
    提供详细的注意力权重输出，便于分析和可视化
    
    参数:
        in_channels: 输入通道数，默认3
        mid_channels: 中间层通道数，默认32
    """
    
    def __init__(self, in_channels=3, mid_channels=32):
        super().__init__()
        
        # 特征提取
        self.feat_extract = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 亮度权重分支（基于输入亮度分析）
        self.luminance_branch = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1),
        )
        
        # 纹理权重分支（基于边缘/纹理分析）
        self.texture_branch = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1),
        )
        
        # 融合控制
        self.fusion_control = nn.Conv2d(2, 2, kernel_size=1)
        
    def forward(self, rgb_out, hvi_out, x_input):
        """
        前向传播
        
        返回:
            (融合输出, RGB权重图, HVI权重图)
        """
        output, w_rgb, lum_attn, tex_attn = self.forward_with_analysis(rgb_out, hvi_out, x_input)
        w_hvi = 1 - w_rgb  # HVI权重 = 1 - RGB权重
        return output, w_rgb, w_hvi
    
    def forward_with_analysis(self, rgb_out, hvi_out, x_input):
        """
        前向传播，返回详细分析信息
        
        返回:
            (融合输出, RGB权重图, 亮度注意力图, 纹理注意力图)
        """
        # 特征提取
        concat = torch.cat([rgb_out, hvi_out, x_input], dim=1)
        feat = self.feat_extract(concat)
        
        # 分析亮度和纹理
        luminance_attn = torch.sigmoid(self.luminance_branch(feat))  # 亮度注意力
        texture_attn = torch.sigmoid(self.texture_branch(feat))       # 纹理注意力
        
        # 融合权重计算
        attn_concat = torch.cat([luminance_attn, texture_attn], dim=1)
        weights = F.softmax(self.fusion_control(attn_concat), dim=1)
        
        w_rgb = weights[:, 0:1, :, :]
        w_hvi = weights[:, 1:2, :, :]
        
        # 加权融合
        output = w_rgb * rgb_out + w_hvi * hvi_out
        
        return output, w_rgb, luminance_attn, texture_attn

class AttentionFusion(nn.Module):
    """
    注意力引导融合模块
    
    使用通道注意力和空间注意力来指导RGB和HVI的融合
    
    参数:
        in_channels: 输入通道数，默认3
        reduction: 通道注意力的压缩比例，默认4
    """
    
    def __init__(self, in_channels=3, reduction=4):
        super().__init__()
        
        # 通道注意力 - 学习每个通道的重要性
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels * 2 // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2 // reduction, in_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 学习每个位置的重要性
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )
        
        # 最终融合权重预测
        self.fusion_conv = nn.Conv2d(in_channels * 2, 2, kernel_size=1)
        
    def forward(self, rgb_out, hvi_out):
        """
        前向传播
        
        参数:
            rgb_out: RGB输出 [B, 3, H, W]
            hvi_out: HVI输出 [B, 3, H, W]
            
        返回:
            融合后的输出 [B, 3, H, W]
        """
        # 拼接特征
        concat = torch.cat([rgb_out, hvi_out], dim=1)  # [B, 6, H, W]
        
        # 通道注意力
        ca = self.channel_attn(concat)  # [B, 6, 1, 1]
        concat = concat * ca
        
        # 空间注意力
        avg_pool = torch.mean(concat, dim=1, keepdim=True)
        max_pool, _ = torch.max(concat, dim=1, keepdim=True)
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attn(sa_input)  # [B, 1, H, W]
        concat = concat * sa
        
        # 预测融合权重
        weights = F.softmax(self.fusion_conv(concat), dim=1)
        w_rgb = weights[:, 0:1, :, :]
        w_hvi = weights[:, 1:2, :, :]
        
        # 加权融合
        output = w_rgb * rgb_out + w_hvi * hvi_out
        
        return output


class GatedFusion(nn.Module):
    """
    门控融合模块
    
    使用门控机制来控制RGB和HVI信息的流动
    借鉴LSTM/GRU的门控思想
    
    参数:
        in_channels: 输入通道数，默认3
        mid_channels: 中间层通道数，默认16
    """
    
    def __init__(self, in_channels=3, mid_channels=16):
        super().__init__()
        
        # 更新门：决定保留多少HVI信息
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 重置门：决定融入多少RGB信息
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 候选状态生成
        self.candidate = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.Tanh()
        )
        
    def forward(self, rgb_out, hvi_out):
        """
        前向传播
        
        参数:
            rgb_out: RGB输出 [B, 3, H, W]
            hvi_out: HVI输出 [B, 3, H, W]
            
        返回:
            融合后的输出 [B, 3, H, W]
        """
        concat = torch.cat([rgb_out, hvi_out], dim=1)
        
        # 计算门控值
        z = self.update_gate(concat)   # 更新门
        r = self.reset_gate(concat)    # 重置门
        
        # 生成候选状态
        reset_concat = torch.cat([r * rgb_out, hvi_out], dim=1)
        candidate = self.candidate(reset_concat)
        
        # 门控融合
        output = (1 - z) * hvi_out + z * candidate
        
        return output





class LightweightFusion(nn.Module):
    """
    轻量级融合模块（建设性建议4的实现 - 轻量化设计）
    
    使用深度可分离卷积减少参数量
    
    参数:
        in_channels: 输入通道数，默认3
    """
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # 深度可分离卷积权重预测
        self.weight_net = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, 
                      padding=1, groups=in_channels * 2),
            nn.InstanceNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            
            # 逐点卷积
            nn.Conv2d(in_channels * 2, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            
            # 深度卷积
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
            nn.ReLU(inplace=True),
            
            # 输出权重
            nn.Conv2d(8, 2, kernel_size=1),
        )
        
    def forward(self, rgb_out, hvi_out):
        """前向传播"""
        concat = torch.cat([rgb_out, hvi_out], dim=1)
        weights = F.softmax(self.weight_net(concat), dim=1)
        
        w_rgb = weights[:, 0:1, :, :]
        w_hvi = weights[:, 1:2, :, :]
        
        output = w_rgb * rgb_out + w_hvi * hvi_out
        return output


# ==========  测试代码  ==========
if __name__ == '__main__':
    print("=" * 60)
    print("可学习权重融合模块单元测试")
    print("=" * 60)
    
    # 测试配置
    batch_size = 2
    height, width = 128, 128
    in_channels = 3
    
    # 模拟输入
    rgb_out = torch.rand(batch_size, in_channels, height, width)
    hvi_out = torch.rand(batch_size, in_channels, height, width)
    x_input = torch.rand(batch_size, in_channels, height, width)
    
    print(f"\n测试配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {height}x{width}")
    print(f"  通道数: {in_channels}")
    print("-" * 40)
    
    # 测试1: LearnableFusion
    print("\n[测试1] LearnableFusion (可学习融合)")
    fusion1 = LearnableFusion(in_channels=3, mid_channels=32, use_input=True)
    out1, w_rgb, w_hvi = fusion1(rgb_out, hvi_out, x_input)
    print(f"  输出形状: {out1.shape}")
    print(f"  RGB权重范围: [{w_rgb.min().item():.4f}, {w_rgb.max().item():.4f}]")
    print(f"  HVI权重范围: [{w_hvi.min().item():.4f}, {w_hvi.max().item():.4f}]")
    print(f"  权重和检验: {(w_rgb + w_hvi).mean().item():.6f} (应接近1.0)")
    print(f"  参数量: {sum(p.numel() for p in fusion1.parameters()):,}")
    assert out1.shape == rgb_out.shape, "输出形状错误！"
    print("  [OK] 测试通过")
    
    # 测试2: AttentionFusion
    print("\n[测试2] AttentionFusion (注意力融合)")
    fusion2 = AttentionFusion(in_channels=3, reduction=4)
    out2 = fusion2(rgb_out, hvi_out)
    print(f"  输出形状: {out2.shape}")
    print(f"  参数量: {sum(p.numel() for p in fusion2.parameters()):,}")
    assert out2.shape == rgb_out.shape, "输出形状错误！"
    print("  [OK] 测试通过")
    
    # 测试3: GatedFusion
    print("\n[测试3] GatedFusion (门控融合)")
    fusion3 = GatedFusion(in_channels=3, mid_channels=16)
    out3 = fusion3(rgb_out, hvi_out)
    print(f"  输出形状: {out3.shape}")
    print(f"  参数量: {sum(p.numel() for p in fusion3.parameters()):,}")
    assert out3.shape == rgb_out.shape, "输出形状错误！"
    print("  [OK] 测试通过")
    
    # 测试4: AdaptiveFusion
    print("\n[测试4] AdaptiveFusion (自适应融合 - 可视化友好)")
    fusion4 = AdaptiveFusion(in_channels=3, mid_channels=32)
    out4, w_rgb4, lum_attn, tex_attn = fusion4.forward_with_analysis(rgb_out, hvi_out, x_input)
    print(f"  输出形状: {out4.shape}")
    print(f"  RGB权重形状: {w_rgb4.shape}")
    print(f"  亮度注意力形状: {lum_attn.shape}")
    print(f"  纹理注意力形状: {tex_attn.shape}")
    print(f"  参数量: {sum(p.numel() for p in fusion4.parameters()):,}")
    assert out4.shape == rgb_out.shape, "输出形状错误！"
    print("  [OK] 测试通过")
    
    # 测试5: LightweightFusion
    print("\n[测试5] LightweightFusion (轻量级融合)")
    fusion5 = LightweightFusion(in_channels=3)
    out5 = fusion5(rgb_out, hvi_out)
    print(f"  输出形状: {out5.shape}")
    print(f"  参数量: {sum(p.numel() for p in fusion5.parameters()):,}")
    assert out5.shape == rgb_out.shape, "输出形状错误！"
    print("  [OK] 测试通过")
    
    # 参数量对比
    print("\n" + "-" * 40)
    print("各融合模块参数量对比:")
    print(f"  LearnableFusion:   {sum(p.numel() for p in fusion1.parameters()):>8,}")
    print(f"  AttentionFusion:   {sum(p.numel() for p in fusion2.parameters()):>8,}")
    print(f"  GatedFusion:       {sum(p.numel() for p in fusion3.parameters()):>8,}")
    print(f"  AdaptiveFusion:    {sum(p.numel() for p in fusion4.parameters()):>8,}")
    print(f"  LightweightFusion: {sum(p.numel() for p in fusion5.parameters()):>8,}")
    
    # GPU测试
    if torch.cuda.is_available():
        print("\n[测试6] GPU兼容性测试")
        rgb_cuda = rgb_out.cuda()
        hvi_cuda = hvi_out.cuda()
        x_cuda = x_input.cuda()
        fusion_cuda = LearnableFusion().cuda()
        out_cuda = fusion_cuda(rgb_cuda, hvi_cuda, x_cuda)
        print(f"  GPU设备: {out_cuda.device}")
        print("  [OK] GPU测试通过")
    else:
        print("\n[测试6] GPU不可用，跳过GPU测试")
    
    print("\n" + "=" * 60)
    print("所有测试通过！可学习权重融合模块可正常使用。")
    print("=" * 60)
