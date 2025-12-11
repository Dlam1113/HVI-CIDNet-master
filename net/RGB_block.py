"""
RGB Pixel-level Block for Low-light Image Enhancement

参考 UIEC²-Net 的 RGB pixel-level block 设计：
- 7层 Conv3×3 + InstanceNorm + LeakyReLU
- 无下采样，保留图像细节
- 用于去噪和基础像素级处理

Author: Based on UIEC²-Net design, adapted for HVI-CIDNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
 

class RGBPixelBlock(nn.Module):
    """
    RGB像素级增强模块
    
    设计思想：
    - 参考UIEC²-Net的RGB pixel-level block
    - 7层卷积网络，无下采样
    - 保留空间分辨率，专注于像素级去噪和色偏校正
    
    Args:
        in_channels (int): 输入通道数，默认3 (RGB)
        hidden_channels (int): 隐藏层通道数，默认64
        out_channels (int): 输出通道数，默认3 (RGB)
        num_layers (int): 卷积层数，默认7
        use_instance_norm (bool): 是否使用InstanceNorm，默认True
    """
    
    def __init__(self, 
                 in_channels=3, 
                 hidden_channels=64, 
                 out_channels=3,
                 num_layers=7,
                 use_instance_norm=True):
        super(RGBPixelBlock, self).__init__()
        
        self.num_layers = num_layers
        
        # 构建卷积层列表
        layers = []
        
        # 第一层：输入通道 -> 隐藏通道
        layers.append(self._make_conv_block(
            in_channels, hidden_channels, use_instance_norm
        ))
        
        # 中间层：隐藏通道 -> 隐藏通道
        for i in range(num_layers - 2):
            layers.append(self._make_conv_block(
                hidden_channels, hidden_channels, use_instance_norm
            ))
        
        # 最后一层：隐藏通道 -> 输出通道（使用Sigmoid输出[0,1]范围）
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()  # 输出范围限制在[0,1]
        )
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _make_conv_block(self, in_ch, out_ch, use_norm=True):
        """
        创建单个卷积块：Conv3×3 + InstanceNorm + LeakyReLU
        
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数
            use_norm: 是否使用InstanceNorm
        
        Returns:
            nn.Sequential: 卷积块
        """
        if use_norm:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
    
    def _init_weights(self):
        """
        使用Xavier初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入RGB图像，shape [B, 3, H, W]，范围[0, 1]
        
        Returns:
            Tensor: 输出增强RGB图像，shape [B, 3, H, W]，范围[0, 1]
        """
        # 通过中间卷积层
        feat = self.conv_layers(x)
        
        # 最后一层输出
        out = self.final_conv(feat)
        
        return out


class RGBPixelBlockWithResidual(nn.Module):
    """
    带残差连接的RGB像素级增强模块
    
    设计思想：
    - 在RGBPixelBlock基础上添加残差连接
    - 学习增量变化而非完整映射
    - 更容易训练，梯度传播更稳定
    
    Args:
        in_channels (int): 输入通道数，默认3
        hidden_channels (int): 隐藏层通道数，默认64
        num_layers (int): 卷积层数，默认7
        use_instance_norm (bool): 是否使用InstanceNorm，默认True
    """
    
    def __init__(self,
                 in_channels=3,
                 hidden_channels=64,
                 num_layers=7,
                 use_instance_norm=True):
        super(RGBPixelBlockWithResidual, self).__init__()
        
        self.rgb_block = RGBPixelBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            num_layers=num_layers,
            use_instance_norm=use_instance_norm
        )
    
    def forward(self, x):
        """
        前向传播（带残差）
        
        Args:
            x (Tensor): 输入RGB图像
        
        Returns:
            Tensor: 输出 = RGB块输出 + 输入（残差连接）
        """
        # RGB块处理
        enhanced = self.rgb_block(x)
        
        # 残差连接：输出 = 增强结果 + 原始输入
        # 注意：由于Sigmoid输出在[0,1]，残差后需要clamp
        out = torch.clamp(enhanced + x, 0, 1)
        
        return out


# 测试代码
if __name__ == "__main__":
    # 创建测试输入
    batch_size = 2
    height, width = 256, 256
    x = torch.rand(batch_size, 3, height, width)
    
    print("=" * 50)
    print("测试 RGBPixelBlock")
    print("=" * 50)
    
    # 测试基础版本
    model = RGBPixelBlock(hidden_channels=64, num_layers=7)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n" + "=" * 50)
    print("测试 RGBPixelBlockWithResidual")
    print("=" * 50)
    
    # 测试残差版本
    model_res = RGBPixelBlockWithResidual(hidden_channels=64, num_layers=7)
    output_res = model_res(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_res.shape}")
    print(f"输出范围: [{output_res.min():.4f}, {output_res.max():.4f}]")
    
    # 计算参数量
    total_params_res = sum(p.numel() for p in model_res.parameters())
    print(f"总参数量: {total_params_res:,}")
