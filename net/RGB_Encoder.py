"""
RGB编码器模块
RGB Pixel-Level Block - 借鉴UIEC²-Net设计
用于在RGB空间进行像素级特征提取

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGB_Encoder(nn.Module):
    """
    RGB像素级编码器
    
    采用7层CNN结构，借鉴UIEC²-Net的RGB Pixel-Level Block设计：
    - 前4层使用LeakyReLU（下行阶段）
    - 后2层使用ReLU（上行阶段）
    - 最后1层通道对齐
    
    特点：
    - 保持原始分辨率，不做下采样
    - 使用InstanceNorm进行归一化（适合图像增强任务）
    - 专注于像素级细节增强
    
    参数:
        in_channels: 输入通道数，默认3（RGB）
        mid_channels: 中间层通道数，默认64
        out_channels: 输出通道数，默认36（与CIDNet的ch1对齐）
    """
    
    def __init__(self, in_channels=3, mid_channels=64, out_channels=36):
        super(RGB_Encoder, self).__init__()
        
        self.out_channels = out_channels
        
        # ==========  编码层1-4：使用LeakyReLU  ==========
        # 层1：输入层，3通道 -> 64通道
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(mid_channels)
        
        # 层2：特征提取
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(mid_channels)
        
        # 层3：特征提取
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.norm3 = nn.InstanceNorm2d(mid_channels)
        
        # 层4：特征提取
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.norm4 = nn.InstanceNorm2d(mid_channels)
        
        # ==========  编码层5-6：使用ReLU  ==========
        # 层5：特征精炼
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.norm5 = nn.InstanceNorm2d(mid_channels)
        
        # 层6：特征精炼
        self.conv6 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.norm6 = nn.InstanceNorm2d(mid_channels)
        
        # ==========  输出层  ==========
        # 层7：通道对齐，使用1x1卷积将64通道映射到out_channels
        self.conv7 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        
        # ==========  激活函数  ==========
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入RGB图像 [B, 3, H, W]
            
        返回:
            特征图 [B, out_channels, H, W]
        """
        # 编码阶段（LeakyReLU）
        h = self.leaky_relu(self.norm1(self.conv1(x)))  # 层1
        h = self.leaky_relu(self.norm2(self.conv2(h)))  # 层2
        h = self.leaky_relu(self.norm3(self.conv3(h)))  # 层3
        h = self.leaky_relu(self.norm4(self.conv4(h)))  # 层4
        
        # 精炼阶段（ReLU）
        h = self.relu(self.norm5(self.conv5(h)))        # 层5
        h = self.relu(self.norm6(self.conv6(h)))        # 层6
        
        # 输出层
        out = self.conv7(h)                              # 层7
        
        return out


class RGB_EncoderWithOutput(nn.Module):
    """
    带直接RGB输出的RGB编码器
    
    除了提供特征图外，还可以直接输出增强后的RGB图像
    用于损失函数计算和可视化
    
    参数:
        in_channels: 输入通道数，默认3
        mid_channels: 中间层通道数，默认64
        out_channels: 特征输出通道数，默认36
    """
    
    def __init__(self, in_channels=3, mid_channels=64, out_channels=36):
        super(RGB_EncoderWithOutput, self).__init__()
        
        # 主编码器
        self.encoder = RGB_Encoder(in_channels, mid_channels, out_channels)
        
        # RGB输出头：将特征图转换为RGB输出
        self.rgb_head = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 3, kernel_size=1, 
                      stride=1, padding=0, bias=False),
            nn.Sigmoid()  # 输出范围[0, 1]
        )
        
    def forward(self, x, return_rgb=False):
        """
        前向传播
        
        参数:
            x: 输入RGB图像 [B, 3, H, W]
            return_rgb: 是否返回RGB输出
            
        返回:
            如果return_rgb=False: 只返回特征图 [B, out_channels, H, W]
            如果return_rgb=True: 返回(特征图, RGB输出)
        """
        feat = self.encoder(x)
        
        if return_rgb:
            rgb_out = self.rgb_head(feat)
            return feat, rgb_out
        else:
            return feat


class RGB_MultiScaleEncoder(nn.Module):
    """
    多尺度RGB编码器（建设性建议1的实现）
    
    在多个尺度上提取RGB特征，便于与HVI分支进行多尺度交互
    
    尺度设计：
    - Scale 0: 原始分辨率 [B, ch1, H, W]
    - Scale 1: 1/2分辨率 [B, ch2, H/2, W/2]
    - Scale 2: 1/4分辨率 [B, ch3, H/4, W/4]
    - Scale 3: 1/8分辨率 [B, ch4, H/8, W/8]
    
    参数:
        in_channels: 输入通道数，默认3
        channels: 各尺度通道数列表，默认[36, 36, 72, 144]
    """
    
    def __init__(self, in_channels=3, channels=[36, 36, 72, 144]):
        super(RGB_MultiScaleEncoder, self).__init__()
        
        [ch1, ch2, ch3, ch4] = channels
        mid_channels = 64
        
        # ==========  初始特征提取  ==========
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # ==========  Scale 0: 原始分辨率  ==========
        self.scale0 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, ch1, kernel_size=1, 
                      stride=1, padding=0, bias=False),
        )
        
        # ==========  Scale 1: 1/2分辨率  ==========
        self.down1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=0.5),  # 下采样
        )
        self.scale1 = nn.Sequential(
            nn.Conv2d(mid_channels, ch2, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch2),
            nn.ReLU(inplace=True),
        )
        
        # ==========  Scale 2: 1/4分辨率  ==========
        self.down2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=0.5),  # 下采样
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(mid_channels, ch3, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch3),
            nn.ReLU(inplace=True),
        )
        
        # ==========  Scale 3: 1/8分辨率  ==========
        self.down3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=0.5),  # 下采样
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(mid_channels, ch4, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch4),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入RGB图像 [B, 3, H, W]
            
        返回:
            多尺度特征列表 [feat0, feat1, feat2, feat3]
            - feat0: [B, ch1, H, W]
            - feat1: [B, ch2, H/2, W/2]
            - feat2: [B, ch3, H/4, W/4]
            - feat3: [B, ch4, H/8, W/8]
        """
        # 初始特征提取
        h = self.stem(x)
        
        # Scale 0
        feat0 = self.scale0(h)
        
        # Scale 1
        h1 = self.down1(h)
        feat1 = self.scale1(h1)
        
        # Scale 2
        h2 = self.down2(h1)
        feat2 = self.scale2(h2)
        
        # Scale 3
        h3 = self.down3(h2)
        feat3 = self.scale3(h3)
        
        return [feat0, feat1, feat2, feat3]


# ==========  测试代码  ==========
if __name__ == '__main__':
    print("=" * 60)
    print("RGB编码器模块单元测试")
    print("=" * 60)
    
    # 测试输入
    batch_size = 2
    height, width = 256, 256
    x = torch.randn(batch_size, 3, height, width)
    
    print(f"\n输入形状: {x.shape}")
    print("-" * 40)
    
    # 测试基础RGB编码器
    print("\n[测试1] RGB_Encoder (基础编码器)")
    encoder = RGB_Encoder(in_channels=3, mid_channels=64, out_channels=36)
    out = encoder(x)
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    assert out.shape == (batch_size, 36, height, width), "输出形状错误！"
    print("  [OK] 测试通过")
    
    # 测试带RGB输出的编码器
    print("\n[测试2] RGB_EncoderWithOutput (带RGB输出)")
    encoder_with_out = RGB_EncoderWithOutput(in_channels=3, mid_channels=64, out_channels=36)
    feat, rgb_out = encoder_with_out(x, return_rgb=True)
    print(f"  特征形状: {feat.shape}")
    print(f"  RGB输出形状: {rgb_out.shape}")
    print(f"  RGB输出范围: [{rgb_out.min().item():.4f}, {rgb_out.max().item():.4f}]")
    print(f"  参数量: {sum(p.numel() for p in encoder_with_out.parameters()):,}")
    assert feat.shape == (batch_size, 36, height, width), "特征形状错误！"
    assert rgb_out.shape == (batch_size, 3, height, width), "RGB输出形状错误！"
    print("  [OK] 测试通过")
    
    # 测试多尺度编码器
    print("\n[测试3] RGB_MultiScaleEncoder (多尺度编码器)")
    multi_encoder = RGB_MultiScaleEncoder(in_channels=3, channels=[36, 36, 72, 144])
    feats = multi_encoder(x)
    print(f"  Scale 0 形状: {feats[0].shape}")
    print(f"  Scale 1 形状: {feats[1].shape}")
    print(f"  Scale 2 形状: {feats[2].shape}")
    print(f"  Scale 3 形状: {feats[3].shape}")
    print(f"  参数量: {sum(p.numel() for p in multi_encoder.parameters()):,}")
    assert feats[0].shape == (batch_size, 36, height, width), "Scale 0 形状错误！"
    assert feats[1].shape == (batch_size, 36, height//2, width//2), "Scale 1 形状错误！"
    assert feats[2].shape == (batch_size, 72, height//4, width//4), "Scale 2 形状错误！"
    assert feats[3].shape == (batch_size, 144, height//8, width//8), "Scale 3 形状错误！"
    print("  [OK] 测试通过")
    
    # GPU测试（如果可用）
    if torch.cuda.is_available():
        print("\n[测试4] GPU兼容性测试")
        x_cuda = x.cuda()
        encoder_cuda = RGB_Encoder().cuda()
        out_cuda = encoder_cuda(x_cuda)
        print(f"  GPU输出形状: {out_cuda.shape}")
        print(f"  GPU设备: {out_cuda.device}")
        print("  [OK] GPU测试通过")
    else:
        print("\n[测试4] GPU不可用，跳过GPU测试")
    
    print("\n" + "=" * 60)
    print("所有测试通过！RGB编码器模块可正常使用。")
    print("=" * 60)
