"""
双空间CIDNet主网络（v3 - RGB后处理版本）
DualSpaceCIDNet - CIDNet(HVI空间处理) → RGB后处理微调

核心设计：
1. CIDNet在HVI空间完成主要增强（完全保留原始能力）
2. 轻量RGB Refiner在RGB空间做残差微调（补偿HVI空间遗漏的细节）
3. 输出与原CIDNet兼容（直接返回tensor）

可选消融实验：
- use_curve: 在I通道解码后添加神经曲线层进行全局调整
- use_rgb_refiner: 是否启用RGB后处理（消融实验用）
"""

import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import NormDownsample, NormUpsample
from net.LCA import HV_LCA, I_LCA
from net.NeuralCurve import NeuralCurveLayer
from huggingface_hub import PyTorchModelHubMixin


class RGBRefiner(nn.Module):
    """
    轻量RGB后处理模块
    
    在CIDNet输出之后，在RGB空间做残差微调。
    设计原则：
    - 极轻量（3层CNN，32通道），避免在小数据集上过拟合
    - 学习残差（输出+输入=最终结果），容易收敛
    - 保持原始分辨率，不做下采样
    
    参数:
        in_channels: 输入通道数，默认3
        mid_channels: 中间层通道数，默认32（比原RGB_Encoder的64更小）
    """
    
    def __init__(self, in_channels=3, mid_channels=32):
        super(RGBRefiner, self).__init__()
        
        # 3层轻量CNN：提取→精炼→输出
        self.refine = nn.Sequential(
            # 层1：特征提取 3→32
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # 层2：特征精炼 32→32
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # 层3：输出映射 32→3（学习残差校正量）
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
    
    def forward(self, x):
        """
        前向传播：学习RGB空间的残差校正
        
        参数:
            x: CIDNet的RGB输出 [B, 3, H, W]
        返回:
            校正后的RGB图像 [B, 3, H, W]
        """
        correction = self.refine(x)            # 学习残差校正量
        out = torch.clamp(x + correction, 0, 1)  # 残差连接 + 截断
        return out


class DualSpaceCIDNet(nn.Module, PyTorchModelHubMixin):
    """
    CIDNet + RGB后处理（v3版本）

    架构：输入 → CIDNet(HVI空间处理) → CIDNet输出 → RGB Refiner残差微调 → 最终输出
    
    相比原CIDNet：
    - CIDNet部分完全不变（保留已验证的HVI空间处理能力）
    - 在RGB空间追加轻量后处理，补偿HVI→RGB转换中可能丢失的细节
    - 返回值与原CIDNet兼容（直接返回tensor）
    
    参数:
        channels: 各层通道数列表，默认[36, 36, 72, 144]
        heads: 各层注意力头数列表，默认[1, 2, 4, 8]
        norm: 是否使用LayerNorm，默认False
        use_rgb_refiner: 是否启用RGB后处理，默认True（设为False可做消融实验）
        refiner_mid_ch: RGB Refiner中间层通道数，默认32
        use_curve: 是否使用神经曲线层对I通道进行全局调整，默认False
        curve_M: 曲线控制点数量，默认11
    """
    
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 use_rgb_refiner=True,
                 refiner_mid_ch=32,
                 use_curve=False,
                 curve_M=11
        ):
        super(DualSpaceCIDNet, self).__init__()
        
        self.use_rgb_refiner = use_rgb_refiner
        self.use_curve = use_curve
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # ========== HVI空间转换 ==========
        self.trans = RGB_HVI()
        
        # ========== HV流编码器（原CIDNet结构，完全不变） ==========
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        # ========== HV流解码器 ==========
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )
        
        # ========== I流编码器（原CIDNet结构，完全不变） ==========
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        # ========== I流解码器 ==========
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )
        
        # ========== HV-I 内部交叉注意力（原LCA模块，完全不变） ==========
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        # ========== 神经曲线层（I分支消融实验） ==========
        if self.use_curve:
            self.i_curve = NeuralCurveLayer(in_channels=ch1, M=curve_M, num_curves=1)
        
        # ========== RGB后处理模块（新增，核心改进） ==========
        if self.use_rgb_refiner:
            self.rgb_refiner = RGBRefiner(in_channels=3, mid_channels=refiner_mid_ch)
    
    def forward(self, x):
        """
        前向传播：CIDNet(HVI空间) → RGB后处理微调
        
        参数:
            x: 输入RGB图像 [B, 3, H, W]，范围[0, 1]
            
        返回:
            output_rgb: 最终增强RGB图像 [B, 3, H, W]（与原CIDNet兼容）
        """
        dtypes = x.dtype
        
        # ========== 第一阶段：CIDNet HVI空间处理（与原CIDNet完全一致） ==========
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        # I流编码
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        
        # HV流编码
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        
        # 保存跳跃连接
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        # 第一次HV-I交叉注意力
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        
        # 继续编码
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        # 第二次HV-I交叉注意力
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        
        # 继续编码（沿用原CIDNet设计）
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        # 第三次HV-I交叉注意力（瓶颈层）
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        # 第四次HV-I交叉注意力
        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        # HVI解码
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        
        # 神经曲线层调整 I 通道（消融实验可选）
        if self.use_curve:
            i_normalized = torch.clamp(i_dec0, 0, 1)
            i_dec0 = self.i_curve(i_dec1, i_normalized)
        
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        # HVI → RGB（CIDNet原始输出）
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        cidnet_rgb = self.trans.PHVIT(output_hvi)
        
        # ========== 第二阶段：RGB后处理微调（新增） ==========
        if self.use_rgb_refiner:
            # 在RGB空间做残差校正，补偿HVI→RGB转换遗漏的细节
            output_rgb = self.rgb_refiner(cidnet_rgb)
        else:
            output_rgb = cidnet_rgb
        
        return output_rgb  # 与原CIDNet返回格式完全兼容
    
    def HVIT(self, x):
        """获取HVI变换结果（兼容原CIDNet接口）"""
        hvi = self.trans.HVIT(x)
        return hvi


# ==========  测试代码  ==========
if __name__ == '__main__':
    print("=" * 60)
    print("DualSpaceCIDNet（v3 RGB后处理版本）单元测试")
    print("=" * 60)
    
    batch_size = 2
    height, width = 256, 256
    x = torch.clamp(torch.randn(batch_size, 3, height, width), 0, 1)
    
    print(f"\n测试配置: batch={batch_size}, size={height}x{width}")
    print("-" * 40)
    
    # 测试1: 带RGB后处理
    print("\n[测试1] CIDNet + RGB Refiner")
    model = DualSpaceCIDNet(use_rgb_refiner=True, refiner_mid_ch=32)
    out = model(x)
    print(f"  输出形状: {out.shape}, 类型: {type(out).__name__}")
    print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
    total_params = sum(p.numel() for p in model.parameters())
    refiner_params = sum(p.numel() for p in model.rgb_refiner.parameters())
    cidnet_params = total_params - refiner_params
    print(f"  CIDNet参数: {cidnet_params:,}")
    print(f"  Refiner参数: {refiner_params:,} (仅增加 {refiner_params/cidnet_params*100:.1f}%)")
    print(f"  总参数量: {total_params:,}")
    assert out.shape == x.shape and isinstance(out, torch.Tensor)
    print("  [OK] 通过")
    
    # 测试2: 不带RGB后处理（消融）
    print("\n[测试2] CIDNet only（消融对照）")
    model_no_refiner = DualSpaceCIDNet(use_rgb_refiner=False)
    out2 = model_no_refiner(x)
    print(f"  输出形状: {out2.shape}")
    print(f"  参数量: {sum(p.numel() for p in model_no_refiner.parameters()):,}")
    assert out2.shape == x.shape
    print("  [OK] 通过")
    
    # 测试3: 带曲线层+RGB后处理
    print("\n[测试3] CIDNet + Curve + RGB Refiner")
    model_full = DualSpaceCIDNet(use_rgb_refiner=True, use_curve=True)
    out3 = model_full(x)
    print(f"  输出形状: {out3.shape}")
    print(f"  参数量: {sum(p.numel() for p in model_full.parameters()):,}")
    assert out3.shape == x.shape
    print("  [OK] 通过")
    
    # 测试4: HVIT兼容性
    print("\n[测试4] HVIT接口")
    hvi = model.HVIT(out)
    assert hvi.shape == (batch_size, 3, height, width)
    print("  [OK] 通过")
    
    # 参数对比
    print("\n" + "-" * 40)
    try:
        from net.CIDNet import CIDNet
        orig = CIDNet()
        orig_p = sum(p.numel() for p in orig.parameters())
        print(f"  原CIDNet:         {orig_p:>10,}")
        print(f"  +RGB Refiner:     {total_params:>10,}  (+{(total_params-orig_p)/orig_p*100:.1f}%)")
    except:
        pass
    
    if torch.cuda.is_available():
        print("\n[测试5] GPU")
        m = DualSpaceCIDNet().cuda()
        o = m(x.cuda())
        print(f"  设备: {o.device}, 形状: {o.shape}")
        print("  [OK] 通过")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
