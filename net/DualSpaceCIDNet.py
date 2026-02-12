"""
双空间CIDNet主网络（串联版本 v2）
DualSpaceCIDNet - RGB初步增强 → HVI深度精化 串联架构

核心设计（借鉴UIEC²-Net的串联思路）：
1. RGB Pixel-Level Block 先做初步增强（去噪、色偏校正）
2. 将增强后的RGB转到HVI空间
3. CIDNet双流结构在HVI空间进行深度精化
4. 输出与原CIDNet兼容（直接返回tensor）

可选消融实验：
- use_curve: 在I通道解码后添加神经曲线层进行全局调整
"""

import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import NormDownsample, NormUpsample
from net.LCA import HV_LCA, I_LCA
from net.RGB_Encoder import RGB_Encoder
from net.NeuralCurve import NeuralCurveLayer
from huggingface_hub import PyTorchModelHubMixin

# ========== 以下模块在并联方案(方案A)中使用，当前串联方案暂不使用 ==========
# from net.RGB_Encoder import RGB_MultiScaleEncoder
# from net.CrossSpaceAttention import CrossSpaceAttention, BidirectionalCrossAttention
# from net.LearnableFusion import LearnableFusion, AdaptiveFusion


class DualSpaceCIDNet(nn.Module, PyTorchModelHubMixin):
    """
    串联双空间CIDNet（v2版本）

    架构：输入 → RGB Block(+残差) → 增强RGB → 转HVI → CIDNet双流处理 → 输出RGB
    
    相比原CIDNet：
    - 在HVI处理之前，先用RGB Pixel-Level Block做初步增强
    - RGB Block使用残差连接，学习增量变化
    - 其余结构与原CIDNet完全一致
    - 返回值与原CIDNet兼容（直接返回tensor）
    
    参数:
        channels: 各层通道数列表，默认[36, 36, 72, 144]
        heads: 各层注意力头数列表，默认[1, 2, 4, 8]
        norm: 是否使用LayerNorm，默认False
        use_curve: 是否使用神经曲线层对I通道进行全局调整，默认False（消融实验用）
        curve_M: 曲线控制点数量，默认11
    """
    
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 use_curve=False,
                 curve_M=11
                 # ========== 以下参数在并联方案(方案A)中使用，当前串联方案暂不使用 ==========
                 # fusion_type='learnable',
                 # cross_space_attn=True,
        ):
        super(DualSpaceCIDNet, self).__init__()
        
        self.use_curve = use_curve
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # ========== HVI空间转换 ==========
        self.trans = RGB_HVI()
        
        # ========== RGB Pixel-Level Block（串联第一阶段） ==========
        # 借鉴UIEC²-Net的RGB像素级处理块
        # 输出3通道，通过残差连接实现初步增强
        self.rgb_encoder = RGB_Encoder(in_channels=3, mid_channels=64, out_channels=3)
        
        # ========== HV流编码器（原CIDNet结构） ==========
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
        
        # ========== I流编码器（原CIDNet结构） ==========
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
        
        # ========== HV-I 内部交叉注意力（原LCA模块） ==========
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
            # 对 I 通道进行全局曲线调整
            self.i_curve = NeuralCurveLayer(in_channels=ch1, M=curve_M, num_curves=1)
        
        # ========== 以下模块在并联方案(方案A)中使用，当前串联方案暂不使用 ==========
        # if cross_space_attn:
        #     self.cross_attn = BidirectionalCrossAttention(ch1, num_heads=head1 if head1 > 0 else 1)
        # if fusion_type == 'learnable':
        #     self.fusion = LearnableFusion(in_channels=3, mid_channels=32, use_input=True)
        # elif fusion_type == 'adaptive':
        #     self.fusion = AdaptiveFusion(in_channels=3, mid_channels=32)
    
    def forward(self, x):
        """
        串联前向传播
        
        流程：输入 → RGB Block(+残差) → 增强RGB → 转HVI → CIDNet双流 → 输出RGB
        
        参数:
            x: 输入RGB图像 [B, 3, H, W]，范围[0, 1]
            
        返回:
            output_rgb: 最终增强RGB图像 [B, 3, H, W]（与原CIDNet兼容）
        """
        dtypes = x.dtype
        
        # ========== 第一阶段：RGB Pixel-Level Block 初步增强 ==========
        # RGB Block输出 + 输入残差连接，学习增量变化
        rgb_residual = self.rgb_encoder(x)               # [B, 3, H, W]
        rgb_enhanced = torch.clamp(rgb_residual + x, 0, 1)  # 残差连接 + 截断到[0,1]
        
        # ========== 第二阶段：转到HVI空间 ==========
        # 对增强后的RGB（而非原始输入）进行HVI转换
        hvi = self.trans.HVIT(rgb_enhanced)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        # ========== 第三阶段：CIDNet HVI双流处理（与原CIDNet完全一致） ==========
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
        
        # ========== HVI分支解码 ==========
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
        
        # ========== 神经曲线层调整 I 通道（消融实验可选） ==========
        if self.use_curve:
            i_normalized = torch.clamp(i_dec0, 0, 1)
            i_dec0 = self.i_curve(i_dec1, i_normalized)
        
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        # ========== 第四阶段：HVI → RGB 输出 ==========
        # 残差连接：解码输出 + HVI输入（与原CIDNet一致）
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)
        
        return output_rgb  # 与原CIDNet返回格式完全兼容
    
    def HVIT(self, x):
        """获取HVI变换结果（兼容原CIDNet接口）"""
        hvi = self.trans.HVIT(x)
        return hvi


# ==========  测试代码  ==========
if __name__ == '__main__':
    print("=" * 60)
    print("DualSpaceCIDNet（串联版本v2）单元测试")
    print("=" * 60)
    
    # 测试配置
    batch_size = 2
    height, width = 256, 256
    
    # 模拟输入
    x = torch.randn(batch_size, 3, height, width)
    x = torch.clamp(x, 0, 1)  # 确保输入在[0,1]范围
    
    print(f"\n测试配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {height}x{width}")
    print("-" * 40)
    
    # 测试1: 基础串联前向传播
    print("\n[测试1] DualSpaceCIDNet 串联前向传播")
    model = DualSpaceCIDNet(
        channels=[36, 36, 72, 144],
        heads=[1, 2, 4, 8],
    )
    out = model(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  输出类型: {type(out)}")  # 应该是tensor，不是dict
    print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    assert out.shape == x.shape, "输出形状错误！"
    assert isinstance(out, torch.Tensor), "输出类型应为tensor！"
    print("  [OK] 测试通过")
    
    # 测试2: 带神经曲线层
    print("\n[测试2] DualSpaceCIDNet + 神经曲线层")
    model_curve = DualSpaceCIDNet(
        channels=[36, 36, 72, 144],
        heads=[1, 2, 4, 8],
        use_curve=True,
        curve_M=11,
    )
    out_curve = model_curve(x)
    print(f"  输出形状: {out_curve.shape}")
    print(f"  总参数量: {sum(p.numel() for p in model_curve.parameters()):,}")
    assert out_curve.shape == x.shape, "输出形状错误！"
    print("  [OK] 测试通过")
    
    # 测试3: HVIT兼容性
    print("\n[测试3] HVIT接口兼容性")
    hvi = model.HVIT(out)
    print(f"  HVI输出形状: {hvi.shape}")
    assert hvi.shape == (batch_size, 3, height, width), "HVIT输出形状错误！"
    print("  [OK] 测试通过")
    
    # 参数量对比
    print("\n" + "-" * 40)
    print("参数量对比:")
    
    try:
        from net.CIDNet import CIDNet
        original_model = CIDNet(channels=[36, 36, 72, 144], heads=[1, 2, 4, 8])
        original_params = sum(p.numel() for p in original_model.parameters())
        print(f"  原CIDNet:           {original_params:>10,}")
    except:
        original_params = 0
        print("  原CIDNet:           (无法加载)")
    
    dual_params = sum(p.numel() for p in model.parameters())
    dual_curve_params = sum(p.numel() for p in model_curve.parameters())
    print(f"  串联CIDNet:         {dual_params:>10,}")
    print(f"  串联CIDNet+曲线层:  {dual_curve_params:>10,}")
    
    if original_params > 0:
        increase = (dual_params - original_params) / original_params * 100
        print(f"  参数增加:           {increase:>9.1f}%")
    
    # GPU测试
    if torch.cuda.is_available():
        print("\n[测试4] GPU兼容性测试")
        x_cuda = x.cuda()
        model_cuda = DualSpaceCIDNet().cuda()
        out_cuda = model_cuda(x_cuda)
        print(f"  GPU设备: {out_cuda.device}")
        print(f"  输出形状: {out_cuda.shape}")
        assert isinstance(out_cuda, torch.Tensor), "GPU输出类型应为tensor！"
        print("  [OK] GPU测试通过")
    else:
        print("\n[测试4] GPU不可用，跳过GPU测试")
    
    print("\n" + "=" * 60)
    print("所有测试通过！串联DualSpaceCIDNet可正常使用。")
    print("=" * 60)
