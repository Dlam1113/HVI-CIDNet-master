"""
双空间CIDNet主网络
DualSpaceCIDNet - HVI + RGB 双空间特征级融合网络

核心创新：
1. 保留原CIDNet的HVI双流(HV+I)结构
2. 新增RGB处理分支
3. 在多尺度特征层进行跨空间交叉注意力
4. 使用可学习权重进行输出融合
"""

import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import NormDownsample, NormUpsample
from net.LCA import HV_LCA, I_LCA
from net.RGB_Encoder import RGB_Encoder, RGB_MultiScaleEncoder
from net.CrossSpaceAttention import CrossSpaceAttention, BidirectionalCrossAttention
from net.LearnableFusion import LearnableFusion, AdaptiveFusion
from net.NeuralCurve import NeuralCurveLayer
from huggingface_hub import PyTorchModelHubMixin


class DualSpaceCIDNet(nn.Module, PyTorchModelHubMixin):
    """
    双空间CIDNet：HVI + RGB 特征级交叉注意力融合
    
    架构设计：
    - HVI分支：保留原CIDNet的双流结构（HV流 + I流）
    - RGB分支：新增7层CNN编码器
    - 跨空间交互：在解码阶段进行RGB-HVI交叉注意力
    - 输出融合：使用可学习权重融合两个分支的输出
    
    参数:
        channels: 各层通道数列表，默认[36, 36, 72, 144]
        heads: 各层注意力头数列表，默认[1, 2, 4, 8]
        norm: 是否使用LayerNorm，默认False
        fusion_type: 融合方式，可选'learnable'或'adaptive'
        cross_space_attn: 是否使用跨空间注意力，默认True
        use_curve: 是否使用神经曲线层对I通道进行全局调整，默认False（消融实验用）
        curve_M: 曲线控制点数量，默认11
    """
    
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 fusion_type='learnable',
                 cross_space_attn=True,
                 use_curve=False,
                 curve_M=11
        ):
        super(DualSpaceCIDNet, self).__init__()
        
        self.cross_space_attn = cross_space_attn
        self.use_curve = use_curve
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # ========== HVI空间转换 ==========
        self.trans = RGB_HVI()
        
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
        
        # ========== RGB分支（新增） ==========
        # RGB_Encoder已直接输出3通道，无需额外的rgb_head
        self.rgb_encoder = RGB_Encoder(in_channels=3, mid_channels=64, out_channels=3)
        
        # ========== RGB-HVI跨空间交叉注意力（核心创新） ==========
        if self.cross_space_attn:
            # RGB特征与HVI特征的双向交叉注意力
            self.cross_attn = BidirectionalCrossAttention(ch1, num_heads=head1 if head1 > 0 else 1)
        
        # ========== 可学习权重融合（新增） ==========
        if fusion_type == 'learnable':
            self.fusion = LearnableFusion(in_channels=3, mid_channels=32, use_input=True)
        elif fusion_type == 'adaptive':
            self.fusion = AdaptiveFusion(in_channels=3, mid_channels=32)
        else:
            raise ValueError("Invalid fusion_type. Must be 'learnable' or 'adaptive'.")
        
        # ========== 神经曲线层（消融实验） ==========
        if self.use_curve:
            # 对 I 通道进行全局曲线调整
            self.i_curve = NeuralCurveLayer(in_channels=ch1, M=curve_M, num_curves=1)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入RGB图像 [B, 3, H, W]
            
        返回:
            dict: {
                'output': 最终输出,
                'rgb_out': RGB分支输出,
                'hvi_out': HVI分支输出,
                'fusion_weights': 融合权重（如果使用LearnableFusion）
        }
        """
        dtypes = x.dtype
        
        # ========== Step 1: HVI空间转换 ==========
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        # ========== Step 2: RGB分支编码 ==========
        rgb_feat = self.rgb_encoder(x)  # [B, ch1, H, W]
        
        # ========== Step 3: HVI分支编码 ==========
        # I流编码
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        
        # HV流编码
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        
        # 保存跳跃连接
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        # ========== Step 4: 第一次HV-I交叉注意力 ==========
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
        
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        # 第三次HV-I交叉注意力（瓶颈层）
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        # 第四次HV-I交叉注意力
        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        # ========== Step 5: HVI分支解码 ==========
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
        
        # ========== Step 5.1: 神经曲线层调整 I 通道（消融实验） ==========
        if self.use_curve:
            # 从 I 解码特征中预测曲线并应用到 I 通道
            # i_dec0 是 I 通道输出 [B, 1, H, W]，需要先归一化到 [0, 1]
            i_normalized = torch.clamp(i_dec0, 0, 1)
            i_dec0 = self.i_curve(i_dec1, i_normalized)
        
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)  # HVI分支正常解码，不使用注意力增强
        
        # ========== Step 6: RGB-HVI跨空间交叉注意力（仅增强RGB分支） ==========
        if self.cross_space_attn:
            hvi_feat = hv_1  # 使用HV特征作为上下文信息
            
            # 下采样RGB特征以匹配HVI特征尺寸
            rgb_feat_down = nn.functional.interpolate(
                rgb_feat, size=hvi_feat.shape[2:], mode='bilinear', align_corners=False
            )
            
            # 单向交叉注意力：只让RGB从HVI学习，HVI不变
            rgb_feat_enhanced, _ = self.cross_attn(rgb_feat_down, hvi_feat)

            # 上采样回原始尺寸
            rgb_feat_enhanced = nn.functional.interpolate(
                rgb_feat_enhanced, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            rgb_feat_enhanced = rgb_feat
        
        # ========== Step 7: 生成两个分支的RGB输出 ==========
        # HVI分支输出
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        hvi_rgb = self.trans.PHVIT(output_hvi)
        
        # RGB分支输出（RGB_Encoder已输出3通道，使用Sigmoid确保输出范围[0,1]）
        rgb_rgb = torch.sigmoid(rgb_feat_enhanced)
        
        # ========== Step 8: 可学习权重融合 ==========
        output, w_rgb, w_hvi = self.fusion(rgb_rgb, hvi_rgb, x)
        fusion_weights = {'rgb': w_rgb, 'hvi': w_hvi}
        
        return {
            'output': output,
            'rgb_out': rgb_rgb,
            'hvi_out': hvi_rgb,
            'fusion_weights': fusion_weights
        }
    
    def HVIT(self, x):
        """获取HVI变换结果（兼容原CIDNet接口）"""
        hvi = self.trans.HVIT(x)
        return hvi


# ==========  测试代码  ==========
if __name__ == '__main__':
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
    
    try:
        from net.CIDNet import CIDNet
        original_model = CIDNet(channels=[36, 36, 72, 144], heads=[1, 2, 4, 8])
        original_params = sum(p.numel() for p in original_model.parameters())
        print(f"  原CIDNet:        {original_params:>10,}")
    except:
        original_params = 0
        print("  原CIDNet:        (无法加载)")
    
    dual_params = sum(p.numel() for p in model.parameters())
    print(f"  DualSpaceCIDNet: {dual_params:>10,}")
    
    if original_params > 0:
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
