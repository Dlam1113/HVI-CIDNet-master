"""
Feature Map 热力图可视化脚本
===========================
功能：加载训练好的 DualSpaceCIDNet 权重，对一张输入图片进行前向推理，
     抓取各个中间层的特征图张量，转换为热力图 (jet colormap) 保存为 PNG。

使用方法：
    python visualize_feature_maps.py \
        --input_image  path/to/low_light.jpg \
        --weights      path/to/model_epoch_xxx.pth \
        --output_dir   ./results/feature_maps/ \
        --device       cuda

输出：在 output_dir 下生成一系列 PNG 热力图，文件名对应网络中的位置。
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 无显示器环境也能用
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DualSpaceCIDNet Feature Map Visualizer')
    parser.add_argument('--input_image', type=str, required=True,
                        help='输入图片路径（低光照/雾天/雨天图片）')
    parser.add_argument('--weights', type=str, required=True,
                        help='训练好的模型权重文件 (.pth)')
    parser.add_argument('--output_dir', type=str, default='./results/feature_maps/',
                        help='热力图输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备 (cuda / cpu)')
    parser.add_argument('--colormap', type=str, default='jet',
                        help='matplotlib colormap 名称 (jet / inferno / viridis / hot)')
    parser.add_argument('--resize', type=int, default=512,
                        help='输入图片 resize 尺寸（0 表示保持原始大小）')
    parser.add_argument('--top_k', type=int, default=4,
                        help='每层最多保存 top_k 个最高激活的通道热力图')
    parser.add_argument('--save_mean', action='store_true', default=True,
                        help='是否保存所有通道平均的热力图（默认开启）')
    return parser.parse_args()


def load_image(image_path, resize=512):
    """
    加载并预处理图片
    
    参数:
        image_path: 图片文件路径
        resize: 缩放尺寸，0 表示保持原始大小
    返回:
        input_tensor: [1, 3, H, W] 的归一化张量
        original_image: PIL Image 原始图片（用于对比展示）
    """
    img = Image.open(image_path).convert('RGB')
    original_image = img.copy()
    
    transform_list = []
    if resize > 0:
        transform_list.append(transforms.Resize((resize, resize)))
    transform_list.append(transforms.ToTensor())
    
    transform = transforms.Compose(transform_list)
    input_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return input_tensor, original_image


def tensor_to_heatmap(feature_tensor, colormap_name='jet'):
    """
    将单通道特征图张量转换为 RGB 热力图 (numpy array)
    
    参数:
        feature_tensor: [H, W] 的张量（单通道）
        colormap_name: matplotlib colormap 名称
    返回:
        heatmap_rgb: [H, W, 3] 的 uint8 numpy 数组
    """
    feat = feature_tensor.detach().cpu().numpy()
    # 归一化到 [0, 1]
    feat_min, feat_max = feat.min(), feat.max()
    if feat_max - feat_min > 1e-6:
        feat = (feat - feat_min) / (feat_max - feat_min)
    else:
        feat = np.zeros_like(feat)
    
    # 应用 colormap
    colormap = cm.get_cmap(colormap_name)
    heatmap = colormap(feat)[:, :, :3]  # 去掉 alpha 通道
    heatmap_rgb = (heatmap * 255).astype(np.uint8)
    return heatmap_rgb


def save_feature_maps(feature_dict, output_dir, colormap_name='jet', top_k=4, save_mean=True):
    """
    将抓取到的所有特征图保存为热力图 PNG
    
    参数:
        feature_dict: {层名称: 特征张量 [B, C, H, W]}
        output_dir: 输出目录
        colormap_name: colormap 名称
        top_k: 每层保存激活最强的 top_k 个通道
        save_mean: 是否保存通道平均热力图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name, feat_tensor in feature_dict.items():
        # 取 batch 中第一张图
        feat = feat_tensor[0]  # [C, H, W]
        C, H, W = feat.shape
        
        # 为每层创建子目录
        layer_dir = os.path.join(output_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        
        # ===== 1. 保存通道平均热力图（最常用） =====
        if save_mean:
            mean_feat = feat.mean(dim=0)  # [H, W]
            heatmap = tensor_to_heatmap(mean_feat, colormap_name)
            save_path = os.path.join(layer_dir, f'{layer_name}_mean.png')
            Image.fromarray(heatmap).save(save_path)
            print(f"  [保存] {save_path}  ({H}x{W}, {C}ch, mean)")
        
        # ===== 2. 保存激活最强的 top_k 个通道 =====
        # 按每个通道的平均激活值排序
        channel_activations = feat.mean(dim=(1, 2))  # [C]
        _, top_indices = torch.topk(channel_activations, min(top_k, C))
        
        for rank, ch_idx in enumerate(top_indices):
            ch_feat = feat[ch_idx]  # [H, W]
            heatmap = tensor_to_heatmap(ch_feat, colormap_name)
            save_path = os.path.join(layer_dir, f'{layer_name}_ch{ch_idx.item():03d}_top{rank+1}.png')
            Image.fromarray(heatmap).save(save_path)
        
        print(f"  [保存] {layer_name}: 共 {min(top_k, C)+1} 张热力图 (分辨率 {H}x{W}, 总通道数 {C})")


def visualize_with_hooks(model, input_tensor, device='cuda'):
    """
    使用 PyTorch register_forward_hook 抓取中间层特征图
    
    参数:
        model: DualSpaceCIDNet 模型实例
        input_tensor: [1, 3, H, W] 输入张量
        device: 推理设备
    返回:
        feature_dict: {层名称: 特征张量}
        output: 模型最终输出
    """
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    
    feature_dict = {}
    hooks = []
    
    def make_hook(name):
        """创建一个 hook 函数，将对应层的输出存入 feature_dict"""
        def hook_fn(module, input, output):
            # 有些模块返回 tuple，取第一个元素
            if isinstance(output, tuple):
                output = output[0]
            feature_dict[name] = output.detach().cpu()
        return hook_fn
    
    # ========== 注册 Hook：覆盖网络所有关键节点 ==========
    
    # --- HV 流编码器 ---
    hooks.append(model.HVE_block0.register_forward_hook(make_hook('01_HV_Enc0')))
    hooks.append(model.HVE_block1.register_forward_hook(make_hook('02_HV_Enc1_down')))
    hooks.append(model.HVE_block2.register_forward_hook(make_hook('04_HV_Enc2_down')))
    hooks.append(model.HVE_block3.register_forward_hook(make_hook('06_HV_Enc3_down')))
    
    # --- I 流编码器 ---
    hooks.append(model.IE_block0.register_forward_hook(make_hook('01_I_Enc0')))
    hooks.append(model.IE_block1.register_forward_hook(make_hook('02_I_Enc1_down')))
    hooks.append(model.IE_block2.register_forward_hook(make_hook('04_I_Enc2_down')))
    hooks.append(model.IE_block3.register_forward_hook(make_hook('06_I_Enc3_down')))
    
    # --- LCA 交叉注意力 (编码阶段) ---
    hooks.append(model.HV_LCA1.register_forward_hook(make_hook('03_HV_LCA1')))
    hooks.append(model.I_LCA1.register_forward_hook(make_hook('03_I_LCA1')))
    hooks.append(model.HV_LCA2.register_forward_hook(make_hook('05_HV_LCA2')))
    hooks.append(model.I_LCA2.register_forward_hook(make_hook('05_I_LCA2')))
    
    # --- LCA 交叉注意力 (瓶颈层) ---
    hooks.append(model.HV_LCA3.register_forward_hook(make_hook('07_HV_LCA3_bottle')))
    hooks.append(model.I_LCA3.register_forward_hook(make_hook('07_I_LCA3_bottle')))
    hooks.append(model.HV_LCA4.register_forward_hook(make_hook('08_HV_LCA4_bottle')))
    hooks.append(model.I_LCA4.register_forward_hook(make_hook('08_I_LCA4_bottle')))
    
    # --- HV 流解码器 ---
    hooks.append(model.HVD_block3.register_forward_hook(make_hook('09_HV_Dec3_up')))
    hooks.append(model.HVD_block2.register_forward_hook(make_hook('11_HV_Dec2_up')))
    hooks.append(model.HVD_block1.register_forward_hook(make_hook('13_HV_Dec1_up')))
    hooks.append(model.HVD_block0.register_forward_hook(make_hook('14_HV_Dec0_out')))
    
    # --- I 流解码器 ---
    hooks.append(model.ID_block3.register_forward_hook(make_hook('09_I_Dec3_up')))
    hooks.append(model.ID_block2.register_forward_hook(make_hook('11_I_Dec2_up')))
    hooks.append(model.ID_block1.register_forward_hook(make_hook('13_I_Dec1_up')))
    hooks.append(model.ID_block0.register_forward_hook(make_hook('14_I_Dec0_out')))
    
    # --- LCA 交叉注意力 (解码阶段) ---
    hooks.append(model.HV_LCA5.register_forward_hook(make_hook('10_HV_LCA5_dec')))
    hooks.append(model.I_LCA5.register_forward_hook(make_hook('10_I_LCA5_dec')))
    hooks.append(model.HV_LCA6.register_forward_hook(make_hook('12_HV_LCA6_dec')))
    hooks.append(model.I_LCA6.register_forward_hook(make_hook('12_I_LCA6_dec')))
    
    # --- 神经曲线层（如果启用） ---
    if hasattr(model, 'i_curve') and model.use_curve:
        hooks.append(model.i_curve.register_forward_hook(make_hook('15_NeuralCurve')))
    
    # --- RGB Refiner（如果启用） ---
    if hasattr(model, 'rgb_refiner') and model.use_rgb_refiner:
        hooks.append(model.rgb_refiner.conv1.register_forward_hook(make_hook('16_Refiner_conv1')))
        hooks.append(model.rgb_refiner.conv2.register_forward_hook(make_hook('17_Refiner_conv2')))
        hooks.append(model.rgb_refiner.conv3.register_forward_hook(make_hook('18_Refiner_conv3_out')))
    
    # ========== 前向推理 ==========
    with torch.no_grad():
        output = model(input_tensor)
    
    # 清理所有 hooks
    for h in hooks:
        h.remove()
    
    return feature_dict, output


def save_comparison_grid(input_tensor, output_tensor, feature_dict, output_dir, colormap_name='jet'):
    """
    生成一张 Overview 对比大图：输入 → 关键层热力图 → 输出
    
    参数:
        input_tensor: [1, 3, H, W] 输入
        output_tensor: [1, 3, H, W] 输出
        feature_dict: 所有层的特征字典
        output_dir: 输出目录
        colormap_name: colormap 名称
    """
    # 选择几个关键层做 overview
    key_layers = [
        '01_I_Enc0', '03_I_LCA1', '07_I_LCA3_bottle',
        '14_I_Dec0_out', '14_HV_Dec0_out',
    ]
    
    # 检查是否有 Refiner 和 Curve
    if '15_NeuralCurve' in feature_dict:
        key_layers.append('15_NeuralCurve')
    if '16_Refiner_conv1' in feature_dict:
        key_layers.append('16_Refiner_conv1')
    
    n_cols = len(key_layers) + 2  # +2 是输入和输出
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))
    
    # 输入图片
    inp = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    inp = np.clip(inp, 0, 1)
    axes[0].imshow(inp)
    axes[0].set_title('Input', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # 关键层热力图
    for idx, layer_name in enumerate(key_layers):
        if layer_name in feature_dict:
            feat = feature_dict[layer_name][0]  # [C, H, W]
            mean_feat = feat.mean(dim=0)  # [H, W]
            heatmap = tensor_to_heatmap(mean_feat, colormap_name)
            axes[idx + 1].imshow(heatmap)
            # 缩短显示名称
            short_name = layer_name.split('_', 1)[1] if '_' in layer_name else layer_name
            axes[idx + 1].set_title(short_name, fontsize=8)
        else:
            axes[idx + 1].set_title(f'{layer_name}\n(N/A)', fontsize=8)
        axes[idx + 1].axis('off')
    
    # 输出图片
    out = output_tensor[0].permute(1, 2, 0).cpu().numpy()
    out = np.clip(out, 0, 1)
    axes[-1].imshow(out)
    axes[-1].set_title('Output', fontsize=10, fontweight='bold')
    axes[-1].axis('off')
    
    plt.suptitle('DualSpaceCIDNet Feature Map Overview', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'overview_grid.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"\n[Overview] 已保存对比概览图: {save_path}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("DualSpaceCIDNet Feature Map 热力图导出工具")
    print("=" * 60)
    
    # ===== 1. 加载模型 =====
    print(f"\n[1/4] 加载模型权重: {args.weights}")
    
    # 导入模型类
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from net.DualSpaceCIDNet import DualSpaceCIDNet
    
    # 先加载权重文件查看配置
    checkpoint = torch.load(args.weights, map_location='cpu')
    
    # 判断权重文件格式（直接 state_dict 还是包了一层）
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 自动检测是否包含曲线层和 RGB Refiner
    has_curve = any('i_curve' in k for k in state_dict.keys())
    has_refiner = any('rgb_refiner' in k for k in state_dict.keys())
    
    print(f"  检测到 Neural Curve Layer: {'✅ 是' if has_curve else '❌ 否'}")
    print(f"  检测到 RGB Refiner:        {'✅ 是' if has_refiner else '❌ 否'}")
    
    # 创建模型实例
    model = DualSpaceCIDNet(
        use_rgb_refiner=has_refiner,
        use_curve=has_curve,
    )
    
    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    print(f"  模型加载成功! 总参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== 2. 加载输入图片 =====
    print(f"\n[2/4] 加载输入图片: {args.input_image}")
    input_tensor, original_image = load_image(args.input_image, resize=args.resize)
    print(f"  图片尺寸: {input_tensor.shape}")
    
    # ===== 3. 前向推理 + Hook 抓取特征图 =====
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n[3/4] 前向推理 (设备: {device})")
    print(f"  注册 Hook 中...")
    
    feature_dict, output = visualize_with_hooks(model, input_tensor, device=device)
    print(f"  已抓取 {len(feature_dict)} 个层的特征图:")
    for name, feat in sorted(feature_dict.items()):
        print(f"    {name}: shape={list(feat.shape)}")
    
    # ===== 4. 保存热力图 =====
    print(f"\n[4/4] 保存热力图到: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存输入和输出图片
    input_save = (input_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(input_save).save(os.path.join(args.output_dir, '00_input.png'))
    
    output_save = (output[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(output_save).save(os.path.join(args.output_dir, '99_output.png'))
    
    # 保存所有层的热力图
    save_feature_maps(
        feature_dict, args.output_dir,
        colormap_name=args.colormap,
        top_k=args.top_k,
        save_mean=args.save_mean
    )
    
    # 生成 overview 对比大图
    save_comparison_grid(input_tensor, output, feature_dict, args.output_dir, args.colormap)
    
    print(f"\n{'=' * 60}")
    print(f"✅ 完成！所有热力图已保存到: {args.output_dir}")
    print(f"  - 各层子目录: 通道平均热力图 + Top-{args.top_k} 通道热力图")
    print(f"  - overview_grid.png: 关键层对比概览图")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
