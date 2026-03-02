"""
FID（弗雷歇初始距离）指标计算脚本

功能：
    计算模型增强图像与真值图像之间的FID分数
    FID越低表示增强效果越好（分布越接近真实图像）

使用方式：
    1. 直接对已有输出目录计算FID：
       python measure_fid.py --output_dir ./results/LOLv1/ --gt_dir ./datasets/LOLdataset/eval15/high/

    2. 先用模型推理再计算FID：
       python measure_fid.py --model_path ./weights/train/epoch_850.pth --output_dir ./results/LOLv1/ --gt_dir ./datasets/LOLdataset/eval15/high/

    3. 指定模型类型（DualSpace或原CIDNet）：
       python measure_fid.py --output_dir ./results/LOLv1/ --gt_dir ./datasets/LOLdataset/eval15/high/ --dual_space --use_curve

依赖安装：
    pip install pytorch-fid
"""

import os
import sys
import argparse
import shutil
import tempfile
import glob

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='计算FID指标')
    
    # ====== 在这里直接填写路径，就不需要在命令行输入了 ======
    parser.add_argument('--output_dir', type=str, 
                        default='./results/LOLv1/',          # ← 增强图片保存到哪里
                        help='模型输出图像目录（增强后的图像）')
    parser.add_argument('--gt_dir', type=str, 
                        default='./datasets/LOLdataset/eval15/high/',  # ← 真值图片目录
                        help='真值图像目录（Ground Truth）')
    
    # 可选：指定模型进行推理（填了就会先跑模型生成增强图，不填就直接用output_dir里已有的图）
    parser.add_argument('--model_path', type=str, 
                        default='./weights/train/best.pth',  # ← 改成你挑出的最优pth路径
                        help='模型权重路径（如需要先推理再计算FID）')
    parser.add_argument('--input_dir', type=str, 
                        default='./datasets/LOLdataset/eval15/low/',   # ← 低光照输入图片目录
                        help='低光照输入图像目录（推理时使用）')
    
    # 模型配置（推理时使用）
    parser.add_argument('--dual_space', type=bool, default=True,
                        help='使用DualSpaceCIDNet（默认启用）')
    parser.add_argument('--use_rgb_refiner', type=bool, default=True,
                        help='是否启用RGB Refiner')
    parser.add_argument('--refiner_mid_ch', type=int, default=64,
                        help='Refiner中间通道数')
    parser.add_argument('--use_curve', type=bool, default=True,
                        help='是否启用神经曲线层（默认启用）')
    parser.add_argument('--curve_M', type=int, default=11,
                        help='曲线控制点数')
    
    # FID计算参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='FID计算的batch size')
    parser.add_argument('--dims', type=int, default=2048,
                        help='Inception特征维度（默认2048）')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')
    
    return parser.parse_args()


def run_inference(model_path, input_dir, output_dir, args):
    """
    使用指定模型对输入图像进行推理

    参数:
        model_path: 模型权重路径
        input_dir: 输入低光照图像目录
        output_dir: 输出增强图像目录
        args: 模型配置参数
    """
    import torch
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm
    
    # 构建模型
    if args.dual_space:
        from net.DualSpaceCIDNet import DualSpaceCIDNet
        model = DualSpaceCIDNet(
            channels=[36, 36, 72, 144],
            heads=[1, 2, 4, 8],
            use_rgb_refiner=args.use_rgb_refiner,
            refiner_mid_ch=args.refiner_mid_ch,
            use_curve=args.use_curve,
            curve_M=args.curve_M
        )
    else:
        from net.CIDNet import CIDNet
        model = CIDNet()
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(args.device)
    model.eval()
    print(f'====> 已加载模型: {model_path}')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集输入图像
    img_files = sorted(
        glob.glob(os.path.join(input_dir, '*.png')) +
        glob.glob(os.path.join(input_dir, '*.jpg')) +
        glob.glob(os.path.join(input_dir, '*.JPG'))
    )
    
    if len(img_files) == 0:
        print(f'错误: 在 {input_dir} 中没有找到图像文件')
        return
    
    print(f'====> 推理 {len(img_files)} 张图像...')
    
    # 逐张推理
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    with torch.no_grad():
        for img_path in tqdm(img_files):
            img = Image.open(img_path).convert('RGB')
            x = to_tensor(img).unsqueeze(0).to(args.device)
            
            output = model(x)
            output = torch.clamp(output, 0, 1)
            
            out_img = to_pil(output.squeeze(0).cpu())
            filename = os.path.basename(img_path)
            out_img.save(os.path.join(output_dir, filename))
    
    print(f'====> 推理完成，结果保存到 {output_dir}')


def calculate_fid(output_dir, gt_dir, batch_size=4, dims=2048, device='cuda:0'):
    """
    计算FID分数

    参数:
        output_dir: 模型输出图像目录
        gt_dir: 真值图像目录
        batch_size: 计算batch size
        dims: Inception特征维度
        device: 计算设备
    
    返回:
        fid_value: FID分数（越低越好）
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print('错误: 未安装pytorch-fid库')
        print('请运行: pip install pytorch-fid')
        sys.exit(1)
    
    # 检查目录是否存在
    if not os.path.isdir(output_dir):
        print(f'错误: 输出目录不存在 {output_dir}')
        sys.exit(1)
    if not os.path.isdir(gt_dir):
        print(f'错误: 真值目录不存在 {gt_dir}')
        sys.exit(1)
    
    # 统计图像数量
    output_imgs = glob.glob(os.path.join(output_dir, '*.png')) + \
                  glob.glob(os.path.join(output_dir, '*.jpg'))
    gt_imgs = glob.glob(os.path.join(gt_dir, '*.png')) + \
              glob.glob(os.path.join(gt_dir, '*.jpg'))
    
    print(f'====> 输出图像数量: {len(output_imgs)}')
    print(f'====> 真值图像数量: {len(gt_imgs)}')
    
    if len(output_imgs) == 0 or len(gt_imgs) == 0:
        print('错误: 某个目录下没有图像文件')
        sys.exit(1)
    
    # 计算FID
    print(f'====> 正在计算FID（dims={dims}）...')
    fid_value = fid_score.calculate_fid_given_paths(
        [output_dir, gt_dir],
        batch_size=batch_size,
        device=device,
        dims=dims
    )
    
    return fid_value


def batch_calculate_fid(model_dir, gt_dir, args):
    """
    批量计算多个checkpoint的FID（可选功能）

    参数:
        model_dir: 包含多个epoch_*.pth的目录
        gt_dir: 真值图像目录
        args: 配置参数
    """
    import re
    
    pth_files = sorted(glob.glob(os.path.join(model_dir, 'epoch_*.pth')))
    if not pth_files:
        print(f'错误: 在 {model_dir} 中没有找到checkpoint文件')
        return
    
    print(f'====> 找到 {len(pth_files)} 个checkpoint')
    
    results = []
    for pth in pth_files:
        # 提取epoch号
        match = re.search(r'epoch_(\d+)', pth)
        if not match:
            continue
        epoch = int(match.group(1))
        
        # 在临时目录中推理
        temp_dir = tempfile.mkdtemp()
        try:
            run_inference(pth, args.input_dir, temp_dir, args)
            fid = calculate_fid(temp_dir, gt_dir, args.batch_size, args.dims, args.device)
            results.append((epoch, fid))
            print(f'  Epoch {epoch}: FID = {fid:.4f}')
        finally:
            shutil.rmtree(temp_dir)
    
    # 打印汇总
    print('\n' + '=' * 40)
    print('FID 汇总')
    print('=' * 40)
    for epoch, fid in sorted(results):
        print(f'  Epoch {epoch:>5d}: FID = {fid:.4f}')
    
    best_epoch, best_fid = min(results, key=lambda x: x[1])
    print(f'\n  最优FID: {best_fid:.4f} (Epoch {best_epoch})')


if __name__ == '__main__':
    args = parse_args()
    
    # 如果指定了模型路径，先进行推理
    if args.model_path is not None:
        if args.input_dir is None:
            # 默认使用LOLv1验证集输入
            args.input_dir = './datasets/LOLdataset/eval15/low'
        run_inference(args.model_path, args.input_dir, args.output_dir, args)
    
    # 计算FID
    fid_value = calculate_fid(
        args.output_dir, 
        args.gt_dir,
        args.batch_size,
        args.dims,
        args.device
    )
    
    print(f'\n{"=" * 40}')
    print(f'  FID = {fid_value:.4f}')
    print(f'{"=" * 40}')
    print(f'  (越低越好，0表示完全一致)')
