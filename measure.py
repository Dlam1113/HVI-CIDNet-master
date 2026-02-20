import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import platform
from loss.niqe_utils import calculate_niqe  # NIQE无参考指标
try:
    import imquality.brisque as brisque_mod  # BRISQUE无参考指标
    HAS_BRISQUE = True
except ImportError:
    HAS_BRISQUE = False
    print("警告: imquality库未安装，BRISQUE指标将跳过。安装: pip install image-quality")

mea_parser = argparse.ArgumentParser(description='Measure')
mea_parser.add_argument('--use_GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
mea_parser.add_argument('--lol', action='store_true', help='measure lolv1 dataset')
mea_parser.add_argument('--lol_v2_real', action='store_true', help='measure lol_v2_real dataset')
mea_parser.add_argument('--lol_v2_syn', action='store_true', help='measure lol_v2_syn dataset')
mea_parser.add_argument('--SICE_grad', action='store_true', help='measure SICE_grad dataset')
mea_parser.add_argument('--SICE_mix', action='store_true', help='measure SICE_mix dataset')
mea_parser.add_argument('--fivek', action='store_true', help='measure fivek dataset')
mea = mea_parser.parse_args()

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:#维度为2也就是灰度图像
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))
    return psnr

def metrics(im_dir, label_dir, use_GT_mean):
    """
    计算图像质量评估指标
    
    全参考指标（需要GT）：PSNR、SSIM、LPIPS
    无参考指标（不需GT）：NIQE、BRISQUE
    
    参数:
        im_dir: 模型输出图像目录
        label_dir: 真值图像目录
        use_GT_mean: 是否使用GT均值校正亮度
    返回:
        avg_psnr, avg_ssim, avg_lpips, avg_niqe, avg_brisque
    """
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    avg_niqe = 0
    avg_brisque = 0
    n = 0
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    
    # 支持多种图片格式匹配
    if '*' in im_dir:
        im_files = sorted(glob.glob(im_dir))
    else:
        im_files = sorted(
            glob.glob(im_dir + '*.png') + 
            glob.glob(im_dir + '*.jpg') + 
            glob.glob(im_dir + '*.JPG') +
            glob.glob(im_dir + '*.jpeg')
        )
    
    for item in tqdm(im_files):
        n += 1
        
        im1 = Image.open(item).convert('RGB') 

        # 跨平台路径处理（Windows用反斜杠，Linux用斜杠）
        os_name = platform.system()
        if os_name.lower() == 'windows':
            name = item.split('\\')[-1]
        elif os_name.lower() == 'linux':
            name = item.split('/')[-1]
        else:
            name = item.split('/')[-1]
        
        # 尝试多种扩展名匹配GT文件
        gt_path = label_dir + name
        if not os.path.exists(gt_path):
            name_without_ext = os.path.splitext(name)[0]
            for ext in ['.JPG', '.jpg', '.png', '.PNG', '.jpeg', '.JPEG']:
                alt_path = label_dir + name_without_ext + ext
                if os.path.exists(alt_path):
                    gt_path = alt_path
                    break
        
        im2 = Image.open(gt_path).convert('RGB')
        (h, w) = im2.size
        im1 = im1.resize((h, w))  
        im1_np = np.array(im1)  # [H, W, 3], uint8
        im2_np = np.array(im2)
        
        if use_GT_mean:
            mean_restored = cv2.cvtColor(im1_np, cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(im2_np, cv2.COLOR_RGB2GRAY).mean()
            im1_np = np.clip(im1_np * (mean_target/mean_restored), 0, 255)
        
        # ========== 全参考指标 ==========
        score_psnr = calculate_psnr(im1_np, im2_np)
        score_ssim = calculate_ssim(im1_np, im2_np)
        ex_p0 = lpips.im2tensor(im1_np).cuda()
        ex_ref = lpips.im2tensor(im2_np).cuda()
        score_lpips = loss_fn.forward(ex_ref, ex_p0)
        
        # ========== 无参考指标 ==========
        # NIQE：自然度评估（越低越好）
        score_niqe = calculate_niqe(im1_np, input_order='HWC', convert_to='y')
        
        # BRISQUE：空间质量评估（越低越好）
        if HAS_BRISQUE:
            try:
                score_brisque = brisque_mod.score(im1)
            except Exception:
                score_brisque = 0  # 某些图像尺寸可能触发库内部bug，跳过
        else:
            score_brisque = 0
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips.item()
        avg_niqe += score_niqe
        avg_brisque += score_brisque
        torch.cuda.empty_cache()
    
    # 防止除以零错误
    if n == 0:
        print(f"警告: 没有找到匹配的图片文件！")
        print(f"  im_dir: {im_dir}")
        print(f"  label_dir: {label_dir}")
        print(f"  请检查路径是否正确，以及文件扩展名是否匹配")
        return 0, 0, 0, 0, 0

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_lpips = avg_lpips / n
    avg_niqe = avg_niqe / n
    avg_brisque = avg_brisque / n
    return avg_psnr, avg_ssim, avg_lpips, avg_niqe, avg_brisque


if __name__ == '__main__':
    
    if mea.lol:
        im_dir = './output/LOLv1/*.png'
        label_dir = './datasets/LOLdataset/eval15/high/'
    if mea.lol_v2_real:
        im_dir = './output/LOLv2_real/*.png'
        label_dir = './datasets/LOLv2/Real_captured/Test/Normal/'
    if mea.lol_v2_syn:
        im_dir = './output/LOLv2_syn/*.png'
        label_dir = './datasets/LOLv2/Synthetic/Test/Normal/'
    if mea.SICE_grad:
        im_dir = './output/SICE_grad/*.png'
        label_dir = './datasets/SICE/SICE_Reshape/'
    if mea.SICE_mix:
        im_dir = './output/SICE_mix/*.png'
        label_dir = './datasets/SICE/SICE_Reshape/'
    if mea.fivek:
        im_dir = './output/fivek/*.jpg'
        label_dir = './datasets/FiveK/test/target/'

    avg_psnr, avg_ssim, avg_lpips, avg_niqe, avg_brisque = metrics(im_dir, label_dir, mea.use_GT_mean)
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
    print("===> Avg.NIQE: {:.4f} ".format(avg_niqe))
    print("===> Avg.BRISQUE: {:.4f} ".format(avg_brisque))
