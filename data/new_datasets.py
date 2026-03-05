import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t


class LoLI_StreetDatasetFromFolder(data.Dataset):
    """
    LoLI_Street数据集类，继承自PyTorch的Dataset基类
    用于加载LoLI_Street数据集中的低光照和正常光照图像对
    """
    
    def __init__(self, data_dir, transform=None):
        """
        构造函数：初始化数据集对象
        
        参数:
            data_dir (str): 数据集根目录路径
            transform: 数据变换函数（可选）
        
        调用时机：创建对象时自动调用
        例如：dataset = LoLI_StreetDatasetFromFolder('./data', transform=None)
        """
        super(LoLI_StreetDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        """
        索引访问方法：根据索引获取数据样本
        
        参数:
            index (int): 数据样本的索引
        
        返回:
            tuple: (低光图像张量, 高光图像张量, 低光文件名, 高光文件名)
        """
        folder = self.data_dir + '/low'
        folder2 = self.data_dir + '/high'
        
        data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        data_filenames2 = sorted([join(folder2, x) for x in listdir(folder2) if is_image_file(x)])
        
        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        
        seed = random.randint(1, 1000000)
        
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
            
        return im1, im2, file1, file2

    def __len__(self):
        """返回数据集大小（动态计算，不再硬编码）"""
        folder = self.data_dir + '/low'
        return len([x for x in listdir(folder) if is_image_file(x)])


class CombinedPedestrianDataset(data.Dataset):
    """
    合并行人数据集类
    
    功能：
        将多个数据源（LoLI-Street低光照、Cityscapes雾天、Cityscapes雨天）
        合并为统一的训练数据集。
        
        每个数据源都有 high/ 和 low/ 两个子文件夹，按文件名排序后一一配对。
    
    使用示例：
        dataset = CombinedPedestrianDataset(
            data_dirs=['./filtered/loli_pedestrian',
                       './filtered/cityscapes_foggy_pedestrian',
                       './filtered/cityscapes_rain_pedestrian'],
            transform=transform1(256)
        )
    """
    
    def __init__(self, data_dirs, transform=None):
        """
        构造函数：初始化合并数据集
        
        参数:
            data_dirs (list): 数据目录列表，每个目录下有 high/ 和 low/ 子文件夹
            transform: 数据变换函数（随机裁剪、翻转等）
        """
        super(CombinedPedestrianDataset, self).__init__()
        self.transform = transform
        
        # 存储所有配对图片的路径: [(low_path, high_path), ...]
        self.pairs = []
        
        for data_dir in data_dirs:
            low_dir = os.path.join(data_dir, 'low')
            high_dir = os.path.join(data_dir, 'high')
            
            if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
                print(f"  警告: 跳过无效目录 {data_dir}（缺少 high/ 或 low/）")
                continue
            
            # 获取文件名列表并排序（确保 high 和 low 配对正确）
            low_files = sorted([f for f in listdir(low_dir) if is_image_file(f)])
            high_files = sorted([f for f in listdir(high_dir) if is_image_file(f)])
            
            # 检查数量是否匹配
            if len(low_files) != len(high_files):
                print(f"  警告: {data_dir} 中 high({len(high_files)}) 和 low({len(low_files)}) 数量不匹配！")
                # 取较少的那个数量
                min_count = min(len(low_files), len(high_files))
                low_files = low_files[:min_count]
                high_files = high_files[:min_count]
            
            # 将配对路径加入列表
            for lf, hf in zip(low_files, high_files):
                self.pairs.append((
                    os.path.join(low_dir, lf),
                    os.path.join(high_dir, hf)
                ))
            
            print(f"  ✅ 加载 {data_dir}: {len(low_files)} 对")
        
        print(f"  📊 合并数据集总计: {len(self.pairs)} 对")
    
    def __getitem__(self, index):
        """
        获取一对训练图像
        
        返回:
            tuple: (低质量图像, 高质量GT图像, 低质量文件名, 高质量文件名)
        """
        low_path, high_path = self.pairs[index]
        
        # 加载图像
        im_low = load_img(low_path)
        im_high = load_img(high_path)
        
        # 提取文件名
        _, file_low = os.path.split(low_path)
        _, file_high = os.path.split(high_path)
        
        # 设置随机种子，确保 low 和 high 应用完全相同的随机裁剪和翻转
        seed = random.randint(1, 1000000)
        
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im_low = self.transform(im_low)
            
            random.seed(seed)
            torch.manual_seed(seed)
            im_high = self.transform(im_high)
        
        return im_low, im_high, file_low, file_high
    
    def __len__(self):
        """返回合并后的数据集总大小"""
        return len(self.pairs)


class CombinedPedestrianEvalDataset(data.Dataset):
    """
    合并行人数据集的验证集类
    
    功能：
        加载验证集中的 low/ 图像用于推理，
        同时提供对应的 high/ GT图像路径用于计算指标。
        与 SICEDatasetFromFolderEval 返回格式兼容。
    """
    
    def __init__(self, data_dirs, transform=None):
        """
        参数:
            data_dirs (list): 验证数据目录列表
            transform: 数据变换（通常只有 ToTensor）
        """
        super(CombinedPedestrianEvalDataset, self).__init__()
        self.transform = transform
        self.low_files = []   # 低质量图像路径列表
        self.high_files = []  # 高质量GT图像路径列表
        
        for data_dir in data_dirs:
            low_dir = os.path.join(data_dir, 'low')
            high_dir = os.path.join(data_dir, 'high')
            
            if not os.path.isdir(low_dir):
                continue
            
            low_list = sorted([os.path.join(low_dir, f) for f in listdir(low_dir) if is_image_file(f)])
            high_list = sorted([os.path.join(high_dir, f) for f in listdir(high_dir) if is_image_file(f)])
            
            self.low_files.extend(low_list)
            self.high_files.extend(high_list)
        
        print(f"  📊 验证集总计: {len(self.low_files)} 张")
    
    def __getitem__(self, index):
        """
        获取验证图像
        
        返回格式与 SICEDatasetFromFolderEval 兼容:
            (input_tensor, filename, original_height, original_width)
        """
        import torch.nn.functional as F
        
        input_img = load_img(self.low_files[index])
        _, file = os.path.split(self.low_files[index])
        
        if self.transform:
            input_img = self.transform(input_img)
            # 填充到8的倍数（模型要求）
            factor = 8
            h, w = input_img.shape[1], input_img.shape[2]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = F.pad(input_img.unsqueeze(0), (0, padw, 0, padh), 'reflect').squeeze(0)
        
        return input_img, file, h, w
    
    def __len__(self):
        return len(self.low_files)
    
    def get_gt_path(self, index):
        """获取GT图像路径（用于计算PSNR/SSIM等指标）"""
        return self.high_files[index]
