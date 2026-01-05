import os                      # 操作系统接口，用于文件路径操作
import random                  # 生成随机数，用于数据增强的随机种子
import torch                   # PyTorch深度学习框架
import torch.utils.data as data  # PyTorch数据加载工具
import numpy as np             # 数值计算库
from os import listdir         # 列出目录中的文件
from os.path import join       # 拼接文件路径
from data.util import *        # 自定义工具函数（图像加载等）
from torchvision import transforms as t  # 图像变换工具

    
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
        self.data_dir = data_dir    # 存储数据集路径
        self.transform = transform  # 存储数据变换函数
        # ImageNet标准化参数，用于预训练模型
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        """
        索引访问方法：根据索引获取数据样本
        
        参数:
            index (int): 数据样本的索引
        
        调用时机：
        - 使用 dataset[index] 时自动调用
        - DataLoader迭代时自动调用
        - 例如：low_img, high_img, file1, file2 = dataset[0]
        
        返回:
            tuple: (低光图像张量, 高光图像张量, 低光文件名, 高光文件名)
        """
        # 构建低光照和高光照图像文件夹路径
        folder = self.data_dir + '/low'   # 低光照图像目录
        folder2 = self.data_dir + '/high' # 正常光照图像目录
        
        # 获取所有图像文件路径列表
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        
        # 加载指定索引的图像
        im1 = load_img(data_filenames[index])    # 加载低光照图像
        im2 = load_img(data_filenames2[index])   # 加载正常光照图像
        
        # 提取文件名
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        
        # 生成随机种子，确保两张图像应用相同的随机变换
        seed = random.randint(1, 1000000)
        
        # 如果有数据变换，应用到两张图像上
        if self.transform:
            random.seed(seed)           # 设置相同随机种子
            torch.manual_seed(seed)     # PyTorch随机种子
            im1 = self.transform(im1)   # 变换低光图像
            
            random.seed(seed)           # 重新设置相同种子
            torch.manual_seed(seed)
            im2 = self.transform(im2)   # 变换高光图像
            
        return im1, im2, file1, file2

    def __len__(self):
        """
        长度方法：返回数据集大小
        
        调用时机：
        - 使用 len(dataset) 时自动调用
        - DataLoader创建时需要知道数据集大小
        - 例如：dataset_size = len(dataset)
        
        返回:
            int: 数据集中样本的数量
        """
        return 30000  # LoLI_Street数据集固定有30000个训练样本
