from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *

def transform1(size=256):
    """训练时使用的数据变换，包含数据增强操作"""
    return Compose([
        RandomCrop((size, size)),      # 随机裁剪到256x256
        RandomHorizontalFlip(),        # 50%概率水平翻转
        RandomVerticalFlip(),          # 50%概率垂直翻转
        ToTensor(),                    # 转换为张量，值域[0,1]
    ])

def transform2():
    """评估时使用的数据变换，不包含随机操作"""
    return Compose([ToTensor()])    # 仅转换为张量



def get_lol_training_set(data_dir,size):
    """加载LOL训练数据集"""
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_training_set(data_dir,size):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir,size):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir,size):
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


def get_SID_training_set(data_dir,size):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir,size):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))

# 通用评估数据集
def get_SICE_eval_set(data_dir):
    """加载SICE评估数据集,用于模型性能测试"""
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    """加载通用评估数据集"""
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_fivek_training_set(data_dir,size):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size))

def get_fivek_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())
