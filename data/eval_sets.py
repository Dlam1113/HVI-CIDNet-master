
import os
import torch.utils.data as data
from os import listdir
from os.path import join
from data.util import *
import torch.nn.functional as F

class SICEDatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SICEDatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
            factor = 8  # 要求尺寸是8的倍数
            h, w = input.shape[1], input.shape[2]  # 获取当前图片的高度和宽度
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor  # "向上取整到因子8的倍数"
            padh = H - h if h % factor != 0 else 0  # 计算需要填充的高度
            padw = W - w if w % factor != 0 else 0  # 计算需要填充的宽度
            input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect').squeeze(0)
            #(0,padw,0,padh)是表示（左右上下）填充的像素值'reflect'是镜像反射边缘的像素。最后squeeze是把第0维度给去掉
            
        return input, file, h, w

    def __len__(self):
        return len(self.data_filenames)
    
    
class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
        return input, file

    def __len__(self):
        return len(self.data_filenames)
