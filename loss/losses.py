import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *


_reduction_modes = ['none', 'mean', 'sum']

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
        
        
        
class EdgeLoss(nn.Module):
    """
    边缘损失函数
    通过拉普拉斯金字塔提取边缘信息,然后计算MSE损失
    
    参数：
        loss_weight (float): 损失权重,默认1.0
        reduction (str): 归约方式,默认'mean'
    """
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        """
        高斯卷积
        
        参数：
            img (Tensor): 输入图像,shape: [N, C, H, W]
        
        返回：
            Tensor: 高斯卷积后的图像,shape: [N, C, H, W]
        """
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class PerceptualLoss(nn.Module):
    """
    感知损失（可选择性包含风格损失）
    
    核心思想：
    - 在VGG特征空间中比较图像相似性
    - 而非在像素空间中比较
    """
    
    def __init__(self,
                 layer_weights,         # 各层的权重字典
                 vgg_type='vgg19',      # VGG类型
                 use_input_norm=True,   # 是否归一化输入
                 range_norm=True,       # 是否范围归一化
                 perceptual_weight=1.0, # 感知损失权重
                 style_weight=0.,       # 风格损失权重
                 criterion='l1'):       # 损失函数类型
        super(PerceptualLoss, self).__init__()
        
        # ========== 保存参数 ==========
        self.perceptual_weight = perceptual_weight  # 默认1.0
        self.style_weight = style_weight            # 默认0（不使用）
        self.layer_weights = layer_weights          # {'conv1_2': 1, ...}
        
        # ========== 创建VGG特征提取器 ==========
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),  # ['conv1_2', 'conv2_2', ...]
            vgg_type=vgg_type,              # 'vgg19'
            use_input_norm=use_input_norm,  # True
            range_norm=range_norm)          # True
        
        # ========== 选择损失函数 ==========
        self.criterion_type = criterion  # 'mse'
        
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
            # MSE = mean((x - y)²)
        elif self.criterion_type == 'fro':
            self.criterion = None
            # Frobenius范数：||A||_F = sqrt(Σ(a_ij²))
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # 图像 → VGG网络 → 特征图 → 比较相似度
                #       ↓
                # 不同层捕获不同信息：
                # - 浅层：纹理、颜色
                # - 中层：形状、边缘
                # - 深层：语义内容
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss感知损失
        # ========== 步骤2：计算感知损失 ==========
        if self.perceptual_weight > 0:
            percep_loss = 0
            
            for k in x_features.keys():
                # 遍历每一层：'conv1_2', 'conv2_2', 'conv3_4', 'conv4_4'
                
                if self.criterion_type == 'fro':
                    # Frobenius范数
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k], p='fro'
                    ) * self.layer_weights[k]
                else:
                    # MSE损失（默认）
                    layer_loss = self.criterion(x_features[k], gt_features[k])
                    # layer_loss = mean((x_features[k] - gt_features[k])²)
                    
                    percep_loss += layer_loss * self.layer_weights[k]
                    # 乘以该层的权重（本例中都是1）
            
            # 乘以总权重
            percep_loss *= self.perceptual_weight
            # percep_loss = (loss_conv1_2 + loss_conv2_2 + loss_conv3_4 + loss_conv4_4) × 1.0
        else:
            percep_loss = None

        # calculate style loss风格损失没用到
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss




class SSIM(torch.nn.Module):
    """
    结构相似性损失函数
    
    参数说明：
        window_size (int): 高斯窗口大小,默认11x11
        size_average (bool): 是否对所有像素取平均,默认True
        weight (float): 损失权重,默认1.0
    """
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size#高斯窗口大小
        self.size_average = size_average#对所有像素取平均
        self.channel = 1#通道数
        self.window = create_window(window_size, self.channel)#创建窗口
        self.weight = weight#权重

    def forward(self, img1, img2):
        """
        计算SSIM损失
        
        参数：
            img1 (Tensor): 预测图像,shape: [N, C, H, W]
            img2 (Tensor): 真实图像,shape: [N, C, H, W]
        
        返回：
            loss (Tensor): SSIM损失值(标量)
        """
        # 步骤1：获取图像的通道数
        (_, channel, _, _) = img1.size()
        # 例如：img1.shape = [8, 3, 256, 256]
        # channel = 3（RGB三通道）
        
        # 步骤2：检查是否需要重新创建窗口
        if channel == self.channel and self.window.data.type() == img1.data.type():
            # 通道数和数据类型都匹配,直接使用已有窗口
            window = self.window
        else:
            # 通道数或数据类型不匹配,重新创建窗口
            window = create_window(self.window_size, channel)
            
            # 如果图像在GPU上,窗口也要移到GPU
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            # 确保窗口的数据类型与图像一致
            window = window.type_as(img1)
            
            # 更新存储的窗口和通道数
            self.window = window
            self.channel = channel
        
        # 步骤3：计算SSIM值并转换为损失
        ssim_value = map_ssim(img1, img2, window, self.window_size, channel, self.size_average)
        # ssim_value: 范围[0, 1],越大越好
        
        # 转换为损失：(1 - SSIM) × weight
        loss = (1. - ssim_value) * self.weight
        
        return loss


class DualSpaceLoss(nn.Module):
    """
    双空间损失函数
    
    用于DualSpaceCIDNet的训练，同时计算：
    1. 最终输出的损失
    2. RGB分支输出的损失（新增）
    3. HVI分支输出的损失（原有）
    
    总损失 = L_output + lambda_rgb * L_rgb + lambda_hvi * L_hvi
    
    参数:
        L1_weight: L1损失权重，默认1.0
        SSIM_weight: SSIM损失权重，默认0.5
        Edge_weight: 边缘损失权重，默认50.0
        Perceptual_weight: 感知损失权重，默认0.01
        RGB_weight: RGB分支损失权重，默认0.5
        HVI_weight: HVI分支损失权重，默认1.0
    """
    
    def __init__(self,
                 L1_weight=1.0,
                 SSIM_weight=0.5,
                 Edge_weight=50.0,
                 Perceptual_weight=0.01,
                 RGB_weight=0.5,
                 HVI_weight=1.0):
        super(DualSpaceLoss, self).__init__()
        
        # 保存权重
        self.L1_weight = L1_weight
        self.SSIM_weight = SSIM_weight
        self.Edge_weight = Edge_weight
        self.Perceptual_weight = Perceptual_weight
        self.RGB_weight = RGB_weight
        self.HVI_weight = HVI_weight
        
        # 各损失函数实例
        self.L1_loss = L1Loss(loss_weight=1.0)
        self.SSIM_loss = SSIM(weight=1.0)
        self.Edge_loss = EdgeLoss(loss_weight=1.0)
        
        # 感知损失（可选）
        if Perceptual_weight > 0:
            self.Perceptual_loss = PerceptualLoss(
                layer_weights={'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                vgg_type='vgg19',
                use_input_norm=True,
                range_norm=True,
                perceptual_weight=1.0,
                style_weight=0,
                criterion='mse'
            )
        else:
            self.Perceptual_loss = None
    
    def forward(self, output, target, rgb_out=None, hvi_out=None):
        """
        计算双空间损失
        
        参数:
            output: 最终融合输出 [B, 3, H, W]
            target: 真值图像 [B, 3, H, W]
            rgb_out: RGB分支输出 [B, 3, H, W]（可选）
            hvi_out: HVI分支输出 [B, 3, H, W]（可选）
        
        返回:
            total_loss: 总损失
            loss_dict: 各部分损失的字典
        """
        loss_dict = {}
        
        # ========== 1. 最终输出损失 ==========
        # L1损失
        loss_L1 = self.L1_loss(output, target) * self.L1_weight
        loss_dict['L1'] = loss_L1
        
        # SSIM损失
        loss_SSIM = self.SSIM_loss(output, target) * self.SSIM_weight
        loss_dict['SSIM'] = loss_SSIM
        
        # 边缘损失
        loss_Edge = self.Edge_loss(output, target) * self.Edge_weight
        loss_dict['Edge'] = loss_Edge
        
        # 感知损失
        if self.Perceptual_loss is not None and self.Perceptual_weight > 0:
            percep_loss, _ = self.Perceptual_loss(output, target)
            loss_Perceptual = percep_loss * self.Perceptual_weight
            loss_dict['Perceptual'] = loss_Perceptual
        else:
            loss_Perceptual = 0
        
        # 最终输出的总损失
        loss_output = loss_L1 + loss_SSIM + loss_Edge + loss_Perceptual
        loss_dict['output'] = loss_output
        
        # ========== 2. RGB分支损失（新增） ==========
        if rgb_out is not None and self.RGB_weight > 0:
            loss_rgb_L1 = self.L1_loss(rgb_out, target)
            loss_rgb_SSIM = self.SSIM_loss(rgb_out, target)
            loss_RGB = (loss_rgb_L1 + loss_rgb_SSIM * 0.5) * self.RGB_weight
            loss_dict['RGB'] = loss_RGB
        else:
            loss_RGB = 0
        
        # ========== 3. HVI分支损失 ==========
        if hvi_out is not None and self.HVI_weight > 0:
            loss_hvi_L1 = self.L1_loss(hvi_out, target)
            loss_hvi_SSIM = self.SSIM_loss(hvi_out, target)
            loss_hvi_Edge = self.Edge_loss(hvi_out, target)
            loss_HVI = (loss_hvi_L1 + loss_hvi_SSIM * 0.5 + loss_hvi_Edge * 50.0) * self.HVI_weight
            loss_dict['HVI'] = loss_HVI
        else:
            loss_HVI = 0
        
        # ========== 总损失 ==========
        total_loss = loss_output + loss_RGB + loss_HVI
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
