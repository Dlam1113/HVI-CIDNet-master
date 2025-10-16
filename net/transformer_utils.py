import torch
import torch.nn as nn
import torch.nn.functional as F

#层归一化：是每个通道的信息进行归一化处理 让神经网络的训练更加稳定和高效
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 可学习的缩放参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # 可学习的偏移参数
        self.eps = eps  # 防止除零的小数值
        self.data_format = data_format  # 数据格式："channels_first" 或 "channels_last"
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            """
            Layer Normalization的详细计算步骤
            输入x的形状：[batch_size, channels, height, width]
            """
            # 步骤1：计算均值（沿通道维度）
            u = x.mean(1, keepdim=True)  # 形状：1是指沿维度1channels进行求均值[batch_size, 1, height, width]，三通道的平均值
            # 解释：对每个样本的所有通道求平均值
            # 步骤2：计算方差（沿通道维度）
            s = (x - u).pow(2).mean(1, keepdim=True)  # 形状：[batch_size, 1, height, width]
            # 解释：计算每个位置上所有通道的方差
            # 步骤3：标准化（零均值，单位方差）
            x_normalized = (x - u) / torch.sqrt(s + self.eps)
            # 解释：减去均值，除以标准差，得到标准正态分布
            # 步骤4：缩放和偏移（恢复表达能力）
            x = self.weight[:, None, None] * x_normalized + self.bias[:, None, None]
            # 解释：通过可学习参数γ和β调整分布，保持网络的表达能力
            return x
def why_layer_normalization():
    """
    层归一化解决的核心问题
    """
    # 1. 内部协变量偏移（Internal Covariate Shift）
    #    - 网络层输入分布在训练过程中不断变化
    #    - 导致训练不稳定，收敛速度慢
    
    # 2. 梯度消失/爆炸问题
    #    - 深层网络中梯度传播困难
    #    - 层归一化有助于梯度稳定传播
    
    # 3. 加速收敛
    #    - 标准化后的数据更容易优化
    #    - 可以使用更大的学习率
def mathematical_formula():
    """
    Layer Normalization的数学公式
    """
    # 给定输入x，对于每个样本：
    # 
    # 1. 计算均值：μ = (1/C) * Σ(x_i)，其中C是通道数
    # 2. 计算方差：σ² = (1/C) * Σ(x_i - μ)²
    # 3. 归一化：x̂ = (x - μ) / √(σ² + ε)
    # 4. 缩放偏移：y = γ * x̂ + β
    #
    # 其中：
    # - γ（weight）：可学习的缩放参数
    # - β（bias）：可学习的偏移参数
    # - ε（eps）：数值稳定性参数
def layernorm_in_hvi_cidnet():
    """
    在HVI-CIDNet中LayerNorm的应用场景
    """
    # 1. LCA模块中的应用
    #    - 在交叉注意力计算前进行归一化
    #    - 稳定注意力权重的计算
    
    # 2. 上采样模块中的应用
    #    - 在特征融合后进行归一化
    #    - 保持特征分布的稳定性
    
    # 3. 主要优势
    #    - 不依赖批次大小，适合小批次训练
    #    - 提高模型在不同输入分辨率下的稳定性
    #    - 加速收敛，提高训练效率
#标准化下采样：逐步“缩小”特征图的空间分辨率（H×W），同时“增大”通道数（C）
class NormDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=0.5, use_norm=False):
        """
        归一化下采样模块
        
        参数:
            in_ch: 输入通道数
            out_ch: 输出通道数  
            scale: 缩放因子，默认0.5表示尺寸缩小一半
            use_norm: 是否使用层归一化
        """
        super(NormDownsample, self).__init__()
        self.use_norm = use_norm
        
        # 可选的层归一化
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
            
        # PReLU激活函数（比ReLU更灵活）
        self.prelu = nn.PReLU()
        
        # 下采样操作：卷积 + 双线性插值缩放
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale)  # scale=0.5，图像缩小为原来的一半，跟池化没啥两样
        )
        
    def forward(self, x):
        """前向传播过程"""
        x = self.down(x)      # 1. 卷积 + 插值缩放
        x = self.prelu(x)     # 2. 激活函数
        if self.use_norm:     # 3. 可选的归一化
            x = self.norm(x)
            return x
        else:
            return x
def why_downsample():
    """
    下采样的优势：
    1. 增大感受野 - 能够看到更大范围的图像内容
    2. 降低计算量 - 减少特征图尺寸，提高计算效率
    3. 抽象特征 - 低分辨率特征图包含更抽象的语义信息
    4. 多尺度处理 - 同时处理不同分辨率的特征
    """
    pass

def why_increase_channels():
    """
    通道数递增的设计思想：
    - 高分辨率阶段：少通道，保留空间细节
    - 低分辨率阶段：多通道，提取丰富的语义特征
    - 符合'分辨率与通道数反比'的经典CNN设计原则
    """
    # 36 → 36 → 72 → 144
    # 空间分辨率: 大 → 中 → 小
    # 特征丰富度: 低 → 中 → 高

#标准化上采样：逐步“放大”特征图的空间分辨率（H×W），同时“减小”通道数（C）
class NormUpsample(nn.Module):
    def __init__(self, in_ch,out_ch,scale=2,use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))#双线性插值放大跟最大池化一个东西
        self.up = nn.Conv2d(out_ch*2,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
    
    
            
    def forward(self, x, y):
        """
        参数说明：
        x: 来自上一层解码器的特征（低分辨率，高语义）
        y: 来自编码器的跳跃连接特征（高分辨率，细节丰富）
        """
        x = self.up_scale(x)        # 1. 上采样x到与y相同尺寸
        x = torch.cat([x, y], dim=1) # 2. 特征融合：沿通道维度拼接
        x = self.up(x)              # 3. 通道压缩：减少通道数
        x = self.prelu(x)           # 4. 激活函数
    
        if self.use_norm:
            return self.norm(x)
        else:
            return x
# def feature_concatenation_example():
#     """
#     特征拼接的详细过程解释了torch.cat和self.up输入通道为什么是输出通道的两倍
#     """
#     # x: 上采样后的特征 [batch, 72, 64, 64]  (高语义信息)
#     # y: 编码器跳跃连接   [batch, 72, 64, 64]  (细节信息)
        
#     # torch.cat沿着通道维度(dim=1)拼接
#     concatenated = torch.cat([x, y], dim=1)
    # 结果: [batch, 144, 64, 64]  (通道数翻倍)
        
    # 为什么拼接？
    # - x包含语义信息，但缺乏细节
    # - y包含丰富的空间细节，但语义层次较低
    # - 拼接后同时具备语义理解和细节保持能力
def importance_in_low_light_enhancement():
    """
    在低光图像增强中的作用
    """
    # 1. 细节保持：
    #    - 跳跃连接保留了编码器中的细节信息
    #    - 避免上采样过程中的信息丢失
    #    - 对低光图像的纹理恢复至关重要
        
    # 2. 多尺度信息融合：
    #    - 编码器特征：包含边缘、纹理等局部信息
    #    - 解码器特征：包含光照、语义等全局信息
    #    - 融合后：既有全局理解，又保持局部细节
        
    # 3. 梯度流动：
    #    - 跳跃连接为梯度提供直接通路
    #    - 解决深层网络的梯度消失问题
    #    - 使训练更加稳定
    
 
