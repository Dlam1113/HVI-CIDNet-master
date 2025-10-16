import torch
import torch.nn as nn
from einops import rearrange
from net.transformer_utils import *

# Cross Attention Block 交叉注意力块：让一个特征图（查询Q）去"关注"另一个特征图（键值K、V）中的重要信息
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # temperature值的影响：
        # - temperature > 1：注意力分布更平滑（关注更广泛）
        # - temperature = 1：标准的点积注意力
        # - temperature < 1：注意力分布更尖锐（关注更集中）

        # 1. 查询生成层
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 作用：将输入x映射为查询的初始表示
        # 特点：1x1卷积，通道数不变，用于特征变换
        # 2. 查询深度卷积层
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 作用：对查询每一个独立通道进行空间信息混合
        # 特点：深度卷积，每个通道独立处理空间信息

        # 3. 键值生成层
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        # 作用：从输入y同时生成键(K)和值(V)
        # 特点：输出通道翻倍，后续会分割为K和V
        # 4. 键值深度卷积层
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        # 作用：对键值进行空间信息混合
        # 特点：深度卷积，保持K和V的空间相关性
        # 5. 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 作用：将注意力输出映射回原始维度
        # 特点：1x1卷积，通道数恢复，最终特征整合

    def forward(self, x, y):
        # 输入参数：
        # x: [batch, dim, H, W] - 查询来源特征（如HV特征）
        # y: [batch, dim, H, W] - 键值来源特征（如I特征）
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        # reshape只是机械地切分内存，而rearrange会按照逻辑含义重新组织数据。
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #重排后c是每个头的特征维度也就是每个头有c个特征向量，(h w)是空间位置，每个特征向量有(h w)个空间位置
        
        # 模为1让注意力计算更关注特征的方向相似性而不是幅度差异，从而获得更稳定、更公平的注意力分布
        # 为什么要归一化？
        # 将特征向量投影到单位球面
        # 让注意力关注方向相似性而非幅度大小
        # 避免某些位置因数值大而占主导地位
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        # @就是矩阵点乘，对k的倒数两个维度进行转置然后跟q进行矩阵点乘得出注意力分数
        # 批量矩阵乘法
        # 这是特征维度之间的注意力分数，不是空间位置的注意力分数！！！！！！
        # 这里面用了广播机制
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 使得每一行求和为1
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
def why_depthwise_conv_enhancement():
    """
    为什么在1x1卷积后添加3x3深度卷积的原因
    """
    # HVI-CIDNet的设计：
    # 第1步：1x1卷积 - 跨通道信息混合
    # 第2步：3x3深度卷积 - 空间信息混合
    
    # 这种设计的优势：
    
    # 1. 空间感知能力
    #    - 1x1卷积：只看单个像素点，无空间感知
    #    - 3x3深度卷积：考虑3×3邻域，具有空间感知
    #    - 结合：既有跨通道混合，又有空间感知
    
    # 2. 局部特征增强
    #    - 图像中相邻像素通常相关
    #    - 3x3卷积捕获局部模式和边缘
    #    - 提高注意力计算的质量
    
    # 3. 适合低光照增强
    #    - 低光照图像噪声较多
    #    - 空间滤波有助于噪声抑制
    #    - 局部特征有助于细节恢复
    
def functional_differentiation_during_training():
    """
    训练过程中K和V功能分化的机制
    """
    # 训练初期：
    # - K通道和V通道的权重都是随机初始化的
    # - 它们生成的特征在语义上是相似的
    
    # 训练中期：
    # - 由于注意力公式的约束：Attention = softmax(Q·K^T)·V
    # - K必须学会与Q进行有效匹配（相似度计算）
    # - V必须学会承载有用的内容信息
    # - 梯度反传会强化这种功能分化
    
    # 训练后期：
    # - K通道专门学习"匹配特征"：
    #   * 边缘、纹理、语义标识等便于匹配的特征
    #   * 有助于计算注意力权重的特征
    # - V通道专门学习"内容特征"：
    #   * 颜色、亮度、细节等需要传递的信息
    #   * 有助于特征融合的特征
    
    # 这个过程是自动的，不需要人工干预！！！！！！！！！！！！
    # 网络会根据任务需求：attn = (q @ k.transpose(-2, -1)) * self.temperature和
    # attn = nn.functional.softmax(attn,dim=-1)和out = (attn @ v)自动优化K和V的功能。重点重点重点！！！！！！！！！！！！！！！
def why_normalize_this_way():
    """
    解释为什么要对空间维度进行归一化
    """
    # 在多头注意力中，q和k的点积计算是：
    # attention_score = q @ k.transpose(-2, -1)
    
    # 这里的运算是：
    # [batch, heads, dim_per_head, spatial] @ [batch, heads, spatial, dim_per_head]
    # = [batch, heads, spatial, spatial]
    
    # 点积的几何意义：
    # - 每个空间位置的特征向量之间计算相似度
    # - 如果特征向量的模长差异很大，会影响相似度计算的公平性
    
    # 归一化的作用：
    # - 将每个位置的特征向量投影到单位球面上
    # - 确保相似度计算主要反映方向相似性，而非幅度差异
    # - 提高注意力权重分布的稳定性
    
    example = """
    例如：
    位置A的特征: [0.1, 0.2, 0.1] (模长小)
    位置B的特征: [10, 20, 10]   (模长大)
    
    不归一化时：B位置会因为数值大而在注意力计算中占主导
    归一化后：A和B位置在相同的尺度上公平竞争
    """
    
    return example

# Intensity Enhancement Layer 强度增强层：专门用于增强图像信息
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        """
        初始化强度增强层
    
         参数说明:
        dim: 输入特征的通道数
        ffn_expansion_factor: 扩展因子（默认2.66）
        bias: 是否使用偏置项
        """
        super(IEL, self).__init__()
        # 计算隐藏层特征数：通道数扩展2.66倍
        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)# project_in: 将输入特征投影到扩展的特征空间
        #dwconv是Depthwise Convolution(深度可分离卷积)groups=channels时候是深度卷积
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)# project_out: 将处理后的特征投影回原始维度空间

        self.Tanh = nn.Tanh()#激活函数值在[-1,1]间
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)#按维度1（通道）二等分
        x1 = self.Tanh(self.dwconv1(x1)) + x1#最后是残差连接保持原有特征
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
def gating_mechanism_explanation():
    """
    门控机制的数学原理和作用
    """
    # 1. 类似LSTM的门控思想
    #    - x1充当"信息门"：决定哪些信息重要
    #    - x2充当"遗忘门"：决定哪些信息需要抑制
    #    - x1 * x2：选择性地保留和增强信息
    
    # 2. Tanh激活函数的作用
    #    - 输出范围[-1, 1]，提供双向调节能力
    #    - 正值：增强对应位置的特征
    #    - 负值：抑制对应位置的特征
    #    - 零值：保持原始特征
    
    # 3. 残差连接的作用
    #    - 保持梯度流动，避免梯度消失
    #    - 保留原始信息，防止信息丢失
    #    - 提高训练稳定性
    
    # 4. 元素级乘法的意义
    #    - 实现软性选择，比硬阈值更平滑
    #    - 允许不同空间位置有不同的增强程度
    #    - 适应性强，能够学习复杂的增强模式
def iel_for_low_light_enhancement():
    """
    IEL在低光照增强中的具体作用机制
    """
    # 1. 自适应亮度调整
    #    - 通过门控机制学习每个像素位置的增强程度
    #    - 暗部区域：更大的增强系数
    #    - 亮部区域：较小的增强系数，避免过曝
    
    # 2. 细节保持
    #    - 残差连接保留原始细节信息
    #    - 门控机制避免过度平滑
    #    - 深度卷积保持空间相关性
    
    # 3. 噪声抑制
    #    - Tanh激活函数的有界性有助于抑制噪声
    #    - 门控机制可以学习抑制噪声区域
    #    - 深度卷积的空间滤波效果
    
    # 4. 非线性映射
    #    - 复杂的非线性变换，适应不同的光照条件
    #    - 比简单的gamma校正更加灵活
    #    - 端到端学习，自动优化增强参数  

def why_element_wise_multiplication():
    """
    为什么使用元素级乘法(*)而不是加法(+)的深层原理
    """
    # 数学原理：门控机制
    # output = x1 * x2
    
    # 乘法的特殊性质：
    # 1. 选择性增强：
    #    - 如果x1[i,j] = 1, x2[i,j] = 0.5 → output[i,j] = 0.5 (保持x2的信息)
    #    - 如果x1[i,j] = 0.8, x2[i,j] = 1.2 → output[i,j] = 0.96 (适度增强)
    #    - 如果x1[i,j] = 0, x2[i,j] = 任意值 → output[i,j] = 0 (完全抑制)
    
    # 2. 互相调制：
    #    - x1控制x2的"开关程度"
    #    - x2控制x1的"调制强度"  
    #    - 两者相互制约，避免过度增强
    
    # 3. 非线性交互：
    #    - 乘法比加法提供更丰富的非线性交互
    #    - 能学习更复杂的特征组合模式
    #    - 提高模型的表达能力


# Lightweight Cross Attention
class HV_LCA(nn.Module):          #HV轻量级交叉注意力
    def __init__(self, dim,num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        self.gdfn = IEL(dim) # IEL and CDL have same structure        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = self.gdfn(self.norm(x))
        return x
def nothing(self):
        """ FFN - Feed Forward Network(前馈神经网络),其实就是交叉注意力模块，实现交叉注意力计算
        让一个特征流去"关注"另一个特征流的重要信息
        在LCA中负责信息交换和融合
        在Transformer中，FFN通常跟在注意力层后面
        虽然这里实现的是注意力机制，但在整体架构中起到FFN的作用
        
        GDFN - Gated Dense Feed-forward Network（门控密集前馈网络）其实就是强度增强层"""
# def lca_architecture_explained(self):
#         """
#         LCA中FFN和GDFN的作用分工
#         """
#         # 步骤1：交叉注意力（FFN实现）
#         x = x + self.ffn(self.norm(x), self.norm(y))
#         # 作用：让x关注y中的重要信息，实现特征交换
        
#         # 步骤2：强度增强（GDFN实现）  
#         x = self.gdfn(self.norm(x))
#         # 作用：通过门控机制增强特征表达，特别是亮度信息
    
class I_LCA(nn.Module):          #I轻量级交叉注意力
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.gdfn = IEL(dim)
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        
        #I_LCA在第二步有残差连接，这是因为I通道代表亮度信息，需要更谨慎地保留原始信息
        x = x + self.gdfn(self.norm(x)) 
        return x
