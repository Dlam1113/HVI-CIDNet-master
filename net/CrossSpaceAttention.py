"""
跨空间交叉注意力模块
Cross-Space Attention - 实现RGB和HVI空间的双向信息交换

核心思想：
- RGB空间擅长保留像素级细节和纹理
- HVI空间擅长亮度增强和色彩调整
- 通过交叉注意力让两个空间互相学习对方的优势

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 尝试导入原有的LayerNorm，如果失败则使用本地定义
try:
    from net.transformer_utils import LayerNorm
except ImportError:
    class LayerNorm(nn.Module):
        """LayerNorm支持channels_first格式"""
        def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps
            self.data_format = data_format
            self.normalized_shape = (normalized_shape, )
        
        def forward(self, x):
            if self.data_format == "channels_first":
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class CrossSpaceAttention(nn.Module):
    """
    跨空间交叉注意力模块
    
    让查询空间（Query Space）从上下文空间（Context Space）学习有用信息
    
    信息流动示例：
    - 当x_query=RGB特征, x_context=HVI特征时：
      RGB特征学习HVI的亮度增强策略
    - 当x_query=HVI特征, x_context=RGB特征时：
      HVI特征学习RGB的纹理细节信息
    
    参数:
        dim: 特征维度（通道数）
        num_heads: 注意力头数
        bias: 是否使用偏置
    """
    
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子 1/sqrt(d_k)
        
        # 可学习的温度参数（控制注意力分布的锐度）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Query投影：从查询空间生成Q
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, 
                                   padding=1, groups=dim, bias=bias)
        
        # Key和Value投影：从上下文空间生成K, V
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1,
                                    padding=1, groups=dim * 2, bias=bias)
        
        # 输出投影
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x_query, x_context):
        """
        前向传播
        
        参数:
            x_query: 查询特征 [B, C, H, W] - 需要被增强的特征
            x_context: 上下文特征 [B, C, H, W] - 提供信息的特征
            
        返回:
            增强后的查询特征 [B, C, H, W]
        """
        B, C, H, W = x_query.shape
        
        # Step 1: 生成Q, K, V
        # Query来自查询空间
        Q = self.q_dwconv(self.q_proj(x_query))  # [B, C, H, W]
        
        # Key和Value来自上下文空间
        kv = self.kv_dwconv(self.kv_proj(x_context))  # [B, 2C, H, W]
        K, V = kv.chunk(2, dim=1)  # 各 [B, C, H, W]
        
        # Step 2: 重塑为多头格式 [B, heads, head_dim, H*W]
        Q = rearrange(Q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        K = rearrange(K, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V = rearrange(V, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        # Step 3: 归一化（稳定训练，防止注意力分布过于尖锐）
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)
        
        # Step 4: 计算注意力权重
        # [B, heads, head_dim, H*W] @ [B, heads, H*W, head_dim] -> [B, heads, head_dim, head_dim]
        attn = (Q @ K.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        
        # Step 5: 应用注意力权重到Value
        # [B, heads, head_dim, head_dim] @ [B, heads, head_dim, H*W] -> [B, heads, head_dim, H*W]
        out = attn @ V
        
        # Step 6: 重塑回原始格式
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        
        # Step 7: 输出投影
        out = self.out_proj(out)
        
        # 残差连接：保留原始查询特征，加上从上下文学到的信息
        return x_query + out


class BidirectionalCrossAttention(nn.Module):
    """
    双向跨空间交叉注意力模块
    
    实现RGB和HVI特征的双向信息交换：
    - RGB特征向HVI空间学习
    - HVI特征向RGB空间学习
    
    参数:
        dim: 特征维度
        num_heads: 注意力头数
    """
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        
        # 归一化层
        self.norm_rgb = LayerNorm(dim)
        self.norm_hvi = LayerNorm(dim)
        
        # RGB → HVI 方向：RGB特征从HVI学习亮度信息
        self.rgb_to_hvi = CrossSpaceAttention(dim, num_heads)
        
        # HVI → RGB 方向：HVI特征从RGB学习细节信息
        self.hvi_to_rgb = CrossSpaceAttention(dim, num_heads)
        
        # 特征精炼FFN
        self.ffn_rgb = FeedForward(dim)
        self.ffn_hvi = FeedForward(dim)
        
    def forward(self, rgb_feat, hvi_feat):
        """
        双向信息交换
        
        参数:
            rgb_feat: RGB空间特征 [B, C, H, W]
            hvi_feat: HVI空间特征 [B, C, H, W]
            
        返回:
            (增强的RGB特征, 增强的HVI特征)
        """
        # 归一化
        rgb_normed = self.norm_rgb(rgb_feat)
        hvi_normed = self.norm_hvi(hvi_feat)
        
        # 双向交叉注意力
        rgb_enhanced = self.rgb_to_hvi(rgb_normed, hvi_normed)
        hvi_enhanced = self.hvi_to_rgb(hvi_normed, rgb_normed)
        
        # FFN精炼
        rgb_enhanced = self.ffn_rgb(rgb_enhanced)
        hvi_enhanced = self.ffn_hvi(hvi_enhanced)
        
        return rgb_enhanced, hvi_enhanced


class FeedForward(nn.Module):
    """
    前馈网络（门控机制）
    
    借鉴原CIDNet中IEL的设计，使用门控机制进行特征精炼
    
    参数:
        dim: 特征维度
        expansion_factor: 扩展因子，默认2.66
        bias: 是否使用偏置
    """
    
    def __init__(self, dim, expansion_factor=2.66, bias=False):
        super().__init__()
        
        hidden_dim = int(dim * expansion_factor)
        
        # 通道扩展
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        
        # 深度卷积
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3,
                                 stride=1, padding=1, groups=hidden_dim * 2, bias=bias)
        
        # 门控分支的额外处理
        self.dwconv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                  stride=1, padding=1, groups=hidden_dim, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                  stride=1, padding=1, groups=hidden_dim, bias=bias)
        
        # 通道压缩
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
        
        # 激活函数
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        """前向传播"""
        # 扩展和深度卷积
        x_proj = self.project_in(x)
        x_proj = self.dwconv(x_proj)
        
        # 分割为两个分支
        x1, x2 = x_proj.chunk(2, dim=1)
        
        # 门控机制：两个分支分别处理后相乘
        x1 = self.tanh(self.dwconv1(x1)) + x1  # 残差
        x2 = self.tanh(self.dwconv2(x2)) + x2  # 残差
        x_gated = x1 * x2  # 门控
        
        # 输出投影 + 残差
        out = self.project_out(x_gated)
        return x + out  # 残差连接


# ==========  测试代码  ==========
if __name__ == '__main__':
    print("=" * 60)
    print("跨空间交叉注意力模块单元测试")
    print("=" * 60)
    
    # 测试配置
    batch_size = 2
    height, width = 64, 64
    dim = 72
    num_heads = 4
    
    # 模拟输入
    rgb_feat = torch.randn(batch_size, dim, height, width)
    hvi_feat = torch.randn(batch_size, dim, height, width)
    
    print(f"\n测试配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  特征尺寸: {height}x{width}")
    print(f"  通道数: {dim}")
    print(f"  注意力头数: {num_heads}")
    print("-" * 40)
    
    # 测试1: CrossSpaceAttention
    print("\n[测试1] CrossSpaceAttention (单向交叉注意力)")
    cross_attn = CrossSpaceAttention(dim, num_heads)
    out = cross_attn(rgb_feat, hvi_feat)
    print(f"  输入形状: {rgb_feat.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in cross_attn.parameters()):,}")
    assert out.shape == rgb_feat.shape, "输出形状错误！"
    print("  [OK] 测试通过")
    
    # 测试2: BidirectionalCrossAttention
    print("\n[测试2] BidirectionalCrossAttention (双向交叉注意力)")
    bi_cross_attn = BidirectionalCrossAttention(dim, num_heads)
    rgb_out, hvi_out = bi_cross_attn(rgb_feat, hvi_feat)
    print(f"  RGB输入形状: {rgb_feat.shape}")
    print(f"  HVI输入形状: {hvi_feat.shape}")
    print(f"  RGB输出形状: {rgb_out.shape}")
    print(f"  HVI输出形状: {hvi_out.shape}")
    print(f"  参数量: {sum(p.numel() for p in bi_cross_attn.parameters()):,}")
    assert rgb_out.shape == rgb_feat.shape, "RGB输出形状错误！"
    assert hvi_out.shape == hvi_feat.shape, "HVI输出形状错误！"
    print("  [OK] 测试通过")
    
    # 测试3: FeedForward
    print("\n[测试3] FeedForward (门控前馈网络)")
    ffn = FeedForward(dim)
    ffn_out = ffn(rgb_feat)
    print(f"  输入形状: {rgb_feat.shape}")
    print(f"  输出形状: {ffn_out.shape}")
    print(f"  参数量: {sum(p.numel() for p in ffn.parameters()):,}")
    assert ffn_out.shape == rgb_feat.shape, "输出形状错误！"
    print("  [OK] 测试通过")
    
    # GPU测试
    if torch.cuda.is_available():
        print("\n[测试4] GPU兼容性测试")
        rgb_cuda = rgb_feat.cuda()
        hvi_cuda = hvi_feat.cuda()
        bi_cross_attn_cuda = BidirectionalCrossAttention(dim, num_heads).cuda()
        rgb_out_cuda, hvi_out_cuda = bi_cross_attn_cuda(rgb_cuda, hvi_cuda)
        print(f"  GPU设备: {rgb_out_cuda.device}")
        print("  [OK] GPU测试通过")
    else:
        print("\n[测试4] GPU不可用，跳过GPU测试")
    
    print("\n" + "=" * 60)
    print("所有测试通过！跨空间交叉注意力模块可正常使用。")
    print("=" * 60)
