import torch
import torch.nn as nn

pi = 3.141592653589793

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2))  # 可学习的密度参数
        self.gated = False      # 门控开关1
        self.gated2= False      # 门控开关2
        self.alpha = 1.0        # 缩放因子1
        self.alpha_s = 1.3      # 饱和度缩放因子
        self.this_k = 0         # 当前k值的缓存
        
    def HVIT(self, img):
        eps = 1e-8    # 防止除零的小数值

        #获取图片的设备信息和数据类型信息确保输入输出的张量保持一致
        device = img.device  # 输出：device(type='cuda', index=0)
        dtypes = img.dtype   # 输出：torch.float32

        #hue张量是 [8, 256, 256] (batch=8, height=256, width=256)随机数
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)

        #channels维度求最大值，维度变为 [8, 256, 256]，max(dim)函数返回一个元组[0]是最大值,[1]是最大值的索引位置
        value = img.max(1)[0].to(dtypes) 
        img_min = img.min(1)[0].to(dtypes)

        #img[:,0]等价于 img[:,0,:,:]表示红色通道其他通道同理
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6
        #布尔索引掩码是PyTorch中一种强大的条件选择机制，它通过布尔值张量来标记哪些位置需要操作
        # mask = img[:,0] == value  布尔掩码，形状: [8, 256, 256]
        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        #unsqueeze(1)的作用是在第1个维度位置插入一个大小为1的新维度最后形状为：[batch, 1, height, width]
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        # .item()的作用：将单元素张量转换为Python标量
        #print(k)              # tensor([0.2000], requires_grad=True)
        #print(k.item())       # 0.2
        k = self.density_k
        self.this_k = k.item()
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I],dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]
        
        # clip
        # torch.clamp(input, min, max)
        # 将张量中的值限制在[min, max]范围内
        # 小于min的值设为min，大于max的值设为max
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
        
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)# 恢复归一化的色调余弦分量
        V = (V) / (color_sensitive + eps)# 恢复归一化的色调正弦分量
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V + eps,H + eps) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)
        
        if self.gated:
            s = s * self.alpha_s
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb
