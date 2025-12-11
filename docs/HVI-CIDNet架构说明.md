# HVI-CIDNet 架构图说明文档

## 概述

HVI-CIDNet是CVPR 2025提出的低光图像增强网络，核心创新包括：
- **HVI颜色空间**：解决HSV中红色不连续和黑色平面噪声问题
- **CIDNet双分支网络**：分离处理色彩(HV)和亮度(I)信息
- **LCA交叉注意力**：实现两分支间的信息交互

---

## 图1：HVI颜色空间转换流程

![HVI颜色空间转换流程图](/home/Bjj/.gemini/antigravity/brain/57a103f0-8379-48d4-ad2d-30e43647a2d8/hvi_color_space_flow_1764831205509.png)

### 转换流程说明

| 阶段 | 颜色空间 | 问题/改进 | 数学操作 |
|------|----------|-----------|----------|
| (a) | sRGB | 高颜色敏感度，色彩失真 | 输入 |
| (b) | HSV | ①红色h=0和h=6不连续 ②黑色区域噪声放大 | Imax, S, H计算 |
| (c) | 极化HS | 解决红色不连续 | h=cos(πH/3), v=sin(πH/3) |
| (d) | HVI | 解决黑色噪声，可学习坍缩 | Ĥ=Ck⊙S⊙h, V̂=Ck⊙S⊙v |

### 关键公式

```
可学习强度坍缩函数:
  Ck = (sin(π·Imax/2) + ε)^(1/k)
  
其中 k 是可训练参数 (density_k)
```

---

## 图2：CIDNet网络整体架构

![CIDNet网络架构图](/home/Bjj/.gemini/antigravity/brain/57a103f0-8379-48d4-ad2d-30e43647a2d8/cidnet_architecture_1764831254280.png)

### 三阶段处理流程

```
Input Image → HVIT → 双分支增强网络 → PHVIT → Enhanced Image
              ↓           ↓              ↓
          HVI转换    I+HV分支处理    HVI逆转换
```

### 双分支设计原理

| 分支 | 输入 | 功能 | 输出 |
|------|------|------|------|
| **I-Branch** | 强度图 (H×W×1) | 亮度增强 | 增强后的亮度图 |
| **HV-Branch** | HV颜色图 (H×W×2) + 强度图 | 去噪、色彩恢复 | 增强后的色彩图 |

### 网络配置

- **编码器**：3层下采样 (通道: 36→36→72→144)
- **解码器**：3层上采样 (通道: 144→72→36→36)
- **LCA模块**：共6个，连接两分支
- **跳跃连接**：3组，连接对应层级的编码器和解码器

---

## 图3：LCA模块详细结构

![LCA模块结构图](/home/Bjj/.gemini/antigravity/brain/57a103f0-8379-48d4-ad2d-30e43647a2d8/lca_module_1764831301351.png)

### 组件说明

**1. CAB (Cross Attention Block)**
```python
Q = DWConv3×3(Conv1×1(x))  # 来自查询流
K = DWConv3×3(Conv1×1(y))  # 来自键值流
V = DWConv3×3(Conv1×1(y))  # 来自键值流
Attn = softmax(Q·K^T / τ) · V  # τ是可学习温度参数
```

**2. IEL (Intensity Enhancement Layer)**
```python
x1, x2 = Split(DWConv(Conv(input)))
output = Tanh(x1) ⊙ Tanh(x2)  # 门控机制
```

### 交叉注意力的作用

- **I→HV方向**：用亮度特征指导色彩去噪
- **HV→I方向**：用去噪后的色彩信息辅助亮度增强

---

## 论文原图参考

````carousel
![Figure 1 - HVI颜色空间转换](/home/Bjj/.gemini/antigravity/brain/57a103f0-8379-48d4-ad2d-30e43647a2d8/uploaded_image_0_1764830906154.png)
<!-- slide -->
![Figure 2 - CIDNet架构](/home/Bjj/.gemini/antigravity/brain/57a103f0-8379-48d4-ad2d-30e43647a2d8/uploaded_image_1_1764830906154.png)
<!-- slide -->
![Figure 5 - 消融实验](/home/Bjj/.gemini/antigravity/brain/57a103f0-8379-48d4-ad2d-30e43647a2d8/uploaded_image_3_1764830906154.png)
````

---

## 总结

HVI-CIDNet的创新点：
1. **HVI颜色空间**：通过极化和可学习坍缩解决HSV的固有缺陷
2. **双分支解耦**：分离处理亮度和色彩，符合物理规律
3. **交叉注意力**：实现两分支信息互补，提升增强效果
4. **轻量高效**：1.88M参数，7.57G FLOPs，性能SOTA
