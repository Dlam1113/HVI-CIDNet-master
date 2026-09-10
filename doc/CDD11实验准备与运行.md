# CDD-11 返修实验准备与运行

## 当前阶段

四任务统一复原为研究主线；CDD-11 保留 11 种退化作为扩展实验。当前先完成数据与运行入口，未产生可写入论文的正式实验成绩。原 RainCityscapes、Foggy Cityscapes 和 LOLv1 数据与结果保持原状。

代码由 Windows 本地修改，经 GitHub 分支 `codex/neucom-cdd11` 同步到 `/home/Bjj/HVI-CIDNet-clean`。服务器运行环境是 `/home/Bjj/anaconda3/envs/CIDNet/bin/python`。以下命令均在该服务器项目根目录执行。

## 数据协议

- 来源：`gy65896/CDD-11`，固定版本 `52a185b4266f6b1093b6391294ce2c230cc35c6c`。
- 数据根目录：`/home/Bjj/HVI-CIDNet-clean/datasets/CDD-11`。
- 原始压缩包保留在 `downloads`，解压到 `official`；不向 GitHub 上传数据或模型权重。
- 官方训练部分 1,183 个清晰场景，抽取 600 个实际训练场景、另 100 个验证场景；483 个场景备用。
- 官方测试部分 200 个清晰场景完整保留。抽样种子固定为 `20260910`。
- 11 任务：训练 6,600 对，验证 1,100 对，测试 2,200 对。
- 4 种单退化：训练 2,400 对，验证 400 对，测试 800 对。它只是四任务子集，不能称为完整 CDD-11 实验。
- 两套清单使用相同场景划分。`scene_id` 按官方划分限定命名空间；跨划分通过清晰图像 RGB 内容摘要检查泄漏。
- 对比工程 `/home/Bjj/comparison_models` 可读取同一批 JSON 清单；适配其他网络的训练入口需要另行完成，不能直接沿用旧模型成绩。

## 数据准备

```bash
/home/Bjj/anaconda3/envs/CIDNet/bin/python -u -B -m revision.prepare_server \
  --root /home/Bjj/HVI-CIDNet-clean/datasets/CDD-11
```

脚本按顺序下载、核验官方 SHA-256、安全解压、检查文件名与尺寸、计算清晰内容摘要、生成清单。`preparation_status.json` 记录阶段和异常；只有 `phase=complete` 才表示全流程完成。下载失败可重跑，已完成文件由下载缓存复用。清单不覆盖既有划分。

输出清单目录：`datasets/CDD-11/manifests/600_100_seed20260910`。

## CPU 流程检查

```bash
/home/Bjj/anaconda3/envs/CIDNet/bin/python -B -m unittest revision.test_cdd11 revision.test_pipeline
/home/Bjj/anaconda3/envs/CIDNet/bin/python -B -m revision.experiment smoke \
  --root datasets/CDD-11/official \
  --manifest datasets/CDD-11/manifests/600_100_seed20260910/all11_train.json \
  --model full --device cpu --output results/cdd11_smoke_full.json
```

流程检查只使用一张图的 32×32 裁剪与 L1 反向传播，不能当作训练完成或性能结果。报告会列出非有限参数；正式训练入口遇到非有限参数会停止。

## 正式训练入口与尚待确定的设置

训练必须指定 `--max-steps`，没有自动启动长训练的默认预算。先结合 GPU 可用时间、收敛情况和审稿实验矩阵确定预算，再执行。

```bash
/home/Bjj/anaconda3/envs/CIDNet/bin/python -u -B -m revision.experiment train \
  --root datasets/CDD-11/official \
  --train-manifest datasets/CDD-11/manifests/600_100_seed20260910/all11_train.json \
  --val-manifest datasets/CDD-11/manifests/600_100_seed20260910/all11_val.json \
  --model full --max-steps <确定后的优化步数> \
  --output results/cdd11_all11_full_seed42
```

- 默认裁剪 256×256、微批量 2、梯度累积 8（有效批量 16）、Adam、学习率 1e-4、梯度范数裁剪 0.01。
- 沿用旧 `train.py` 的 RGB/HVI 双域损失：L1=1、SSIM=0.5、边缘=50、VGG 感知=0.01。仅调整损失模块设备迁移方式，使 CPU 检查可运行；未改变数学表达式。
- 新入口以优化步数计数，先线性预热再单次余弦下降；默认关闭随机 gamma，等概率采样任务。此为明确的新训练协议，不能声称与旧训练完全相同。
- `cidnet`、`refiner`、`curve`、`full` 是现有网络的四种配置。当前未修改神经曲线公式、单调性或初始化；这些属于后续审稿意见的核查范围。若出现非有限参数，先解决原因再做长训练。
- 只允许 train/val 清单进入训练入口，验证任务平均 PSNR 用于选最佳模型。官方 test 清单不参与模型选择。
- `config.json`、`train.jsonl`、`last.pt`、`best.pt` 保存配置、指标、权重和优化器状态。用 `--resume` 指向已有 checkpoint，配置、代码及清单来源需保持一致。

## 最终测试

```bash
/home/Bjj/anaconda3/envs/CIDNet/bin/python -u -B -m revision.experiment test \
  --root datasets/CDD-11/official \
  --manifest datasets/CDD-11/manifests/600_100_seed20260910/all11_test.json \
  --checkpoint results/cdd11_all11_full_seed42/best.pt \
  --lpips --output results/cdd11_all11_full_seed42_final_test
```

测试保留原图分辨率，只补齐到 8 的倍数后裁回；不使用亮度缩放或其他额外后处理。PSNR/SSIM 使用浮点 RGB [0,1]、无裁边，高斯 SSIM 窗口；LPIPS 使用 AlexNet，输入映射到 [-1,1]。输出逐图 CSV、分任务指标以及单重/双重/三重退化汇总，保留场景编号供后续统计。

## 原始来源

- [OneRestore 官方代码](https://github.com/gy65896/OneRestore)
- [CDD-11 官方数据](https://huggingface.co/datasets/gy65896/CDD-11)
- [OneRestore 原论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02906.pdf)
