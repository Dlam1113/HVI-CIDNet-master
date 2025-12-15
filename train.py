import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from net.DualSpaceCIDNet import DualSpaceCIDNet  # 双空间CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # TensorBoard支持

opt = option().parse_args()
# opt 是一个 argparse.Namespace 对象
# 这个对象是一个"空的盒子"，可以动态地往里面放东西

def seed_torch(seed = 42):
    """设置所有相关的随机种子以确保实验的可重复性"""
    print(f"使用随机种子：{seed}")
    
    random.seed(seed)           # Python随机数种子
    np.random.seed(seed)        # NumPy随机数种子
    torch.manual_seed(seed)     # PyTorch CPU随机数种子
    torch.cuda.manual_seed(seed)     # PyTorch单GPU随机数种子
    torch.cuda.manual_seed_all(seed) # PyTorch多GPU随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希种子，针对dict和set中元素的存储顺序和遍历顺序
    
    # 确保完全可重复性（会稍微降低训练速度）因为显存满了所以更改了记一下
    #torch.backends.cudnn.deterministic = True  # 使用确定性算法
    torch.backends.cudnn.benchmark = True     # 关闭自动优化，确保可重复
    
def train_init():
    """初始化训练环境"""
    seed_torch()                    # 设置随机种子
    # cudnn.benchmark 已在 seed_torch() 中设置为 False 以确保可重复性
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用第0号GPU
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("No GPU found, please run without --cuda")
    
def train(epoch, writer=None):
    """
    训练一个epoch
    
    Args:
        epoch: 当前epoch编号
        writer: TensorBoard的SummaryWriter对象，用于记录训练过程
    
    Returns:
        epoch_loss: 当前epoch的总损失
        batch_count: 处理的batch数量
    """
    model.train()
    epoch_loss = 0      # 累积整个epoch的总损失
    batch_count = 0     # 统计整个epoch处理的batch数量
    train_len = len(training_data_loader)  # DataLoader的长度
    iter = 0            # 当前epoch中已处理的batch计数器
    
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            input_img = im1 ** gamma
        else:
            input_img = im1
        
        gt_rgb = im2
        
        # ========== 前向传播（根据模型类型选择） ==========
        if opt.dual_space:
            # DualSpaceCIDNet：获取中间结果用于计算双空间损失
            results = model.forward_with_intermediates(input_img)
            output_rgb = results['output']
            rgb_out = results['rgb_out']
            hvi_out = results['hvi_out']
            
            # 计算双空间损失
            output_hvi = model.HVIT(output_rgb)
            gt_hvi = model.HVIT(gt_rgb)
            
            # 最终输出损失
            loss_output = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
            
            # RGB分支损失（新增）
            loss_rgb_branch = (L1_loss(rgb_out, gt_rgb) + D_loss(rgb_out, gt_rgb) * 0.5) * opt.RGB_loss_weight
            
            # HVI分支损失
            loss_hvi_branch = (L1_loss(hvi_out, gt_rgb) + D_loss(hvi_out, gt_rgb) + E_loss(hvi_out, gt_rgb)) * opt.HVI_weight
            
            # HVI空间损失
            loss_hvi_space = (L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi)) * opt.HVI_weight
            
            # 总损失
            loss = loss_output + loss_rgb_branch + loss_hvi_branch + loss_hvi_space
        else:
            # 原CIDNet训练逻辑
            output_rgb = model(input_img)
            output_hvi = model.HVIT(output_rgb)
            gt_hvi = model.HVIT(gt_rgb)
            loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
            loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
            loss = loss_rgb + opt.HVI_weight * loss_hvi
        
        iter += 1
        
        # 梯度裁剪防止梯度爆炸
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        # 标准的反向传播过程
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新模型参数
        
        # 累积损失
        epoch_loss += loss.item()
        batch_count += 1
        
        # 每个epoch结束时打印平均损失和学习率，并保存样本图像
        if iter == train_len:
            avg_loss = epoch_loss / batch_count
            current_lr = optimizer.param_groups[0]['lr']
            
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={:.6f}".format(
                epoch, avg_loss, current_lr))
            
            # 【TensorBoard记录】记录训练损失和学习率
            if writer is not None:
                writer.add_scalar('Train/Loss', avg_loss, epoch)
                writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
                
                # 记录训练图像（可选）
                # 将第一个batch的第一张图像记录到TensorBoard
                writer.add_image('Train/Output_Image', output_rgb[0], epoch, dataformats='CHW')
                writer.add_image('Train/Ground_Truth', gt_rgb[0], epoch, dataformats='CHW')
            
            # 保存训练样本到本地
            output_img = transforms.ToPILImage()(output_rgb[0].squeeze(0))
            gt_img = transforms.ToPILImage()(gt_rgb[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    
    return epoch_loss, batch_count
                

def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print('===> Loading datasets')
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad or opt.fivek:
        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lol_blur:
            train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.lolv2_real:
            train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_real)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lolv2_syn:
            train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_syn)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        
        if opt.SID:
            train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_SID)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.fivek:
            train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_fivek_eval_set(opt.data_val_fivek)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:
        raise ValueError("should choose a dataset")
    return training_data_loader, testing_data_loader

def build_model():
    """构建CIDNet或DualSpaceCIDNet模型"""
    print('===> Building model ')
    
    # 根据配置选择模型
    if opt.dual_space:
        print('===> 使用 DualSpaceCIDNet (HVI+RGB双空间融合)')
        model = DualSpaceCIDNet(
            channels=[36, 36, 72, 144],
            heads=[1, 2, 4, 8],
            fusion_type=opt.fusion_type,
            cross_space_attn=opt.cross_space_attn
        ).cuda()
    else:
        print('===> 使用原始 CIDNet')
        model = CIDNet().cuda()
    
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
        print(f'===> 已加载预训练模型: {pth}')
    return model

def make_scheduler():
    """创建优化器和学习率调度器"""
    # 步骤1: 创建Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    
    # 步骤2: 根据配置选择调度器
    if opt.cos_restart_cyclic:  # 使用循环余弦退火
        if opt.start_warmup:  # 如果启用warmup
            # 2.1 先创建余弦退火调度器（作为主调度器）
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer, 
                periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4],
                restart_weights=[1, 1], # 重启时的权重
                eta_mins=[0.0002, 0.0000001] # 每个周期的最小学习率
            )
            # 2.2 用warmup调度器包装主调度器
            scheduler = GradualWarmupScheduler(
                optimizer, 
                multiplier=1, 
                total_epoch=opt.warmup_epochs, 
                after_scheduler=scheduler_step
            )
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise ValueError("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda() # 创建L1损失函数
    D_loss = SSIM(weight=D_weight).cuda() # 创建SSIM损失函数
    E_loss = EdgeLoss(loss_weight=E_weight).cuda() # 创建边缘损失函数
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda() # 创建感知损失函数
    return L1_loss,P_loss,E_loss,D_loss 

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    '''
    train
    '''
    #峰值信噪比
    psnr = []
    #结构相似性：是一种衡量两幅图像相似度的指标，常用于评估图像失真前后的相似性
    ssim = []
    #学习感知图像块相似度也称感知损失，越低越好
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) #opt.val_folder = 'results/'
    
    # 【TensorBoard初始化】创建TensorBoard写入器
    # 生成带时间戳的日志目录，避免不同实验的日志混在一起
    log_dir = f'./runs/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    print(f"===> TensorBoard日志保存在: {log_dir}")
    print(f"===> 启动TensorBoard: tensorboard --logdir=runs")
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        # 训练一个epoch，传入writer用于记录
        epoch_loss, batch_num = train(epoch, writer=writer)
        scheduler.step()  # 通过调度器更新学习率

        
        norm_size = True #是否将图像归一化（统一）到固定尺寸

        # LOL three subsets
        if opt.lol_v1:
            output_folder = 'LOLv1/'#模型生成的增强图像保存路径，保存在results/LOLv1/文件夹下
            label_dir = opt.data_valgt_lol_v1#验证集真实图像保存路径，保存在datasets/LOLdataset/eval15/high/文件夹下
        if opt.lolv2_real:
            output_folder = 'LOLv2_real/'
            label_dir = opt.data_valgt_lolv2_real
        if opt.lolv2_syn:
            output_folder = 'LOLv2_syn/'
            label_dir = opt.data_valgt_lolv2_syn
            
        # LOL-blur dataset with low_blur and high_sharp_scaled
        if opt.lol_blur:
            output_folder = 'LOL_blur/'
            label_dir = opt.data_valgt_lol_blur
                
        if opt.SID:
            output_folder = 'SID/'
            label_dir = opt.data_valgt_SID
            npy = True #没用到
        if opt.SICE_mix:
            output_folder = 'SICE_mix/'
            label_dir = opt.data_valgt_SICE_mix
            norm_size = False
        if opt.SICE_grad:
            output_folder = 'SICE_grad/'
            label_dir = opt.data_valgt_SICE_grad
            norm_size = False
                
        if opt.fivek:
            output_folder = 'fivek/'
            label_dir = opt.data_valgt_fivek
            norm_size = False
        

        im_dir = opt.val_folder + output_folder + '*.png' #模型生成的增强图像的路径模式，匹配所有png文件
        # 每隔一定epoch进行模型评估
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) #每隔opt.snapshots个epoch保存一次模型
            # 在验证集上评估模型性能
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)
        # 计算评估指标
        avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)#metric就是评价指标的意思，use_GT_mean：是否使用亮度校正
        print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
        print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
        print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            
        # 保存指标到列表
        psnr.append(avg_psnr)
        ssim.append(avg_ssim)
        lpips.append(avg_lpips)
            
        # 【TensorBoard记录】记录评估指标
        writer.add_scalar('Eval/PSNR', avg_psnr, epoch)
        writer.add_scalar('Eval/SSIM', avg_ssim, epoch)
        writer.add_scalar('Eval/LPIPS', avg_lpips, epoch)
            
        # 同时在一个图中显示所有指标的变化趋势
        writer.add_scalars('Eval/All_Metrics', {
            'PSNR': avg_psnr,
            'SSIM': avg_ssim * 30,  # 放大SSIM以便在同一图中观察
            'LPIPS': avg_lpips * 100  # 放大LPIPS以便在同一图中观察
        }, epoch)
            
        
        torch.cuda.empty_cache()
    # 【训练完成】关闭TensorBoard写入器
    print("\n===> 训练完成！")
    
    # 记录最终的最佳结果到TensorBoard
    if len(psnr) > 0:
        best_psnr = max(psnr)
        best_ssim = max(ssim)
        best_lpips = min(lpips)
        best_psnr_epoch = (psnr.index(best_psnr) + 1) * opt.snapshots
        
        writer.add_text('Final_Results/Best_PSNR', f'{best_psnr:.4f} at Epoch {best_psnr_epoch}')
        writer.add_text('Final_Results/Best_SSIM', f'{best_ssim:.4f}')
        writer.add_text('Final_Results/Best_LPIPS', f'{best_lpips:.4f}')
    
    writer.close()
    print(f"===> TensorBoard日志已保存到: {log_dir}")
    
    #将训练过程中的所有评估指标（PSNR、SSIM、LPIPS）和训练配置保存到一个格式化的 Markdown 文件中，便于后续查看和比较实验结果。    
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")#获取当前日期并格式化时间字符串，（）里都是对应日期缩写eg：%Y是year
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("dataset: "+ output_folder + "\n")  
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"L1_weight: {opt.L1_weight}\n")  
        f.write(f"D_weight: {opt.D_weight}\n")  
        f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"P_weight: {opt.P_weight}\n")
        f.write(f"TensorBoard日志: {log_dir}\n\n")
        best_psnr_idx = psnr.index(max(psnr))
        f.write("## 最佳结果\n\n")
        f.write(f"- **最佳PSNR**: {max(psnr):.4f} (Epoch {(best_psnr_idx+1)*opt.snapshots})\n")
        f.write(f"- **最佳SSIM**: {max(ssim):.4f}\n")
        f.write(f"- **最低LPIPS**: {min(lpips):.4f}\n\n")  
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | { psnr[i*opt.snapshots]:.4f} | {ssim[i*opt.snapshots]:.4f} | {lpips[i*opt.snapshots]:.4f} |\n")  
        