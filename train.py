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
    
    # 梯度累积设置：
    accum_steps = opt.accum_steps  # 累积步数（每accum_steps个batch更新一次参数）
    
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
        
       
        # 串联DualSpaceCIDNet与原CIDNet使用完全一致的损失结构
        # DualSpaceCIDNet: 输入 → RGB Block → 增强RGB → HVI空间CIDNet处理 → 输出
        # 原CIDNet:        输入 → HVI空间CIDNet处理 → 输出
        output_rgb = model(input_img)  # 两种模型都直接返回tensor
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        
        iter += 1
        
        # 梯度累积：损失除以累积步数，保证梯度大小一致
        loss = loss / accum_steps
        loss.backward()  # 累积梯度（不清零）
        
        # 每accum_steps步更新一次参数
        if iter % accum_steps == 0:
            if opt.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()  # 更新后才清零
        
        # 累积损失（还原为原始损失值用于显示）
        epoch_loss += loss.item() * accum_steps
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
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad or opt.fivek or opt.LoLI_Street:
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
        
        if opt.LoLI_Street:
            train_set = get_LoLI_Street_training_set(opt.data_LoLI_Street, size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_LoLI_Street)  # 使用 SICE eval（返回4个值）
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:
        raise ValueError("should choose a dataset")
    return training_data_loader, testing_data_loader

def build_model():
    """构建CIDNet或DualSpaceCIDNet模型"""
    print('===> Building model ')
    
    # 根据配置选择模型
    if opt.dual_space:
        print('===> 使用 DualSpaceCIDNet (v3: CIDNet + RGB后处理)')
        if opt.use_curve:
            print('===> 启用神经曲线层消融实验 (I通道全局调整)')
        if opt.use_rgb_refiner:
            print(f'===> 启用RGB后处理微调 (mid_ch={opt.refiner_mid_ch})')
        model = DualSpaceCIDNet(
            channels=[36, 36, 72, 144],
            heads=[1, 2, 4, 8],
            use_rgb_refiner=opt.use_rgb_refiner,
            refiner_mid_ch=opt.refiner_mid_ch,
            use_curve=opt.use_curve,
            curve_M=opt.curve_M
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
        # 两段周期与两阶段训练对齐：
        # 第一段 = freeze_epoch（联合训练期，lr: 0.0001 → 1e-5）
        # 第二段 = 剩余epoch（只训Refiner，lr: 0.0001 → 1e-6）
        freeze_ep = opt.freeze_epoch if opt.freeze_epoch > 0 else opt.nEpochs // 4
        remaining = opt.nEpochs - freeze_ep - opt.start_epoch
        if opt.start_warmup:  # 如果启用warmup
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer, 
                periods=[freeze_ep - opt.warmup_epochs, remaining],
                restart_weights=[1, 1],
                eta_mins=[1e-5, 1e-6]  # 必须低于base_lr=0.0001
            )
            scheduler = GradualWarmupScheduler(
                optimizer, 
                multiplier=1, 
                total_epoch=opt.warmup_epochs, 
                after_scheduler=scheduler_step
            )
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer, 
                periods=[freeze_ep - opt.start_epoch, remaining],
                restart_weights=[1, 1],
                eta_mins=[1e-5, 1e-6])  # 必须低于base_lr=0.0001
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
    #结构相似性
    ssim = []
    #学习感知图像块相似度（越低越好）
    lpips = []
    # NIQE自然度（越低越好，无参考指标）
    niqe = []
    # BRISQUE空间质量（越低越好，无参考指标）
    brisque = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) #opt.val_folder = 'results/'
    
    # 【TensorBoard初始化】创建TensorBoard写入器
    # 生成带时间戳的日志目录，避免不同实验的日志混在一起
    log_dir = f'./runs/exp{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    print(f"===> TensorBoard日志保存在: {log_dir}")
    print(f"===> 启动TensorBoard: tensorboard --logdir=runs")
    
    
        
    # 标记是否已冻结CIDNet
    cidnet_frozen = False
    
    for epoch in range(start_epoch+1, opt.nEpochs + 1):
        # ========== 两阶段训练：到达freeze_epoch时冻结CIDNet ==========
        if (opt.dual_space and opt.use_rgb_refiner and opt.freeze_epoch > 0
                and epoch == opt.freeze_epoch and not cidnet_frozen):
            print(f"\n{'='*60}")
            print(f"===> Epoch {epoch}: 冻结CIDNet，只训练RGB Refiner")
            print(f"{'='*60}")
            
            # 冻结所有非Refiner的参数
            frozen_count = 0
            trainable_count = 0
            for name, param in model.named_parameters():
                if 'rgb_refiner' not in name:
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    trainable_count += 1
            
            print(f"  冻结参数组数: {frozen_count}")
            print(f"  可训练参数组数: {trainable_count}")
            
            # 重建优化器，只包含Refiner参数
            refiner_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(refiner_params, lr=opt.lr)
            # 重建调度器绑定新optimizer，第二段余弦衰减
            # +1 是因为冻结当轮也会调用scheduler.step()，避免最后一轮超出周期
            remaining_epochs = opt.nEpochs - opt.freeze_epoch + 1
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[remaining_epochs],
                restart_weights=[1],
                eta_mins=[1e-6])  # 与初始调度器第二段一致
            
            cidnet_frozen = True
            
            # TensorBoard记录冻结事件
            if writer is not None:
                writer.add_text('Training/Event', 
                    f'Epoch {epoch}: CIDNet frozen, only training Refiner')
        
        # 训练一个epoch，传入writer
        epoch_loss, batch_num = train(epoch, writer=writer)
        scheduler.step()  # 通过调度器更新学习率

        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) #每隔opt.snapshots个epoch保存一次模型
            # 在验证集上评估模型性能
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
            if opt.LoLI_Street:
                output_folder = 'LoLI_Street/'
                label_dir = opt.data_valgt_LoLI_Street
                norm_size = False
            
            im_dir = opt.val_folder + output_folder  # 只传目录路径
            # 每隔一定epoch进行模型评估
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                    norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)
            
            # 计算评估指标
            avg_psnr, avg_ssim, avg_lpips, avg_niqe, avg_brisque = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            print("===> Avg.NIQE: {:.4f} ".format(avg_niqe))
            print("===> Avg.BRISQUE: {:.4f} ".format(avg_brisque))
                
            # 保存指标到列表
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            niqe.append(avg_niqe)
            brisque.append(avg_brisque)
                
            # 【TensorBoard记录】记录评估指标
            writer.add_scalar('Eval/PSNR', avg_psnr, epoch)
            writer.add_scalar('Eval/SSIM', avg_ssim, epoch)
            writer.add_scalar('Eval/LPIPS', avg_lpips, epoch)
            writer.add_scalar('Eval/NIQE', avg_niqe, epoch)
            writer.add_scalar('Eval/BRISQUE', avg_brisque, epoch)
                
            # 同时在一个图中显示所有指标的变化趋势
            writer.add_scalars('Eval/All_Metrics', {
                'PSNR': avg_psnr,
                'SSIM': avg_ssim * 30,
                'LPIPS': avg_lpips * 100,
                'NIQE': avg_niqe,
                'BRISQUE': avg_brisque / 5  # 缩小BRISQUE以便在同一图中观察
            }, epoch)
            print(psnr)
            print(ssim)
            print(lpips)
            
        
        torch.cuda.empty_cache()
    # 【训练完成】关闭TensorBoard写入器
    print("\n===> 训练完成！")
    
    # 记录最终的最佳结果到TensorBoard
    if len(psnr) > 0:
        best_psnr = max(psnr)
        best_ssim = max(ssim)
        best_lpips = min(lpips)
        best_niqe = min(niqe) if niqe else 0
        best_brisque = min(brisque) if brisque else 0
        best_psnr_epoch = (psnr.index(best_psnr) + 1) * opt.snapshots
        best_ssim_epoch = (ssim.index(best_ssim) + 1) * opt.snapshots
        best_lpips_epoch = (lpips.index(best_lpips) + 1) * opt.snapshots
        best_niqe_epoch = (niqe.index(best_niqe) + 1) * opt.snapshots if niqe else 0
        best_brisque_epoch = (brisque.index(best_brisque) + 1) * opt.snapshots if brisque else 0
        
        writer.add_text('Final_Results/Best_PSNR', f'{best_psnr:.4f} at Epoch {best_psnr_epoch}')
        writer.add_text('Final_Results/Best_SSIM', f'{best_ssim:.4f} at Epoch {best_ssim_epoch}')
        writer.add_text('Final_Results/Best_LPIPS', f'{best_lpips:.4f} at Epoch {best_lpips_epoch}')
        writer.add_text('Final_Results/Best_NIQE', f'{best_niqe:.4f} at Epoch {best_niqe_epoch}')
        writer.add_text('Final_Results/Best_BRISQUE', f'{best_brisque:.4f} at Epoch {best_brisque_epoch}')
    
    writer.close()
    print(f"===> TensorBoard日志已保存到: {log_dir}")
    
    # 保存所有指标到Markdown文件
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/metrics/metrics{now}.md", "w") as f:
        f.write("dataset: "+ output_folder + "\n")  
        f.write("dual_space: " + str(opt.dual_space) + "\n")
        f.write("use_rgb_refiner: " + str(opt.use_rgb_refiner) + "\n")
        f.write("refiner_mid_ch: " + str(opt.refiner_mid_ch) + "\n")
        f.write("use_curve: " + str(opt.use_curve) + "\n")
        f.write("curve_M: " + str(opt.curve_M) + "\n")
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"accum_steps: {opt.accum_steps}\n")
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"L1_weight: {opt.L1_weight}\n")  
        f.write(f"D_weight: {opt.D_weight}\n")  
        f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"P_weight: {opt.P_weight}\n")
        f.write(f"freeze_epoch: {opt.freeze_epoch}\n")
        f.write(f"TensorBoard日志: {log_dir}\n\n")
        
        # 最佳结果汇总
        best_psnr_idx = psnr.index(max(psnr))
        best_ssim_idx = ssim.index(max(ssim))
        best_lpips_idx = lpips.index(min(lpips))
        best_niqe_idx = niqe.index(min(niqe)) if niqe else 0
        best_brisque_idx = brisque.index(min(brisque)) if brisque else 0
        
        f.write("## 最佳结果\n\n")
        f.write(f"- **最佳PSNR**: {max(psnr):.4f} (Epoch {(best_psnr_idx+1)*opt.snapshots})\n")
        f.write(f"- **最佳SSIM**: {max(ssim):.4f} (Epoch {(best_ssim_idx+1)*opt.snapshots})\n")
        f.write(f"- **最低LPIPS**: {min(lpips):.4f} (Epoch {(best_lpips_idx+1)*opt.snapshots})\n")
        f.write(f"- **最低NIQE**: {min(niqe):.4f} (Epoch {(best_niqe_idx+1)*opt.snapshots})\n")
        f.write(f"- **最低BRISQUE**: {min(brisque):.4f} (Epoch {(best_brisque_idx+1)*opt.snapshots})\n\n")
        
        # 指标表格（含新指标）
        f.write("| Epochs | PSNR | SSIM | LPIPS | NIQE | BRISQUE |\n")  
        f.write("|--------|------|------|-------|------|---------|\n")  
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | {psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} | {niqe[i]:.4f} | {brisque[i]:.4f} |\n")
        
        f.write(f"\n## 最终结果\n\n")
        f.write(f"| 指标 | 最佳值 | 对应Epoch |\n")
        f.write(f"|------|--------|----------|\n")
        f.write(f"| PSNR ↑ | {max(psnr):.4f} | {(best_psnr_idx+1)*opt.snapshots} |\n")
        f.write(f"| SSIM ↑ | {max(ssim):.4f} | {(best_ssim_idx+1)*opt.snapshots} |\n")
        f.write(f"| LPIPS ↓ | {min(lpips):.4f} | {(best_lpips_idx+1)*opt.snapshots} |\n")
        f.write(f"| NIQE ↓ | {min(niqe):.4f} | {(best_niqe_idx+1)*opt.snapshots} |\n")
        f.write(f"| BRISQUE ↓ | {min(brisque):.4f} | {(best_brisque_idx+1)*opt.snapshots} |\n")