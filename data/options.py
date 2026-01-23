import argparse

def option():
    # Training settings
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='CIDNet')

    # 添加各种命令行参数
    parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=5, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')
    parser.add_argument('--accum_steps', type=int, default=1, help='gradient accumulation steps')

    # choose a scheduler 学习率调度器的作用是使学习率周期性变化帮助模型跳过局部最优解
    parser.add_argument('--cos_restart_cyclic', type=bool, default=False)
    parser.add_argument('--cos_restart', type=bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=bool, default=True, help='turn False to train without warmup') 

    # train datasets  训练数据路径
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')
    parser.add_argument('--data_train_fivek'        , type=str, default='./datasets/FiveK/train')
    parser.add_argument('--data_LoLI_Street'        , type=str, default='./datasets/LoLI-Street/Train') 

    # validation input   验证输入路径
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='./datasets/LOLdataset/eval15/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_test_fivek'         , type=str, default='./datasets/FiveK/test/input')
    parser.add_argument('--data_val_LoLI_Street'    , type=str, default='./datasets/LoLI-Street/Val/low')

    # validation groundtruth   验证真值路径
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='./datasets/LOLdataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_fivek'        , type=str, default='./datasets/FiveK/test/target/')
    parser.add_argument('--data_valgt_LoLI_Street'  , type=str, default='./datasets/LoLI-Street/Val/high')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights 损失权重
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)
    
    # use random gamma function (enhancement curve) to improve generalization 使用随机gamma函数提高泛化能力
    parser.add_argument('--gamma', type=bool, default=True)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=bool, default=True)  # 梯度爆炸检测
    parser.add_argument('--grad_clip', type=bool, default=True)     # 梯度裁剪
    
    # ========== 双空间CIDNet配置（新增） ==========
    parser.add_argument('--dual_space', type=bool, default=True, 
                        help='是否使用DualSpaceCIDNet（HVI+RGB双空间融合）')
    parser.add_argument('--RGB_loss_weight', type=float, default=0.5, 
                        help='RGB分支损失权重')
    parser.add_argument('--cross_space_attn', type=bool, default=False, 
                        help='是否使用跨空间交叉注意力')
    parser.add_argument('--fusion_type', type=str, default='learnable', 
                        help='融合类型：learnable/adaptive')
    
    # ========== 神经曲线层消融实验 ==========
    parser.add_argument('--use_curve', type=bool, default=False,
                        help='是否使用神经曲线层对I通道进行全局调整（消融实验）')
    parser.add_argument('--curve_M', type=int, default=11,
                        help='曲线控制点数量')
    
    
    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--lol_v1', type=bool, default=False)
    parser.add_argument('--lolv2_real', type=bool, default=False)
    parser.add_argument('--lolv2_syn', type=bool, default=False)
    parser.add_argument('--lol_blur', type=bool, default=False)
    parser.add_argument('--SID', type=bool, default=False)
    parser.add_argument('--SICE_mix', type=bool, default=True)
    parser.add_argument('--SICE_grad', type=bool, default=False)
    parser.add_argument('--fivek', type=bool, default=False)
    parser.add_argument('--LoLI_Street', type=bool, default=False)
    return parser
