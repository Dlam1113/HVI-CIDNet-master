"""
数据集二次缩减脚本 (v2)
=============================
功能：将三个子数据集的训练集各缩减至 600 张，验证集各缩减至 100 张。
- 支持通过 --base_dir 参数指定数据集根目录（适配服务器路径）
- 多余图片对移动到 unused_high / unused_low 子目录，不会直接删除
- 使用固定随机种子(seed=42)保证可复现性
- 自动处理配对文件（high/low 必须同名）

服务器用法示例:
    python reduce_datasets_v2.py --base_dir /home/Bjj/HVI-CIDNet-clean/filtered
本地用法示例:
    python reduce_datasets_v2.py --base_dir d:\\HVI-CIDNet-master\\new_datasets\\filtered
"""

import os
import random
import shutil
import argparse


def reduce_dataset(dataset_dir, target_count):
    """
    将指定目录下的 high/low 配对图片缩减到 target_count 张。

    参数:
        dataset_dir (str): 包含 high/ 和 low/ 子目录的数据集路径
        target_count (int): 目标保留的图片对数
    """
    train_high_dir = os.path.join(dataset_dir, 'high')
    train_low_dir = os.path.join(dataset_dir, 'low')

    # 检查目录是否存在
    if not os.path.exists(train_high_dir) or not os.path.exists(train_low_dir):
        print(f"  [跳过] {dataset_dir}: high/ 或 low/ 目录不存在")
        return

    # 获取当前文件列表（只统计图片文件，忽略隐藏文件等）
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    high_files = sorted([
        f for f in os.listdir(train_high_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    ])
    low_files = sorted([
        f for f in os.listdir(train_low_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    ])

    current_count = len(high_files)
    dir_name = os.path.basename(dataset_dir)
    print(f"  [{dir_name}] 当前图片对数: {current_count}")

    # 配对检查
    if len(high_files) != len(low_files):
        print(f"  [警告] high 与 low 图片数量不匹配 ({len(high_files)} vs {len(low_files)})")

    # 已经达标则跳过
    if current_count <= target_count:
        print(f"  -> 已经 <= {target_count}，无需缩减")
        return

    # 创建 unused 目录存放多余文件
    unused_high_dir = os.path.join(dataset_dir, 'unused_high')
    unused_low_dir = os.path.join(dataset_dir, 'unused_low')
    os.makedirs(unused_high_dir, exist_ok=True)
    os.makedirs(unused_low_dir, exist_ok=True)

    # 使用固定种子随机选取要保留的文件
    random.seed(42)
    files_to_keep = set(random.sample(high_files, target_count))
    files_to_move = [f for f in high_files if f not in files_to_keep]

    print(f"  -> 即将移动 {len(files_to_move)} 对图片到 unused 目录...")

    moved_count = 0
    for filename in files_to_move:
        high_src = os.path.join(train_high_dir, filename)
        high_dst = os.path.join(unused_high_dir, filename)
        low_src = os.path.join(train_low_dir, filename)
        low_dst = os.path.join(unused_low_dir, filename)

        # 确保配对的 low 文件存在再一起移动
        if os.path.exists(low_src):
            shutil.move(high_src, high_dst)
            shutil.move(low_src, low_dst)
            moved_count += 1
        else:
            print(f"  [警告] low 目录中未找到配对文件: {filename}，跳过该文件")

    # 验证结果
    remaining = len([
        f for f in os.listdir(train_high_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    ])
    print(f"  -> 完成! 已移动 {moved_count} 对，剩余 {remaining} 对")


def main():
    """主函数：解析参数并依次处理三个训练集和三个验证集"""
    parser = argparse.ArgumentParser(
        description="将三个子数据集训练集缩减至600张，验证集缩减至100张"
    )
    parser.add_argument(
        '--base_dir', type=str,
        default=r'd:\HVI-CIDNet-master\new_datasets\filtered',
        help='数据集根目录路径 (服务器上改为 /home/Bjj/HVI-CIDNet-clean/filtered)'
    )
    args = parser.parse_args()
    base_dir = args.base_dir

    # ==================== 配置 ====================
    # 训练集目录名 -> 目标数量
    train_datasets = {
        'loli_pedestrian': 600,
        'cityscapes_foggy_pedestrian': 600,
        'cityscapes_rain_pedestrian': 600,
    }
    # 验证集目录名 -> 目标数量
    val_datasets = {
        'loli_pedestrian_val': 100,
        'cityscapes_foggy_pedestrian_val': 100,
        'cityscapes_rain_pedestrian_val': 100,
    }

    print("=" * 55)
    print(f"数据集二次缩减脚本 v2")
    print(f"根目录: {base_dir}")
    print("=" * 55)

    # 处理训练集
    print("\n--- 训练集缩减 (目标: 各600张) ---")
    for name, target in train_datasets.items():
        dataset_path = os.path.join(base_dir, name)
        reduce_dataset(dataset_path, target_count=target)
        print()

    # 处理验证集
    print("--- 验证集缩减 (目标: 各100张) ---")
    for name, target in val_datasets.items():
        dataset_path = os.path.join(base_dir, name)
        reduce_dataset(dataset_path, target_count=target)
        print()

    print("=" * 55)
    print("全部完成！")
    print("=" * 55)


if __name__ == '__main__':
    main()
