#!/usr/bin/env bash
# 用户手动启动；保留原40k实验，从其best.pt开启独立40k续训阶段。
set -euo pipefail

main() {
  # 固定完整模型和数据，检查输出、GPU占用及截止日期后才启动。
  local project python_bin manifest run_dir source_run gpu_jobs remaining
  project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  python_bin=/home/Bjj/anaconda3/envs/CIDNet/bin/python
  manifest="$project/datasets/CDD-11/manifests/600_100_seed20260910"
  source_run="$project/results/cdd11_full_40k_s42"
  run_dir="$project/results/cdd11_full_continue40k_s42"
  cd "$project"
  mkdir -p "$project/results"
  exec 9>"${run_dir}.lock"
  flock -n 9 || { echo '同一续训已有启动进程，拒绝重复运行。' >&2; return 2; }
  local -a start_args=()
  if [[ $# -eq 1 && "$1" == --resume ]]; then
    [[ -f "$run_dir/last.pt" ]] || { echo '新阶段没有last.pt。' >&2; return 2; }
    start_args=(--resume "$run_dir/last.pt")
  elif [[ $# -ne 0 ]]; then
    echo '用法：bash revision/run_cdd11_full_continue_40k.sh [--resume]' >&2
    return 2
  else
    [[ ! -e "$run_dir" ]] || { echo '新阶段目录已存在，不覆盖；中断恢复使用--resume。' >&2; return 2; }
    [[ -f "$source_run/best.pt" ]] || { echo '找不到原实验best.pt。' >&2; return 2; }
    start_args=(--continue-from "$source_run/best.pt")
  fi
  gpu_jobs="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits)"
  [[ -z "$gpu_jobs" ]] || { echo "GPU已有计算进程：$gpu_jobs" >&2; return 2; }
  # 全部返修9月22日截止，训练最晚9月19日晚停止，为写作与核查留出时间。
  remaining=$(( $(date -d '2026-09-19 20:00:00 +0800' +%s) - $(date +%s) ))
  [[ $remaining -gt 0 ]] || { echo '已到训练截止时间，拒绝启动。' >&2; return 2; }
  [[ $remaining -ge 50400 ]] || { echo '距训练截止不足14小时规划窗口，请先重新核查预算。' >&2; return 2; }
  echo "代码版本：$(git rev-parse HEAD)"
  echo '独立追加40000更新；完整模型64通道/11点，batch16，累积1，256裁剪，FP32。'
  echo '继承最佳权重及Adam；500步由源LR升至1e-5，再余弦下降至1e-7。'
  echo '每2000更新完整验证1100对并打印PSNR、SSIM、LPIPS；按PSNR保存最佳。'
  echo '追加耗时需要第一轮三指标验证校准，14小时是规划窗口而非完成保证。'
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  exec timeout --signal=INT --kill-after=120 "${remaining}s" "$python_bin" -u -B -m revision.experiment train \
    --root "$project/datasets/CDD-11/official" \
    --train-manifest "$manifest/all11_train.json" --val-manifest "$manifest/all11_val.json" \
    --output "$run_dir" --model full --device cuda --crop-size 256 \
    --batch-size 16 --accum-steps 1 --max-steps 40000 --warmup-steps 500 \
    --eval-every 2000 --lr 1e-5 --scheduler continuation_cosine --seed 42 \
    --workers 2 --cpu-threads 2 --val-lpips --progress bar "${start_args[@]}"
}

main "$@"
