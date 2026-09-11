#!/usr/bin/env bash
# 用户手动启动完整模型；仅运行此实验，不自动衔接对比或消融。
set -euo pipefail

main() {
  # 固定数据、更新预算和学习率协议，并阻止覆盖或重复启动。
  local project python_bin manifest run_dir remaining gpu_jobs
  project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  python_bin=/home/Bjj/anaconda3/envs/CIDNet/bin/python
  manifest="$project/datasets/CDD-11/manifests/600_100_seed20260910"
  run_dir="$project/results/cdd11_full_40k_s42"
  cd "$project"
  mkdir -p "$project/results"
  exec 9>"${run_dir}.lock"
  if ! flock -n 9; then
    echo '同一实验已有启动进程，请勿重复运行。' >&2
    return 2
  fi
  local -a resume_args=()
  if [[ $# -eq 1 && "$1" == --resume ]]; then
    [[ -f "$run_dir/last.pt" ]] || { echo '没有可恢复的last.pt。' >&2; return 2; }
    resume_args=(--resume "$run_dir/last.pt")
  elif [[ $# -ne 0 ]]; then
    echo '用法：bash revision/run_cdd11_full_40k.sh [--resume]' >&2
    return 2
  elif [[ -e "$run_dir" ]]; then
    echo '输出目录已存在，不覆盖；如需恢复请使用--resume。' >&2
    return 2
  fi
  gpu_jobs="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits)"
  [[ -z "$gpu_jobs" ]] || { echo "GPU已有计算进程：$gpu_jobs" >&2; return 2; }
  remaining=$(( $(date -d '2026-09-18 20:00:00 +0800' +%s) - $(date +%s) ))
  [[ $remaining -gt 0 ]] || { echo '已到训练截止时间，拒绝开始。' >&2; return 2; }
  echo "实验版本：$(git rev-parse HEAD)"
  echo '完整模型64通道/11点；40000更新，750预热，每2000更新完整验证。'
  echo '原始曲线首点保持固定0；FP32，batch16，累积1；从头训练。'
  echo '调度：0→1e-4预热，第一阶段升至2e-4；第10001次更新重启至1e-4并降至1e-7。'
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  exec timeout --signal=INT --kill-after=120 "${remaining}s" "$python_bin" -u -B -m revision.experiment train \
    --root "$project/datasets/CDD-11/official" \
    --train-manifest "$manifest/all11_train.json" \
    --val-manifest "$manifest/all11_val.json" \
    --output "$run_dir" --model full --device cuda \
    --crop-size 256 --batch-size 16 --accum-steps 1 \
    --max-steps 40000 --warmup-steps 750 --eval-every 2000 \
    --lr 1e-4 --scheduler legacy_two_stage --seed 42 \
    --workers 2 --cpu-threads 2 "${resume_args[@]}"
}

main "$@"
