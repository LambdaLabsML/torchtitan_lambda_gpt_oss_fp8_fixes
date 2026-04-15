#!/bin/bash
#SBATCH --job-name=torchtitan_20b
#SBATCH --nodes=1
#SBATCH --partition=dev
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j_torchtitan.out
#SBATCH --error=logs/%j_torchtitan.err

# Activate your virtual environment
source /data/dj/.venv/bin/activate

# Set working directory
cd /data/dj/torchtitan

# Distributed training env vars
export NCCL_IB_DISABLE=0          # Enable InfiniBand if available
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# PyTorch memory: expandable_segments avoids contiguous-block OOM from fragmentation.
# TORCH_NCCL_AVOID_RECORD_STREAMS reduces peak memory by not caching NCCL comm buffers.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  -m torchtitan.train \
  --module gpt_oss \
  --config gpt_oss_20b_dense_fp8_only
