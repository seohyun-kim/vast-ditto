#!/bin/bash

# mkdir -p logs
# TS=$(date +%Y%m%d_%H%M%S)
# LOG="runs/ae_stage1/logs/train_${TS}.log"

# nohup env PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128' \
#           TORCHINDUCTOR_CUDA_GRAPHS=0 \
#     python autoencoder/train_autoencoder_stage1.py \
#       --embed-dir dataset/embed_trace\
#       --spec dataset/embed_trace/rowemb_spec.json \
#       --batch-size 8 \
#       --lr 1e-4 \
#   >"$LOG" 2>&1 & echo $! > runs/ae_stage1/train.pid

# echo "Started PID $(cat runs/ae_stage1/train.pid). Tail: tail -f $LOG"



python autoencoder/train_autoencoder_stage1.py \
  --embed-dir dataset/emb128/embed_trace\
  --spec dataset/emb128/spec.json \
  --batch-size 128 --lr 1e-4 \
  --loss-scale 1000