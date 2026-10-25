#!/usr/bin/env bash
# Experiment queue for the Colab L4 session (bf16). Run from the project root on the VM:
#   nohup bash scripts/colab_chain.sh > logs/chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
PY=${PY:-python}
run() { local name=$1; shift; echo; echo "\$ [$(date +%H:%M:%S)] $PY -u -m $*"; "$PY" -u -m "$@" > "logs/$name.log" 2>&1; echo "-> $name exit $? [$(date +%H:%M:%S)]"; }
# 1) DeBERTa-v3-base, 5 folds, bf16 (fp16 overflows for DeBERTa-v3; NaN on MPS)
run deberta_v3_base_len256 src.train_plm --model microsoft/deberta-v3-base --name deberta_v3_base_len256 --lr 2e-5 --epochs 3 --max-len 256 --batch-size 16 --precision bf16 --eval-batch-size 32
# 2) Qwen2.5-0.5B + LoRA (r=16), classification head trainable, bf16 weights
run qwen2.5_0.5b_lora src.train_plm --model Qwen/Qwen2.5-0.5B --name qwen2.5_0.5b_lora16_len256 --lora --lora-r 16 --lora-alpha 32 --lr 2e-4 --epochs 3 --max-len 256 --batch-size 16 --precision bf16 --load-dtype bf16 --eval-batch-size 32
# 3) Qwen2.5-1.5B + LoRA (r=16), gradient checkpointing
run qwen2.5_1.5b_lora src.train_plm --model Qwen/Qwen2.5-1.5B --name qwen2.5_1.5b_lora16_len256 --lora --lora-r 16 --lora-alpha 32 --lr 2e-4 --epochs 3 --max-len 256 --batch-size 16 --precision bf16 --load-dtype bf16 --gradient-checkpointing --eval-batch-size 32
# 4) frozen features + LR: Qwen2.5-1.5B (last token) and roberta-base (mean) — the "feature extraction vs fine-tuning" rows
run qwen2.5_1.5b_frozen src.train_frozen --model Qwen/Qwen2.5-1.5B --pooling last --load-dtype bf16 --name qwen2.5_1.5b_frozen_last_lr --batch-size 32
run roberta_frozen src.train_frozen --model roberta-base --pooling mean --name roberta_base_frozen_mean_lr --batch-size 64
# 5) roberta-base, two more seeds (same config as the MPS run) for mean ± std over seeds
run roberta_seed2024 src.train_plm --model roberta-base --name roberta_base_len256 --seed 2024 --lr 3e-5 --epochs 3 --max-len 256 --batch-size 16 --precision bf16 --eval-batch-size 32
run roberta_seed7 src.train_plm --model roberta-base --name roberta_base_len256 --seed 7 --lr 3e-5 --epochs 3 --max-len 256 --batch-size 16 --precision bf16 --eval-batch-size 32
echo; echo "CHAIN FINISHED [$(date +%H:%M:%S)]"
