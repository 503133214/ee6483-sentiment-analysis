#!/usr/bin/env bash
# Newest-Qwen experiments on a Colab A100 40GB. Run from the project root on the VM:
#   nohup bash scripts/colab_chain_qwen3.sh > logs/chain.log 2>&1 &
# 1) Qwen3.8-27B (post-trained, multimodal; 55.6 GB bf16) as a ZERO-SHOT classifier, 4-bit nf4, thinking off
# 2) Qwen3.5-2B-Base + LoRA r=16: the smallest Base model of the Qwen3.5/3.8 architecture, same recipe as Qwen2.5-1.5B
set -u
cd "$(dirname "$0")/.."
mkdir -p logs results/cache
PY=${PY:-python}
run() { local name=$1; shift; echo; echo "\$ [$(date +%H:%M:%S)] $PY -u -m $*"; "$PY" -u -m "$@" > "logs/$name.log" 2>&1; echo "-> $name exit $? [$(date +%H:%M:%S)]"; }
run qwen3.8_27b_zeroshot src.llm_zero_shot --model Qwen/Qwen3.8-27B --quant 4bit --name qwen3.8_27b_zeroshot_4bit --batch-size 16 --scores-cache results/cache/qwen3.8_27b_zeroshot_scores.npz
run qwen3.5_2b_lora src.train_plm --model Qwen/Qwen3.5-2B-Base --name qwen3.5_2b_base_lora16_len256 --lora --lora-r 16 --lora-alpha 32 --lr 2e-4 --epochs 3 --max-len 256 --batch-size 16 --precision bf16 --load-dtype bf16 --eval-batch-size 32
echo; echo "CHAIN FINISHED [$(date +%H:%M:%S)]"
