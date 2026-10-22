#!/usr/bin/env bash
# GloVe + BiLSTM / TextCNN: full 5-fold runs, the random-embedding ablation, then small 2-fold sweeps.
#   bash scripts/rnn_runs.sh            (~30-40 min on Apple M5, MPS)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
run() { echo; echo "\$ $PY -u -m src.train_rnn $*"; "$PY" -u -m src.train_rnn "$@"; echo "-> exit $?"; }
# 5-fold main rows
run --arch bilstm  --name glove_bilstm
run --arch textcnn --name glove_textcnn
run --arch bilstm  --name rand_bilstm --no-pretrained
# 2-fold sweeps (report section d): learning rate, capacity, frozen embeddings
for lr in 5e-4 2e-3; do run --arch bilstm --name glove_bilstm_lr$lr --lr $lr --folds 0,1 --allow-partial; done
run --arch bilstm  --name glove_bilstm_h128 --hidden 128 --folds 0,1 --allow-partial
run --arch bilstm  --name glove_bilstm_frozen --freeze-emb-epochs 99 --folds 0,1 --allow-partial
run --arch textcnn --name glove_textcnn_lr5e-4 --lr 5e-4 --folds 0,1 --allow-partial
echo; echo "all runs finished"
