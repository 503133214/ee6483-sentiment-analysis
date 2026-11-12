# EE6483 Product Review Sentiment Classification

**Language:** **English** | [简体中文](README.zh-CN.md)

[GitHub repository](https://github.com/503133214/ee6483-sentiment-analysis) · [Experiment results](docs/experiment_results.md) · [Assignment](docs/assignment/EE6483-Project1.pdf)

Course project for NTU EE6483 Artificial Intelligence and Data Mining: predict sentiment from English product reviews, with `0` for negative and `1` for positive. The project compares five representation and adaptation approaches on shared data splits—TF-IDF, pretrained word vectors, encoder fine-tuning, compact decoder LLMs with LoRA, and a 27B zero-shot reference—to answer: **how much does each approach improve negative-review recognition when 85.4% of reviews are positive, and what computation does it require?**

On 7,185 exact-deduplicated training reviews with shared five-fold validation, **Qwen3.5-2B-Base + LoRA** achieves the highest adapted-model scores: **0.971 accuracy, 0.942 macro-F1, and 0.902 negative-class F1**. Qwen2.5-1.5B + LoRA, the submitted model, reaches 0.970 / 0.940 / 0.897 while updating 0.28% of its parameters. Relative to negation-aware TF-IDF + Logistic Regression (0.854 macro-F1; 1.14 seconds for five-fold vectorization and fitting), the two LoRA models improve macro-F1 by **8.79 / 8.55 percentage points** and negative-class F1 by **14.97 / 14.53 points**. The task-label-free Qwen3.8-27B reference reaches 0.940 macro-F1 at threshold 0.5; calibrating its threshold with training labels gives **0.9509 threshold-cross-fit macro-F1** (0.9534 on the same OOF scores used to fit the threshold). Hardware-specific training, inference, and memory measurements are documented in the results page.

See [experiment results](docs/experiment_results.md) for full metrics and methodology, and [report guidance](docs/project_advice.md) for the report structure and project plan. These supporting documents are in Chinese.

## Key Results

The following are development-validation results on the shared five-fold split, with main run `seed=42` and threshold 0.5. Accuracy and macro-F1 are fold means ± population standard deviations; F1(neg) is the fold mean.

| Model | Accuracy | Macro-F1 | F1(neg) |
|---|---|---|---|
| TF-IDF + NOT_ + LR | 0.927 ± 0.001 | 0.854 ± 0.001 | 0.752 |
| GloVe + BiLSTM | 0.930 ± 0.007 | 0.860 ± 0.013 | 0.761 |
| RoBERTa-base | 0.962 ± 0.003 | 0.923 ± 0.006 | 0.868 |
| DeBERTa-v3-base (Colab L4, bf16) | 0.967 ± 0.002 | 0.934 ± 0.004 | 0.887 |
| Qwen2.5-0.5B + LoRA r=16 (Colab L4) | 0.967 ± 0.002 | 0.932 ± 0.004 | 0.884 |
| **Qwen2.5-1.5B + LoRA r=16 (Colab L4, submitted)** | **0.970 ± 0.002** | **0.940 ± 0.005** | **0.897** |
| Qwen3.5-2B-Base + LoRA r=16 (Colab A100) | 0.971 ± 0.003 | 0.942 ± 0.005 | 0.902 |
| Qwen3.8-27B zero-shot, 4-bit, threshold 0.5 (Colab A100) | 0.968 ± 0.003 | 0.940 ± 0.005 | 0.898 |

RoBERTa-base records macro-F1 of 0.9227, 0.9230, and 0.9243 for training seeds 42 (local MPS fp16), 2024, and 7 (Colab L4 bf16). Frozen-feature controls use the same backbones and shared splits: fine-tuned RoBERTa-base exceeds frozen mean pooling + LR (0.848) by 7.49 points, and Qwen2.5-1.5B LoRA exceeds frozen last-token pooling + LR (0.827) by 11.34 points. These compare complete frozen-feature and task-adapted pipelines.

**Evaluation protocol:** neural checkpoints are selected and scored on the held-out fold, so these are development-validation metrics; last-epoch predictions are also saved for comparison. Threshold cross-fitting holds out folds for threshold selection only, not for the entire training and hyperparameter-selection procedure. The test set has no reference labels, so no test accuracy is reported. Full five-fold results, two-fold sweeps, resource measurements, and threshold analysis are in [experiment results](docs/experiment_results.md).

## Documentation

| Document | Contents |
|---|---|
| [Experiment results and evaluation protocol](docs/experiment_results.md) | Full metrics, ablations, sweeps, resource measurements, data and model settings; Chinese |
| [Report guidance and project plan](docs/project_advice.md) | Writing reference, evidence for questions a–i, and project plan; Chinese |
| [Assignment](docs/assignment/EE6483-Project1.pdf) | Original task and marking requirements; English |

## Setup and Usage

Use the `ee6483` conda environment with Python 3.13. `requirements.txt` pins the recorded base dependencies; `requirements-lock.txt` is a local Apple MPS environment snapshot, not a CUDA lockfile. Keep the environment, Hugging Face cache, and GloVe downloads outside cloud-synced folders.

```bash
git clone https://github.com/503133214/ee6483-sentiment-analysis.git
cd ee6483-sentiment-analysis
conda create -y -n ee6483 python=3.13
conda activate ee6483
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ee6483 --display-name "Python (ee6483)"
python -c "import torch, transformers, sklearn; print(torch.__version__, torch.backends.mps.is_available(), transformers.__version__, sklearn.__version__)"
```

LoRA additionally requires `peft`; the installed local version is 0.21.2. CUDA 4-bit / 8-bit inference also requires `bitsandbytes`, with a version compatible with the target CUDA / PyTorch environment. A complete version lock for that GPU environment is not included.

```bash
# Install before running LoRA experiments
python -m pip install peft==0.21.2
# Install only in a CUDA environment used for quantized inference
python -m pip install bitsandbytes
```

### Quick Start: TF-IDF Baseline

Run all commands from the project root. Create the shared folds and reproduce the baseline; its separate output file preserves the final LoRA submission:

```bash
python -m src.data
python -m src.train_tfidf --name tfidf_lr_neg
python -m src.make_submission --summary results/summary/tfidf_lr_neg_seed42.json --out submissions/tfidf_submission.csv
python -m src.check_submission submissions/tfidf_submission.csv
```

### Pretrained Encoders

Fine-tune a pretrained language model. Apple Silicon automatically uses MPS + fp16; the first run downloads the model to the Hugging Face cache.

```bash
# Small training-pipeline check
python -m src.train_plm --model distilbert-base-uncased --folds 0 --limit 256 --epochs 1 --no-save
# Full five-fold RoBERTa; recorded cumulative training time is about 57 minutes
python -m src.train_plm --model roberta-base --name roberta_base_len256 --lr 3e-5 --max-len 256 --batch-size 16
# Resume an interrupted run with the same settings
python -m src.train_plm --model roberta-base --name roberta_base_len256 --lr 3e-5 --max-len 256 --batch-size 16 --resume
# DistilBERT two-fold learning-rate sweep, weighting ablation, and five-fold RoBERTa
python -m scripts.plm_sweep
```

### Word-Vector Models

The first run downloads `glove-wiki-gigaword-300` through gensim (about 376 MB), stored in `~/gensim-data` by default:

```bash
python -m src.train_rnn --arch bilstm --name glove_bilstm
python -m src.train_rnn --arch textcnn --name glove_textcnn
python -m src.train_rnn --arch bilstm --name rand_bilstm --no-pretrained
bash scripts/rnn_runs.sh
```

### CUDA and Google Colab

DeBERTa-v3 uses bf16, and Qwen uses LoRA. `train_frozen.py` extracts frozen features for logistic regression; `llm_zero_shot.py` runs 27B zero-shot classification. For Colab, install and authenticate `google-colab-cli` locally. The L4 queue is defined in `scripts/colab_chain.sh`; the A100 queue (Qwen3.8-27B zero-shot and Qwen3.5-2B-Base LoRA) is in `scripts/colab_chain_qwen3.sh`. Training runs as a background process on the VM; `colab exec` starts it and checks progress.

The L4 workflow is shown below. For A100 experiments, change session creation to `--gpu A100` and replace the default launch command with the alternative queue command below. Install `bitsandbytes` in that session for 27B quantized inference.

```bash
colab new -s ee6483 --gpu L4                                   # L4 24 GB; bf16; recorded usage about 1.5 compute units/hour
colab install -s ee6483 peft==0.21.2
echo 'import subprocess,sys; subprocess.run([sys.executable,"-m","pip","uninstall","-y","torchao"])' | colab exec -s ee6483   # Preinstalled torchao 0.10 conflicted with peft
tar czf project-colab.tar.gz --exclude=__pycache__ --exclude=.DS_Store src scripts data/raw/train.json data/raw/test.json data/folds.csv results/runs.csv requirements.txt
colab upload -s ee6483 project-colab.tar.gz /content/project.tar.gz
colab exec -s ee6483 -f scripts/colab_launch.py                # Extract and start the L4 queue in the background
colab exec -s ee6483 -f scripts/colab_poll.py                  # Inspect progress, logs, and completed runs
# Replace RUN_ID with the desired run; download probabilities and curves similarly
colab download -s ee6483 /content/project/results/summary/RUN_ID.json results/summary/RUN_ID.json
colab stop -s ee6483                                           # Stop billing
```

```bash
# Alternative: start the A100 queue in a prepared A100 session instead of the L4 queue
colab install -s ee6483 bitsandbytes
colab exec -s ee6483 --env CHAIN=scripts/colab_chain_qwen3.sh -f scripts/colab_launch.py
```

Equivalent individual experiment commands for a CUDA / bf16 machine:

```bash
python -m src.train_plm --model microsoft/deberta-v3-base --name deberta_v3_base_len256 --lr 2e-5 --max-len 256 --batch-size 16 --precision bf16 --eval-batch-size 32
python -m src.train_plm --model Qwen/Qwen2.5-1.5B --name qwen2.5_1.5b_lora16_len256 --lora --lora-r 16 --lora-alpha 32 --lr 2e-4 --max-len 256 --batch-size 16 --precision bf16 --load-dtype bf16 --gradient-checkpointing --eval-batch-size 32
python -m src.train_frozen --model Qwen/Qwen2.5-1.5B --pooling last --load-dtype bf16 --name qwen2.5_1.5b_frozen_last_lr
python -m src.train_frozen --model roberta-base --pooling mean --name roberta_base_frozen_mean_lr --batch-size 64
python -m src.train_plm --model Qwen/Qwen3.5-2B-Base --name qwen3.5_2b_base_lora16_len256 --lora --lora-r 16 --lora-alpha 32 --lr 2e-4 --epochs 3 --max-len 256 --batch-size 16 --precision bf16 --load-dtype bf16 --eval-batch-size 32
python -m src.llm_zero_shot --model Qwen/Qwen3.8-27B --quant 4bit --name qwen3.8_27b_zeroshot_4bit --batch-size 16   # No training; threshold 0.5 is the task-label-free operating point
python -m scripts.label_noise_candidates   # 92 high-confidence label disagreements; 57 supported by all three supervised OOF models
```

`--folds 0,1` or `--limit` requests a partial run and requires `--no-save` or `--allow-partial`. The latter saves artifacts under a suffixed `run_id`, such as `_f0-1` or `_n64`. TF-IDF ablation options include `--no-negation`, `--class-weight none`, `--C 1`, `--ngram-max 1`, and `--min-df 1`.

### Regression Tests

```bash
python -m tests.test_pipeline
python -m tests.test_train_plm
python -m tests.test_train_rnn
python -m tests.test_results
```

## Submission

The current `submissions/submission.csv` uses **Qwen2.5-1.5B + LoRA**:

- Run ID: `qwen2.5_1.5b_lora16_len256_seed42`.
- Average test probabilities from the five fold models, then apply OOF-selected threshold **0.45**; no separate full-data refit is used.
- 1,851 records: 1,595 positive and 256 negative predictions (86.17% positive).
- Accuracy / macro-F1 on the OOF scores used to fit the threshold: 0.9706 / 0.9404; threshold-cross-fit estimates: 0.9676 / 0.9357.

The repository stores probabilities and experiment summaries, not trained model weights. The following commands **reconstruct the CSV from saved probabilities without downloading or retraining models**. To obtain new predictions directly from reviews, rerun the training workflows above.

```bash
python -m src.make_submission --summary results/summary/qwen2.5_1.5b_lora16_len256_seed42.json --out submissions/reproduced_submission.csv --compare submissions/submission.csv
python -m src.check_submission submissions/reproduced_submission.csv --compare submissions/submission.csv
python -m src.make_submission --summary results/summary/roberta_base_len256_seed42.json --out submissions/roberta_submission.csv --compare submissions/submission.csv
```

**Always specify `--out` to preserve the final file.** Set it to `submissions/submission.csv` only when intentionally updating the final submission. Example ensemble:

```bash
python -m src.make_submission --summary results/summary/tfidf_lr_neg_seed42.json results/summary/roberta_base_len256_seed42.json --out submissions/ensemble_submission.csv
```

The submission has exactly two columns, `reviews,predictions`, with 1,851 texts in the original `test.json` order and integer 0/1 predictions. The generator writes a temporary file, reads it back for validation, and only then updates the destination. The report should identify the final model, seed, threshold, and generation command.

The assignment requires one report and `submission.csv` per three-person group, with member contributions and references stated in the report. The deadline is **13 November 2026, 23:59**; see the [assignment](docs/assignment/EE6483-Project1.pdf).

## Analysis and Report

Available evidence includes full five-fold results, negation and embedding-initialization ablations, frozen versus adapted pipeline controls, three RoBERTa seeds, the 27B zero-shot reference, and hardware-specific resource measurements.

- **Negative-review recognition:** pooled OOF confusion matrices show 252 missed negatives for TF-IDF versus 119 for Qwen1.5B, a 52.8% reduction.
- **Label disagreements:** `results/analysis/label_noise_candidates.csv` contains 92 training reviews where the 27B reference confidently disagrees with the supplied label; 57 are also supported by all three supervised OOF models. These form a manual-review queue, not confirmed annotation errors or a measured noise rate.
- **Transfer and noise robustness:** adapting to unrated hotel reviews and inaccurate rating-derived labels is discussed as a proposed extension, not a completed experiment.

See [experiment results](docs/experiment_results.md) for detailed measurements and [report guidance](docs/project_advice.md) for section organization.

## Contributions

Men Zhaolin, Gao Zhihan, and Zhou Junchong contributed **equally to the project and report (one third each)**.

## Repository Layout

```text
.
├── data/
│   ├── raw/train.json                # Original labeled reviews
│   ├── raw/test.json                 # Unlabeled test reviews
│   └── folds.csv                     # Shared five-fold mapping keyed by text hash
├── docs/
│   ├── assignment/EE6483-Project1.pdf # Assignment
│   ├── examples/tfidf_features_example.html
│   ├── experiment_results.md         # Detailed metrics, resources, and evaluation protocol
│   └── project_advice.md             # Report guidance and project plan
├── notebooks/tfidf_features_example.ipynb
├── src/
│   ├── data.py / metrics.py          # Data protocol, metrics, and threshold rules
│   ├── features/negation.py          # NOT_ negation-scope marking
│   ├── train_tfidf.py                # TF-IDF + LR
│   ├── train_plm.py                  # Full fine-tuning or LoRA
│   ├── train_frozen.py               # Frozen features + logistic regression
│   ├── llm_zero_shot.py              # Label-logit LLM zero-shot classification
│   ├── train_rnn.py                  # GloVe + BiLSTM / TextCNN
│   ├── utils/                       # Resource measurement and result persistence
│   └── make_submission.py / check_submission.py
├── scripts/                         # Sweeps, Colab L4 / A100 queues, disagreement analysis
├── results/
│   ├── runs.csv                     # One summary row per run_id
│   ├── oof/                         # Out-of-fold probabilities in deduplicated training order
│   ├── test_probs/                  # Test probabilities averaged over folds
│   ├── summary/                     # Settings, metrics, thresholds, and training curves
│   └── analysis/                    # Label-disagreement candidates
├── tests/                           # Regression tests
├── submissions/                     # Course template and predictions
├── environment.yml
├── requirements.txt / requirements-lock.txt
├── README.md                        # English
└── README.zh-CN.md                  # Simplified Chinese
```

The HTML file archives the original course example; use `notebooks/` for interactive execution. Neural runs also save `*_last.npy` predictions for checkpoint comparisons. Each summary's `artifacts` field lists its output paths.
