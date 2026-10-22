# EE6483 商品评论情感分类

NTU EE6483 Artificial Intelligence and Data Mining 课程项目：根据英文商品评论预测情感，`0` 表示负面，`1` 表示正面。

项目建议、评分要点、模型路线、分工与时间线见 [docs/project_advice.md](docs/project_advice.md)。

## 目录

```text
.
├── data/
│   ├── raw/train.json                # 7,401 条带标签评论 (去重后 7,185)
│   ├── raw/test.json                 # 1,851 条无标签评论
│   └── folds.csv                     # 共享的 5 折划分 (sha1(text) → fold), 由 python -m src.data 生成
├── docs/
│   ├── assignment/EE6483-Project1.pdf
│   ├── examples/tfidf_features_example.html
│   └── project_advice.md             # 项目建议文档
├── notebooks/tfidf_features_example.ipynb
├── src/
│   ├── data.py                       # 读数据、去重、5 折协议
│   ├── metrics.py                    # 指标 + OOF 阈值规则
│   ├── features/negation.py          # 词袋模型用的 NOT_ 否定范围标记
│   ├── utils/resources.py            # 计时 / 峰值内存 (报告 g 用)
│   ├── train_tfidf.py                # TF-IDF + LogisticRegression 基线 (5 折)
│   ├── train_plm.py                  # 预训练模型微调 (distilbert / bert / roberta), 5 折, MPS fp16
│   ├── make_submission.py            # 测试概率 → submission.csv (+ 自检)
│   └── check_submission.py           # 提交文件格式校验
├── results/
│   ├── runs.csv                      # 每次实验一行 (指标 + 资源)
│   ├── oof/*.npy                     # out-of-fold 概率 (与去重后的训练集顺序一致); PLM 另有 *_last.npy
│   ├── test_probs/*.npy              # 测试集概率 (5 折模型平均)
│   └── summary/*.json                # 每次运行的完整摘要 (配置、各折指标、阈值); PLM 另有 *_curves.csv; plm_sweep.json 记录扫参决策
├── scripts/plm_sweep.py              # distilbert 扫参 → 类权重消融 → roberta-base 5 折 的顺序驱动
├── tests/                            # 回归测试 (test_pipeline.py, test_train_plm.py)
├── submissions/
│   ├── submission_template.csv       # 课程给的模板 (全 0)
│   └── submission.csv                # 当前提交文件
├── environment.yml / requirements.txt / requirements-lock.txt
└── README.md
```

HTML 文件是课程示例的原始导出归档；运行时使用 `notebooks/` 中的 notebook。

## 环境

使用 conda 环境 `ee6483`（Python 3.13；torch 在 Apple Silicon 上自动使用 MPS）。环境、HuggingFace 缓存、GloVe、checkpoint 都放在 OneDrive 目录之外。

```bash
conda create -y -n ee6483 python=3.13
conda activate ee6483
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ee6483 --display-name "Python (ee6483)"
python -c "import torch, transformers, sklearn; print(torch.__version__, torch.backends.mps.is_available(), transformers.__version__, sklearn.__version__)"
```

`requirements.txt` 固定了已验证的版本；`requirements-lock.txt` 是完整的 `pip freeze`。本机原有的 conda 环境 `torch`（Python 3.10）的 scipy 在 macOS 27 上无法加载，请勿使用。

## 运行（在项目根目录）

```bash
python -m src.data                                   # 打印数据统计, 生成/校验 data/folds.csv
python -m src.train_tfidf --name tfidf_lr_neg        # 1-2gram + NOT_ 否定标记 + LR(C=4, balanced), 5 折
python -m src.make_submission --summary results/summary/tfidf_lr_neg_seed42.json   # 写 submissions/submission.csv 并自检
python -m src.check_submission submissions/submission.csv                           # 单独校验任意提交文件
python -m src.make_submission --summary results/summary/a.json results/summary/b.json  # 集成: 平均概率, 阈值在平均 OOF 上重选
python -m tests.test_pipeline                                                       # 回归测试 (否定标记 / 折协议 / 指标 / 校验器)
```

预训练模型微调（Apple Silicon 上自动用 MPS + fp16；首次运行会从 HuggingFace 下载模型到 `~/.cache/huggingface`）：

```bash
python -m src.train_plm --model distilbert-base-uncased --folds 0 --limit 256 --epochs 1 --no-save   # 冒烟测试 (约 20 秒)
python -m src.train_plm --model roberta-base --name roberta_base_len256 --lr 3e-5 --max-len 256 --batch-size 16   # 5 折, 约 50 分钟
python -m src.train_plm --model roberta-base --name roberta_base_len256 --lr 3e-5 --max-len 256 --batch-size 16 --resume   # 中断后续跑
python -m scripts.plm_sweep                                                         # distilbert lr 扫参 (folds 0,1) → 类权重消融 → roberta 5 折
python -m tests.test_train_plm                                                      # 微调脚本的 CPU 单元测试
```

`--folds 0,1` 或 `--limit` 的运行是部分运行：必须加 `--no-save`（不写任何文件）或 `--allow-partial`（run_id 带 `_f0-1` / `_n64` 后缀，不会覆盖完整运行）。

消融示例：`--no-negation`、`--class-weight none`、`--C 1`、`--ngram-max 1`、`--min-df 1`（`--help` 查看全部参数）。

## 评估协议（所有模型共用）

1. 训练集按文本去重：7,401 → 7,185 行（重复文本之间标签无冲突）。
2. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，划分保存在 `data/folds.csv`，以 `sha1(text)` 为键，所有模型使用同一组折。
3. 所有可学习的预处理（TF-IDF 词表/IDF、词向量词表、tokenizer 统计）只在训练折上拟合；测试集文本不参与任何训练、阈值选择或早停。
4. 指标：accuracy、macro-F1、F1(neg)、precision/recall(neg)、ROC-AUC、混淆矩阵；报 5 折 mean ± std（总体标准差，ddof=0）。全预测正类的 accuracy 是 0.854，单看 accuracy 没有意义。
5. 决策阈值在全部 OOF 概率上选：macro-F1 最大且 accuracy 不低于最优值 0.003。在同一 OOF 上评估该阈值是样本内估计（summary 里的 `oof_at_threshold`）；报告阈值效果时用嵌套估计 `oof_at_threshold_nested`（在其余 4 折的 OOF 上选阈值、在第 5 折上打分、汇总）。
6. 最终预测 = 5 个折模型测试概率的平均 + OOF 阈值；不做全量重训。多模型集成时阈值在成员的平均 OOF 上重新选（`make_submission.py` 自动处理）。
7. `results/runs.csv` 每个 run_id 只保留一行（重跑同名实验会替换而不是追加）。
8. 预训练模型微调（`train_plm.py`）每半个 epoch 在留出折上评估一次，保留 macro-F1 最高的 checkpoint，用它产生该折的 OOF 概率和测试概率。这个 checkpoint 是在同一个留出折上挑出来的，所以 PLM 行的 CV 指标比 TF-IDF 行（没有任何选择）略偏乐观；每次运行同时保存最后一个 checkpoint 的无选择估计（`results/oof/<run_id>_last.npy`，summary 里的 `cv_mean_std_last`，runs.csv 里的 `acc_mean_last` / `macro_f1_mean_last`），报告里两者都要给。
9. 类权重用 `sqrt`（按频率平方根的倒数，正负约 1 : 2.4），比完全反频率温和；超长文本用 head+tail 截断（前 75% + 后 25% 的 token 预算）。

`results/runs.csv` 列口径：`train_s_total` = 产生 5 个折模型所需的全部时间（TF-IDF：向量化 + 拟合；PLM：分词 + 训练步 + 用于选 checkpoint 的留出折评估），`infer_test_s` = 用一个模型对 1851 条测试评论做特征构建 + 预测的时间，`feature_dim` = TF-IDF 词表大小 / PLM 隐层维度，`n_params` = 可训练参数量，`device` = 设备/精度（有 fp32 回退的折时标 mixed），`device_mem_mb` = MPS/CUDA 峰值分配，`partial` = 是否只跑了部分折或子样本，`selection` = checkpoint 选择方式（none / best_val_macro_f1），`*_insample` / `*_nested` = 阈值效果的样本内 / 嵌套估计。

## 当前结果（折均值 ± 总体标准差, 去重后, 阈值 0.5）

| run_id | 配置 | accuracy | macro-F1 | F1(neg) | ROC-AUC | 最后 checkpoint acc / macro-F1 | 嵌套阈值 acc / macro-F1 |
|---|---|---|---|---|---|---|---|
| 全预测 1 | — | 0.854 | 0.461 | 0 | — | — | — |
| tfidf_lr_notebook | TF-IDF unigram, C=1, 无类权重, 无否定标记 (示例 notebook 设置) | 0.886 ± 0.003 | 0.663 ± 0.010 | 0.389 | 0.939 | — | 0.914 / 0.820 |
| tfidf_lr_noneg | TF-IDF 1-2gram, C=4, balanced | 0.922 ± 0.002 | 0.846 ± 0.005 | 0.737 | 0.950 | — | 0.922 / 0.842 |
| tfidf_lr_neg | TF-IDF 1-2gram + NOT_ 否定标记, C=4, balanced | 0.927 ± 0.001 | 0.854 ± 0.001 | 0.752 | 0.954 | — | 0.925 / 0.851 |
| distilbert_len128_lr2e-05_f0-1 | distilbert-base-uncased, len 128, bs 32, lr 2e-5, sqrt 类权重 (折 0,1) | 0.942 ± 0.001 | 0.884 ± 0.002 | 0.801 | 0.970 | 0.940 / 0.878 | 0.943 / 0.885 |
| distilbert_len128_lr3e-05_f0-1 | distilbert, lr 3e-5 (折 0,1) | 0.945 ± 0.000 | 0.887 ± 0.001 | 0.807 | 0.972 | 0.945 / 0.887 | 0.945 / 0.887 |
| distilbert_len128_lr5e-05_f0-1 | distilbert, lr 5e-5 (折 0,1) | 0.946 ± 0.001 | 0.893 ± 0.003 | 0.818 | 0.970 | 0.946 / 0.891 | 0.949 / 0.895 |
| distilbert_len128_lr5e-05_cwnone_f0-1 | distilbert, lr 5e-5, 无类权重 (折 0,1) | 0.948 ± 0.001 | 0.892 ± 0.002 | 0.815 | 0.969 | 0.943 / 0.882 | 0.942 / 0.884 |
| roberta_base_len256 | **roberta-base, len 256 head+tail, bs 16, lr 3e-5, sqrt 类权重 (5 折)** | **0.962 ± 0.003** | **0.923 ± 0.006** | 0.868 | 0.980 | 0.961 / 0.921 | 0.962 / 0.922 |

- distilbert 行是扫参用的部分运行（只跑折 0、1，run_id 带 `_f0-1`），用来选学习率和检验类权重，不与 5 折行直接比较。lr 2e-5 / 3e-5 / 5e-5 的 macro-F1 差距在折间方差量级内；去掉类权重后 accuracy 略高、macro-F1 持平，最终保留 sqrt 类权重。扫参决策记录在 `results/summary/plm_sweep.json`。
- roberta-base 的学习率按规则 min(最优 distilbert lr, 3e-5) = 3e-5（roberta-base 在 5e-5 下微调不稳定）。5 折 OOF：acc 0.9617，macro-F1 0.9227，F1(neg) 0.8679，AUC 0.9779；最后 checkpoint 口径 0.9612 / 0.9210，嵌套阈值 0.9616 / 0.9224，三者几乎一致，说明在留出折上选 checkpoint 带来的乐观偏差很小（≤ 0.2 pt）。
- 资源（Apple M5, MPS fp16）：roberta-base 1.25 亿参数，每折训练约 11 分钟（3 epoch），5 折共 55 分钟训练步 + 1.4 分钟评估，MPS 峰值 9.4 GB（padding 对齐 64 并定期清缓存后约 5.3 GB），对 1851 条测试评论推理 3.6 秒；TF-IDF + LR 3.9 万维特征、训练 1.2 秒、推理 0.04 秒、内存 0.3 GB。
- 内存注意：16 GB 机器上 PyTorch 默认允许 MPS 占到推荐值的 1.7 倍（约 21 GB），roberta len 256 会把机器拖进换页；`train_plm.py` 已把高/低水位设为 0.85 / 0.3 并把 batch padding 对齐到 64 的倍数。

当前 `submissions/submission.csv` 仍由 `tfidf_lr_neg_seed42` 在 OOF 阈值 0.48 下生成（测试正类比例 0.854）。roberta 的测试概率已在 `results/test_probs/roberta_base_len256_seed42.npy`，与当前提交的预测一致率 0.943（约 105 条测试评论的预测不同）；切换到 roberta 提交：

```bash
python -m src.make_submission --summary results/summary/roberta_base_len256_seed42.json --compare submissions/submission.csv
```

## 数据观察

- 正面评论占 85.38%；训练集内 216 行重复；87 条文本同时出现在训练集与测试集（test 101 行，train 136 行；按文本 71 正 / 16 负）。对重叠样本不做任何特殊处理，预测全部由模型生成。
- 约 76% 的评论末尾字符被截断（如 "quick shippin"），训练集和测试集同样受影响，不做修复。
- 约 1-2% 的训练标签与文本明显不符（标签来自星级），可用于错例分析与噪声标签讨论。
- 训练集与测试集同分布（长度、词频一致，token 级 OOV 仅 1.8%）。

## 后续实验

1. GloVe + BiLSTM / TextCNN（满足题目对 RNN 的鼓励），纳入资源-精度对照表。
2. roberta-base 多 seed（`--seed 2024 --seed 7`）取均值 ± 标准差；bert-base-uncased 一组对照；roberta 冻结特征 + LR（同一 PLM"特征提取 vs 微调"）。
3. 可选：deberta-v3-base（MPS 上 loss 会 NaN，放 Colab）、Qwen 冻结表示 + 分类器、Qwen + LoRA，作为对照表的补充行。

## 课程提交

按课程说明，三人一组提交报告及 `submission.csv`，报告注明成员分工并引用使用的资料。截止时间：**2026 年 11 月 13 日 23:59**；详细要求见 [课程说明](docs/assignment/EE6483-Project1.pdf)。
