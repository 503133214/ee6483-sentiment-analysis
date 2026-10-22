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
│   ├── make_submission.py            # 测试概率 → submission.csv (+ 自检)
│   └── check_submission.py           # 提交文件格式校验
├── results/
│   ├── runs.csv                      # 每次实验一行 (指标 + 资源)
│   ├── oof/*.npy                     # out-of-fold 概率 (与去重后的训练集顺序一致)
│   ├── test_probs/*.npy              # 测试集概率 (5 折模型平均)
│   └── summary/*.json                # 每次运行的完整摘要 (配置、各折指标、阈值)
├── tests/test_pipeline.py            # 回归测试
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

消融示例：`--no-negation`、`--class-weight none`、`--C 1`、`--ngram-max 1`、`--min-df 1`（`--help` 查看全部参数）。

## 评估协议（所有模型共用）

1. 训练集按文本去重：7,401 → 7,185 行（重复文本之间标签无冲突）。
2. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，划分保存在 `data/folds.csv`，以 `sha1(text)` 为键，所有模型使用同一组折。
3. 所有可学习的预处理（TF-IDF 词表/IDF、词向量词表、tokenizer 统计）只在训练折上拟合；测试集文本不参与任何训练、阈值选择或早停。
4. 指标：accuracy、macro-F1、F1(neg)、precision/recall(neg)、ROC-AUC、混淆矩阵；报 5 折 mean ± std（总体标准差，ddof=0）。全预测正类的 accuracy 是 0.854，单看 accuracy 没有意义。
5. 决策阈值在全部 OOF 概率上选：macro-F1 最大且 accuracy 不低于最优值 0.003。在同一 OOF 上评估该阈值是样本内估计（summary 里的 `oof_at_threshold`）；报告阈值效果时用嵌套估计 `oof_at_threshold_nested`（在其余 4 折的 OOF 上选阈值、在第 5 折上打分、汇总）。
6. 最终预测 = 5 个折模型测试概率的平均 + OOF 阈值；不做全量重训。多模型集成时阈值在成员的平均 OOF 上重新选（`make_submission.py` 自动处理）。
7. `results/runs.csv` 每个 run_id 只保留一行（重跑同名实验会替换而不是追加）。

## 当前结果（5 折均值 ± 总体标准差, 去重后, 阈值 0.5; 最后一列是嵌套阈值的诚实估计）

| 运行 | 配置 | accuracy | macro-F1 | F1(neg) | ROC-AUC | 嵌套阈值 acc / macro-F1 |
|---|---|---|---|---|---|---|
| 全预测 1 | — | 0.854 | 0.461 | 0 | — | — |
| tfidf_lr_notebook | unigram, C=1, 无类权重, 无否定标记 (示例 notebook 设置) | 0.886 | 0.663 | 0.389 | 0.939 | 0.914 / 0.820 |
| tfidf_lr_noneg | 1-2gram, C=4, balanced | 0.922 | 0.846 | 0.737 | 0.950 | 0.922 / 0.842 |
| **tfidf_lr_neg** | 1-2gram + NOT_ 否定标记, C=4, balanced | **0.927 ± 0.001** | **0.854 ± 0.001** | 0.752 | 0.954 | 0.925 / 0.851 |

当前 `submissions/submission.csv` 由 `tfidf_lr_neg_seed42` 在 OOF 阈值 0.48 下生成，测试集预测正类比例 0.854（训练先验 0.854）。

## 数据观察

- 正面评论占 85.38%；训练集内 216 行重复；87 条文本同时出现在训练集与测试集（test 101 行，train 136 行；按文本 71 正 / 16 负）。对重叠样本不做任何特殊处理，预测全部由模型生成。
- 约 76% 的评论末尾字符被截断（如 "quick shippin"），训练集和测试集同样受影响，不做修复。
- 约 1-2% 的训练标签与文本明显不符（标签来自星级），可用于错例分析与噪声标签讨论。
- 训练集与测试集同分布（长度、词频一致，token 级 OOV 仅 1.8%）。

## 后续实验

1. GloVe + BiLSTM / TextCNN（满足题目对 RNN 的鼓励）。
2. roberta-base 微调（主模型；MPS fp16 下 5 折约 23 分钟），distilbert / bert-base 对照，roberta 冻结特征 + LR。
3. 可选：deberta-v3-base（Colab）、Qwen 冻结表示 + 分类器、Qwen + LoRA，作为资源-精度对照表的补充行。

## 课程提交

按课程说明，三人一组提交报告及 `submission.csv`，报告注明成员分工并引用使用的资料。截止时间：**2026 年 11 月 13 日 23:59**；详细要求见 [课程说明](docs/assignment/EE6483-Project1.pdf)。
