# EE6483 商品评论情感分类

NTU EE6483 Artificial Intelligence and Data Mining 课程项目：根据英文商品评论预测情感，`0` 表示负面，`1` 表示正面。项目比较 TF-IDF、预训练词向量和预训练语言模型三类表示，重点考察**负面评论识别能力与计算成本的权衡**。

在去重后的 7,185 条训练评论和共享的 5 折划分上，RoBERTa-base 达到 **0.962 accuracy、0.923 macro-F1 和 0.868 负类 F1**。相较于带否定标记的 TF-IDF + Logistic Regression，macro-F1 提高 **6.83 个百分点**，负类 F1 提高 **11.60 个百分点**。TF-IDF 则以约 1.14 秒的五折向量化与拟合时间取得 0.854 macro-F1，为低成本方案提供参照。

完整指标与方法口径见 [实验结果](docs/experiment_results.md)，报告组织与后续安排见 [项目建议](docs/project_advice.md)。

## 核心结果

以下为共享五折划分、`seed=42`、阈值 0.5 下的已保存结果；accuracy 和 macro-F1 为折均值 ± 总体标准差，F1(neg) 为折均值。

| 模型 | accuracy | macro-F1 | F1(neg) |
|---|---|---|---|
| TF-IDF + NOT_ + LR | 0.927 ± 0.001 | 0.854 ± 0.001 | 0.752 |
| GloVe + BiLSTM | 0.930 ± 0.007 | 0.860 ± 0.013 | 0.761 |
| **RoBERTa-base** | **0.962 ± 0.003** | **0.923 ± 0.006** | **0.868** |

神经模型在留出折选择 checkpoint，并记录末轮预测作对照。完整五折表、两折扫参、资源计时、阈值分析和评估口径统一维护在 [实验结果](docs/experiment_results.md)。

## 文档导航

| 文档 | 内容 |
|---|---|
| [实验结果与评估口径](docs/experiment_results.md) | 完整指标、消融与扫参、资源统计、数据与模型配置 |
| [报告主线与项目计划](docs/project_advice.md) | 报告草稿、a–i 各节证据、分工和后续里程碑 |
| [课程说明](docs/assignment/EE6483-Project1.pdf) | 原始任务与评分要求 |

## 环境与运行

使用 conda 环境 `ee6483`（Python 3.13）；`requirements.txt` 固定项目已记录的依赖版本，`requirements-lock.txt` 保存完整环境快照。将环境、Hugging Face 缓存和 GloVe 放在本地非同步目录中。

```bash
conda create -y -n ee6483 python=3.13
conda activate ee6483
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ee6483 --display-name "Python (ee6483)"
python -c "import torch, transformers, sklearn; print(torch.__version__, torch.backends.mps.is_available(), transformers.__version__, sklearn.__version__)"
```

以下命令均在项目根目录运行。先建立共享划分并复现基线：

```bash
python -m src.data
python -m src.train_tfidf --name tfidf_lr_neg
python -m src.make_submission --summary results/summary/tfidf_lr_neg_seed42.json
python -m src.check_submission submissions/submission.csv
```

微调预训练语言模型（Apple Silicon 自动使用 MPS + fp16；首次运行下载模型至 Hugging Face 缓存）：

```bash
# 小规模验证训练流程
python -m src.train_plm --model distilbert-base-uncased --folds 0 --limit 256 --epochs 1 --no-save
# 完整五折 RoBERTa；已记录的累计训练计时约 57 分钟
python -m src.train_plm --model roberta-base --name roberta_base_len256 --lr 3e-5 --max-len 256 --batch-size 16
# 按相同配置恢复中断的运行
python -m src.train_plm --model roberta-base --name roberta_base_len256 --lr 3e-5 --max-len 256 --batch-size 16 --resume
# DistilBERT 两折学习率扫描、类权重消融及 RoBERTa 五折实验
python -m scripts.plm_sweep
```

训练词向量模型（首次通过 gensim 下载 `glove-wiki-gigaword-300`，约 376 MB，默认存至 `~/gensim-data`）：

```bash
python -m src.train_rnn --arch bilstm --name glove_bilstm
python -m src.train_rnn --arch textcnn --name glove_textcnn
python -m src.train_rnn --arch bilstm --name rand_bilstm --no-pretrained
bash scripts/rnn_runs.sh
```

`--folds 0,1` 或 `--limit` 表示部分运行，需搭配 `--no-save` 或 `--allow-partial`；后者使用带 `_f0-1` / `_n64` 等后缀的 `run_id` 保存产物。TF-IDF 消融参数包括 `--no-negation`、`--class-weight none`、`--C 1`、`--ngram-max 1` 和 `--min-df 1`。

运行现有回归测试：

```bash
python -m tests.test_pipeline
python -m tests.test_train_plm
python -m tests.test_train_rnn
python -m tests.test_results
```

## 提交文件

当前 `submissions/submission.csv` 的预测与 `tfidf_lr_neg_seed42` 在 OOF 阈值 0.48 下的输出一致，预测正类比例为 85.36%。RoBERTa 的五折测试概率已保存，可用以下命令生成对应的提交文件并与当前文件比较：

```bash
python -m src.make_submission --summary results/summary/roberta_base_len256_seed42.json --compare submissions/submission.csv
```

该命令校验通过后会更新 `submissions/submission.csv`；可添加 `--out submissions/roberta_submission.csv` 保存单独版本。集成示例：

```bash
python -m src.make_submission --summary results/summary/tfidf_lr_neg_seed42.json results/summary/roberta_base_len256_seed42.json --out submissions/ensemble_submission.csv
```

提交文件使用 `reviews,predictions` 两列，1,851 条记录的文本及顺序与 `test.json` 一致，预测为整数 0/1。生成脚本先写临时文件、读回校验，再更新目标文件。报告应记录最终使用的模型、seed、阈值和生成命令。

课程要求三人一组提交报告与 `submission.csv`，报告注明成员分工并引用使用的资料。截止时间为 **2026 年 11 月 13 日 23:59**，详细要求见 [课程说明](docs/assignment/EE6483-Project1.pdf)。

## 报告完善与后续实验

当前已有三类特征的完整五折结果、否定标记与词向量初始化消融，以及资源记录。后续工作优先补强核心结论：

1. **多 seed 稳定性。** 分别以 `--seed 2024`、`--seed 7` 运行 RoBERTa，报告不同训练 seed 的均值与标准差。
2. **机制与错例。** 对照 TF-IDF 与 RoBERTa 的 OOF 预测，分析否定、转折和混合情感；按课程 f) 要求补充 1–2 条测试集案例，注明人工判断依据；用固定标注规则复核疑似标签分歧。
3. **表示方式对照。** 补充冻结 RoBERTa 特征 + LR，检验同一预训练模型下任务微调的作用。其他模型扩展按报告论证需要安排。

具体章节证据、人员分工与里程碑见 [项目建议](docs/project_advice.md)。

## 目录与产物

```text
.
├── data/
│   ├── raw/train.json                # 原始带标签评论
│   ├── raw/test.json                 # 无标签测试评论
│   └── folds.csv                     # 文本哈希对应的共享五折划分
├── docs/
│   ├── assignment/EE6483-Project1.pdf # 课程说明
│   ├── examples/tfidf_features_example.html
│   ├── experiment_results.md         # 详细结果、资源统计与评估协议
│   └── project_advice.md             # 报告主线、章节证据与后续计划
├── notebooks/tfidf_features_example.ipynb
├── src/
│   ├── data.py / metrics.py          # 数据协议、指标与阈值规则
│   ├── features/negation.py          # NOT_ 否定范围标记
│   ├── train_tfidf.py                # TF-IDF + LR
│   ├── train_plm.py                  # 预训练语言模型微调
│   ├── train_rnn.py                  # GloVe + BiLSTM / TextCNN
│   ├── utils/                       # 资源测量与结果保存
│   └── make_submission.py / check_submission.py
├── scripts/                         # PLM 与词向量模型实验驱动
├── results/
│   ├── runs.csv                     # 每个 run_id 一行汇总
│   ├── oof/                         # 去重训练集顺序下的折外概率
│   ├── test_probs/                  # 各折平均后的测试概率
│   └── summary/                     # 配置、指标、阈值与训练曲线
├── tests/                           # 现有回归测试
├── submissions/                     # 课程模板与预测提交文件
├── environment.yml
├── requirements.txt / requirements-lock.txt
└── README.md
```

HTML 是课程示例的原始导出归档；交互运行使用 `notebooks/` 中的 notebook。神经模型另存 `*_last.npy` 供 checkpoint 对照，summary 中的 `artifacts` 给出每次实验的具体产物路径。
