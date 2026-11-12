# EE6483 商品评论情感分类

**语言：** [English](README.md) | **简体中文**

[GitHub 仓库](https://github.com/503133214/ee6483-sentiment-analysis) · [实验结果](docs/experiment_results.md) · [课程说明](docs/assignment/EE6483-Project1.pdf)

NTU EE6483 Artificial Intelligence and Data Mining 课程项目：根据英文商品评论预测情感，`0` 表示负面，`1` 表示正面。项目在同一组数据划分上比较五个层次的文本表示——TF-IDF、预训练词向量、编码器微调、解码器 LLM 的 LoRA 微调和 27B LLM 零样本——回答一个问题：**在正类占 85.4% 的评论数据上，负面评论识别能力随表示能力如何提升，每一级要付出多少计算成本。**

在去重后的 7,185 条训练评论和共享的 5 折划分上，LoRA 微调的小型解码器 LLM 取得全部微调模型中的最高指标：Qwen3.5-2B-Base + LoRA 达到 **0.971 accuracy、0.942 macro-F1 和 0.902 负类 F1**，Qwen2.5-1.5B + LoRA（当前提交模型）达到 0.970 / 0.940 / 0.897，可训练参数只占全模型的 0.28%。相较于带否定标记的 TF-IDF + Logistic Regression（0.854 macro-F1，五折向量化与拟合 1.14 秒），macro-F1 提高 **8.79 / 8.55 个百分点**，负类 F1 提高 **14.97 / 14.53 个百分点**。不使用任务标签的 Qwen3.8-27B 零样本参考在阈值 0.5 下达到 0.940 macro-F1；使用训练标签校准阈值后，阈值交叉拟合 macro-F1 为 **0.9509**（在用于拟合阈值的同一组 OOF 分数上为 0.9534）。各硬件上的训练、推理和内存开销见实验结果页。

完整指标与方法口径见 [实验结果](docs/experiment_results.md)，报告组织与后续安排见 [项目建议](docs/project_advice.md)。

## 核心结果

以下为共享五折划分、主运行 `seed=42`、阈值 0.5 下的开发验证结果；accuracy 和 macro-F1 为折均值 ± 总体标准差，F1(neg) 为折均值。

| 模型 | accuracy | macro-F1 | F1(neg) |
|---|---|---|---|
| TF-IDF + NOT_ + LR | 0.927 ± 0.001 | 0.854 ± 0.001 | 0.752 |
| GloVe + BiLSTM | 0.930 ± 0.007 | 0.860 ± 0.013 | 0.761 |
| RoBERTa-base | 0.962 ± 0.003 | 0.923 ± 0.006 | 0.868 |
| DeBERTa-v3-base (Colab L4, bf16) | 0.967 ± 0.002 | 0.934 ± 0.004 | 0.887 |
| Qwen2.5-0.5B + LoRA r=16 (Colab L4) | 0.967 ± 0.002 | 0.932 ± 0.004 | 0.884 |
| **Qwen2.5-1.5B + LoRA r=16 (Colab L4，提交模型)** | **0.970 ± 0.002** | **0.940 ± 0.005** | **0.897** |
| Qwen3.5-2B-Base + LoRA r=16 (Colab A100) | 0.971 ± 0.003 | 0.942 ± 0.005 | 0.902 |
| Qwen3.8-27B 零样本, 4-bit, 阈值 0.5 (Colab A100) | 0.968 ± 0.003 | 0.940 ± 0.005 | 0.898 |

RoBERTa-base 的三个训练 seed（42：本机 MPS fp16；2024、7：Colab L4 bf16）macro-F1 分别为 0.9227、0.9230、0.9243。冻结特征对照使用相同骨干和共享划分：RoBERTa-base 微调比冻结 mean 池化 + LR（0.848）高 7.49 个百分点，Qwen2.5-1.5B LoRA 比冻结末 token + LR（0.827）高 11.34 个百分点。这些结果比较的是完整特征提取与任务适配流程。

**评估口径：** 神经模型在留出折选择 checkpoint，并在同一折评分，因此这里报告开发验证指标；同时保存末轮预测作对照。阈值交叉拟合只对阈值选择留折，不是覆盖训练和超参数选择的完整嵌套交叉验证。测试集没有参考标签，不报告测试准确率。完整五折表、两折扫参、资源计时和阈值分析见 [实验结果](docs/experiment_results.md)。

## 文档导航

| 文档 | 内容 |
|---|---|
| [实验结果与评估口径](docs/experiment_results.md) | 完整指标、消融与扫参、资源统计、数据与模型配置 |
| [报告主线与项目计划](docs/project_advice.md) | 报告写作参考、a–i 各节证据和项目计划 |
| [课程说明](docs/assignment/EE6483-Project1.pdf) | 原始任务与评分要求 |

## 环境与运行

使用 conda 环境 `ee6483`（Python 3.13）。`requirements.txt` 固定已记录的基础依赖，`requirements-lock.txt` 是本地 Apple MPS 环境快照，不是 CUDA 环境锁文件。将环境、Hugging Face 缓存和 GloVe 放在本地非同步目录中。

```bash
git clone https://github.com/503133214/ee6483-sentiment-analysis.git
cd ee6483-sentiment-analysis
conda create -y -n ee6483 python=3.13
conda activate ee6483
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ee6483 --display-name "Python (ee6483)"
python -c "import torch, transformers, sklearn; print(torch.__version__, torch.backends.mps.is_available(), transformers.__version__, sklearn.__version__)"
```

LoRA 额外需要 `peft`；本地已安装版本为 0.21.2。CUDA 上的 4-bit / 8-bit 推理还需要 `bitsandbytes`，其版本应与目标 CUDA / PyTorch 环境匹配；本仓库未保存该 GPU 环境的完整版本锁。

```bash
# 运行 LoRA 实验前安装
python -m pip install peft==0.21.2
# 仅在需要量化推理的 CUDA 环境安装
python -m pip install bitsandbytes
```

### 快速开始：TF-IDF 基线

以下命令均在项目根目录运行。先建立共享划分并复现基线；基线使用独立输出文件，不会覆盖最终 LoRA 提交：

```bash
python -m src.data
python -m src.train_tfidf --name tfidf_lr_neg
python -m src.make_submission --summary results/summary/tfidf_lr_neg_seed42.json --out submissions/tfidf_submission.csv
python -m src.check_submission submissions/tfidf_submission.csv
```

### 预训练编码器

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

### 词向量模型

首次通过 gensim 下载 `glove-wiki-gigaword-300`，约 376 MB，默认存至 `~/gensim-data`：

```bash
python -m src.train_rnn --arch bilstm --name glove_bilstm
python -m src.train_rnn --arch textcnn --name glove_textcnn
python -m src.train_rnn --arch bilstm --name rand_bilstm --no-pretrained
bash scripts/rnn_runs.sh
```

### CUDA 与 Google Colab

DeBERTa-v3 使用 bf16；Qwen 用 LoRA；`train_frozen.py` 用冻结模型抽特征再接逻辑回归；`llm_zero_shot.py` 做 27B 零样本分类。在 Colab 上运行时，本机需安装并登录 `google-colab-cli`。L4 队列定义在 `scripts/colab_chain.sh`，A100 队列（Qwen3.8-27B 零样本、Qwen3.5-2B-Base LoRA）在 `scripts/colab_chain_qwen3.sh`；训练在 VM 上以后台进程运行，`colab exec` 只用于启动和轮询。

L4 工作流如下；A100 实验需将创建会话的 `--gpu` 改为 `A100`，并用下方备用队列命令替代默认启动命令。27B 量化推理还需在该会话安装 `bitsandbytes`。

```bash
colab new -s ee6483 --gpu L4                                   # L4 24 GB, 支持 bf16, 本次实测约 1.5 计算单元/小时
colab install -s ee6483 peft==0.21.2
echo 'import subprocess,sys; subprocess.run([sys.executable,"-m","pip","uninstall","-y","torchao"])' | colab exec -s ee6483   # 预装的 torchao 0.10 与 peft 冲突
tar czf project-colab.tar.gz --exclude=__pycache__ --exclude=.DS_Store src scripts data/raw/train.json data/raw/test.json data/folds.csv results/runs.csv requirements.txt
colab upload -s ee6483 project-colab.tar.gz /content/project.tar.gz
colab exec -s ee6483 -f scripts/colab_launch.py                # 解压并后台启动 scripts/colab_chain.sh（L4 队列）
colab exec -s ee6483 -f scripts/colab_poll.py                  # 查看进度、当前任务日志、已完成的 runs.csv 行
# 将 RUN_ID 替换为要取回的运行标识；其他概率与曲线文件按相同方式下载
colab download -s ee6483 /content/project/results/summary/RUN_ID.json results/summary/RUN_ID.json
colab stop -s ee6483                                           # 结束计费
```

```bash
# 备用：在已安装量化依赖的 A100 会话中启动 A100 队列，而不是默认 L4 队列
colab install -s ee6483 bitsandbytes
colab exec -s ee6483 --env CHAIN=scripts/colab_chain_qwen3.sh -f scripts/colab_launch.py
```

单个实验的本地等价命令（在有 CUDA/bf16 的机器上直接运行）：

```bash
python -m src.train_plm --model microsoft/deberta-v3-base --name deberta_v3_base_len256 --lr 2e-5 --max-len 256 --batch-size 16 --precision bf16 --eval-batch-size 32
python -m src.train_plm --model Qwen/Qwen2.5-1.5B --name qwen2.5_1.5b_lora16_len256 --lora --lora-r 16 --lora-alpha 32 --lr 2e-4 --max-len 256 --batch-size 16 --precision bf16 --load-dtype bf16 --gradient-checkpointing --eval-batch-size 32
python -m src.train_frozen --model Qwen/Qwen2.5-1.5B --pooling last --load-dtype bf16 --name qwen2.5_1.5b_frozen_last_lr
python -m src.train_frozen --model roberta-base --pooling mean --name roberta_base_frozen_mean_lr --batch-size 64
python -m src.train_plm --model Qwen/Qwen3.5-2B-Base --name qwen3.5_2b_base_lora16_len256 --lora --lora-r 16 --lora-alpha 32 --lr 2e-4 --epochs 3 --max-len 256 --batch-size 16 --precision bf16 --load-dtype bf16 --eval-batch-size 32
python -m src.llm_zero_shot --model Qwen/Qwen3.8-27B --quant 4bit --name qwen3.8_27b_zeroshot_4bit --batch-size 16   # 零样本，不训练；阈值 0.5 为纯零样本点
python -m scripts.label_noise_candidates   # 零样本 LLM 以 ≥ 95% 置信度反对训练标签的 92 条名单，并记录三个监督模型 OOF 中与 LLM 一致的个数（57 条全部一致）
```

`--folds 0,1` 或 `--limit` 表示部分运行，需搭配 `--no-save` 或 `--allow-partial`；后者使用带 `_f0-1` / `_n64` 等后缀的 `run_id` 保存产物。TF-IDF 消融参数包括 `--no-negation`、`--class-weight none`、`--C 1`、`--ngram-max 1` 和 `--min-df 1`。

### 回归测试

```bash
python -m tests.test_pipeline
python -m tests.test_train_plm
python -m tests.test_train_rnn
python -m tests.test_results
```

## 提交文件

当前 `submissions/submission.csv` 来自 **Qwen2.5-1.5B + LoRA**：

- `run_id`：`qwen2.5_1.5b_lora16_len256_seed42`。
- 对五折模型的测试概率取均值，再应用 OOF 选择的阈值 **0.45**；没有额外的全量数据重训。
- 1,851 条记录，1,595 条正面、256 条负面（正类比例 86.17%）。
- 同集拟合阈值后的 OOF accuracy / macro-F1 为 0.9706 / 0.9404；阈值交叉拟合为 0.9676 / 0.9357。

仓库保存了概率和实验摘要，没有保存训练后的模型权重。以下命令从已有概率**重建 CSV，无需下载模型或重新训练**；若需从评论重新推理，应先按上方命令重新训练：

```bash
python -m src.make_submission --summary results/summary/qwen2.5_1.5b_lora16_len256_seed42.json --out submissions/reproduced_submission.csv --compare submissions/submission.csv
python -m src.check_submission submissions/reproduced_submission.csv --compare submissions/submission.csv
python -m src.make_submission --summary results/summary/roberta_base_len256_seed42.json --out submissions/roberta_submission.csv --compare submissions/submission.csv
```

**请始终显式指定 `--out`，避免覆盖最终文件。** 如确需更新最终提交，将其设为 `submissions/submission.csv`。集成示例：

```bash
python -m src.make_submission --summary results/summary/tfidf_lr_neg_seed42.json results/summary/roberta_base_len256_seed42.json --out submissions/ensemble_submission.csv
```

提交文件使用 `reviews,predictions` 两列，1,851 条记录的文本及顺序与 `test.json` 一致，预测为整数 0/1。生成脚本先写临时文件、读回校验，再更新目标文件。报告应记录最终使用的模型、seed、阈值和生成命令。

课程要求三人一组提交报告与 `submission.csv`，报告注明成员分工并引用使用的资料。截止时间为 **2026 年 11 月 13 日 23:59**，详细要求见 [课程说明](docs/assignment/EE6483-Project1.pdf)。

## 分析与报告

现有证据包括完整五折结果、否定标记和词向量初始化消融、冻结与适配流程对照、RoBERTa 三个 seed、27B 零样本参考，以及按硬件记录的资源开销。

- **负类识别：** OOF 混淆矩阵中，TF-IDF 漏判 252 条负面评论，Qwen1.5B 漏判 119 条，减少 52.8%。
- **标签分歧：** `results/analysis/label_noise_candidates.csv` 包含 92 条 27B 参考高置信度反对给定标签的训练评论，其中 57 条也得到三个监督模型 OOF 预测的一致支持。这是人工复核队列，不是已确认的错误标签或噪声率。
- **迁移与抗噪：** 无评分酒店评论和不准确评分标签的适配方案属于讨论性扩展，不作为已完成实验。

具体指标见 [实验结果](docs/experiment_results.md)，章节组织参考见 [项目建议](docs/project_advice.md)。

## 成员贡献

Men Zhaolin、Gao Zhihan、Zhou Junchong 对项目和报告作出**平等贡献（各三分之一）**。

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
│   ├── train_plm.py                  # 预训练语言模型微调（全参数或 LoRA）
│   ├── train_frozen.py               # 冻结模型特征 + 逻辑回归
│   ├── llm_zero_shot.py              # LLM 零样本分类（标签词 logits）
│   ├── train_rnn.py                  # GloVe + BiLSTM / TextCNN
│   ├── utils/                       # 资源测量与结果保存
│   └── make_submission.py / check_submission.py
├── scripts/                         # 扫参、Colab 队列（L4 / A100）与标签分歧名单脚本
├── results/
│   ├── runs.csv                     # 每个 run_id 一行汇总
│   ├── oof/                         # 去重训练集顺序下的折外概率
│   ├── test_probs/                  # 各折平均后的测试概率
│   ├── summary/                     # 配置、指标、阈值与训练曲线
│   └── analysis/                    # 标签分歧候选名单（label_noise_candidates.csv）
├── tests/                           # 现有回归测试
├── submissions/                     # 课程模板与预测提交文件
├── environment.yml
├── requirements.txt / requirements-lock.txt
├── README.md                        # English
└── README.zh-CN.md                  # 简体中文
```

HTML 是课程示例的原始导出归档；交互运行使用 `notebooks/` 中的 notebook。神经模型另存 `*_last.npy` 供 checkpoint 对照，summary 中的 `artifacts` 给出每次实验的具体产物路径。
