# 实验结果与评估口径

本页汇总截至 **2026-10-23** 已保存在项目中的实验结果，作为报告结果表、消融分析和资源对照的依据。所有数字来自 [runs.csv](../results/runs.csv)、[summary JSON](../results/summary/) 或原始数据统计；未完成实验的安排见 [项目建议](project_advice.md)，复现与提交命令见 [README](../README.md)。

当前共 16 个完整五折运行（其中 7 个在 Google Colab 的 L4 GPU、2 个在 A100 上完成）和 9 个两折运行；除 RoBERTa-base 另有 seed 2024、7 两次重复外，训练 seed 均为 42。核心结论是：RoBERTa 提升负类识别，TF-IDF 提供低计算成本参照，GloVe 初始化改善 BiLSTM 的分类表现。

## 主要结果

所有行使用同一组去重后的数据划分；模型行均为完整 5 折、阈值 0.5 的结果。accuracy 和 macro-F1 报告折均值 ± 总体标准差（`ddof=0`），负类 F1 和 ROC-AUC 报告折均值。神经模型使用留出折选择 checkpoint，具体口径见下文“评估协议”。

| 模型 / 实验名 | accuracy | macro-F1 | F1(neg) | ROC-AUC | 比较目的 |
|---|---|---|---|---|---|
| 全预测正类 | 0.854 | 0.461 | 0.000 | — | 类别先验参照 |
| TF-IDF unigram + LR (`tfidf_lr_notebook`) | 0.886 ± 0.003 | 0.663 ± 0.010 | 0.389 | 0.939 | 课程示例配置 |
| TF-IDF 1–2gram + LR (`tfidf_lr_noneg`) | 0.922 ± 0.002 | 0.846 ± 0.005 | 0.737 | 0.950 | 否定标记消融 |
| TF-IDF 1–2gram + NOT_ + LR (`tfidf_lr_neg`) | 0.927 ± 0.001 | 0.854 ± 0.001 | 0.752 | 0.954 | 低成本基线 |
| 随机词向量 + BiLSTM (`rand_bilstm`) | 0.910 ± 0.007 | 0.824 ± 0.010 | 0.701 | 0.932 | 词向量初始化消融 |
| GloVe + TextCNN (`glove_textcnn`) | 0.922 ± 0.007 | 0.842 ± 0.010 | 0.730 | 0.945 | 局部上下文表示 |
| GloVe + BiLSTM (`glove_bilstm`) | 0.930 ± 0.007 | 0.860 ± 0.013 | 0.761 | 0.953 | 序列表示 |
| RoBERTa-base 冻结 mean 池化 + LR (`roberta_base_frozen_mean_lr`) | 0.921 ± 0.005 | 0.848 ± 0.008 | 0.743 | 0.953 | 同一模型：特征提取 vs 微调 |
| Qwen2.5-1.5B 冻结末 token + LR (`qwen2.5_1.5b_frozen_last_lr`) | 0.911 ± 0.006 | 0.827 ± 0.014 | 0.705 | 0.915 | 解码器 LLM 的冻结表示 |
| RoBERTa-base (`roberta_base_len256`) | 0.962 ± 0.003 | 0.923 ± 0.006 | 0.868 | 0.980 | 本机 MPS 可完成的全参数微调主模型 |
| DeBERTa-v3-base (`deberta_v3_base_len256`) | 0.967 ± 0.002 | 0.934 ± 0.004 | 0.887 | 0.985 | 更强的编码器，需 bf16（Colab L4） |
| Qwen2.5-0.5B + LoRA r=16 (`qwen2.5_0.5b_lora16_len256`) | 0.967 ± 0.002 | 0.932 ± 0.004 | 0.884 | 0.981 | 解码器 LLM 的参数高效微调（可训练参数 0.44%） |
| **Qwen2.5-1.5B + LoRA r=16 (`qwen2.5_1.5b_lora16_len256`)** | **0.970 ± 0.002** | **0.940 ± 0.005** | **0.897** | **0.984** | **当前完整实验中的最高分类指标** |
| Qwen3.5-2B-Base + LoRA r=16 (`qwen3.5_2b_base_lora16_len256`) | 0.971 ± 0.003 | 0.942 ± 0.005 | 0.902 | 0.981 | 最新架构（qwen3_5）的最小 Base 模型微调 |
| Qwen3.8-27B 零样本, 4-bit, 阈值 0.5 (`qwen3.8_27b_zeroshot_4bit`) | 0.968 ± 0.003 | 0.940 ± 0.005 | 0.898 | 0.988 | 不训练的 LLM 对照；阈值经训练标签校准后 acc 0.976 / macro-F1 0.953 |

这组结果支持三个结论：

- **上下文预训练模型的主要收益体现在负类识别。** RoBERTa 相对 TF-IDF 的负类 F1 增益为 11.60 个百分点，同时保持 0.962 的整体 accuracy，契合正类占多数时对少数类识别的需求。
- **预训练词向量为序列模型提供有效起点。** 在相同词表规则和 BiLSTM 架构下，GloVe 初始化的 macro-F1 为 0.860，随机初始化为 0.824，相差 3.64 个百分点。
- **简单的文本特征也有明确价值。** 在相同 TF-IDF 与 LR 配置下，增加 NOT_ 否定标记使 macro-F1 从 0.846 提升至 0.854；整个基线保留秒级训练和推理优势。

- **更大的预训练模型继续提升负类识别，但边际收益递减。** 在同一划分上，DeBERTa-v3-base 与 Qwen2.5-0.5B + LoRA 的 macro-F1 分别为 0.934 和 0.932，Qwen2.5-1.5B + LoRA 为 0.940（负类 F1 0.897），比 RoBERTa-base 高 1.7 个百分点，而可训练参数只有 436 万（LoRA 权重 + 分类头）。
- **微调而非特征提取是收益的来源。** 同一个 RoBERTa-base 冻结后接逻辑回归的 macro-F1 为 0.848，比全参数微调低 7.5 个百分点；Qwen2.5-1.5B 的冻结末 token 表示（0.827）甚至低于 TF-IDF 基线。
- **RoBERTa-base 结果对随机种子与硬件稳定。** seed 42（本机 MPS，fp16）与 seed 2024、7（Colab L4，bf16）的 macro-F1 为 0.923, 0.923, 0.924，均值 0.923、标准差 0.001；accuracy 均值 0.962、标准差 0.0004。

- **不训练的 27B LLM 零样本分类已与微调的小模型持平。** Qwen3.8-27B（4-bit，thinking 关闭，只读取标签词 logits）在阈值 0.5 下 acc 0.968、macro-F1 0.940、负类召回 0.961、负类精确率 0.843（略偏向判负）；仅用训练标签把阈值校准到 0.17 后 acc 0.976、macro-F1 0.953，嵌套估计 0.975 / 0.951，高于所有微调模型；AUC 0.988。代价是单条推理开销最大（见计算成本）。
- **零样本 LLM 揭示了标签噪声。** 有 92 条训练评论（1.28%）LLM 以 ≥ 95% 置信度给出与标签相反的类别，其中 57 条上 TF-IDF、RoBERTa 与 Qwen2.5-1.5B LoRA 的 OOF 预测全部与 LLM 一致而非与标签一致；名单在 `results/analysis/label_noise_candidates.csv`（`python -m scripts.label_noise_candidates` 重新生成），可直接用于错例分析与噪声标注讨论。
- **同架构最小 Base 模型 Qwen3.5-2B 的 LoRA 微调与 Qwen2.5-1.5B 相当**（macro-F1 0.942 对 0.940），说明在这个 7k 条数据集上更新的预训练并未带来额外收益。

DistilBERT 学习率、类权重及 GloVe 超参数扫描使用折 0、1，作为配置选择的补充证据；明细保存在 [runs.csv](../results/runs.csv)，PLM 选择规则见 [plm_sweep.json](../results/summary/plm_sweep.json)。两折结果单独分析，主表统一使用完整五折结果。

### 两折配置比较

以下实验仅使用共享划分的折 0、1，全部为 `seed=42`、阈值 0.5。表中“默认配置”的数字从对应完整运行的这两折提取，用于同折比较；各配置完整记录见 [结果摘要目录](../results/summary/)。

| 实验名 | 配置变化 | accuracy | macro-F1 | F1(neg) |
|---|---|---|---|---|
| `distilbert_len128_lr2e-05_f0-1` | DistilBERT，lr=2e-5 | 0.9419 | 0.8837 | 0.8014 |
| `distilbert_len128_lr3e-05_f0-1` | DistilBERT，lr=3e-5 | 0.9447 | 0.8872 | 0.8067 |
| `distilbert_len128_lr5e-05_f0-1` | DistilBERT，lr=5e-5 | 0.9464 | 0.8931 | 0.8175 |
| `distilbert_len128_lr5e-05_cwnone_f0-1` | DistilBERT，lr=5e-5，无类权重 | 0.9478 | 0.8922 | 0.8148 |
| `glove_bilstm` | BiLSTM 默认配置（截取折 0、1） | 0.9252 | 0.8532 | 0.7504 |
| `glove_bilstm_lr5e-4_f0-1` | BiLSTM，lr=5e-4 | 0.9210 | 0.8455 | 0.7375 |
| `glove_bilstm_lr2e-3_f0-1` | BiLSTM，lr=2e-3 | 0.9248 | 0.8475 | 0.7390 |
| `glove_bilstm_h128_f0-1` | BiLSTM，hidden=128 | 0.9210 | 0.8372 | 0.7205 |
| `glove_bilstm_frozen_f0-1` | BiLSTM，全程冻结词向量 | 0.9259 | 0.8512 | 0.7457 |
| `glove_textcnn` | TextCNN 默认配置（截取折 0、1） | 0.9154 | 0.8358 | 0.7215 |
| `glove_textcnn_lr5e-4_f0-1` | TextCNN，lr=5e-4 | 0.9217 | 0.8436 | 0.7332 |

在这两折上，DistilBERT 的 `lr=5e-5` 配置取得最高 macro-F1（0.8931）；同学习率下取消类权重的 macro-F1 为 0.8922。扫参脚本据此保留 `sqrt` 类权重，并按 `min(选定学习率, 3e-5)` 设定 RoBERTa 学习率。GloVe BiLSTM 的默认配置 macro-F1 为 0.8532，全程冻结词向量为 0.8512；TextCNN 的 `lr=5e-4` 为 0.8436，默认配置为 0.8358。这些数字用于确定后续验证候选，完整五折结果承担主结论。

### 计算成本

下表取自已保存的实验摘要和 `runs.csv`，运行平台为 Apple M5 / 16 GB。训练列累计五折的计时项，推理列为单个折模型处理全部 1,851 条测试评论的平均时间；最终提交使用五折模型的概率平均。

| 模型 | 设备 / 精度 | 训练计时（5 折） | 推理计时（单模型） | 进程峰值 RSS | MPS 折末分配量（最大值） |
|---|---|---|---|---|---|
| TF-IDF + NOT_ + LR | CPU | 1.14 s | 0.039 s | 313 MB | — |
| GloVe + TextCNN | MPS / fp32 | 104.7 s | 0.150 s | 1,421 MB | 1,112 MB |
| GloVe + BiLSTM | MPS / fp32 | 215.1 s | 0.364 s | 1,330 MB | 4,779 MB |
| RoBERTa-base | MPS / fp16 | 3,399.9 s（56.7 min） | 3.558 s | 800 MB | 9,350 MB |

Colab L4（24 GB，bf16）上的运行与上表不在同一硬件，耗时不可直接与 MPS 比较；表中可训练参数为 `n_params`（LoRA 行只含 LoRA 权重与分类头），显存为 `torch.cuda.max_memory_allocated` 的最大值。L4 队列约 2.5 小时、约 4.4 个计算单元；A100 上的 Qwen3.8 零样本（含 55.6 GB 下载与量化）与 Qwen3.5-2B 微调约 1.2 小时。Qwen3.5-2B 的显存偏高是因为 VM 没有线性注意力的 fla / causal-conv1d kernel，回退实现更吃内存。

| 模型 | 设备 / 精度 | 训练计时（5 折） | 推理计时（单模型） | 可训练参数 | GPU 峰值显存 |
|---|---|---|---|---|---|
| RoBERTa-base 冻结 + LR | Colab L4 / fp32 | 28.8 s（0.5 min） | 4.46 s | 0（LR 另有数千） | 1,118 MB |
| Qwen2.5-1.5B 冻结 + LR | Colab L4 / bf16 | 58.8 s（1.0 min） | 12.62 s | 0（LR 另有数千） | 3,927 MB |
| RoBERTa-base（seed 2024） | Colab L4 / bf16 | 638.8 s（10.6 min） | 1.53 s | 124,647,170 | 3,452 MB |
| DeBERTa-v3-base | Colab L4 / bf16 | 1,291.3 s（21.5 min） | 3.04 s | 184,423,682 | 6,994 MB |
| Qwen2.5-0.5B + LoRA | Colab L4 / bf16 | 1,395.4 s（23.3 min） | 5.41 s | 2,164,480 | 6,562 MB |
| Qwen2.5-1.5B + LoRA（梯度检查点） | Colab L4 / bf16 | 4,820.7 s（80.3 min） | 14.06 s | 4,361,216 | 4,148 MB |
| Qwen3.5-2B-Base + LoRA | Colab A100 40GB / bf16 | 2,640.8 s（44.0 min） | 8.82 s | 6,197,248 | 27,591 MB |
| Qwen3.8-27B 零样本 | Colab A100 40GB / 4-bit nf4 | 0（无训练；加载 1.6 min） | 121.8 s（66 s / 1000 条） | 0 | 23,320 MB |

TF-IDF 训练计时包含向量化与拟合，另有约 0.16 秒文本预处理；神经模型包含训练文本分词、训练步和留出折评估，GloVe 模型另含每折词表与嵌入矩阵构建。GloVe 加载约 35 秒，单独记录在 `glove_load_s`。推理计时包含对应的特征构建和预测，TF-IDF 的否定预处理单独计时。RSS 为进程峰值；MPS 数值取各折结束时驱动分配量的最大值，两者分列展示。恢复运行的 `wall_s` 只记录该次进程耗时，五折成本使用累计训练计时。

### Checkpoint 与阈值对照

| 实验名 | 最后 checkpoint accuracy / macro-F1（折均值） | 阈值交叉估计 accuracy / macro-F1（合并 OOF） |
|---|---|---|
| tfidf_lr_notebook | — | 0.914 / 0.820 |
| tfidf_lr_noneg | — | 0.922 / 0.842 |
| tfidf_lr_neg | — | 0.925 / 0.851 |
| rand_bilstm | 0.903 / 0.802 | 0.909 / 0.822 |
| glove_textcnn | 0.924 / 0.829 | 0.922 / 0.841 |
| glove_bilstm | 0.925 / 0.843 | 0.930 / 0.859 |
| roberta_base_len256 | 0.961 / 0.921 | 0.962 / 0.922 |

RoBERTa 在最优 checkpoint、最后 checkpoint 和阈值交叉估计下的 macro-F1 分别为 0.923、0.921 和 0.922，说明本次结果对这些选择较稳定。这些对照用于检查结果对 checkpoint 与阈值规则的敏感性，各自的选择流程见下文。

## 评估协议

1. **数据与划分。** 训练集按完全相同的文本去重：7,401 → 7,185 行，重复文本的标签一致。使用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，以 `sha1(text)` 为键保存至 `data/folds.csv`。所有模型及不同训练 seed 共享这一划分。
2. **特征构建。** TF-IDF 词表、IDF 和词向量模型的任务词表仅在训练折上拟合；PLM 使用随预训练模型提供的 tokenizer。测试文本仅用于预测和描述性统计，模型训练、阈值选择和 checkpoint 选择均使用训练集。
3. **指标。** 同时记录 accuracy、macro-F1、负类 precision / recall / F1、ROC-AUC 和混淆矩阵。去重后正类占 85.40%，因此用 macro-F1 和负类指标补充整体 accuracy。折间标准差反映本次运行各折得分的离散程度，多 seed 稳定性需另行评估。
4. **Checkpoint。** PLM 每半个 epoch 在留出折评估，记录 macro-F1 最优时点和固定训练轮数结束时的 OOF 预测与指标；词向量模型每个 epoch 评估，`patience=3` 早停，记录最优时点和停止时刻的 OOF 预测与指标。主表采用最优 checkpoint，`cv_mean_std_last` 提供对照。最优 checkpoint 的选择与评分使用同一留出折；词向量模型的停止时刻也由该折决定。PLM 的最后 checkpoint 消除了训练过程中的最优时点选择，但结果仍受实验配置选择影响。
5. **阈值。** 在候选阈值中，先筛选 accuracy 距最高值不超过 0.003 的候选，再选择 macro-F1 最大者。最终提交阈值使用全部 OOF（out-of-fold，折外）概率确定。`oof_at_threshold` 是同一组 OOF 上选阈值并评分的结果；`oof_at_threshold_nested` 在其他折的 OOF 上选阈值，再应用到当前折并汇总。后者对阈值选择做交叉估计，checkpoint 与超参数选择仍沿用上述流程。
6. **最终预测。** 五个折模型的测试概率取平均，再应用 OOF 阈值，不进行全量重训。多模型集成由 `make_submission.py` 对成员概率取平均，并在平均 OOF 上重新选择阈值。
7. **实验记录。** `results/runs.csv` 每个 `run_id` 保留一行，同名重跑会替换已有记录；`partial` 标识部分折或子样本实验，`selection` 记录 checkpoint 选择方式。完整配置、各折指标和资源统计保存在对应的 summary JSON 中。

## 数据与模型配置

原始训练集正类占 85.38%，去重后为 85.40%。训练集与测试集有 87 条相同文本，涉及 136 条原始训练记录和 101 条测试记录；按唯一文本计为 71 正、16 负。所有测试记录统一通过模型预测，重叠信息用于数据描述。

RoBERTa 的训练 / 测试子词长度中位数分别为 51 / 50，长度上限 256 对应的截断比例分别为 2.17% / 2.16%（见 [RoBERTa 摘要](../results/summary/roberta_base_len256_seed42.json)）。这一统计为长度预算提供依据；文本与标签不一致的样本可在人工复核后用于错例分析。

| 分支 | 当前主配置 |
|---|---|
| TF-IDF + LR | unigram + bigram；NOT_ 否定范围标记；`C=4`；`class_weight=balanced`；`min_df=2`；`max_df=0.95`；sublinear TF |
| GloVe + BiLSTM | 300 维词向量；每方向 hidden size 256；mean + max 池化；`lr=1e-3`；第 1 个 epoch 后解冻词向量 |
| GloVe + TextCNN | 300 维词向量；卷积核 {3, 4, 5}，每种 128 个 filters；`lr=1e-3` |
| RoBERTa-base | `max_len=256`；head + tail 截断（75% + 25%）；`batch_size=16`；`lr=3e-5`；3 epochs；`sqrt` 类权重 |

词向量模型采用小写和标点分词，每折任务词表仅由训练文本构建：GloVe 中已有的词出现一次即收录，其余词至少出现两次，未登录词映射到 `<unk>`。BiLSTM 在每条序列的有效长度内翻转反向输入，避免 padding 影响反向表示。神经模型的 `sqrt` 类权重按类别频率的平方根倒数设置；具体参数均可通过 `--help` 查看。

