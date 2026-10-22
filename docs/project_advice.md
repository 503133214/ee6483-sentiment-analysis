# EE6483 Mini Project (Option 1) 项目建议

> 生成日期 2026-10-22。截止 2026-11-13 (周五 23:59), 还有 22 天。
> 文中所有标 **[实测]** 的数字都是今天在这台 Apple M5 / 16 GB 上用 `data/raw/train.json` / `test.json` 真实跑出来的; 标 **[估算]** 的是推断。

---

## 0. 一页总览: 最重要的 10 件事 (按优先级)

1. **分值结构决定策略**: 写作占 60% (a 20 + f 10 + g 10 + h 10 + i 10), 建模占 40% (b 5 + c 15 + d 5 + e 15)。文献综述和 h/i 两道开放题不依赖代码, **第一天就动笔**, 不要等模型。
2. **D1-D2 先交保底**: TF-IDF 1-2gram + 否定标记 + LogisticRegression(C=4, class_weight=balanced) **[实测]** 5 折 accuracy 0.929 / macro-F1 0.860, 几秒跑完; 用它先生成一版 `submission.csv`, 之后随时可交。
3. **环境一次定死**: 现有 conda 环境 `torch` (Python 3.10) 的 scipy 在这台 macOS 27 上 dlopen 失败, sklearn/transformers 都装不上 **[实测]**。用 base 的 Python 3.13 新建 venv 到 `~/venvs/ee6483` (不要建在 OneDrive 目录里)。torch 2.14 + transformers 5.x + sklearn 1.7 在 MPS 上实测可用。
4. **评估协议统一**: 文本去重 (7401 → 7185) → `StratifiedKFold(5, shuffle, random_state=42)` → fold id 存成文件, **所有模型共用同一组折**; 指标固定为 accuracy / macro-F1 / F1(neg) / ROC-AUC + 混淆矩阵; 每次实验追加一行 `results/runs.csv`。否则 g) 的对照表和 d) 的超参曲线都不可比。
5. **类别不平衡必须处理**: 正面 85.4%, 全预测 1 就是 85.4% accuracy; 不处理时 TF-IDF+LR 的负类 F1 只有 0.42-0.56 **[实测]**。最终只保留一个旋钮 (OOF 阈值或加权 CE, 不要叠加), 以 macro-F1 选模型。
6. **主模型 roberta-base**: MPS fp16 autocast 下 3 epoch 一折 5.9 分钟, 5 折 23 分钟, 3 seed × 5 折约 70 分钟 **[实测]**。本机完全够用。deberta-v3-base 在 MPS 上 loss 变 NaN **[实测]**, 只能放 Colab 当可选对照。
7. **RNN 一行不能少**: PDF 明确鼓励 RNN / BERT; GloVe + BiLSTM (mask 池化, 不要用 pack_padded_sequence, MPS 上慢 12 倍 **[实测]**) 和 TextCNN 各跑一组, 分钟级。README 里的 Qwen 路线放在所有必做项之后, 只作为 g) 表里 "解码器 LLM" 的可选一行。
8. **合规红线**: test 文本不得参与任何训练, 包括 TF-IDF fit、TAPT 继续预训练、self-training、阈值选择; 87 条 train/test 重叠文本不查表覆盖, 报告里如实说明; h) 若用外部酒店数据只能当评估集并声明不参与 submission。
9. **f) 要有无偏的 test 子集**: 从 test 随机抽 200-300 条, 三人独立标注、多数投票、报告 Cohen's kappa, 得到诚实的 test 准确率点估计; 再从 OOF 错误里挑 "做对的难例" + "做错的" 各 1 条做词级归因。
10. **11/12 提交, 留一天缓冲**; 最晚降级点 11/5: 若 roberta 仍不稳定, 用 TF-IDF 结果写 c/d/e 保证报告完整。

---

## 1. 数据实测结论 (直接可写进报告 b)

| 项目 | 数值 |
|---|---|
| train / test | 7401 / 1851 条; train 有 `reviews`+`sentiments`, test 只有 `reviews` |
| 类别 | 正 6319 (85.38%) / 负 1082 (14.62%); 全预测 1: acc 0.854, macro-F1 0.461 |
| 词数 (空格分词) | train mean 56.7 / median 42 / p95 143 / max 1066; test 几乎相同 (KS p=0.157) |
| RoBERTa 子词数 **[实测]** | median 54, p95 182; >128 token 占 11.5%, >256 占 2.3%, >512 占 0.3% |
| 负面更长 | 负 mean 62 词 vs 正 56 词; 错误率随长度上升 (<20 词 6% → >160 词 10.7%) |
| train 内重复 | 151 个文本重复共 367 行, 冗余 216 行; 重复间标签无冲突; 去重后 7185 |
| test 内重复 | 18 个文本共 39 行 |
| **train/test 重叠** | 87 个唯一文本 = test 101 行 (5.46%) = train 136 行; 按文本 71 正 / 16 负; 都是普通评论不是模板 |
| 末尾截断 | 只有 24% 的文本以标点结尾, 7.7% 末尾词疑似被截 ("quick shippin", "agai", "servic"); train/test 同样受影响, 不需要也无法修复, 但要在 b) 里写出来, 错例分析时别当成用户拼写错误 |
| 标签噪声 | OOF 高置信反向样本约 1-2%; 实锤例子: 标 0 的 "My tween daughter loves these Saucony Jazz shoes... comfortable and stylish", 标 1 的 "i regret to say that i did not receive the shoes..!!" |
| 分布偏移 | **无**: 前 2000 高频词重合 83.5%, unigram JS 散度 0.04 bit, token 级 OOV 仅 1.76% (品牌名/错字); CV 分数可直接预测 test |
| 最强判别词 | 负: not, return(ed), disappointed, cheap, poor, refund, waste, money, sent, company; 正: great, comfortable, love, perfect, nice, well, best |
| 转折/否定 | 含 but/however 的评论 acc 0.890 vs 不含 0.948; 含否定词 0.900 vs 0.954 → BoW 的两大失败源, 正是 RNN/BERT 的卖点 |

要点: 负面信号大量是 "退货 / 退款 / 客服 / 物流" 的事务性词, 这是电商服装类评论的领域特点, 也是 h) 题 (迁移到酒店评论) 论证 "词级模型会明显掉分" 的数据依据。

---

## 2. 评估与复现协议 (所有模型共用)

```
读 json → 记录重复/重叠统计 → 按 reviews 去重 (7401 → 7185)
→ StratifiedKFold(n_splits=5, shuffle=True, random_state=42), fold id 存 data/folds.csv
→ 每种特征各自 tokenize (TF-IDF / GloVe / PLM), 所有 fit 只在训练折
→ 指标: accuracy, macro-F1, F1(neg), precision/recall(neg), ROC-AUC, 混淆矩阵
→ 保存每折 OOF 概率 results/oof/{model}_{seed}.npy 和 test 概率 results/test_probs/
→ 主模型 3 seed (42, 2024, 7) × 5 折, 报 mean ± std
→ 最终预测 = 15 个折模型的 test 概率平均 (软投票) + OOF 上选的阈值; 不做全量重训
```

- 去重后 CV 会比不去重低约 0.3 pt (0.923 → 0.920) **[实测]**, 那是去掉折间泄漏后的真实值, 报告用去重后的数字。
- **不要从 train 删掉与 test 重叠的 87 条** (它们是合法训练数据), 也不要用 train 标签覆盖 test 预测。全量 LR 对这 101 行预测与 train 标签一致 99/101, 差异最多 2 行。
- 固定种子: `random / numpy / torch.manual_seed / torch.mps.manual_seed / PYTHONHASHSEED / DataLoader generator + worker_init_fn`。MPS 和 CUDA 的部分算子不保证逐位确定, 报告里写 "固定种子 + 3 seed 平均" 而不是声称完全确定。
- 阈值规则 (评分口径未知时的折中): 在 OOF 上取 "macro-F1 最大且 accuracy 不低于最大值 0.3 pt" 的阈值。实测 acc 最优阈值 0.4、macro-F1 最优 0.5, 这条曲线放进 d)。
- `make_submission.py` 两个模式: `--from-probs` (从版本库里的 .npy 逐字节复现, 评分人可直接跑) 和 `--retrain` (约 70 min, 复现到 ±std)。.npy 和 fold id 一起提交。

---

## 3. 建模路线

### 3.1 模型梯队

| 层 | 特征 / 模型 | 作用 | 5 折 acc / macro-F1 |
|---|---|---|---|
| A | TF-IDF 1-2gram + NOT_ 否定标记 + LR(C=4, balanced) 或 LinearSVC | baseline; g) 的便宜端; 错例分析参照 | **[实测]** 0.929 / 0.860; LinearSVC 0.927 / 0.852 |
| B | GloVe 300d + BiLSTM (mask mean+max 池化) / TextCNN | 满足 "RNN" 要求; c) 架构图; g) 中间档 | **[估算]** 0.92-0.93 / 0.84-0.86 |
| C | **roberta-base 微调 (主模型)**; distilbert 扫参 + 中间档; bert-base-uncased 一组对照 (PDF 引用 [4]) | 最终提交; g) 的贵端 | **[估算]** 0.935-0.95 / 0.88-0.90 |
| C' | roberta-base 冻结, [CLS]/mean-pool 特征 + LR | 同一 PLM "特征提取 vs 微调", g) 最干净的对比 | — |
| D (可选) | 15 模型软投票集成; deberta-v3-base (Colab); Qwen2.5-0.5B/1.5B + LoRA; LLM zero-shot 对照 | 加分项, 全部必做项完成后再碰 | — |

合理预期: 本数据有 1-2% 明显标签噪声且负类只有 15%, roberta 合理目标是 acc 0.935-0.95 / macro-F1 0.88-0.90。**不要因为没到 95% 就反复调参或怀疑实现有误**; 若低于 0.93 多半是 lr / 不平衡 / 阈值没处理好。

### 3.2 训练时间与内存 **[实测, M5 MPS, pad 到 max_len, 不含评估]**

| 模型 | 精度 | max_len/bs | 参数 | min/epoch | 5 折 × 3 ep | 峰值内存 | 推理 1851 条 |
|---|---|---|---|---|---|---|---|
| distilbert-base-uncased | fp32 | 128/32 | 67M | 1.8 | 22 min | 3.3 GB | 7.5 s |
| roberta-base | fp32 | 128/32 | 125M | 3.6 | 44 min | 6.9 GB | 13.7 s |
| **roberta-base** | **fp16 autocast** | 128/32 | 125M | 2.0 | **23 min** | 6.5 GB | 5.5 s |
| roberta-base | fp32 | 256/16 | 125M | 7.9 | 94 min | 7.8 GB | 29 s |
| bert-base-uncased | fp32 | 128/32 | 110M | 3.6 | 43 min | 6.8 GB | 13.4 s |
| deberta-v3-base | fp32 | 128/32 | 184M | — | **loss=NaN, MPS 不可用** | 7.6 GB | — |
| BiLSTM 300/256 (mask 池化) | fp32 | 128/64 | 3.5M | 0.075 | 5 折 × 10 ep 3 min | <1 GB | 秒级 |
| BiLSTM (pack_padded_sequence) | fp32 | 128/64 | 3.5M | 0.93 | 37 min | — | — |
| TextCNN | fp32 | 128/64 | 2.8M | 0.03 | 1 min | — | — |
| TF-IDF uni+bi + LR | CPU | — | 38k 维 | 0.41 s fit + 1.8 s 训练 | 秒级 | 278 MB | 毫秒级 |

Colab 免费 T4 **[估算]**: roberta fp16 len128 约 1.1 min/epoch, 5 折 15-20 min; 只在跑 deberta-v3 或赶时间时用, 每折结束立刻把概率存到 Drive。

### 3.3 超参数 (对应报告 d)

**max_len**: 扫参用 128 (覆盖 88.5% 子词长度); 最终用 256 + 动态 padding (按 batch 内最长 pad, 中位数只有 54 所以额外开销小) + 对超长样本 head+tail 截断 (前 ~75% + 后 ~25%, 因为结论常在结尾, 引 Sun et al. 2019)。d) 里用**子词**统计论证, 不要用空格词数 (93.6% 那个数字是词数口径)。

**A 层**: `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w[\w']*\b")` (默认 pattern 会把 don't 切成 don); 否定标记: not/no/never/n't/cannot/without 之后到下一个标点的词加 `NOT_` 前缀 (+1.2 pt macro-F1, 零成本, 引 Pang et al. 2002)。LR C ∈ {0.5,1,2,4,8}, LinearSVC C ∈ {0.1,0.25,0.5,1}, `scoring='f1_macro'`。实测 C=1 时 uni+bi 反而比 unigram 差 (正则太强退化到多数类), C=4-10 才正常 — d) 可用这组数字讲超参交互。不去停用词 (not/but/the/was 都是强判别词), 不词干化, 不拼写纠正。

**B 层**: 小写 + 保留标点 token, 与 GloVe 6B 分词风格一致; 词表 min_count=2 (约 7.5k 词), 记录 GloVe 覆盖率; 词级 max_len 160; emb 先冻结 1-2 epoch 再解冻 (或 emb lr 降 10 倍); hidden {128, 256}, layers {1, 2}, dropout {0.3, 0.5}; Adam lr {5e-4, 1e-3, 2e-3}; 梯度裁剪 1.0; ≤15 epoch, 按验证 macro-F1 早停 (patience 3); 加权 CE。不做否定标记 (会让词全部 OOV)。

**C 层**:

| 超参 | 初始 | 范围 | 理由 |
|---|---|---|---|
| lr | 2e-5 | {1e-5, 2e-5, 3e-5, 5e-5} | BERT/RoBERTa 标准范围; 小数据偏小更稳 (Mosbach 2021) |
| batch | 32 (fp16) / 16 (len256) | {16, 32} | 16 GB 下 roberta len128 bs32 占 6.5-6.9 GB |
| epochs | 3 | {2, 3, 4} | 7k × 3 ≈ 700 步, 再多过拟合; 每半 epoch 评估取最佳 |
| warmup | 10% | {6%, 10%} | 线性 warmup + 线性衰减 |
| weight decay | 0.01 | {0, 0.01, 0.1} | AdamW 默认 |
| dropout | backbone 0.1 不动, head 0.1-0.3 | — | — |
| loss | CE 或温和加权 CE (sqrt 反频率, 约 2.5:1) + label smoothing 0.05-0.1 (可选, 顺便对应 i) | — | 见第 2 节阈值规则 |
| 梯度裁剪 | 1.0 | — | — |

扫描策略: distilbert 在 5 折上扫 lr × epochs (每组约 22 min) → 把最优区间带到 roberta 细扫 lr ∈ {1e-5, 2e-5, 3e-5} (每组 fp16 约 23 min) → 全部一个下午。
给 PLM 的文本**几乎不做预处理**: 不小写 (roberta 区分大小写)、不去标点/表情、不去停用词、不做否定标记、不要先 nltk 分词再拼回去。

### 3.4 MPS 专属的坑 **[实测踩到]**

- `pack_padded_sequence` 在 MPS 上慢 12 倍 (479 vs 39 ms/step): BiLSTM 用 mask 做 mean/max 池化。
- `nn.Embedding` 索引越界在 MPS 上**不报错**, 换 CPU 立刻 IndexError: 任何新模型先在 CPU 跑 2 个 batch 做 sanity check。
- fp16 autocast 对 roberta/bert/distilbert 正常 (loss 0.295 无 NaN, 快 1.85 倍); 每次训练打印前 50 步 loss, NaN 则回退 fp32。bf16 不要用。
- deberta-v3-base fp32 在 MPS 上 23 步后 loss=NaN; 报告里只写 "我们在 MPS 上观察到 NaN, 故改在 Colab 跑", 不要写成 "已知问题"。
- `export PYTORCH_ENABLE_MPS_FALLBACK=1` 让未实现算子回退 CPU。
- 跑 roberta 时关掉浏览器等大应用; OOM 就 batch 8 + gradient accumulation。
- transformers 4.41+ 把 Trainer 的 `evaluation_strategy`/`tokenizer=` 改成 `eval_strategy`/`processing_class=`, 网上旧教程会报错; 建议自己写 100 行以内的 PyTorch 训练循环, c) 里反而更好解释 loss/训练策略, 也便于插加权 CE、head+tail 截断、多 seed。
- 只存 OOF 和 test 概率 (.npy), 不要存 15 个 roberta checkpoint (7.5 GB); 最多存最优 seed 的一折权重用于演示。

---

## 4. 环境与仓库

### 4.1 环境搭建 (推荐, 已验证可用的组合)

```bash
python3 -m venv ~/venvs/ee6483          # base 的 Python 3.13; 不要建在 OneDrive 目录里
source ~/venvs/ee6483/bin/activate
python -m pip install -U pip
python -m pip install torch             # macOS arm64 的 PyPI wheel 自带 MPS, 不需要 --index-url
python -m pip install transformers accelerate datasets scikit-learn pandas numpy scipy gensim nltk matplotlib seaborn tqdm jupyter ipykernel sentencepiece
python -m ipykernel install --user --name ee6483 --display-name "Python (ee6483)"
python -c "import torch, transformers, sklearn; print(torch.__version__, torch.backends.mps.is_available(), transformers.__version__, sklearn.__version__)"
pip freeze > requirements.txt           # 交报告时附上
```

- **不要用现有的 conda `torch` 环境**: Python 3.10 的 scipy wheel 在 macOS 27 上 `scipy.sparse/linalg/stats` 全部 dlopen 失败 (`__thread_bss ... offset field is not zero`), sklearn 装上也 import 不了 **[实测]**。
- README 里的 `python3 -m venv .venv` 会把 venv 建进 OneDrive, 建议改成上面的路径。
- `.git` 放在 OneDrive 同步目录里是已知的仓库损坏源: 建议 `git clone` 到 `~/code/ee6483` 工作, OneDrive 只放交付物 (报告、submission.csv)。HF cache (`HF_HOME`)、GloVe (`~/gensim-data`)、checkpoint 同样放本地。
- GloVe 用 `gensim.downloader.load("glove-wiki-gigaword-300")` (约 376 MB) 最省事。

### 4.2 仓库结构建议

```
ee6483-sentiment/
  README.md             # 环境安装、每个实验一条命令、如何复现 submission.csv
  requirements.txt
  configs/              # tfidf_lr.yaml, glove_bilstm.yaml, roberta.yaml (所有超参在这里)
  data/raw/             # train.json, test.json
  data/folds.csv        # 去重后的 fold id (所有模型共用)
  src/
    data.py             # 读 json、去重、切折
    features/           # tfidf.py, glove.py, tokenize.py (head+tail 截断)
    models/             # linear.py, rnn.py, cnn.py, plm.py
    train.py            # --config --seed --fold → results/runs.csv + results/oof/
    predict.py          # 集成 → 阈值 → outputs/submission.csv
    metrics.py  utils/seed.py  utils/resources.py (计时/内存)
  notebooks/            # 01_eda, 02_tfidf_baseline, 03_error_analysis, 04_resource_table
  results/              # runs.csv, oof/*.npy, test_probs/*.npy, figures/
  outputs/submission.csv
```

---

## 5. 报告结构、分工、时间线

### 5.1 报告结构 (LaTeX/Overleaf, IEEE 或 ACL 模板, 正文 20-25 页 + 附录; 章节标题直接带 a)-i))

| 章节 | 内容与图表 | 篇幅 | 分值 |
|---|---|---|---|
| 封面 + 贡献表 | 姓名/学号; **按题 a-i 列每人贡献** (PDF 明确要求) | 0.5 页 | 必需 |
| 1 引言 | 任务、数据来源 (Blitzer 2007, 内容推断为服装鞋饰类)、一句话说明 PDF 笔误的理解 (json 非 csv; test.json; (i) 的 "(g)" 指 (h)) | 0.5 页 | — |
| 2 文献综述 (a) | 2.1 问题定义 + 挑战 + **3×2 设定对照表**; 2.2 检索方法 + **20-30 篇论文表** + taxonomy 图; 2.3 两个研究组精读; 2.4 baseline 选择与改进点 | 5-6 页 | 20 |
| 3 数据与预处理 (b) | 第 1 节的数据画像表; 长度直方图 (标 128/256 线); 重复/重叠/截断说明; 三分支预处理流程图; **写明不做什么及为什么** | 1.5 页 | 5 |
| 4 模型 (c) | 每个模型: 自绘架构图 (token ids [B,256] → 12 层 → [CLS] 768 → dropout → Linear 768→2)、维度表、损失公式、训练策略; 复现协议; 代码/README 指向附录 | 3-4 页 | 15 |
| 5 超参数 (d) | 超参表 (范围/选定/理由); lr × epoch × max_len 验证曲线; C × class_weight × 阈值交互表 | 1-1.5 页 | 5 |
| 6 结果与提交 (e) | **主结果表** (所有模型 CV mean±std, **第一行放 "全预测 1 = 85.4%"**); 混淆矩阵; test 预测正负比例; submission 生成过程; 声明测试标签未知、test 文本未参与任何训练 | 1.5 页 | 15 |
| 7 错误分析 (f) | OOF 混淆矩阵; 错误类型占比表; 人工标注 test 子集的结果 + kappa; 1-2 案例 + 词级归因 (LR 系数 / Captum IG) | 1.5-2 页 | 10 |
| 8 特征格式对比 (g) | 资源-精度对照表 (写明计时条件); 训练时间(log) vs macro-F1 散点图; 场景化建议 | 1.5 页 | 10 |
| 9 无标签酒店评论 (h) | 见 6.8 | 1.5-2 页 | 10 |
| 10 噪声标签 (i) | 见 6.9 | 1.5-2 页 | 10 |
| 11 结论 | 3-5 句 | 0.3 页 | — |
| 参考文献 | IEEE 编号, 正文每个论断带标号, 含 PDF 给的 5 条 | 1-2 页 | 必需 |
| 附录 | A 代码 + README; B 完整超参网格; C 额外样本; D GenAI 使用声明 (按课程规定) | 不限 | — |

### 5.2 三人分工 (按题分, 写作 60% 必须有人全职写)

| 成员 | 负责 | 具体 |
|---|---|---|
| A 建模与工程 | b, c, d, e (40%) | 环境、数据模块与 fold 文件、TF-IDF baseline + 保底 submission (D2)、roberta 主模型、3 seed 协议、`make_submission.py`、README |
| B 实验与分析 | f, g, d 的实验部分 | GloVe + BiLSTM/TextCNN、冻结 roberta 特征 + LR、资源测量脚本、超参曲线、OOF 错例分析、test 子集标注组织、所有图表 |
| C 综述与开放题 + 整合 | a, h, i (40%) | 文献检索 (Zotero)、a(1)-(4)、h/i 撰写、噪声注入小实验、参考文献核对、报告模板与整合、贡献表 |

补充: 30 篇 ★ 快读论文三人各读 10 篇, 填共享阅读表 (论文 / 会议年 / 谁读 / 一句话 / 引用在哪节), 既分摊工作也是贡献证明。
依赖: f 等 A 的主模型冻结 (D12), 但 B 可先用 baseline OOF 搭流程; g 要求三条流水线同一协议 (D0 定死); a(4) 等 D7 三种特征的 CV 数字出来再定稿; 每人 review 另一人的章节。

### 5.3 22 天时间线

| 日期 | 里程碑 |
|---|---|
| 10/22-24 (D0-2) | 启动会定协议; venv + 仓库; A: EDA + **TF-IDF baseline + submission v1**; B: GloVe 下载、DataLoader、资源测量脚本; C: 读 2-3 篇综述, 建 Zotero |
| 10/25-28 (D3-6) | A: distilbert/roberta 在 MPS 跑通 (fp16, 前 50 步查 NaN); B: BiLSTM/TextCNN + 冻结特征 + LR; C: a(1)(2) 初稿 |
| **10/29 (D7)** | 里程碑 1: 三种特征在同一组折上都有数字; 定主模型路线; C 与 A 对齐 a(4) |
| 10/30-11/2 (D8-11) | A: 主模型扫参, 3 seed; B: g 表、d 曲线、OOF 错例流程, 组织 test 子集标注; C: a(3) 精读、h/i 初稿、噪声注入实验 (TF-IDF 版很便宜) |
| **11/3 (D12)** | 里程碑 2: 主模型冻结; OOF + test 概率入库; submission v2 通过自检 |
| 11/4-6 (D13-15) | B: f 案例 + 归因图; A: 写 b/c/d/e, 画架构图; C: 完成 a/h/i。**11/5 最晚降级点** |
| **11/7 (D16)** | 里程碑 3: 报告初稿 v1, 所有图表入文 |
| 11/8-10 (D17-19) | 交叉审稿; 补漏; 代码清理; **全新 venv 按 README 复跑**, 确认 submission.csv 逐字节一致; 一人逐条核对参考文献 |
| 11/11 (D20) | 终稿; 最终 submission 自检; 贡献表; 确认 NTULearn 上传格式 (zip 还是分开) 与代码提交方式 |
| 11/12 (D21) | **提交** |
| 11/13 | 只做应急 |

---

## 6. 各题答题要点与失分点

### 6.1 a) 文献综述 (20%)
- **a(1)** 问题定义: 输入文本 x → y ∈ {0,1}; 粒度 (文档/句子/方面级); 本项目 = 文档级、二分类、closed-set、单领域。**写明标签来源**: Blitzer 数据标签由星级转换 (论文里 processed 版本规则: >3 星正, <3 星负, 3 星剔除), 写成 "likely derived from star ratings", 这直接连到 h (无评分) 和 i (评分噪声)。PDF 引言写 "positive, negative and neutral" 而本数据是二分类且 3 星被剔除 — 一句话就能把 closed-set/open-set 和 h) 的酒店场景 (评论常是混合/中性) 连起来。
- 挑战清单 (每条配引用): 否定与转折、反讽、领域依赖 (small/tight 在鞋类偏负)、类别不平衡、隐式情感、长文本截断、标注噪声、评测可靠性。
- **3×2 设定对照表** (评分人明示要看): supervised (BoW+SVM → CNN/RNN → PLM 微调 → 情感专用预训练) vs unsupervised (词典 VADER/SO-CAL、LOTClass、NLI 零样本、LLM 零样本); closed-set vs open-set (OOD 检测 max-softmax、多语 MARC、方面级/四元组); with vs without domain shift (SCL → SFA → DANN → PBLM/PERL → DAPT/TAPT → UDALM → PADA; Hendrycks 2020 "预训练本身提升 OOD 鲁棒性")。
- **a(2)**: 20-30 行论文表 (论文/年份/会议/方法家族/数据集/关键结果/与本项目关系) + taxonomy 图; **必须写明检索方法** (关键词、ACL Anthology、Semantic Scholar 引用图从 BERT 和 Blitzer 2007 前后扩展、Papers with Code SST-2/IMDb/Amazon Polarity 榜单、年份范围、筛选标准)。
- **a(3)** 研究组 (必须是真正的一个组, 不要把 Google/Meta/微软的 BERT 谱系凑成一组): 首选 **Lidong Bing / Wenxuan Zhang 团队** (ABSA 综述 TKDE; "Sentiment Analysis in the Era of LLMs: A Reality Check", Findings of NAACL **2024**; 生成式 ABSA), 覆盖 closed→open 与微调小模型 vs LLM 两条主线; 第二个 **Roi Reichart 团队 (Technion)** (PBLM NAACL 2018 → PERL TACL 2020 → PADA TACL 2022), 与 Blitzer 数据同源, 直接服务 h)。每篇 0.5-1 页: 动机、方法 (自己重画示意图)、实验、局限、对本项目的启示。
- **a(4)** 论证链: baseline = TF-IDF + LR/SVM (NBSVM Wang & Manning 2012; 老师给了示例; 秒级可跑; 系数可解释); 主模型 = 微调 PLM (BERT → RoBERTa/DeBERTa 在 SST-2/IMDb 持续领先; Hendrycks 2020 抗偏移; Zhang 2024 证明有标注数据时微调小模型 ≥ 零样本 LLM, 不必追 LLM)。改进点要**针对本数据**: 不平衡 → 加权 CE/阈值 + macro-F1; 小数据微调不稳 → 小 lr + 多 seed (Mosbach 2021); 截断 → 256 + head+tail (Sun 2019); 重复 → 去重分层 K 折; 87 条重叠 → 只讨论不利用; 标签噪声 → confident learning 找可疑样本。
- 失分点: 只罗列不综合; 没有检索方法; 小标题不对应 (1)-(4); 漏 open-set; 精读变摘要翻译; 引用不统一或正文无标号; 大段复述原文。

### 6.2 b) 特征与预处理 (5%)
数据画像表 + 三分支流程图 + 每分支的参数 (TF-IDF 词表 38k/40k、GloVe 覆盖率、PLM 子词截断比例) + **明确写 "所有 fit 只在训练折"** + 写明**不做什么及理由** (BERT 不去停用词/不词干化; TF-IDF 不去停用词因为 not/but 是强特征)。

### 6.3 c) 模型 (15%)
自绘架构图 (不要截 BERT 论文的图) + 维度表 + 损失公式 (CE / 加权 CE) + 训练策略 (AdamW, lr, warmup, 早停指标) + 复现协议 (seed 列表、折数、mean±std) + 代码附录 + README (精确命令、requirements 版本、硬件、每个实验耗时)。至少 baseline + 主模型两个; RNN 一行满足题目的鼓励。

### 6.4 d) 超参数 (5%)
表格 (超参 | 范围 | 选定 | 依据) + 验证曲线 (lr × epoch × max_len) + C × class_weight × 阈值交互表 + 原理 (小 lr 防破坏预训练权重; max_len 由**子词** p95 决定; batch 受内存限制; 早停防过拟合)。失分点: "用默认值"; 没证据; 偷看 test 预测分布调参。

### 6.5 e) 提交 (15%)
见第 7 节清单。报告写明生成文件的模型 / seed / 命令、test 预测正负比例 (与 train 先验 85/15 对比)、CV 结果作为预期表现, 声明 test 标签未知且 test 文本未参与任何训练或词表构建。

### 6.6 f) 错例分析 (10%)
1. 主分析放在 train 的 OOF 预测: 混淆矩阵 + 错误类型占比 (转折/混合、否定、标签噪声、短文本、反讽、领域词义、末尾截断)。
2. **test 无偏子集**: 随机抽 200-300 条, 三人独立标注、多数投票、报告 kappa → 诚实的 test 准确率点估计和混淆矩阵, 也为 e) 提供预期。
3. 有偏补充: 概率 0.4-0.6 的低置信样本 (实测 121 行)、TF-IDF 与 roberta 意见不一致的样本。
4. 案例: 1 个 "难但做对" (否定/转折) + 1 个 "做错", 配词级归因 (LR 系数; Captum Integrated Gradients), 错因归到模型设计 (截断、负类欠学习、[CLS] 对长评论后半段不敏感)。现成材料: 标签对但 BoW 错的 "Looks great on the web, don't think many would buy it if they could examine it in-person..." (反讽+转折)、"This shoe isn't as comfortable as my other Keds shoes..." (比较级否定)、"...If you want any kind of foot comfort...skip it" (条件句, 结论在末尾)。
5. 失分点: 只贴样本不解释; 声称 "测试准确率" 却没有标注依据; 选平凡样本; 不归因到模型设计。

### 6.7 g) 特征格式 vs 资源与精度 (10%)
从 `runs.csv` 生成 8-9 行对照表 (由便宜到贵): TF-IDF unigram + LR → uni+bi + LinearSVC → +char n-gram (维度翻倍无收益的例子) → GloVe 平均池化 + LR → TextCNN → BiLSTM → **roberta 冻结特征 + LR** → distilbert 微调 → roberta 微调。列: feature_dim、参数量、特征构建时间、训练时间、推理 1851 条时间、峰值内存、模型/依赖磁盘 (GloVe ~1 GB, roberta ~500 MB)、是否需要 GPU、acc/macro-F1/F1(neg)/AUC。**写明计时条件** (同一台机器、同 max_len、动态 padding、推理 batch、是否预热)。画训练时间 (log) vs macro-F1 散点图。结论形态: "贵 3-4 个数量级换 2-3 pt macro-F1"。

### 6.8 h) 酒店评论、无标签 (10%)
1. 诊断: 源域 apparel (有标签) → 目标域 hotel (无标签) = unsupervised domain adaptation; 酒店评论更长、多方面 (房间/服务/位置)。
2. **先答现有模型会怎样**: TF-IDF 退化最重 (负面信号依赖 return/refund/size/shipping 等电商事务词, 到酒店域失效; 词表不匹配); GloVe 中等; 微调 PLM 退化最小但仍下降 (Blitzer 2007 跨域损失 5-15 pt; Hendrycks 2020); **label prior shift**: 模型和阈值都是按 85/15 调的, 酒店正负比例不同, 即使特征能迁移阈值也要在目标域重新校准 (单独写成小节); **没有标签就无法评估**, 必须先手标 200-300 条评估集并报告标注一致性。
3. 按标注预算分层 (做成表): 零标签 — 词典 (VADER/SO-CAL)、NLI 零样本 (Yin 2019)、LLM 零样本/标注 (Zhang 2024; Gilardi 2023)、LOTClass、UDA (DANN; SCL→PBLM→PERL; DAPT/TAPT Gururangan 2020; **UDALM** Karouzos 2021 源域分类 + 目标域 MLM 多任务, 最值得推荐)、self-training/pseudo-labeling; 弱标签 — 关键词/表情远程监督、Snorkel、LLM 伪标签蒸馏到小模型; 少量标签 — active learning、SetFit、PET、UDA 一致性 (Xie 2020)。
4. 推荐组合: DAPT 在酒店无标签文本 + LLM/词典伪标签启动 + self-training, 用小人工标注集做模型选择; 提 ABSA 扩展。
5. 若做实验: 只用无标签酒店文本 + 自己手标的小评估集; 外部带标签集 (如 TripAdvisor) 只能当评估集, 写明来源与许可, 声明不参与 submission 的模型训练。

### 6.9 i) 酒店评论、有评分但噪声 (10%)
1. 噪声类型: symmetric / class-conditional (3 星边界) / instance-dependent (文本与评分不符, 如物流差评但夸产品); 本数据 1-2% 真实噪声样本可直接当例子。
2. 现有模型会怎样: 过参数化 PLM + CE 先学干净模式再记忆噪声 (Arpit 2017 memorization), 准确率随噪声率下降, 少数类受害更重, 验证集含噪则模型选择失效, 校准变差 (Zhu 2022 "Is BERT robust to label noise": 对随机噪声较鲁棒, 对特征相关噪声脆弱, 早停最关键)。
3. **醒目编号三种方法**: (1) 噪声鲁棒损失: GCE (Zhang & Sabuncu 2018)、Symmetric CE (Wang 2019)、label smoothing; (2) 样本选择/协同训练: Co-teaching (Han 2018)、DivideMix (Li 2020)、小损失准则 + 早停; (3) **利用评分的序数结构** (比通用方法更贴题): 只用 1/5 星做高置信标签、按评分极端程度加权、评分当软标签/序数回归、剔除 3 星边界; 另可提 confident learning (Northcutt 2021, cleanlab)、噪声转移矩阵损失修正 (Patrini 2017; Jindal 2019 文本版)、Meta-Weight-Net。
4. 加分小实验 (便宜): 在 train 注入 10/20/30% 对称噪声, 比较 TF-IDF+LR 与 roberta 在 CE / label smoothing / 早停 / cleanlab 过滤下的 macro-F1 曲线, 一张图把 "会怎样" 变成证据; 结合第 1 节找到的真实标错样本。

---

## 7. 合规与提交检查清单

**合规红线**
- test 文本不参与任何 fit: TF-IDF 词表、GloVe 词表统计、阈值、早停、TAPT 继续预训练、self-training 全部只用 train (PDF 第 4 条原话: "your algorithm should take only the inputs from the training set")。TAPT 若做只用 train 文本并声明。
- 87 条 train/test 重叠: 不查表覆盖; 报告统一口径 "87 unique texts, appearing in 101 test rows and 136 train rows; 71 pos / 16 neg by text", 声明未做特殊处理, 并指出 test 真实表现会因此略乐观 (约 +0.3 pt)。
- 不要用已在情感数据上微调好的模型 (如 siebert/sentiment-roberta-large) 冒充自己训练; 最多作为明确标注的 zero-shot 对照。
- 不要手动删除可疑标签样本后训练 (会被质疑挑数据); 做成对照实验放 i)。
- 综述不能是 LLM 生成的无来源文本; 每个论断有引用; 不引用不存在的论文 (录入前逐条在 ACL Anthology / DOI / arXiv 核对); 查阅 NTU 对 GenAI 使用声明的规定。

**submission.csv**
- 文件名精确 `submission.csv` (PDF 原话; 不是 submission_template.csv); 列名 `reviews,predictions`; UTF-8; 无 index 列。
- 推荐做法: 读入课程给的模板, 断言 `reviews` 与 `test.json` 逐行相同, 只替换 `predictions` 为 int 0/1, `to_csv(index=False)`, **再用 `pd.read_csv` 读回**: `len(df) == 1851` 且 `df.reviews.equals(test_df.reviews)`。**评论含换行, `wc -l` 是 2781 行, 不要用行数工具检查。**
- 正类比例 0.82-0.88 (train 0.854); 与 TF-IDF baseline 预测一致率 > 93%, 否则大概率错位。
- 报告描述的模型 / 超参 / seed 必须与生成文件的代码一致; 提交前在全新 venv 里按 README 复跑一次。

**报告**
- 成员姓名、学号; 按题 a-i 的贡献表; 章节标题带题号; 结果表第一行放 "全预测 1 = 85.4%"。
- 引用统一 IEEE 格式, 含 PDF 给的 5 条; 图表编号并在正文引用; 架构图自绘。
- 提前一天上传 NTULearn 并下载回来核对; 确认上传格式 (zip 或 PDF + csv 分开) 与代码提交方式 (附录 / 可访问的仓库链接)。

---

## 8. 文献清单 (★ = 建议纳入 "数十篇" 核心清单; ✔ = 2026-10-22 已在 arXiv / ACL Anthology 核实; 无标记为公认经典但录入前仍需核对; [未核实] 不要直接引用)

**综述 / 教材 (a1)**: Pang & Lee 2008 (FnTIR); Bing Liu 2012 (Morgan & Claypool) / 2015, 2nd ed. 2020 (CUP); Zhang, Wang & Liu 2018 WIREs DMKD (arXiv 1801.07883 ✔); Wankhade et al. 2022 AI Review ✔; **Zhang, Deng, Liu, Pan & Bing 2024, Findings of NAACL, pp. 3881-3906 ✔** (引用用 2024, 不是 2023); Ramponi & Plank 2020 COLING (UDA 综述); Song et al. 2022 IEEE TNNLS (噪声标签综述); Jurafsky & Martin SLP 3rd ed. (NB/LR 章)。

**经典基线 (7)**: ★ Pang, Lee & Vaithyanathan 2002 EMNLP; ★ Turney 2002 ACL; Hu & Liu 2004 KDD; ★ Wang & Manning 2012 ACL (NBSVM); Maas 2011 ACL (IMDb); ★ Socher 2013 EMNLP (SST); Zhang, Zhao & LeCun 2015 NeurIPS。

**词向量 + 神经网络 (5)**: ★ Kim 2014 EMNLP (TextCNN); Tang, Qin & Liu 2015 EMNLP; ★ Yang 2016 NAACL (HAN); Joulin 2017 EACL (fastText); ★ Howard & Ruder 2018 ACL (ULMFiT)。

**PLM 与微调 (14)**: Peters 2018 (ELMo); ★ Devlin 2019 NAACL (BERT); ★ Liu 2019 (RoBERTa, arXiv 1907.11692); Yang 2019 (XLNet); Lan 2020 ICLR (ALBERT); ★ Clark 2020 ICLR (ELECTRA); ★ Sanh 2019 (DistilBERT); ★ He 2021 ICLR (DeBERTa) / He, Gao & Chen 2023 ICLR (DeBERTaV3); Warner 2024 (ModernBERT, arXiv 2412.13663 ✔); ★ Sun, Qiu, Xu & Huang 2019 CCL (How to fine-tune BERT: head+tail); ★ Mosbach 2021 ICLR (微调稳定性); Zhang, Wu et al. 2021 ICLR (few-sample fine-tuning); Hu 2022 ICLR (LoRA); ★ Hendrycks 2020 ACL ✔ (预训练 Transformer 抗 OOD)。

**情感专用 / 领域预训练 (5)**: Xu, Liu, Shu & Yu 2019 NAACL; ★ Gururangan 2020 ACL (DAPT/TAPT); ★ Tian 2020 ACL (SKEP); Yin, Meng & Chang 2020 ACL (SentiBERT); Ke 2020 EMNLP (SentiLARE)。

**少样本 / LLM (11)**: Brown 2020 NeurIPS; Schick & Schütze 2021 EACL (PET); ★ Gao, Fisch & Chen 2021 ACL (LM-BFF); ★ Tunstall 2022 (SetFit, arXiv 2209.11055 ✔); Liu 2023 ACM CSUR (prompting 综述); ★ Wang et al. 2023 (Is ChatGPT a good sentiment analyzer, arXiv 2304.04339 ✔); ★ Zhang 2024 Findings NAACL ✔; Qin 2023 EMNLP; Zhong 2023 (arXiv 2302.10198); Kocoń 2023 Information Fusion; ★ Gilardi 2023 PNAS (LLM 标注); Sun 2023 Findings of EMNLP ✔ (CARP); Fei 2023 ACL (隐式情感 CoT)。

**噪声标签 (13, 服务 i)**: Natarajan 2013 NeurIPS; Arpit 2017 ICML; ★ Han 2018 NeurIPS (Co-teaching); ★ Zhang & Sabuncu 2018 NeurIPS (GCE); ★ Wang 2019 ICCV (SCE); Müller 2019 NeurIPS (label smoothing); Li 2020 ICLR (DivideMix); ★ Northcutt 2021 JAIR (confident learning); Northcutt 2021 NeurIPS D&B (基准标签错误); ★ Jindal 2019 NAACL ✔ (文本噪声转移层); ★ Zhu 2022 ACL Insights workshop ✔ (Is BERT robust to label noise); Wu 2023 Findings ACL ✔ (NoisywikiHow); Swayamdipta 2020 EMNLP (dataset cartography)。

**领域自适应 / 自训练 / 零样本 (14, 服务 h)**: ★ Blitzer 2007 ACL (SCL); Pan 2010 WWW (SFA); Glorot 2011 ICML; ★ Ganin 2016 JMLR (DANN); Ziser & Reichart 2018 NAACL (PBLM); Ben-David 2020 TACL (PERL); Han & Eisenstein 2019 EMNLP (AdaptaBERT); ★ Karouzos 2021 NAACL ✔ (UDALM); Ben-David 2022 TACL ✔ (PADA); Lee 2013 (pseudo-label); ★ Xie 2020 NeurIPS (UDA 一致性); Du 2021 NAACL (self-training); ★ Meng 2020 EMNLP (LOTClass); Yin, Hay & Roth 2019 EMNLP (NLI 零样本); Hutto & Gilbert 2014 ICWSM (VADER); Taboada 2011 CL (SO-CAL)。

**类别不平衡 (4)**: ★ Lin 2017 ICCV (focal loss); Cui 2019 CVPR; Li 2020 ACL (Dice loss, 已核实 ACL 2020 main pp. 465-476); Johnson & Khoshgoftaar 2019 J. Big Data。

需格外小心的条目: ABSA 综述的 TKDE 卷期; ModernBERT 引 arXiv 即可; Blitzer 数据的域名 — JHU 数据页不列 unprocessed 版本的域名, 报告里写 "推断为 apparel/shoes" 而不是事实。实际引用控制在 40-50 条, 不要为了 "数十篇" 硬塞。

---

## 9. 本次实测脚本位置 (临时目录, 可复制进项目当 `notebooks/01_eda` 的素材; 路径写死了旧的 `Project/train.json`, 复用时改成 `data/raw/`)

`/private/tmp/claude-501/-Users-saber812-Library-CloudStorage-OneDrive-NanyangTechnologicalUniversity-EE6483-Project/4bbb100c-9485-48b7-8c44-b0337b07f1dc/scratchpad/` 下:
`01_data_stats.py` (统计/重复/重叠/词表/判别词), `02_baseline_cv.py` (TF-IDF 配置对比), `03_noise_and_extras.py` (OOF 噪声候选、分桶错误率), `04_truncation_check.py` (末尾截断), `baseline.py`, `oof.py`, `bench_plm.py` + `plm_bench.log` (PLM 速度/内存/子词长度), `bench_lstm.py`。
