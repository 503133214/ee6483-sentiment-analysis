# EE6483 商品评论情感分类

NTU EE6483 Artificial Intelligence and Data Mining 课程项目：根据英文商品评论预测情感，`0` 表示负面，`1` 表示正面。

目前包含课程资料、原始数据、TF-IDF 特征提取示例和提交模板，尚未实现完整的分类训练、评估或 Qwen 微调流程。

## 目录

```text
.
├── data/raw/
│   ├── train.json                    # 7,401 条带标签评论
│   └── test.json                     # 1,851 条无标签评论
├── docs/
│   ├── assignment/EE6483-Project1.pdf
│   └── examples/tfidf_features_example.html
├── notebooks/tfidf_features_example.ipynb
├── submissions/submission_template.csv
├── requirements.txt
└── README.md
```

HTML 文件是课程示例的原始导出归档，可能保留旧路径和旧输出；运行时使用 `notebooks/` 中的 notebook。

## 运行示例

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/tfidf_features_example.ipynb
```

Notebook 支持从项目根目录或 `notebooks/` 目录运行。它读取原始 JSON，在训练数据上拟合 TF-IDF，再用同一向量器转换测试数据。

## 数据与评估

- `train.json` 包含 `reviews` 和 `sentiments`；正面评论占 85.38%。
- `test.json` 仅包含 `reviews`，不能直接计算测试准确率。
- 数据存在重复评论。验证划分应按规范化文本分组，避免同一评论跨训练集与验证集；TF-IDF 等可学习的预处理仅在训练折拟合。
- 同时报告 accuracy、macro-F1、负类召回率和混淆矩阵，并固定随机种子、记录实验配置。

`submission_template.csv` 包含 `reviews,predictions` 两列，当前 1,851 个预测全部为 `0`，仅用作格式模板。最终提交需替换为模型预测，保持测试评论顺序不变，并输出为 `submission.csv`。

## 后续实验

1. TF-IDF + LinearSVC 基线。
2. 冻结 Qwen 文本表示 + 分类器。
3. Qwen + LoRA + 二分类头，与前两种方案比较效果及计算成本。

## 课程提交

按课程说明，三人一组提交报告及 `submission.csv`，报告注明成员分工并引用使用的资料。截止时间：**2026 年 11 月 13 日 23:59**；详细要求见 [课程说明](docs/assignment/EE6483-Project1.pdf)。
