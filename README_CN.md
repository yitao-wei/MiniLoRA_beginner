# Qwen2.5-0.5B 医疗健康 LoRA 微调学习项目

一个从零开始学 LLM 微调的完整教程项目。7 个模块，覆盖数据处理、SFT 训练、推理对比、消融实验的全流程。

适合有一定 Python 和 PyTorch 基础，想动手学 LoRA 微调的人。

## 项目做了什么

用 Qwen2.5-0.5B-Instruct（498M 参数），在 640 条中文医疗问答数据上做 LoRA SFT 微调，然后对比微调前后的效果。

```
原始数据 → 清洗划分 → SFT 数据预处理 → LoRA 训练 → 推理对比 → 批量评测 → 消融实验
  模块1        模块1        模块2          模块3-4     模块5       模块6       模块7
```

## 7 个模块

| 模块 | 文件 | 学什么 |
|------|------|--------|
| 1. 数据准备 | `scripts/my_prepare_data.py` | jsonl 读写、数据清洗、数据集划分 |
| 2. SFT 预处理 | `scripts/my_train_lora.py` | messages 格式、tokenize、assistant-only loss mask（-100） |
| 3. LoRA 配置 | `scripts/my_train_lora.py` | LoRA 原理、get_peft_model、target_modules |
| 4. 训练 | `scripts/my_train_lora.py` | TrainingArguments、Trainer、梯度累积、bf16 |
| 5. 推理对比 | `scripts/my_infer_compare.py` | PeftModel 加载、采样生成、显存管理 |
| 6. 批量评测 | `scripts/my_eval_lora.py` | 批量推理、结果保存为 jsonl |
| 7. 消融实验 | 同模块 4 的脚本 | 对比不同 rank 和数据量对 loss 的影响 |

每个模块对应 `scripts/` 下一个 `my_*.py` 文件，旁边有同名的参考代码（`scripts/train_lora.py` 等）供对照。

## 核心知识点

### SFT 数据处理

```python
# 原始数据
{"instruction": "...", "input": "高血压怎么办？", "output": "高血压患者应该..."}

# 转成 messages 格式
[
    {"role": "system",    "content": "你是一名谨慎的医疗健康问答助手。"},
    {"role": "user",      "content": "请以谨慎、专业的方式回答...\n\n问题：高血压怎么办？"},
    {"role": "assistant", "content": "高血压患者应该..."},
]

# tokenize 后构造 labels（assistant-only loss mask）
# input_ids: [prompt tokens...] + [answer tokens...]
# labels:    [-100, -100, ...]    + [answer tokens...]
# → 模型只在回答部分计算 loss，不需要学习复述问题
```

### LoRA 原理

```
h = Wx + (alpha/r) * BAx

W: 原始权重（冻结）
A: d×r 矩阵（训练）
B: r×d 矩阵（训练）
alpha/r: 缩放因子

本项目：r=8, alpha=16, 可训练参数 4.4M（占总参数 0.88%）
```

### 消融实验结论

| 实验 | 发现 |
|------|------|
| rank 4/8/16 对比 | loss 几乎一样（~2.764），瓶颈在数据量而非模型容量 |
| 200 vs 640 条数据 | 更多数据能降低 train_loss（2.823 → 2.720），但提升有限 |

## 数据集

使用 [shibing624/medical](https://huggingface.co/datasets/shibing624/medical) 数据集，这是一个中文医疗健康问答数据集，包含 instruction（指令）、input（问题）、output（回答）三个字段。

原始数据约 8000+ 条中文医疗问答对，覆盖高血压、糖尿病、感冒、失眠等常见疾病。

数据示例：

```json
{
    "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
    "input": "高血压应该怎么办？",
    "output": "高血压患者应该控制盐分摄入，保持规律运动，定期监测血压..."
}
```

本项目从中取 800 条，按 80/10/10 划分为 train（640 条）、valid（160 条）、test（200 条）。

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.1+（推荐 2.6+cu124）
- NVIDIA GPU，6GB+ 显存（RTX 3060 / 4050 及以上）
- 约 3GB 磁盘空间（模型 + 数据）

### 安装

**注意：** 仓库只包含代码，不包含模型和数据。需要自行下载模型和生成数据（共约 3GB）。

```bash
# 1. 克隆仓库并创建环境
git clone https://github.com/SoloCalm/MiniLoRA.git
cd MiniLoRA
conda create -n minillm python=3.10
conda activate minillm

# 2. 安装依赖
pip install torch>=2.1.0 transformers>=4.45.0 datasets>=2.19.0 peft>=0.12.0 accelerate>=0.33.0

# 3. 下载模型
# 从 HuggingFace 下载 Qwen2.5-0.5B-Instruct 到 models/ 目录
# 或者修改脚本中的 model_name 为远程模型名（会自动下载）
# 模型地址：https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

# 4. 下载数据
python download_dataset.py
# 原始数据会保存到 data/medical_raw/finetune/

# 国内用户可用镜像加速：
# HF_ENDPOINT=https://hf-mirror.com python download_dataset.py
```

### 按顺序跑 7 个模块

```bash
# 模块 1：数据准备（生成 data/medical/*.jsonl）
python scripts/my_prepare_data.py

# 模块 2-4：训练（用 50 条数据快速测试）
python scripts/my_train_lora.py --max-train-samples 50 --max-valid-samples 20 --epochs 1 --grad-accum 4 --max-length 256

# 模块 5：单条推理对比
python scripts/my_infer_compare.py --question "高血压患者日常生活中应该注意什么？"

# 模块 6：批量评测
python scripts/my_eval_lora.py

# 模块 7：消融实验（对比不同 rank）
python scripts/my_train_lora.py --r 4 --output-dir outputs/lora-r4 --max-train-samples 200 --epochs 1
python scripts/my_train_lora.py --r 8 --output-dir outputs/lora-r8 --max-train-samples 200 --epochs 1
python scripts/my_train_lora.py --r 16 --output-dir outputs/lora-r16 --max-train-samples 200 --epochs 1
```

## 项目结构

```
MiniLoRA/
├── README.md
├── requirements.txt                    # Python 依赖
├── download_dataset.py                 # 数据下载脚本
├── scripts/
│   ├── prepare_medical_sft.py          # [参考] 数据准备
│   ├── train_lora.py                   # [参考] LoRA 训练
│   ├── infer_compare.py                # [参考] 推理对比
│   ├── my_prepare_data.py              # [学习] 模块 1：数据准备
│   ├── my_train_lora.py                # [学习] 模块 2-4：预处理 + LoRA + 训练
│   ├── my_infer_compare.py             # [学习] 模块 5：推理对比
│   └── my_eval_lora.py                 # [学习] 模块 6：批量评测
├── data/
│   ├── medical/
│   │   ├── train.jsonl                 # 训练集（640 条）
│   │   ├── valid.jsonl                 # 验证集（160 条）
│   │   ├── test.jsonl                  # 测试集（200 条）
│   │   └── eval_prompts.jsonl          # 评测问题（10 个）
│   └── medical_raw/                    # 原始下载数据
├── outputs/                            # LoRA adapter 输出
│   ├── lora-r4-1000/                   # rank=4 实验
│   ├── lora-r8-1000/                   # rank=8 实验
│   └── ...
└── results/
    ├── base_vs_lora.jsonl              # base vs LoRA 对比结果
    └── lora_ablation_summary.csv       # 消融实验汇总表
```

### scripts/ 下的文件说明

- `scripts/xxx.py`（无 my_ 前缀）：参考代码，已写好可运行
- `scripts/my_xxx.py`（有 my_ 前缀）：空白模板，带有 TODO 提示，需要你来填写

学习流程：先看参考代码理解逻辑，再关掉参考代码，填写 `my_*.py` 中的 `TODO` 部分。

## 代码详解

### 模块 1：数据准备 (`my_prepare_data.py`)

4 个函数：

```python
read_jsonl(path)                    # 读取 jsonl 文件
normalize_item(item)                # 清洗一条数据（统一 instruction，过滤空 output）
split_data(items, 0.8, 0.1, 42)     # 划分 train/valid/test（80/10/10）
write_jsonl(path, items)            # 写出 jsonl（ensure_ascii=False）
```

关键点：`ensure_ascii=False` 保证中文不被转义成 `\uXXXX`。

### 模块 2-4：训练 (`my_train_lora.py`)

```python
build_messages(example)              # 转成 messages 格式
preprocess(example, tokenizer, max_length)  # tokenize + 构造 labels
load_model_and_lora(model_name, r)   # 加载模型 + LoRA
main()                               # 训练全流程
```

关键点：
- `apply_chat_template` 返回值类型不固定，需要用 `hasattr` 检查
- labels 中 prompt 部分设为 -100（CrossEntropyLoss 的 ignore_index）
- `DataCollatorForSeq2Seq` 动态 padding，pad 位置的 label 设为 -100
- `remove_unused_columns=False` 防止 Trainer 删除自定义列

### 模块 5：推理对比 (`my_infer_compare.py`)

```python
load_model(model_name, adapter_dir)  # 加载 base 或 LoRA 模型
generate(tokenizer, model, question) # 生成回答
```

关键点：
- 6GB 显存放不下两个模型，需要先 `del model` + `gc.collect()` + `torch.cuda.empty_cache()`
- `temperature=0.7` 控制随机性，`top_p=0.9` 做 nucleus sampling
- `torch.no_grad()` 关闭梯度计算，省显存加速

### 模块 6：批量评测 (`my_eval_lora.py`)

流程：读取评测问题 → 跑 base 模型全部问题 → 释放显存 → 跑 LoRA 模型全部问题 → 保存对比结果。

### 模块 7：消融实验

用 `--r` 参数指定不同的 LoRA rank，用 `--max-train-samples` 控制数据量，各训练一次，记录 loss。

## 后续可以拓展的方向

- [ ] 增大训练数据量（几万条以上）
- [ ] 用更大的模型（Qwen2.5-7B）
- [ ] 加入 BLEU/ROUGE 评估指标
- [ ] 用 GPT-4 做生成质量打分
- [ ] 部署推理服务（vLLM / TGI）
- [ ] 做 DPO/RLHF 偏好对齐
- [ ] 做 RAG（检索增强生成）

## 依赖

```
torch>=2.1.0
transformers>=4.45.0
datasets>=2.19.0
peft>=0.12.0
accelerate>=0.33.0
huggingface_hub>=0.24.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```
