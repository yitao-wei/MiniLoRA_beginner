"""
模块 2-4：SFT 数据预处理 + LoRA 配置 + 训练

模块 2：build_messages + preprocess（数据转 token 序列）
模块 3：load_model_and_lora（加载模型 + LoRA）
模块 4：main()（训练全流程）

运行方式（模块 2 测试）：
    python scripts/my_train_lora.py

运行方式（模块 4 训练）：
    python scripts/my_train_lora.py --max-train-samples 50 --max-valid-samples 20 --epochs 1 --grad-accum 4 --max-length 256

参考代码：scripts/train_lora.py（先看懂，关掉，再自己写）
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ============================================================
# 常量
# ============================================================

# -100 是 PyTorch CrossEntropyLoss 的 ignore_index
# labels 中值为 -100 的位置不参与 loss 计算
# SFT 中，prompt 部分设为 -100，只对 assistant 回复计算 loss
IGNORE_INDEX = -100


# ============================================================
# 模块 1 复用：数据加载
# ============================================================

def load_jsonl(path):
    """读取 jsonl 文件，返回 list[dict]"""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()# 去掉首尾空格
            if line:# 跳过空行
                items.append(json.loads(line))
    return items


# ============================================================
# 模块 2：SFT 数据预处理
# ============================================================

def build_messages(example):
    """将一条数据转成 messages 格式

    输入格式：
    {
        "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
        "input": "高血压应该怎么办？",
        "output": "高血压患者应该..."
    }

    输出格式（3 条 message）：
    [
        {"role": "system",    "content": "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"},
        {"role": "user",      "content": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。\n\n问题：高血压应该怎么办？"},
        {"role": "assistant", "content": "高血压患者应该..."},
    ]
    """
    user_content = example["instruction"].strip()
    input_content = example["input"].strip()
    if input_content:
        user_content = f"{user_content}\n\n问题：{input_content}"
    return [
        {"role": "system",    "content": "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": example["output"].strip()},
    ]



def preprocess(example, tokenizer, max_length):
    """将一条数据转成训练用的 token 序列

    返回格式：
    {
        "input_ids": [token1, token2, ...],        # prompt tokens + answer tokens
        "attention_mask": [1, 1, 1, ...],          # 全 1
        "labels": [-100, -100, ..., ans_tok1, ...]  # prompt 部分 -100，answer 部分真实 id
    }

    步骤：
    1. 调用 build_messages 获取 messages
    2. 分离 prompt_messages = messages[:-1] 和 answer = messages[-1]["content"]
    3. 用 tokenizer.apply_chat_template 把 prompt 转成 token ids
       - 参数: tokenize=True, add_generation_prompt=True, return_tensors=None
       - ⚠️ 注意: apply_chat_template 可能返回 BatchEncoding 对象而不是 list
         需要检查并提取 input_ids，再转成 list
    4. 把 answer 单独 tokenize:
       tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    5. 拼接: input_ids = prompt_ids + answer_ids
    6. 构造 labels:
       - prompt 部分: [-100] * len(prompt_ids)  ← 不计算 loss
       - answer 部分: answer_ids                 ← 计算 loss
    7. 如果总长度 > max_length，三个列表都截断到 max_length
    8. attention_mask = [1] * len(input_ids)

    为什么要这样设计 labels：
    - -100 在 CrossEntropyLoss 中被忽略
    - prompt 部分设为 -100：模型不需要学习"复述问题"
    - answer 部分保留真实 token：模型学习"怎么回答"
    - 这叫 assistant-only loss mask，是 SFT 的标准做法
    """
    messages = build_messages(example)
    prompt_messages = messages[:-1]
    answer = messages[-1]["content"]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,             # 返回 token ids，而不是字符串
        add_generation_prompt=True,  # 在末尾加上 assistant 的开头标记
        return_tensors=None,       # 返回 list，不返回 tensor（方便后续处理）
    )

    # apply_chat_template 返回值类型不固定，需要检查并转换
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids   # BatchEncoding → 提取 input_ids
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()    # tensor → list
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = attention_mask[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }



# ============================================================
# 模块 3：模型加载和 LoRA 配置
# ============================================================

def load_model_and_lora(model_name, r=8, lora_alpha=16, lora_dropout=0.05):
    """加载模型并配置 LoRA

    返回: (tokenizer, lora_model)

    步骤：
    1. 加载 tokenizer
       - AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
       - 如果 pad_token 为 None，设为 eos_token

    2. 加载模型
       - AutoModelForCausalLM.from_pretrained(
           model_name,
           torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
           device_map="auto" if torch.cuda.is_available() else None,
           trust_remote_code=True,
       )

    3. 关闭 use_cache（训练时不需要 KV Cache）
       - model.config.use_cache = False

    4. 启用输入梯度（PEFT 需要）
       - model.enable_input_require_grads()

    5. 配置 LoRA
       - LoraConfig(
           task_type=TaskType.CAUSAL_LM,    # 因果语言模型
           r=r,                               # rank，低秩矩阵维度
           lora_alpha=lora_alpha,             # 缩放因子
           lora_dropout=lora_dropout,         # dropout
           target_modules=[                   # 对哪些层加 LoRA
               "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
               "gate_proj", "up_proj", "down_proj",      # FFN 层
           ],
       )

    6. 用 get_peft_model 把 LoRA 挂到模型上
       - model = get_peft_model(model, lora_config)

    7. 打印可训练参数
       - model.print_trainable_parameters()

    LoRA 公式: h = Wx + (alpha/r) * BAx
    - W: 原始权重（冻结）
    - A: d×r 矩阵（训练）
    - B: r×d 矩阵（训练）
    - alpha/r: 缩放因子
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
           model_name,
           torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
           device_map="auto" if torch.cuda.is_available() else None,
           trust_remote_code=True,
    )

    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,    # 因果语言模型
        r=r,                               # rank，低秩矩阵维度
        lora_alpha=lora_alpha,             # 缩放因子
        lora_dropout=lora_dropout,         # dropout
        target_modules=[                   # 对哪些层加 LoRA
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
            "gate_proj", "up_proj", "down_proj",      # FFN 层
        ],
    )


    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    return tokenizer, model


# ============================================================
# 模块 4：训练
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Qwen LoRA on Chinese medical SFT data.")
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/medical/train.jsonl"))
    parser.add_argument("--valid-file", type=Path, default=Path("data/medical/valid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--r", type=int, default=8)
    args = parser.parse_args()

    # 步骤 1 - 加载 tokenizer 和模型
    tokenizer, model = load_model_and_lora(args.model_name, r=args.r)

    # 步骤 2 - 加载数据
    train_items = load_jsonl(args.train_file)
    valid_items = load_jsonl(args.valid_file)

    # 步骤 3 - 如果 max_train_samples > 0，截取前 N 条
    if args.max_train_samples > 0:
        train_items = train_items[:args.max_train_samples]
    if args.max_valid_samples > 0:
        valid_items = valid_items[:args.max_valid_samples]

    # 步骤 4 - 构建 Dataset
    train_dataset = Dataset.from_list(train_items).map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        remove_columns=list(train_items[0].keys()),
    )
    valid_dataset = Dataset.from_list(valid_items).map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        remove_columns=list(valid_items[0].keys()),
    )

    # 步骤 5 - 配置 TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )



    # 步骤 6 - 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # 步骤 7 - 训练
    trainer.train()

    # 步骤 8 - 保存 adapter 和 tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")



# ============================================================
# 模块 2 测试代码
# 如果直接运行此文件（不带命令行参数），测试 preprocess
# ============================================================

if __name__ == "__main__":
    import sys

    # 如果没有命令行参数，运行模块 2 测试
    if len(sys.argv) == 1:
        print("=" * 50)
        print("模块 2 测试：验证 preprocess 输出")
        print("=" * 50)

        tokenizer = AutoTokenizer.from_pretrained(
            "models/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载 1 条数据测试
        items = load_jsonl(Path("data/medical/train.jsonl"))
        if not items:
            print("错误: data/medical/train.jsonl 为空或不存在")
            print("请先运行模块 1 (my_prepare_data.py) 生成数据")
            sys.exit(1)

        result = preprocess(items[0], tokenizer, max_length=256)

        if result is None:
            print("错误: preprocess 返回 None，请检查代码")
            sys.exit(1)

        print(f"input_ids 长度: {len(result['input_ids'])}")
        print(f"labels 长度: {len(result['labels'])}")
        print(f"attention_mask 长度: {len(result['attention_mask'])}")
        print(f"labels 中 -100 的数量（prompt 部分）: {result['labels'].count(-100)}")
        print(f"labels 中非 -100 的数量（answer 部分）: {len(result['labels']) - result['labels'].count(-100)}")
        print()
        print("[OK] 预期：三个列表长度相同，-100 + 非-100 == 总长度")

        # 模块 3 测试
        print()
        print("=" * 50)
        print("模块 3 测试：加载模型 + LoRA")
        print("=" * 50)
        tokenizer, model = load_model_and_lora("models/Qwen2.5-0.5B-Instruct")
        print("[OK] 预期: trainable params ~4,399,104 / all params ~498,431,872 / ~0.88%")

    else:
        # 有命令行参数，运行训练
        main()
