"""
模块 6：批量评测 eval_lora.py

目标：批量跑 base vs LoRA 对比，保存结果到 results/base_vs_lora.jsonl

运行方式：
    python scripts/my_eval_lora.py --model-name models/Qwen2.5-0.5B-Instruct

本模块没有参考代码，需要你自己写。
可以复用模块 5 的 load_model 和 generate 函数。
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"


# ============================================================
# 模型加载（复用模块 5）
# ============================================================

def load_model(model_name, adapter_dir=None):
    """加载模型，支持 base 和 LoRA 两种模式

    adapter_dir=None  → 加载 base 模型
    adapter_dir="路径" → 加载 base + LoRA adapter
    """
    # 从 adapter_dir 或 model_name 加载 tokenizer
    if adapter_dir:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 确保 pad_token 存在，否则会报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 base 模型权重
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    # 如果有 adapter，用 PeftModel 挂载 LoRA 权重
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    # 切换到推理模式（关闭 dropout）
    model.eval()
    return tokenizer, model


# ============================================================
# 文本生成（复用模块 5）
# ============================================================

def generate(tokenizer, model, question, max_new_tokens=256):
    """用模型生成回答

    流程：构造 messages → tokenize → model.generate() → 解码
    """
    # 构造对话格式（只有 system + user，assistant 由模型生成）
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请以谨慎、专业、易懂的方式回答下面的医疗健康问题。\n\n问题：{question}"},
    ]
    # tokenize，return_tensors="pt" 返回 tensor（推理时用）
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(inputs, "input_ids"):
        inputs = inputs.input_ids
    inputs = inputs.to(model.device)

    # 采样生成，不计算梯度（省显存）
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,            # 采样模式
            temperature=0.7,           # 随机性控制
            top_p=0.9,                 # nucleus sampling
            repetition_penalty=1.05,   # 防重复
            eos_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成的 token（去掉 prompt 部分）
    new_tokens = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# 文件读写工具
# ============================================================

def load_jsonl(path):
    """读取 jsonl 文件，返回 list[dict]"""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, items):
    """写出 jsonl 文件，ensure_ascii=False 保证中文不被转义"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/medical/eval_prompts.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("results/base_vs_lora.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    # 检查评测文件是否存在
    if not args.eval_file.exists():
        print(f"评测文件不存在: {args.eval_file}")
        print("请先创建 data/medical/eval_prompts.jsonl")
        print("每行一个 JSON: {\"question\": \"你的问题\"}")
        return

    # 步骤 1: 读取评测问题
    questions = load_jsonl(args.eval_file)
    print(f"共 {len(questions)} 个评测问题")

    # 步骤 2: 加载 base 模型
    print("Loading base model...")
    base_tokenizer, base_model = load_model(args.model_name, None)

    # 步骤 3: 对每个问题生成 base 回答
    # 先跑完 base 全部问题，再释放模型，避免反复加载
    results = []
    for i, q in enumerate(questions):
        question = q["question"]
        base_answer = generate(base_tokenizer, base_model, question, args.max_new_tokens)
        # 用 dict 存储，后续直接追加 lora_answer 字段
        results.append({"question": question, "base_answer": base_answer})
        print(f"[{i+1}/{len(questions)}] Base done: {question[:30]}...")

    # 步骤 4: 释放 base 模型显存，加载 LoRA 模型
    # RTX 4050 6GB 显存装不下两个模型，必须先释放再加载
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading LoRA adapter...")
    lora_tokenizer, lora_model = load_model(args.model_name, args.adapter_dir)

    # 步骤 5: 对每个问题生成 LoRA 回答，追加到已有 results 中
    for i, r in enumerate(results):
        lora_answer = generate(lora_tokenizer, lora_model, r["question"], args.max_new_tokens)
        r["lora_answer"] = lora_answer
        # 记录训练配置，方便消融实验区分不同组的结果
        r["train_samples"] = 1000
        r["lora_r"] = 8
        print(f"[{i+1}/{len(questions)}] LoRA done: {r['question'][:30]}...")

    # 步骤 6: 保存结果到 jsonl
    # 格式: {"question": "...", "base_answer": "...", "lora_answer": "...", "train_samples": 1000, "lora_r": 8}
    write_jsonl(args.output_file, results)
    print(f"\n结果保存到: {args.output_file}")


if __name__ == "__main__":
    main()
