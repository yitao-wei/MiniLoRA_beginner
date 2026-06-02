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


def load_model(model_name, adapter_dir=None):
    """加载模型（复用模块 5）

    adapter_dir=None  → 加载 base 模型
    adapter_dir="路径" → 加载 base + LoRA adapter
    """
    # TODO: 你的代码（从模块 5 复制过来）
    pass


def generate(tokenizer, model, question, max_new_tokens=256):
    """用模型生成回答（复用模块 5）"""
    # TODO: 你的代码（从模块 5 复制过来）
    pass


def load_jsonl(path):
    """读取 jsonl 文件，返回 list[dict]"""
    # TODO: 你的代码
    pass


def write_jsonl(path, items):
    """写出 jsonl 文件，ensure_ascii=False 保证中文不被转义"""
    # TODO: 你的代码
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/medical/eval_prompts.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("results/base_vs_lora.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.eval_file.exists():
        print(f"评测文件不存在: {args.eval_file}")
        print("请先创建 data/medical/eval_prompts.jsonl")
        print("每行一个 JSON: {\"question\": \"你的问题\"}")
        return

    # TODO: 步骤 1 - 读取评测问题
    # TODO: 步骤 2 - 加载 base 模型，对所有问题生成回答
    #         用 list 存结果: [{"question": ..., "base_answer": ...}, ...]
    # TODO: 步骤 3 - 释放显存（del + gc.collect + empty_cache）
    # TODO: 步骤 4 - 加载 LoRA 模型，对所有问题生成回答
    #         追加到已有结果: result["lora_answer"] = ...
    # TODO: 步骤 5 - 用 write_jsonl 保存结果到 jsonl
    pass


if __name__ == "__main__":
    main()
