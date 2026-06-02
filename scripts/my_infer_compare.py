"""
模块 5：推理和 base vs LoRA 对比

目标：加载 base 模型和 LoRA adapter，对比生成效果。

运行方式：
    python scripts/my_infer_compare.py --question "高血压患者日常生活中应该注意什么？"

参考代码：scripts/infer_compare.py（先看懂，关掉，再自己写）
"""

import argparse
import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"


def load_model(model_name, adapter_dir=None):
    """加载模型，支持 base 和 LoRA 两种模式

    参数：
        model_name: 模型路径，如 "models/Qwen2.5-0.5B-Instruct"
        adapter_dir: LoRA adapter 路径，None 表示加载 base 模型

    返回：(tokenizer, model)

    步骤：
    1. 从 adapter_dir（如有）或 model_name 加载 tokenizer
    2. 设置 pad_token = eos_token（如果没有的话）
    3. 用 AutoModelForCausalLM 加载模型（bf16，device_map="auto"）
    4. 如果 adapter_dir 不为 None，用 PeftModel.from_pretrained 加载 LoRA
    5. model.eval() 切到推理模式
    """
    # TODO: 你的代码
    pass


def generate(tokenizer, model, question, max_new_tokens=256):
    """用模型生成回答

    参数：
        tokenizer: 分词器
        model: 模型
        question: 用户问题
        max_new_tokens: 最大生成 token 数

    返回：str（生成的回答文本）

    步骤：
    1. 构造 messages（system + user，不包含 assistant）
    2. 用 apply_chat_template tokenize（return_tensors="pt"）
    3. 注意 apply_chat_template 可能返回 BatchEncoding，需要检查 .input_ids
    4. inputs 移到 model.device
    5. 用 torch.no_grad() + model.generate() 生成
       参数建议：do_sample=True, temperature=0.7, top_p=0.9
    6. 只取新生成的 token（去掉 prompt 部分），解码返回
    """
    # TODO: 你的代码
    pass


def main():
    parser = argparse.ArgumentParser(description="Compare base and LoRA responses.")
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--question", default="高血压患者日常生活中应该注意什么？")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    # TODO: 步骤 1 - 加载 base 模型，生成回答
    # TODO: 步骤 2 - del model + gc.collect() + torch.cuda.empty_cache() 释放显存
    #         6GB 显存放不下两个模型，必须先释放再加载
    # TODO: 步骤 3 - 检查 adapter_dir 是否存在，加载 LoRA 模型，生成回答
    pass


if __name__ == "__main__":
    main()
