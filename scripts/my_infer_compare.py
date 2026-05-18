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
    """
    if adapter_dir:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 确保 pad_token 存在，否则会报错
    # 例如：Qwen2.5-0.5B-Instruct 没有 pad_token，需要手动设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 base 模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        )
    # 加载 LoRA adapter
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    # 切换到推理模式
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, question, max_new_tokens=256):
    """用模型生成回答

    参数：
        tokenizer: 分词器
        model: 模型
        question: 用户问题
        max_new_tokens: 最大生成 token 数

    返回：str（生成的回答文本）
    """
    messages = [
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": f"请以谨慎、专业、易懂的方式回答下面的医疗健康问题。\n\n问题：{question}"},
    ]

    inputs = tokenizer.apply_chat_template(
           messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
       )
    if hasattr(inputs, "input_ids"):
        inputs = inputs.input_ids
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # 启用采样（非贪心）
            temperature=0.7,         # 随机性：0=确定，1=随机
            top_p=0.9,               # nucleus sampling
            repetition_penalty=1.05, # 重复惩罚
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()



def main():
    parser = argparse.ArgumentParser(description="Compare base and LoRA responses.")
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--question", default="高血压患者日常生活中应该注意什么？")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    # 步骤 1 - 加载 base 模型，生成回答
    base_tokenizer, base_model = load_model(args.model_name, None)
    print("\n[Base Model]")
    print(generate(base_tokenizer, base_model, args.question, args.max_new_tokens))

    # 步骤 2 - 释放显存，加载 LoRA 模型，生成回答
    # RTX 4050 只有 6GB 显存，装不下两个模型同时存在
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    if args.adapter_dir.exists():
        print("\nLoading LoRA adapter...")
        lora_tokenizer, lora_model = load_model(args.model_name, args.adapter_dir)
        print("\n[LoRA Model]")
        print(generate(lora_tokenizer, lora_model, args.question, args.max_new_tokens))
    else:
        print(f"\nLoRA adapter not found: {args.adapter_dir}")
    


if __name__ == "__main__":
    main()
