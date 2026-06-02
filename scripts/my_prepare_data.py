"""
模块 1：数据准备和预处理

目标：从原始 json 文件生成训练用的 jsonl 文件。

运行方式：
    python scripts/my_prepare_data.py

运行后检查 data/medical/ 目录下是否生成了 train.jsonl、valid.jsonl、test.jsonl

参考代码：scripts/prepare_medical_sft.py（先看懂，关掉，再自己写）
"""

import json
import random
from pathlib import Path


def read_jsonl(path):
    """读取 jsonl 文件，返回 list[dict]

    jsonl 格式：每行一个 JSON 对象
    {"instruction": "...", "input": "...", "output": "..."}
    {"instruction": "...", "input": "...", "output": "..."}
    ...

    提示：
    - 用 path.open("r", encoding="utf-8") 打开文件
    - 逐行读取，每行用 json.loads() 解析
    - 跳过空行
    """
    # TODO: 你的代码
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()           # 去掉每行末尾的换行符和空白
            if not line:                   # 跳过空行
                continue
            try:
                items.append(json.loads(line))  # 把 JSON 字符串解析成 Python 字典
            except json.JSONDecodeError:
                continue                   # 如果某行 JSON 格式有误，跳过而不是报错
    return items
    pass


def normalize_item(item):
    """清洗一条原始数据，返回统一格式，或返回 None（丢弃）

    输入格式（原始 Alpaca）：
    {
        "instruction": "各种不同的指令，有的为空",   <- 可能为空
        "input": "具体问题",
        "output": "回答内容"                         <- 如果为空则丢弃
    }

    输出格式（统一后的）：
    {
        "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
        "input": "具体问题",
        "output": "回答内容"
    }

    提示：
    - 用 item.get("instruction", "") 获取字段，str() 转换后 .strip() 去空白
    - output 为空 → return None（这条数据丢弃）
    - instruction 统一替换为："请以谨慎、专业、易懂的方式回答下面的医疗健康问题。"
    - 如果 input 不为空，把它作为问题；如果 input 为空但 instruction 不为空，用 instruction 作为问题
    - 但因为 instruction 已经被统一了，所以只需要关注 input
    """
    # TODO: 你的代码
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    output = str(item.get("output", "")).strip()
    if not output:
        return None
    if input_text:
        # 情况 B 或 C: input 不为空
        if instruction:
            # 情况 C: instruction 和 input 都有，拼起来作为完整问题
            prompt = f"{instruction}\n{input_text}"
        else:
            # 情况 B: 只有 input，直接用它作为问题
            prompt = input_text
    else:
        # 情况 A: 只有 instruction，用它作为问题
        prompt = instruction
    if not prompt:
        return None
    return {
        "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
        "input": prompt,
        "output": output,
    }
    
    pass


def split_data(items, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """将数据划分为 train / valid / test

    参数：
        items: 清洗后的数据列表
        train_ratio: 训练集比例（默认 0.8）
        valid_ratio: 验证集比例（默认 0.1），剩余为 test
        seed: 随机种子（保证可复现）

    返回：(train_data, valid_data, test_data) 三个 list

    提示：
    - 先 random.seed(seed)
    - 再 random.shuffle(items) 打乱
    - 按比例切分
    """
    # TODO: 你的代码
    random.seed(seed)
    
    # 浅拷贝原始列表，避免 random.shuffle 修改传入的原始数据
    data = list(items)
    
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    
    return train_data, valid_data, test_data
    pass


def write_jsonl(path, items):
    """将数据列表写入 jsonl 文件

    提示：
    - path.parent.mkdir(parents=True, exist_ok=True) 确保目录存在
    - 每条数据用 json.dumps(item, ensure_ascii=False) 转成字符串
    - 写入时加换行符 "\n"
    - ensure_ascii=False 保证中文不被转义
    """
    # TODO: 你的代码
    path.parent.mkdir(parents=True, exist_ok=True)  # 如果目录不存在就创建
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            # json.dumps 把字典转成 JSON 字符串
            # ensure_ascii=False 保证中文正常显示，不会变成 中文
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    pass


def main():
    # 原始数据路径
    raw_path = Path("data/medical_raw/finetune/train_zh_0.json")
    out_dir = Path("data/medical")

    if not raw_path.exists():
        print(f"原始数据不存在: {raw_path}")
        print("请先运行 python download_dataset.py 下载数据集")
        return
    


    # TODO: 步骤 1 - 用 read_jsonl 读取原始数据
    items = read_jsonl(raw_path)
    # TODO: 步骤 2 - 逐条调用 normalize_item 清洗，过滤掉 None
    items = [normalize_item(item) for item in items]
    # TODO: 步骤 3 - 调用 split_data 划分数据集
    train_data, valid_data, test_data = split_data(items)
    # TODO: 步骤 4 - 用 write_jsonl 写出 train.jsonl、valid.jsonl、test.jsonl
    write_jsonl(out_dir / "train.jsonl", train_data)
    write_jsonl(out_dir / "valid.jsonl", valid_data)
    write_jsonl(out_dir / "test.jsonl", test_data)
    pass


if __name__ == "__main__":
    main()
