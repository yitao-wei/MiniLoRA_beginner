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
    # TODO: 步骤 2 - 逐条调用 normalize_item 清洗，过滤掉 None
    # TODO: 步骤 3 - 调用 split_data 划分数据集
    # TODO: 步骤 4 - 用 write_jsonl 写出 train.jsonl、valid.jsonl、test.jsonl
    pass


if __name__ == "__main__":
    main()
