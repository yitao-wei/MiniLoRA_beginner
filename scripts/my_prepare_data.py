"""
模块 1：数据准备和预处理

目标：从原始 json 文件生成训练用的 jsonl 文件。

运行方式：
    python scripts/my_prepare_data.py

运行后检查 data/medical/ 目录下是否生成了 train.jsonl、valid.jsonl、test.jsonl

参考代码：scripts/prepare_medical_sft.py（先看懂，关掉，再自己写）

原始数据集 shibing624/medical 已经有 train/valid/test 的分割：
- train_zh_0.json: 训练集
- valid_zh_0.json: 验证集
- test_zh_0.json:  测试集

本脚本分别读取三个文件，清洗后直接使用原始分割，无需重新划分。
"""

import argparse
import json
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
        "instruction": "各种不同的指令，有的为空",
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
    parser = argparse.ArgumentParser(description="Prepare medical SFT data from raw JSON files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/medical_raw"),
                        help="Raw data directory")
    parser.add_argument("--out-dir", type=Path, default=Path("data/medical"),
                        help="Output directory for processed data")
    args = parser.parse_args()

    finetune_dir = args.raw_dir / "finetune"
    train_path = finetune_dir / "train_zh_0.json"
    valid_path = finetune_dir / "valid_zh_0.json"
    test_path = finetune_dir / "test_zh_0.json"

    # 检查文件是否存在
    if not any(p.exists() for p in [train_path, valid_path, test_path]):
        print(f"原始数据不存在: {finetune_dir}")
        print("请先运行 python download_dataset.py 下载数据集")
        return

    # --- 读取三个文件，分别清洗 ---
    print(f"读取原始文件: train_zh_0.json, valid_zh_0.json, test_zh_0.json")

    # TODO: 你的代码
    # 分别读取三个文件，分别清洗，直接使用原始分割
    # 提示：
    #   train_data = [item for item in (normalize_item(i) for i in read_jsonl(train_path)) if item is not None]
    #   valid_data = [item for item in (normalize_item(i) for i in read_jsonl(valid_path)) if item is not None]
    #   test_data  = [item for item in (normalize_item(i) for i in read_jsonl(test_path))  if item is not None]
    train_data, valid_data, test_data = [], [], []

    print(f"清洗后: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")

    # --- 写出 jsonl 文件 ---
    # TODO: 你的代码
    # 提示：
    #   write_jsonl(args.out_dir / "train.jsonl", train_data)
    #   write_jsonl(args.out_dir / "valid.jsonl", valid_data)
    #   write_jsonl(args.out_dir / "test.jsonl", test_data)

    print(f"\n输出到 {args.out_dir}/")
    print(f"  train.jsonl: {len(train_data)} 条")
    print(f"  valid.jsonl: {len(valid_data)} 条")
    print(f"  test.jsonl:  {len(test_data)} 条")


if __name__ == "__main__":
    main()
