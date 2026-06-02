"""
模块 1：数据准备和预处理

目标：从原始 json 文件生成训练用的 jsonl 文件。

运行方式：
    # 方法1：只使用 train_zh_0.json（默认，推荐）
    python scripts/my_prepare_data.py

    # 方法2：使用所有原始文件（train + valid + test），直接使用原始分割
    python scripts/my_prepare_data.py --use-all-files

运行后检查 data/medical/ 目录下是否生成了 train.jsonl、valid.jsonl、test.jsonl

参考代码：scripts/prepare_medical_sft.py（先看懂，关掉，再自己写）

两种方法的区别：
- 方法1（默认）：只读取 train_zh_0.json，从中随机抽取 800 条重新划分
  - 优点：速度快，只需要下载一个文件
  - 优点：划分结果完全随机，更均匀
  - 缺点：只用了原始数据的一部分

- 方法2（--use-all-files）：读取 train_zh_0.json + valid_zh_0.json + test_zh_0.json，直接使用原始分割
  - 优点：使用了所有原始数据
  - 优点：保留了原始的 train/valid/test 分割
  - 缺点：需要下载三个文件，速度较慢
  - 缺点：原始分割可能有偏差（如某些疾病类别分布不均）
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Prepare medical SFT data from raw JSON files.")
    parser.add_argument("--use-all-files", action="store_true",
                        help="Use all raw files (train + valid + test) with original split")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/medical_raw"),
                        help="Raw data directory")
    parser.add_argument("--out-dir", type=Path, default=Path("data/medical"),
                        help="Output directory for processed data")
    parser.add_argument("--max-samples", type=int, default=800,
                        help="Maximum number of samples to use (default: 800, only for method 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    finetune_dir = args.raw_dir / "finetune"

    if args.use_all_files:
        # ============================================================
        # 方法2：读取三个文件，直接使用原始的 train/valid/test 分割
        # ============================================================
        print("方法2：使用所有原始文件（train + valid + test），直接使用原始分割")

        train_path = finetune_dir / "train_zh_0.json"
        valid_path = finetune_dir / "valid_zh_0.json"
        test_path = finetune_dir / "test_zh_0.json"

        if not any(p.exists() for p in [train_path, valid_path, test_path]):
            print(f"原始数据不存在: {finetune_dir}")
            print("请先运行 python download_dataset.py 下载数据集")
            return

        # TODO: 你的代码
        # 分别读取三个文件，分别清洗，直接写出
        # 提示：
        #   train_data = [item for item in (normalize_item(i) for i in read_jsonl(train_path)) if item is not None]
        #   valid_data = [item for item in (normalize_item(i) for i in read_jsonl(valid_path)) if item is not None]
        #   test_data  = [item for item in (normalize_item(i) for i in read_jsonl(test_path))  if item is not None]
        train_data, valid_data, test_data = [], [], []

        # TODO: 用 write_jsonl 分别写出
        # 提示：
        #   write_jsonl(args.out_dir / "train.jsonl", train_data)
        #   write_jsonl(args.out_dir / "valid.jsonl", valid_data)
        #   write_jsonl(args.out_dir / "test.jsonl", test_data)

    else:
        # ============================================================
        # 方法1：只读取 train_zh_0.json，随机抽取后重新划分
        # ============================================================
        print("方法1：只使用 train_zh_0.json（默认）")

        raw_path = finetune_dir / "train_zh_0.json"
        if not raw_path.exists():
            print(f"原始数据不存在: {raw_path}")
            print("请先运行 python download_dataset.py 下载数据集")
            return

        # TODO: 你的代码（用 read_jsonl 读取）
        raw_items = []
        # TODO: 在这里写代码
        # 提示：raw_items = read_jsonl(raw_path)
        print(f"原始数据: {len(raw_items)} 条")

        # TODO: 你的代码（用 normalize_item 清洗）
        normalized = []
        # TODO: 在这里写代码
        # 提示：normalized = [item for item in (normalize_item(i) for i in raw_items) if item is not None]
        print(f"清洗后: {len(normalized)} 条")

        # 如果数据量超过 max_samples，随机抽取
        if len(normalized) > args.max_samples:
            random.seed(args.seed)
            random.shuffle(normalized)
            normalized = normalized[:args.max_samples]
            print(f"随机抽取: {args.max_samples} 条")

        # TODO: 你的代码（用 split_data 划分）
        train_data, valid_data, test_data = [], [], []
        # TODO: 在这里写代码
        # 提示：train_data, valid_data, test_data = split_data(normalized, seed=args.seed)

        # TODO: 你的代码（用 write_jsonl 写出）
        # TODO: 在这里写代码
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
