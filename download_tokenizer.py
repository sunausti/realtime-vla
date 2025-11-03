#!/usr/bin/env python3
"""
下载 PaliGemma tokenizer 到本地

使用方法:
    python download_tokenizer.py
    python download_tokenizer.py --output_dir ./my_tokenizer
"""

import argparse
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="下载 PaliGemma tokenizer")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./paligemma-3b-pt-224',
        help='tokenizer 保存路径'
    )
    args = parser.parse_args()
    
    print(f"正在下载 PaliGemma tokenizer...")
    print(f"保存路径: {args.output_dir}")
    
    # 下载并保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/paligemma-3b-pt-224",
        cache_dir=None  # 使用默认缓存
    )
    
    # 保存到指定目录
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✓ Tokenizer 已下载到: {args.output_dir}")
    print(f"\n使用方法:")
    print(f"  --tokenizer_path {args.output_dir}")


if __name__ == "__main__":
    main()
