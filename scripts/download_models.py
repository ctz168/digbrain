#!/usr/bin/env python3
"""
模型下载脚本
"""

import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def download_qwen():
    """下载Qwen3.5-0.8B"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        return False
    
    model_dir = os.path.join(MODELS_DIR, "Qwen3.5-0.8B")
    
    print(f"下载 Qwen3.5-0.8B 到 {model_dir}")
    print("模型大小约 1.7GB...\n")
    
    snapshot_download(
        repo_id="Qwen/Qwen3.5-0.8B",
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
    
    print("✅ 下载完成！")
    return True

def main():
    print("\n类脑AI - 模型下载")
    print("=" * 40)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("\n1. 下载Qwen3.5-0.8B (必需)")
    print("2. 退出")
    
    choice = input("\n选择: ").strip()
    
    if choice == "1":
        download_qwen()

if __name__ == "__main__":
    main()
