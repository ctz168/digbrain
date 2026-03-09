#!/usr/bin/env python3
"""
Qwen模型下载脚本
Qwen Model Download Script

支持下载：
- Qwen2.5-0.5B-Instruct (约954MB)
- Qwen2.5-0.5B-Instruct 是最新稳定版本
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
import subprocess

# 配置
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# 模型配置
MODELS = {
    "Qwen2.5-0.5B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "size": "约954MB",
        "description": "轻量级语言模型，适合快速推理",
        "params": "0.5B"
    },
    "Qwen2-VL-2B-Instruct": {
        "repo_id": "Qwen/Qwen2-VL-2B-Instruct",
        "size": "约4.2GB",
        "description": "视觉语言模型，支持图像和视频理解",
        "params": "2B"
    }
}


def check_huggingface_cli() -> bool:
    """检查huggingface-cli是否可用"""
    try:
        result = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_dependencies():
    """安装必要依赖"""
    print("安装必要依赖...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "huggingface_hub", "transformers", "torch", "--quiet"
    ])


def download_model_hf(model_name: str, target_dir: Path) -> bool:
    """使用Hugging Face Hub下载模型"""
    from huggingface_hub import snapshot_download
    
    if model_name not in MODELS:
        print(f"未知模型: {model_name}")
        return False
    
    model_info = MODELS[model_name]
    repo_id = model_info["repo_id"]
    
    print(f"\n{'='*60}")
    print(f"下载模型: {model_name}")
    print(f"仓库: {repo_id}")
    print(f"大小: {model_info['size']}")
    print(f"目标: {target_dir}")
    print(f"{'='*60}\n")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"\n✅ 模型下载完成: {target_dir}")
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def download_model_transformers(model_name: str, target_dir: Path) -> bool:
    """使用transformers下载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if model_name not in MODELS:
        print(f"未知模型: {model_name}")
        return False
    
    model_info = MODELS[model_name]
    repo_id = model_info["repo_id"]
    
    print(f"\n{'='*60}")
    print(f"下载模型: {model_name}")
    print(f"仓库: {repo_id}")
    print(f"{'='*60}\n")
    
    try:
        # 下载tokenizer
        print("下载Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(str(target_dir))
        print("✅ Tokenizer下载完成")
        
        # 下载模型
        print("下载模型权重（这可能需要几分钟）...")
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        model.save_pretrained(str(target_dir))
        print("✅ 模型下载完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def verify_model(model_dir: Path) -> Dict:
    """验证模型完整性"""
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    result = {
        "valid": True,
        "missing_files": [],
        "total_size": 0
    }
    
    for file in required_files:
        file_path = model_dir / file
        if not file_path.exists():
            result["missing_files"].append(file)
            result["valid"] = False
    
    # 计算总大小
    if model_dir.exists():
        for file in model_dir.rglob("*"):
            if file.is_file():
                result["total_size"] += file.stat().st_size
    
    return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="下载Qwen模型")
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()) + ["all"],
        default="Qwen2.5-0.5B-Instruct",
        help="要下载的模型"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="模型保存目录"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出可用模型"
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="验证已下载模型"
    )
    
    args = parser.parse_args()
    
    # 列出模型
    if args.list:
        print("\n可用模型：")
        print("-" * 60)
        for name, info in MODELS.items():
            print(f"  📦 {name}")
            print(f"     大小: {info['size']}")
            print(f"     参数: {info['params']}")
            print(f"     描述: {info['description']}")
            print()
        return
    
    # 确定模型目录
    models_dir = Path(args.dir) if args.dir else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定要下载的模型
    if args.model == "all":
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = [args.model]
    
    # 验证模式
    if args.verify:
        print("\n验证模型完整性：")
        print("-" * 60)
        for model_name in models_to_download:
            model_dir = models_dir / model_name.replace("-", "_")
            if model_dir.exists():
                result = verify_model(model_dir)
                status = "✅" if result["valid"] else "❌"
                size_mb = result["total_size"] / (1024 * 1024)
                print(f"{status} {model_name}")
                print(f"   大小: {size_mb:.1f}MB")
                if result["missing_files"]:
                    print(f"   缺失文件: {result['missing_files']}")
            else:
                print(f"❌ {model_name} - 未下载")
        return
    
    # 安装依赖
    try:
        import huggingface_hub
    except ImportError:
        install_dependencies()
    
    # 下载模型
    success_count = 0
    for model_name in models_to_download:
        model_dir = models_dir / model_name.replace("-", "_")
        
        # 检查是否已存在
        if model_dir.exists():
            result = verify_model(model_dir)
            if result["valid"]:
                print(f"\n✅ {model_name} 已存在，跳过下载")
                success_count += 1
                continue
        
        # 下载
        if download_model_hf(model_name, model_dir):
            success_count += 1
        elif download_model_transformers(model_name, model_dir):
            success_count += 1
    
    # 总结
    print("\n" + "=" * 60)
    print(f"下载完成: {success_count}/{len(models_to_download)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
