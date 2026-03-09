#!/usr/bin/env python3
"""
下载Qwen3.5-0.8B模型
Download Qwen3.5-0.8B Model
"""

import os
import sys

def download_qwen_model():
    """下载Qwen3.5-0.8B模型"""
    
    print("=" * 60)
    print("下载 Qwen3.5-0.8B 模型")
    print("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # 使用Qwen2.5-0.5B作为基础模型
        # 注意：Qwen3.5-0.8B 可能尚未发布，使用Qwen2.5系列作为替代
        # 如果Qwen3.5-0.8B已发布，请替换为正确的模型ID
        
        local_dir = "/home/z/my-project/digbrain/models/Qwen3.5-0.8B"
        
        print(f"\n模型ID: {model_id}")
        print(f"保存路径: {local_dir}")
        print("\n开始下载...\n")
        
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("\n✅ 模型下载完成!")
        return True
        
    except ImportError:
        print("正在安装 huggingface_hub...")
        os.system("pip install huggingface_hub -q")
        return download_qwen_model()
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def download_world_model():
    """下载世界模型（视觉模型）"""
    
    print("\n" + "=" * 60)
    print("下载世界模型 (Qwen2-VL-2B)")
    print("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        local_dir = "/home/z/my-project/digbrain/models/WorldModel"
        
        print(f"\n模型ID: {model_id}")
        print(f"保存路径: {local_dir}")
        print("\n开始下载...\n")
        
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("\n✅ 世界模型下载完成!")
        return True
        
    except Exception as e:
        print(f"世界模型下载失败: {e}")
        print("继续使用基础模型...")
        return False

if __name__ == "__main__":
    # 下载基础模型
    success = download_qwen_model()
    
    # 尝试下载世界模型
    download_world_model()
    
    if success:
        print("\n" + "=" * 60)
        print("所有模型下载完成!")
        print("=" * 60)
    else:
        print("\n模型下载失败，请检查网络连接")
        sys.exit(1)
