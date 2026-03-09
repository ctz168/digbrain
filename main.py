#!/usr/bin/env python3.13
"""
类脑AI系统 - 主程序
Brain-like AI System - Main Entry Point

使用真实的Qwen3.5-0.8B模型
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Generator, Dict, Any, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.streaming_engine import BrainLikeAI
from core.stdp_learning import STDPOnlineLearning
from core.memory_system import HippocampalMemory
from core.multimodal import MultimodalProcessor

# 默认路径
DEFAULT_MODEL_PATH = "./models/Qwen3.5-0.8B"
DEFAULT_WEIGHTS_PATH = "./weights"
DEFAULT_MEMORY_PATH = "./memory_storage"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="类脑AI系统")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--refresh-rate", type=int, default=60, help="刷新率")
    parser.add_argument("--mode", type=str, default="chat", choices=["chat", "eval", "train", "api"], help="运行模式")
    parser.add_argument("--port", type=int, default=8000, help="API端口")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🧠 类脑AI系统 | Brain-like AI System")
    print("=" * 60)
    print(f"\n模型: Qwen3.5-0.8B")
    print(f"刷新率: {args.refresh_rate}Hz")
    print(f"模式: {args.mode}")
    print("=" * 60)
    
    if args.mode == "chat":
        run_chat(args)
    elif args.mode == "eval":
        run_evaluation(args)
    elif args.mode == "train":
        run_training(args)
    elif args.mode == "api":
        run_api(args)


def run_chat(args):
    """运行对话模式"""
    print("\n初始化系统...")
    
    ai = BrainLikeAI(
        model_path=args.model,
        refresh_rate=args.refresh_rate
    )
    ai.initialize()
    
    print("\n开始对话 (输入 'quit' 退出)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            
            if not user_input:
                continue
            
            print("\n助手: ", end="", flush=True)
            
            # 流式输出
            for chunk in ai.stream_chat(user_input):
                if chunk.type == "text":
                    print(chunk.content, end="", flush=True)
                elif chunk.type == "memory_call":
                    print(f"\n[记忆] {chunk.content}", flush=True)
                elif chunk.type == "learning":
                    pass  # 静默处理学习事件
                elif chunk.type == "control":
                    print(f"\n\n[完成] {chunk.metadata.get('total_tokens', 0)} tokens, "
                          f"{chunk.metadata.get('avg_tokens_per_second', 0):.1f} t/s")
            
        except KeyboardInterrupt:
            print("\n\n中断")
            break
        except Exception as e:
            print(f"\n错误: {e}")
    
    # 保存状态
    ai.save_state(os.path.join(DEFAULT_WEIGHTS_PATH, "final_state.json"))
    print("\n状态已保存")


def run_evaluation(args):
    """运行评估模式"""
    print("\n运行基准测试...")
    
    from evaluation.benchmark import run_benchmark
    
    ai = BrainLikeAI(
        model_path=args.model,
        refresh_rate=args.refresh_rate
    )
    ai.initialize()
    
    results = run_benchmark(ai)
    
    # 保存结果
    output_path = os.path.join(DEFAULT_WEIGHTS_PATH, f"eval_{int(time.time())}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_path}")


def run_training(args):
    """运行训练模式"""
    print("\n运行训练...")
    
    from training.offline_trainer import OfflineTrainer
    
    trainer = OfflineTrainer(
        model_path=args.model,
        output_path=DEFAULT_WEIGHTS_PATH
    )
    
    trainer.train(epochs=10)
    trainer.save_weights()
    
    print("\n训练完成")


def run_api(args):
    """运行API模式"""
    print(f"\n启动API服务器 (端口: {args.port})...")
    
    from api.server import run_server
    
    run_server(port=args.port, model_path=args.model)


if __name__ == "__main__":
    main()
