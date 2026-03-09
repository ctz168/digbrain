#!/usr/bin/env python3
"""
类脑AI系统 - 主程序入口
Brain-like AI System - Main Entry Point

功能：
1. 交互式对话
2. 流式处理演示
3. 训练模式
4. API服务器
5. 基准测试
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.brain_engine import BrainLikeStreamingEngine


def interactive_mode(engine: BrainLikeStreamingEngine):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("交互式对话模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'status' 查看系统状态")
    print("输入 'memory' 查看记忆统计")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            
            if user_input.lower() == 'status':
                status = engine.get_status()
                print(f"\n系统状态:")
                print(f"  模型加载: {status['model_loaded']}")
                print(f"  处理次数: {status['processing_count']}")
                if status['stdp']:
                    print(f"  STDP更新: {status['stdp']['update_count']}")
                if status['memory']:
                    print(f"  记忆数量: {status['memory']['total_memories']}")
                print()
                continue
            
            if user_input.lower() == 'memory':
                if engine.memory:
                    stats = engine.memory.get_stats()
                    print(f"\n记忆统计:")
                    print(f"  瞬时记忆: {stats['sensory_count']}")
                    print(f"  短期记忆: {stats['short_term_count']}")
                    print(f"  长期记忆: {stats['long_term_count']}")
                    print(f"  神经增长: {stats['neuron_growth_events']}")
                    print()
                else:
                    print("\n记忆系统未启用\n")
                continue
            
            # 流式处理
            print("\nAI: ", end='', flush=True)
            
            for chunk in engine.stream_process(user_input, max_tokens=300):
                if chunk.type == "text":
                    print(chunk.content, end='', flush=True)
                elif chunk.type == "memory_call":
                    if chunk.content:
                        print(f"\n[记忆] {chunk.content}", flush=True)
                        print("AI: ", end='', flush=True)
                elif chunk.type == "wiki_search":
                    if chunk.content:
                        print(f"\n[维基] {chunk.content[:100]}...", flush=True)
                        print("AI: ", end='', flush=True)
                elif chunk.type == "control":
                    print(f"\n")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def demo_mode(engine: BrainLikeStreamingEngine):
    """演示模式"""
    print("\n" + "=" * 60)
    print("演示模式 - 展示类脑AI系统核心功能")
    print("=" * 60)
    
    demo_questions = [
        "请解释什么是量子纠缠？",
        "TCP三次握手是什么？",
        "什么是机器学习？",
        "请写一首关于春天的短诗"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n[问题 {i}] {question}")
        print("-" * 40)
        print("AI: ", end='', flush=True)
        
        for chunk in engine.stream_process(question, max_tokens=200):
            if chunk.type == "text":
                print(chunk.content, end='', flush=True)
        
        print("\n")
    
    # 显示统计
    print("\n" + "=" * 60)
    print("系统统计")
    print("=" * 60)
    status = engine.get_status()
    print(f"处理次数: {status['processing_count']}")
    if status['stdp']:
        print(f"STDP更新: {status['stdp']['update_count']}")
        print(f"LTP/LTD: {status['stdp']['ltp_count']}/{status['stdp']['ltd_count']}")
    if status['memory']:
        print(f"记忆数量: {status['memory']['total_memories']}")


def train_mode(args):
    """训练模式"""
    from training.offline_trainer import OfflineTrainer
    
    print("\n" + "=" * 60)
    print("训练模式")
    print("=" * 60)
    
    trainer = OfflineTrainer(
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    
    if args.module:
        print(f"\n训练模块: {args.module}")
        result = trainer.train_module(args.module, epochs=args.epochs)
    else:
        print(f"\n综合训练 (并行: {args.parallel})")
        result = trainer.train_all(parallel=args.parallel, epochs=args.epochs)
    
    trainer.export_training_report(result)
    trainer.save_weights()
    
    print("\n训练完成！")


def benchmark_mode(engine: BrainLikeStreamingEngine):
    """基准测试模式"""
    from evaluation.benchmark import BenchmarkSuite, StreamingBenchmark
    
    print("\n" + "=" * 60)
    print("基准测试模式")
    print("=" * 60)
    
    # 能力评估
    benchmark = BenchmarkSuite(engine)
    report = benchmark.run_full_assessment()
    benchmark.save_report(report)
    
    # 性能测试
    streaming_benchmark = StreamingBenchmark(engine)
    perf_report = streaming_benchmark.run_full_benchmark()
    
    # 保存
    output_path = os.path.join(BASE_DIR, "evaluation/results")
    os.makedirs(output_path, exist_ok=True)
    
    perf_path = os.path.join(output_path, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(perf_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n性能报告已保存: {perf_path}")


def api_mode(args):
    """API服务器模式"""
    from api.server import BrainAPIServer
    
    server = BrainAPIServer(
        host=args.host,
        port=args.port,
        refresh_rate=args.refresh_rate
    )
    
    server.start(use_flask=args.flask)


def web_mode(args):
    """Web界面模式"""
    from web.server import BrainWebServer
    
    server = BrainWebServer(
        host=args.host,
        port=args.port,
        refresh_rate=args.refresh_rate
    )
    
    server.start()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="类脑AI系统 - Brain-like AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          # 交互式对话
  python main.py --demo                   # 演示模式
  python main.py --train --epochs 10      # 训练模式
  python main.py --api --port 8000        # API服务器
  python main.py --web --port 8000        # Web界面
  python main.py --benchmark              # 基准测试
        """
    )
    
    # 模式选择
    parser.add_argument("--demo", action="store_true", help="演示模式")
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--api", action="store_true", help="API服务器模式")
    parser.add_argument("--web", action="store_true", help="Web界面模式")
    parser.add_argument("--benchmark", action="store_true", help="基准测试模式")
    
    # 通用参数
    parser.add_argument("--refresh-rate", type=int, default=60, help="刷新率 (Hz)")
    parser.add_argument("--no-stdp", action="store_true", help="禁用STDP学习")
    parser.add_argument("--no-memory", action="store_true", help="禁用记忆系统")
    parser.add_argument("--no-wiki", action="store_true", help="禁用维基百科搜索")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="学习率")
    parser.add_argument("--module", type=str, help="训练特定模块")
    parser.add_argument("--parallel", action="store_true", help="并行训练")
    
    # API参数
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--flask", action="store_true", help="使用Flask服务器")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("类脑AI系统 | Brain-like AI System")
    print("=" * 60)
    print("\n核心特性:")
    print("  • Qwen2.5-0.5B-Instruct 语言模型")
    print("  • Qwen2-VL-2B 世界模型 (视觉)")
    print("  • STDP在线学习")
    print("  • 高刷新率流式处理 (60Hz)")
    print("  • 海马体记忆系统")
    print("  • 维基百科知识扩展")
    print("=" * 60)
    
    # API模式
    if args.api:
        api_mode(args)
        return
    
    # Web模式
    if args.web:
        web_mode(args)
        return
    
    # 训练模式
    if args.train:
        train_mode(args)
        return
    
    # 初始化引擎
    engine = BrainLikeStreamingEngine(
        refresh_rate=args.refresh_rate,
        enable_stdp=not args.no_stdp,
        enable_memory=not args.no_memory,
        enable_wiki=not args.no_wiki,
        learning_rate=args.learning_rate
    )
    
    if not engine.load_models():
        print("模型加载失败，退出")
        return 1
    
    # 基准测试模式
    if args.benchmark:
        benchmark_mode(engine)
        return
    
    # 演示模式
    if args.demo:
        demo_mode(engine)
        return
    
    # 默认：交互式模式
    interactive_mode(engine)
    
    # 保存权重
    engine.save_weights()
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
