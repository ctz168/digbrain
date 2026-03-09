#!/usr/bin/env python3
"""
测试评分模块
Benchmark and Assessment Module

提供多维度能力评估
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

OUTPUT_PATH = os.path.join(BASE_DIR, "evaluation/results")


@dataclass
class AssessmentResult:
    """评估结果"""
    dimension: str
    score: float
    max_score: float = 100.0
    details: Dict = None
    
    @property
    def percentage(self) -> float:
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0


class BenchmarkSuite:
    """
    基准测试套件
    多维度能力评估
    """
    
    def __init__(self, engine=None):
        self.engine = engine
        self.results: List[AssessmentResult] = []
        
        # 测试数据集
        self.test_sets = {
            "math": [
                {"question": "计算: 15 + 27 = ?", "answer": "42", "type": "arithmetic"},
                {"question": "计算: 100 - 37 = ?", "answer": "63", "type": "arithmetic"},
                {"question": "计算: 8 × 7 = ?", "answer": "56", "type": "arithmetic"},
                {"question": "计算: 144 ÷ 12 = ?", "answer": "12", "type": "arithmetic"},
                {"question": "如果一个三角形的底是6，高是4，面积是多少？", "answer": "12", "type": "geometry"},
                {"question": "求方程 2x + 5 = 13 的解", "answer": "4", "type": "algebra"},
            ],
            "code": [
                {"question": "写一个Python函数，计算列表中所有数字的和", "keywords": ["def", "sum", "return"], "type": "basic"},
                {"question": "写一个函数判断一个数是否为质数", "keywords": ["def", "if", "for", "return"], "type": "algorithm"},
                {"question": "写一个Python类表示一个简单的银行账户", "keywords": ["class", "def", "self", "__init__"], "type": "oop"},
                {"question": "写一个函数实现二分查找算法", "keywords": ["def", "while", "mid", "return"], "type": "algorithm"},
            ],
            "knowledge": [
                {"question": "中国的首都是哪里？", "answer": "北京", "type": "geography"},
                {"question": "地球绕太阳一周需要多长时间？", "answer": "一年", "type": "science"},
                {"question": "谁发明了电话？", "answer": "贝尔", "type": "history"},
                {"question": "水的化学式是什么？", "answer": "H2O", "type": "chemistry"},
                {"question": "《红楼梦》的作者是谁？", "answer": "曹雪芹", "type": "literature"},
                {"question": "光速大约是多少？", "answer": "30万公里每秒", "type": "physics"},
            ],
            "reasoning": [
                {"question": "如果所有的猫都是动物，所有的动物都需要水，那么猫需要水吗？", "answer": "需要", "type": "deduction"},
                {"question": "A比B高，B比C高，谁最高？", "answer": "A", "type": "transitive"},
                {"question": "如果今天是星期三，后天是星期几？", "answer": "星期五", "type": "temporal"},
                {"question": "一个房间里有3盏灯，外面有3个开关，每个开关控制一盏灯。你只能进入房间一次，如何确定哪个开关控制哪盏灯？", "type": "puzzle"},
            ],
            "creativity": [
                {"question": "请写一首关于春天的短诗", "type": "poetry"},
                {"question": "请编一个关于机器人和人类友谊的短故事", "type": "story"},
                {"question": "如果可以时间旅行，你会去哪个时代？为什么？", "type": "imagination"},
                {"question": "设计一个未来城市的交通系统", "type": "design"},
            ]
        }
    
    def set_engine(self, engine):
        """设置引擎"""
        self.engine = engine
    
    def assess_math(self) -> AssessmentResult:
        """数学能力评估"""
        if not self.engine:
            return AssessmentResult("math", 0, details={"error": "No engine"})
        
        correct = 0
        total = len(self.test_sets["math"])
        details = []
        
        for test in self.test_sets["math"]:
            # 获取答案
            response = ""
            for chunk in self.engine.stream_process(test["question"], max_tokens=50, search_wiki=False):
                if chunk.type == "text":
                    response += chunk.content
            
            # 检查答案
            is_correct = test["answer"] in response
            if is_correct:
                correct += 1
            
            details.append({
                "question": test["question"],
                "expected": test["answer"],
                "response": response[:100],
                "correct": is_correct
            })
        
        score = (correct / total) * 100
        return AssessmentResult("math", score, details={"correct": correct, "total": total, "tests": details})
    
    def assess_code(self) -> AssessmentResult:
        """代码能力评估"""
        if not self.engine:
            return AssessmentResult("code", 0, details={"error": "No engine"})
        
        score = 0
        total = len(self.test_sets["code"])
        details = []
        
        for test in self.test_sets["code"]:
            response = ""
            for chunk in self.engine.stream_process(test["question"], max_tokens=200, search_wiki=False):
                if chunk.type == "text":
                    response += chunk.content
            
            # 检查关键词
            keywords_found = sum(1 for kw in test.get("keywords", []) if kw in response)
            keyword_score = (keywords_found / len(test["keywords"])) * 100 if test.get("keywords") else 50
            
            # 检查代码结构
            has_def = "def " in response
            has_return = "return" in response
            structure_score = (has_def + has_return) * 25
            
            test_score = (keyword_score + structure_score) / 2
            score += test_score
            
            details.append({
                "question": test["question"],
                "response": response[:200],
                "keywords_found": keywords_found,
                "score": test_score
            })
        
        avg_score = score / total
        return AssessmentResult("code", avg_score, details={"tests": details})
    
    def assess_knowledge(self) -> AssessmentResult:
        """知识问答评估"""
        if not self.engine:
            return AssessmentResult("knowledge", 0, details={"error": "No engine"})
        
        correct = 0
        total = len(self.test_sets["knowledge"])
        details = []
        
        for test in self.test_sets["knowledge"]:
            response = ""
            for chunk in self.engine.stream_process(test["question"], max_tokens=50, search_wiki=True):
                if chunk.type == "text":
                    response += chunk.content
            
            # 检查答案
            is_correct = test["answer"] in response or test["answer"].lower() in response.lower()
            if is_correct:
                correct += 1
            
            details.append({
                "question": test["question"],
                "expected": test["answer"],
                "response": response[:100],
                "correct": is_correct
            })
        
        score = (correct / total) * 100
        return AssessmentResult("knowledge", score, details={"correct": correct, "total": total, "tests": details})
    
    def assess_reasoning(self) -> AssessmentResult:
        """逻辑推理评估"""
        if not self.engine:
            return AssessmentResult("reasoning", 0, details={"error": "No engine"})
        
        score = 0
        total = len(self.test_sets["reasoning"])
        details = []
        
        for test in self.test_sets["reasoning"]:
            response = ""
            for chunk in self.engine.stream_process(test["question"], max_tokens=100, search_wiki=False):
                if chunk.type == "text":
                    response += chunk.content
            
            # 检查答案
            if "answer" in test:
                is_correct = test["answer"] in response
                test_score = 100 if is_correct else 0
            else:
                # 主观评分
                test_score = 50  # 基础分
                if "因为" in response or "所以" in response:
                    test_score += 25
                if len(response) > 50:
                    test_score += 25
            
            score += test_score
            
            details.append({
                "question": test["question"],
                "response": response[:150],
                "score": test_score
            })
        
        avg_score = score / total
        return AssessmentResult("reasoning", avg_score, details={"tests": details})
    
    def assess_creativity(self) -> AssessmentResult:
        """创造性写作评估"""
        if not self.engine:
            return AssessmentResult("creativity", 0, details={"error": "No engine"})
        
        score = 0
        total = len(self.test_sets["creativity"])
        details = []
        
        for test in self.test_sets["creativity"]:
            response = ""
            for chunk in self.engine.stream_process(test["question"], max_tokens=200, search_wiki=False):
                if chunk.type == "text":
                    response += chunk.content
            
            # 评估创造性
            test_score = 0
            
            # 长度评分
            if len(response) > 100:
                test_score += 30
            elif len(response) > 50:
                test_score += 15
            
            # 内容丰富度
            unique_words = len(set(response.split()))
            if unique_words > 30:
                test_score += 30
            elif unique_words > 15:
                test_score += 15
            
            # 结构性
            if "。" in response or "！" in response or "？" in response:
                test_score += 20
            
            # 创意元素
            creative_words = ["想象", "未来", "美丽", "神奇", "奇妙", "独特", "创新"]
            creative_count = sum(1 for w in creative_words if w in response)
            test_score += min(creative_count * 5, 20)
            
            score += test_score
            
            details.append({
                "question": test["question"],
                "response": response[:200],
                "length": len(response),
                "unique_words": unique_words,
                "score": test_score
            })
        
        avg_score = score / total
        return AssessmentResult("creativity", avg_score, details={"tests": details})
    
    def run_full_assessment(self) -> Dict:
        """运行完整评估"""
        print("\n" + "=" * 60)
        print("开始多维度能力评估")
        print("=" * 60)
        
        start_time = time.time()
        
        # 运行各项评估
        assessments = {
            "math": self.assess_math(),
            "code": self.assess_code(),
            "knowledge": self.assess_knowledge(),
            "reasoning": self.assess_reasoning(),
            "creativity": self.assess_creativity()
        }
        
        # 计算总分
        total_score = sum(a.score for a in assessments.values()) / len(assessments)
        
        # 生成报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_time": time.time() - start_time,
            "overall_score": total_score,
            "dimensions": {
                name: {
                    "score": result.score,
                    "percentage": result.percentage,
                    "details": result.details
                }
                for name, result in assessments.items()
            }
        }
        
        # 打印结果
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        
        for name, result in assessments.items():
            print(f"  {name:12s}: {result.score:6.1f}%")
        
        print(f"\n  {'综合得分':12s}: {total_score:6.1f}%")
        print("=" * 60)
        
        return report
    
    def save_report(self, report: Dict, path: str = None):
        """保存报告"""
        path = path or os.path.join(OUTPUT_PATH, f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存: {path}")


class StreamingBenchmark:
    """
    流式处理性能基准测试
    """
    
    def __init__(self, engine=None):
        self.engine = engine
    
    def set_engine(self, engine):
        """设置引擎"""
        self.engine = engine
    
    def measure_latency(self, prompt: str, max_tokens: int = 100) -> Dict:
        """测量延迟"""
        if not self.engine:
            return {"error": "No engine"}
        
        start_time = time.time()
        first_token_time = None
        token_times = []
        token_count = 0
        
        for chunk in self.engine.stream_process(prompt, max_tokens=max_tokens):
            if chunk.type == "text":
                if first_token_time is None:
                    first_token_time = time.time()
                token_times.append(time.time())
                token_count += 1
        
        total_time = time.time() - start_time
        
        return {
            "prompt": prompt,
            "total_time": total_time,
            "time_to_first_token": first_token_time - start_time if first_token_time else 0,
            "tokens_generated": token_count,
            "tokens_per_second": token_count / total_time if total_time > 0 else 0,
            "avg_token_interval": (token_times[-1] - token_times[0]) / (len(token_times) - 1) if len(token_times) > 1 else 0
        }
    
    def measure_memory_performance(self, queries: List[str]) -> Dict:
        """测量记忆性能"""
        if not self.engine or not self.engine.memory:
            return {"error": "Memory not available"}
        
        results = []
        
        for query in queries:
            start_time = time.time()
            memories = self.engine.memory.search(query)
            search_time = time.time() - start_time
            
            results.append({
                "query": query,
                "search_time": search_time,
                "results_found": len(memories)
            })
        
        avg_time = sum(r["search_time"] for r in results) / len(results)
        
        return {
            "queries": len(queries),
            "avg_search_time": avg_time,
            "details": results
        }
    
    def measure_stdp_learning(self, iterations: int = 100) -> Dict:
        """测量STDP学习性能"""
        if not self.engine or not self.engine.stdp:
            return {"error": "STDP not available"}
        
        initial_updates = self.engine.stdp.update_count
        
        # 运行一些处理
        for i in range(3):
            for chunk in self.engine.stream_process(f"测试问题 {i}", max_tokens=30):
                pass
        
        final_updates = self.engine.stdp.update_count
        
        return {
            "iterations": iterations,
            "updates_performed": final_updates - initial_updates,
            "total_updates": final_updates,
            "ltp_count": self.engine.stdp.ltp_count,
            "ltd_count": self.engine.stdp.ltd_count
        }
    
    def run_full_benchmark(self) -> Dict:
        """运行完整基准测试"""
        print("\n" + "=" * 60)
        print("流式处理性能基准测试")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "latency": None,
            "memory": None,
            "stdp": None
        }
        
        # 延迟测试
        print("\n[1/3] 延迟测试...")
        results["latency"] = self.measure_latency("请解释什么是人工智能？")
        print(f"  首token延迟: {results['latency']['time_to_first_token']:.3f}s")
        print(f"  生成速度: {results['latency']['tokens_per_second']:.1f} tokens/s")
        
        # 记忆测试
        print("\n[2/3] 记忆性能测试...")
        results["memory"] = self.measure_memory_performance([
            "人工智能",
            "机器学习",
            "神经网络"
        ])
        print(f"  平均搜索时间: {results['memory']['avg_search_time']*1000:.2f}ms")
        
        # STDP测试
        print("\n[3/3] STDP学习测试...")
        results["stdp"] = self.measure_stdp_learning()
        print(f"  总更新次数: {results['stdp']['total_updates']}")
        print(f"  LTP/LTD: {results['stdp']['ltp_count']}/{results['stdp']['ltd_count']}")
        
        print("\n" + "=" * 60)
        
        return results


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("类脑AI系统 - 测试评分模块")
    print("=" * 60)
    
    # 导入引擎
    from core.brain_engine import BrainLikeStreamingEngine
    
    # 初始化引擎
    engine = BrainLikeStreamingEngine(
        refresh_rate=60,
        enable_stdp=True,
        enable_memory=True,
        enable_wiki=True
    )
    
    if not engine.load_models():
        print("模型加载失败")
        return
    
    # 运行评估
    benchmark = BenchmarkSuite(engine)
    report = benchmark.run_full_assessment()
    benchmark.save_report(report)
    
    # 运行性能测试
    streaming_benchmark = StreamingBenchmark(engine)
    perf_report = streaming_benchmark.run_full_benchmark()
    
    # 保存性能报告
    perf_path = os.path.join(OUTPUT_PATH, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(perf_report, f, ensure_ascii=False, indent=2)
    print(f"\n性能报告已保存: {perf_path}")


if __name__ == "__main__":
    main()
