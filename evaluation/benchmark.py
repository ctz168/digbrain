"""
真实模型基准测试
Real Model Benchmark

无作弊的真实评估
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TestQuestion:
    """测试问题"""
    id: str
    category: str
    question: str
    expected_keywords: List[str]
    difficulty: int = 1


# 真实测试数据集
BENCHMARK_DATASET = {
    "knowledge": {
        "name": "知识问答",
        "questions": [
            TestQuestion("k001", "知识", "中国的首都是哪个城市？", ["北京"], 1),
            TestQuestion("k002", "知识", "《红楼梦》的作者是谁？", ["曹雪芹"], 1),
            TestQuestion("k003", "知识", "水的化学式是什么？", ["H2O", "H₂O"], 1),
            TestQuestion("k004", "知识", "世界上最高的山峰是哪座？", ["珠穆朗玛峰", "珠峰"], 1),
            TestQuestion("k005", "知识", "人体最大的器官是什么？", ["皮肤"], 2),
            TestQuestion("k006", "知识", "太阳系中最大的行星是哪颗？", ["木星"], 2),
        ]
    },
    "reasoning": {
        "name": "逻辑推理",
        "questions": [
            TestQuestion("r001", "推理", "如果所有的A都是B，所有的B都是C，那么所有的A都是C吗？", ["是", "对", "正确"], 2),
            TestQuestion("r002", "推理", "小明比小红高，小红比小华高，谁最矮？", ["小华"], 2),
            TestQuestion("r003", "推理", "有5个苹果，拿走3个，还剩几个？", ["2", "二"], 1),
            TestQuestion("r004", "推理", "如果A比B大2岁，B比C大3岁，A比C大几岁？", ["5", "五"], 2),
        ]
    },
    "math": {
        "name": "数学能力",
        "questions": [
            TestQuestion("m001", "数学", "123 + 456 = ?", ["579"], 1),
            TestQuestion("m002", "数学", "15 × 15 = ?", ["225"], 2),
            TestQuestion("m003", "数学", "100 - 37 = ?", ["63"], 1),
            TestQuestion("m004", "数学", "81 ÷ 9 = ?", ["9"], 1),
        ]
    },
    "coding": {
        "name": "代码能力",
        "questions": [
            TestQuestion("c001", "代码", "HTTP状态码200表示什么？", ["成功", "OK"], 1),
            TestQuestion("c002", "代码", "快速排序的时间复杂度是多少？", ["O(n log n)", "O(nlogn)"], 3),
            TestQuestion("c003", "代码", "SQL的SELECT语句用于什么？", ["查询", "检索"], 1),
        ]
    },
    "complex": {
        "name": "复杂问题",
        "questions": [
            TestQuestion("x001", "复杂", "请解释什么是死锁，以及死锁产生的四个必要条件。", ["互斥", "占有", "循环", "等待"], 4),
            TestQuestion("x002", "复杂", "请解释TCP三次握手的过程。", ["SYN", "ACK", "连接"], 3),
            TestQuestion("x003", "复杂", "请解释量子纠缠的基本原理。", ["量子", "纠缠", "叠加"], 4),
        ]
    }
}

# GLM-5基准分数
GLM5_BASELINE = {
    "knowledge": 0.85,
    "reasoning": 0.78,
    "math": 0.68,
    "coding": 0.72,
    "complex": 0.65
}


class RealModelEvaluator:
    """
    真实模型评估器
    
    无作弊，使用真实模型推理
    """
    
    def __init__(self, ai_instance):
        """
        初始化评估器
        
        Args:
            ai_instance: BrainLikeAI实例
        """
        self.ai = ai_instance
        self.results = {}
        
    def evaluate_question(self, question: TestQuestion) -> Dict:
        """
        评估单个问题
        
        Args:
            question: 测试问题
            
        Returns:
            评估结果
        """
        start_time = time.time()
        
        # 使用真实模型生成回答
        response = self.ai.chat(question.question)
        
        elapsed = time.time() - start_time
        
        # 评估答案
        is_correct, matched = self._evaluate_response(
            response,
            question.expected_keywords
        )
        
        return {
            "id": question.id,
            "category": question.category,
            "question": question.question,
            "response": response[:200],
            "expected": question.expected_keywords,
            "matched": matched,
            "correct": is_correct,
            "difficulty": question.difficulty,
            "time": elapsed
        }
    
    def _evaluate_response(
        self,
        response: str,
        expected_keywords: List[str]
    ) -> tuple:
        """
        评估回答
        
        Args:
            response: 模型回答
            expected_keywords: 期望关键词
            
        Returns:
            (是否正确, 匹配的关键词)
        """
        response_lower = response.lower()
        matched = []
        
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                matched.append(keyword)
        
        # 至少匹配一半关键词才算正确
        is_correct = len(matched) >= max(1, len(expected_keywords) // 2)
        
        return is_correct, matched
    
    def run_full_evaluation(self) -> Dict:
        """
        运行完整评估
        
        Returns:
            评估结果
        """
        print("\n" + "=" * 60)
        print("开始真实模型评估")
        print("=" * 60)
        
        all_results = {}
        total_correct = 0
        total_questions = 0
        
        for category, data in BENCHMARK_DATASET.items():
            print(f"\n【{data['name']}】")
            
            correct = 0
            total = len(data['questions'])
            category_results = []
            
            for q in data['questions']:
                result = self.evaluate_question(q)
                category_results.append(result)
                
                if result['correct']:
                    correct += 1
                
                status = "✅" if result['correct'] else "❌"
                print(f"  {status} {q.question[:40]}...")
            
            accuracy = correct / total if total > 0 else 0
            
            all_results[category] = {
                "name": data['name'],
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "details": category_results
            }
            
            total_correct += correct
            total_questions += total
            
            # 与GLM-5对比
            glm5 = GLM5_BASELINE.get(category, 0.7)
            diff = accuracy - glm5
            print(f"\n  得分: {correct}/{total} ({accuracy*100:.1f}%)")
            print(f"  GLM-5对比: {diff*100:+.1f}%")
        
        overall = total_correct / total_questions if total_questions > 0 else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen3.5-0.8B",
            "evaluation_type": "real_model_no_cheating",
            "dimensions": all_results,
            "overall": {
                "correct": total_correct,
                "total": total_questions,
                "accuracy": overall
            },
            "comparison": {
                "glm5_baseline": GLM5_BASELINE,
                "difference": {
                    k: all_results[k]['accuracy'] - GLM5_BASELINE.get(k, 0.7)
                    for k in all_results
                }
            }
        }


def run_benchmark(ai_instance) -> Dict:
    """
    运行基准测试
    
    Args:
        ai_instance: BrainLikeAI实例
        
    Returns:
        测试结果
    """
    evaluator = RealModelEvaluator(ai_instance)
    return evaluator.run_full_evaluation()
