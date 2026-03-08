#!/usr/bin/env python3.13
"""
类脑AI系统 - 真实模型多维度能力测评
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/home/z/my-project/brain-like-ai/models/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "/home/z/my-project/brain-like-ai/evaluation/results"

# 测试数据集
TEST_DATASETS = {
    "knowledge": {
        "name": "知识问答",
        "questions": [
            {"id": "k001", "q": "中国的首都是哪个城市？", "a": ["北京"]},
            {"id": "k002", "q": "《红楼梦》的作者是谁？", "a": ["曹雪芹"]},
            {"id": "k003", "q": "水的化学式是什么？", "a": ["H2O", "H₂O"]},
            {"id": "k004", "q": "世界上最高的山峰是哪座？", "a": ["珠穆朗玛峰", "珠峰"]},
            {"id": "k005", "q": "人体最大的器官是什么？", "a": ["皮肤"]},
            {"id": "k006", "q": "太阳系中最大的行星是哪颗？", "a": ["木星"]},
        ]
    },
    "reasoning": {
        "name": "逻辑推理",
        "questions": [
            {"id": "r001", "q": "如果所有的A都是B，所有的B都是C，那么所有的A都是C吗？", "a": ["是", "对", "正确"]},
            {"id": "r002", "q": "小明比小红高，小红比小华高，谁最矮？", "a": ["小华"]},
            {"id": "r003", "q": "有5个苹果，拿走3个，还剩几个？", "a": ["2", "二"]},
            {"id": "r004", "q": "如果A比B大2岁，B比C大3岁，A比C大几岁？", "a": ["5", "五"]},
        ]
    },
    "math": {
        "name": "数学能力",
        "questions": [
            {"id": "m001", "q": "123 + 456 = ?", "a": ["579"]},
            {"id": "m002", "q": "15 × 15 = ?", "a": ["225"]},
            {"id": "m003", "q": "100 - 37 = ?", "a": ["63"]},
            {"id": "m004", "q": "81 ÷ 9 = ?", "a": ["9"]},
        ]
    },
    "coding": {
        "name": "代码能力",
        "questions": [
            {"id": "c001", "q": "HTTP状态码200表示什么？", "a": ["成功", "OK"]},
            {"id": "c002", "q": "快速排序的时间复杂度是多少？", "a": ["O(n log n)", "O(nlogn)"]},
            {"id": "c003", "q": "SQL的SELECT语句用于什么？", "a": ["查询", "检索"]},
        ]
    },
    "language": {
        "name": "语言理解",
        "questions": [
            {"id": "l001", "q": "画蛇添足是什么意思？", "a": ["多余", "多此一举"]},
            {"id": "l002", "q": "守株待兔是褒义还是贬义？", "a": ["贬义"]},
            {"id": "l003", "q": "一石二鸟是什么意思？", "a": ["一举两得"]},
        ]
    }
}

GLM5_BASELINE = {
    "knowledge": 0.85,
    "reasoning": 0.78,
    "math": 0.68,
    "coding": 0.72,
    "language": 0.82
}

class RealModelEvaluator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        print("=" * 60)
        print("加载真实Qwen模型")
        print("=" * 60)
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        print(f"模型路径: {self.model_path}")
        
        config_file = os.path.join(self.model_path, "config.json")
        model_file = os.path.join(self.model_path, "model.safetensors")
        
        print(f"✅ 配置文件: {os.path.getsize(config_file)} bytes")
        print(f"✅ 模型文件: {os.path.getsize(model_file) / 1024 / 1024:.1f} MB")
        
        print("\n正在加载配置...")
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        print(f"✅ 模型类型: {config.model_type}")
        
        print("\n正在加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        print("✅ Tokenizer加载完成")
        
        print("\n正在加载模型权重（需要几分钟）...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        print("✅ 模型加载完成")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"参数量: {total_params / 1e6:.2f}M")
        
        self.model.eval()
        return True
    
    def generate_response(self, prompt: str, max_new_tokens: int = 30) -> str:
        import torch
        
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in response:
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        return response.strip()
    
    def evaluate_answer(self, response: str, expected: List[str]) -> Tuple[bool, float, str]:
        response_lower = response.lower().strip()
        
        if not expected:
            return True, 0.8, "开放式问题"
        
        for exp in expected:
            if exp.lower() in response_lower:
                return True, 1.0, f"匹配: {exp}"
        
        return False, 0.0, f"期望: {expected}"
    
    def run_evaluation(self) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("开始真实模型评估")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen2.5-0.5B-Instruct",
            "evaluation_type": "real_model_inference",
            "dimensions": {},
            "overall": {}
        }
        
        total_correct = 0
        total_questions = 0
        
        for dim_key, dataset in TEST_DATASETS.items():
            print(f"\n【{dataset['name']}】")
            
            correct = 0
            total = len(dataset['questions'])
            
            for q in dataset['questions']:
                print(f"\n  问题: {q['q']}")
                
                start_time = time.time()
                response = self.generate_response(q['q'])
                inference_time = time.time() - start_time
                
                is_correct, score, match_info = self.evaluate_answer(response, q['a'])
                
                if is_correct:
                    correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  回答: {response[:80]}")
                print(f"  结果: {status} {match_info} ({inference_time:.1f}s)")
            
            accuracy = correct / total if total > 0 else 0
            results['dimensions'][dim_key] = {
                "name": dataset['name'],
                "correct": correct,
                "total": total,
                "accuracy": accuracy
            }
            
            total_correct += correct
            total_questions += total
            
            glm5 = GLM5_BASELINE.get(dim_key, 0.7)
            diff = accuracy - glm5
            print(f"\n  得分: {correct}/{total} ({accuracy*100:.1f}%) | GLM-5: {diff*100:+.1f}%")
        
        overall = total_correct / total_questions if total_questions > 0 else 0
        results['overall'] = {
            "correct": total_correct,
            "total": total_questions,
            "accuracy": overall
        }
        
        return results
    
    def print_summary(self, results: Dict):
        print("\n" + "=" * 60)
        print("评估结果摘要")
        print("=" * 60)
        
        print(f"\n模型: {results['model']}")
        print(f"类型: 真实模型推理")
        
        print("\n各维度得分:")
        print("-" * 50)
        
        for dim_key, dim_data in results['dimensions'].items():
            acc = dim_data['accuracy']
            glm5 = GLM5_BASELINE.get(dim_key, 0.7)
            diff = acc - glm5
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  {dim_data['name']:8s} [{bar}] {acc*100:5.1f}% ({diff*100:+.1f}%)")
        
        print("-" * 50)
        overall = results['overall']['accuracy']
        glm5_overall = sum(GLM5_BASELINE.values()) / len(GLM5_BASELINE)
        diff = overall - glm5_overall
        bar = "█" * int(overall * 20) + "░" * (20 - int(overall * 20))
        print(f"  {'综合得分':8s} [{bar}] {overall*100:5.1f}% ({diff*100:+.1f}%)")
        
        print("\n" + "=" * 60)

def main():
    print("\n" + "=" * 60)
    print("类脑AI系统 - 真实模型评估")
    print("=" * 60)
    print("\n⚠️  使用真实Qwen模型推理，无预设答案\n")
    
    evaluator = RealModelEvaluator(MODEL_PATH)
    
    if not evaluator.load_model():
        print("模型加载失败")
        return
    
    results = evaluator.run_evaluation()
    evaluator.print_summary(results)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    result_file = os.path.join(OUTPUT_PATH, f"real_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {result_file}")

if __name__ == "__main__":
    main()
