#!/usr/bin/env python3.13
"""
类脑AI系统 - 复杂问题真实模型评估
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Generator

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/home/z/my-project/brain-like-ai/models/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "/home/z/my-project/brain-like-ai/evaluation/results"

# 复杂测试问题
COMPLEX_TEST_QUESTIONS = {
    "complex_reasoning": {
        "name": "复杂逻辑推理",
        "questions": [
            {
                "id": "cr001",
                "q": "一个房间里有3个开关，分别控制另一个房间里的3盏灯。你只能进入有灯的房间一次。如何确定哪个开关控制哪盏灯？请详细说明你的推理过程。",
                "evaluation": ["温度", "热", "摸", "等待"],
                "min_tokens": 100
            },
            {
                "id": "cr002",
                "q": "有12个球，其中11个重量相同，1个重量不同（可能更重或更轻）。用天平最少称几次能找出这个球？请详细说明步骤。",
                "evaluation": ["3", "三次", "三步"],
                "min_tokens": 150
            },
            {
                "id": "cr003",
                "q": "甲、乙、丙、丁四人参加比赛，已知：甲不是第一，乙不是最后，丙比丁高一个名次，乙比甲高一个名次。请推断四人的排名。",
                "evaluation": ["丙", "乙", "甲", "丁"],
                "min_tokens": 100
            }
        ]
    },
    "complex_math": {
        "name": "复杂数学问题",
        "questions": [
            {
                "id": "cm001",
                "q": "一个水池有两个进水管和一个出水管。甲管单独注满需要6小时，乙管单独注满需要8小时，丙管单独放空需要12小时。如果三管同时开放，多少小时能注满水池？请写出计算过程。",
                "evaluation": ["4", "四"],
                "min_tokens": 100
            },
            {
                "id": "cm002",
                "q": "一个等差数列的前n项和为Sn，已知S10=100，S20=400，求S30的值。请详细说明解题步骤。",
                "evaluation": ["900", "九百"],
                "min_tokens": 100
            }
        ]
    },
    "complex_coding": {
        "name": "复杂编程问题",
        "questions": [
            {
                "id": "cc001",
                "q": "请解释什么是死锁？死锁产生的四个必要条件是什么？如何预防和避免死锁？请详细回答。",
                "evaluation": ["互斥", "占有", "抢占", "循环"],
                "min_tokens": 200
            },
            {
                "id": "cc002",
                "q": "请解释CAP定理。为什么分布式系统不能同时满足一致性、可用性和分区容错性？",
                "evaluation": ["CAP", "一致性", "可用性", "分区"],
                "min_tokens": 150
            },
            {
                "id": "cc003",
                "q": "请解释TCP三次握手的过程。为什么需要三次握手而不是两次？",
                "evaluation": ["SYN", "ACK", "三次", "连接"],
                "min_tokens": 150
            }
        ]
    },
    "complex_knowledge": {
        "name": "深度知识问答",
        "questions": [
            {
                "id": "ck001",
                "q": "请详细解释量子纠缠的原理。为什么爱因斯坦称之为幽灵般的超距作用？这与相对论是否矛盾？",
                "evaluation": ["纠缠", "量子", "叠加", "测量"],
                "min_tokens": 200
            },
            {
                "id": "ck002",
                "q": "请解释黑洞的形成过程、事件视界的概念，以及霍金辐射的原理。",
                "evaluation": ["黑洞", "事件视界", "霍金辐射", "引力"],
                "min_tokens": 200
            }
        ]
    },
    "creative_writing": {
        "name": "创造性写作",
        "questions": [
            {
                "id": "cw001",
                "q": "请写一个短篇科幻故事，主题是：人类发明了一种可以读取他人思想的设备，但使用者会发现一些意想不到的真相。字数200字以上。",
                "evaluation": [],
                "min_tokens": 250
            }
        ]
    }
}

class StreamingEvaluator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        print("=" * 60)
        print("加载真实Qwen模型")
        print("=" * 60)
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"模型路径: {self.model_path}")
        
        config_file = os.path.join(self.model_path, "config.json")
        model_file = os.path.join(self.model_path, "model.safetensors")
        
        print(f"✅ 配置文件: {os.path.getsize(config_file)} bytes")
        print(f"✅ 模型文件: {os.path.getsize(model_file) / 1024 / 1024:.1f} MB")
        
        print("\n正在加载模型...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        print("✅ Tokenizer加载完成")
        
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
    
    def generate_streaming(self, prompt: str, max_new_tokens: int = 300) -> Generator[str, None, None]:
        """流式生成输出"""
        import torch
        
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                
                next_token_id = logits.argmax(dim=-1, keepdim=True)
                
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                next_token = self.tokenizer.decode(next_token_id[0])
                yield next_token
                
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id], dim=-1)
                
                if inputs['input_ids'].shape[1] > 2048:
                    break
    
    def generate_full(self, prompt: str, max_new_tokens: int = 300) -> str:
        """完整生成"""
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
    
    def evaluate_response(self, response: str, evaluation_keywords: List[str]) -> Tuple[bool, float, List[str]]:
        if not evaluation_keywords:
            return True, 0.8, []
        
        response_lower = response.lower()
        matched = []
        
        for keyword in evaluation_keywords:
            if keyword.lower() in response_lower:
                matched.append(keyword)
        
        score = len(matched) / len(evaluation_keywords) if evaluation_keywords else 0.8
        is_correct = score >= 0.5
        
        return is_correct, score, matched
    
    def run_evaluation(self) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("开始复杂问题评估")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen2.5-0.5B-Instruct",
            "evaluation_type": "complex_questions_streaming",
            "dimensions": {},
            "streaming_tests": [],
            "overall": {}
        }
        
        total_correct = 0
        total_questions = 0
        
        for dim_key, dataset in COMPLEX_TEST_QUESTIONS.items():
            print(f"\n{'='*60}")
            print(f"【{dataset['name']}】")
            print(f"{'='*60}")
            
            correct = 0
            total = len(dataset['questions'])
            dim_results = []
            
            for q in dataset['questions']:
                print(f"\n{'─'*60}")
                print(f"问题 [{q['id']}]: {q['q'][:60]}...")
                print(f"{'─'*60}")
                
                # 测试流式输出
                print("\n📤 流式输出测试:")
                stream_start = time.time()
                stream_chunks = []
                stream_response = ""
                
                try:
                    for chunk in self.generate_streaming(q['q'], max_new_tokens=q.get('min_tokens', 200)):
                        stream_chunks.append(chunk)
                        stream_response += chunk
                        print(chunk, end='', flush=True)
                except Exception as e:
                    print(f"\n流式输出错误: {e}")
                    stream_response = ""
                
                stream_time = time.time() - stream_start
                print()
                
                # 流式统计
                stream_stats = {
                    "total_time": stream_time,
                    "chunk_count": len(stream_chunks),
                    "total_chars": len(stream_response),
                    "chars_per_second": len(stream_response) / stream_time if stream_time > 0 else 0
                }
                
                print(f"\n📊 流式统计:")
                print(f"   总时间: {stream_time:.2f}s")
                print(f"   输出块数: {len(stream_chunks)}")
                print(f"   总字符数: {len(stream_response)}")
                print(f"   速度: {stream_stats['chars_per_second']:.1f} 字符/秒")
                
                # 如果流式失败，使用完整生成
                if not stream_response:
                    print("\n⚠️ 流式输出失败，使用完整生成...")
                    stream_response = self.generate_full(q['q'], q.get('min_tokens', 200))
                
                # 评估
                is_correct, score, matched = self.evaluate_response(
                    stream_response, 
                    q.get('evaluation', [])
                )
                
                if is_correct:
                    correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"\n📝 评估结果: {status}")
                print(f"   匹配关键词: {matched}")
                print(f"   得分: {score*100:.1f}%")
                
                result = {
                    "id": q['id'],
                    "question": q['q'],
                    "response": stream_response[:500],
                    "correct": is_correct,
                    "score": score,
                    "matched_keywords": matched,
                    "streaming_stats": stream_stats
                }
                dim_results.append(result)
                
                results['streaming_tests'].append({
                    "id": q['id'],
                    "streaming_success": len(stream_chunks) > 0,
                    "chunk_count": len(stream_chunks),
                    "stream_time": stream_time
                })
            
            accuracy = correct / total if total > 0 else 0
            results['dimensions'][dim_key] = {
                "name": dataset['name'],
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "details": dim_results
            }
            
            total_correct += correct
            total_questions += total
            
            print(f"\n维度得分: {correct}/{total} ({accuracy*100:.1f}%)")
        
        overall = total_correct / total_questions if total_questions > 0 else 0
        results['overall'] = {
            "correct": total_correct,
            "total": total_questions,
            "accuracy": overall
        }
        
        return results
    
    def print_summary(self, results: Dict):
        print("\n" + "=" * 60)
        print("复杂问题评估结果摘要")
        print("=" * 60)
        
        print(f"\n模型: {results['model']}")
        
        print("\n各维度得分:")
        print("-" * 50)
        
        for dim_key, dim_data in results['dimensions'].items():
            acc = dim_data['accuracy']
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  {dim_data['name']:12s} [{bar}] {acc*100:5.1f}%")
        
        print("-" * 50)
        overall = results['overall']['accuracy']
        bar = "█" * int(overall * 20) + "░" * (20 - int(overall * 20))
        print(f"  {'综合得分':12s} [{bar}] {overall*100:5.1f}%")
        
        # 流式统计
        print("\n流式输出统计:")
        print("-" * 50)
        streaming_tests = results.get('streaming_tests', [])
        success_count = sum(1 for t in streaming_tests if t['streaming_success'])
        total_chunks = sum(t['chunk_count'] for t in streaming_tests)
        avg_time = sum(t['stream_time'] for t in streaming_tests) / len(streaming_tests) if streaming_tests else 0
        
        print(f"  流式成功率: {success_count}/{len(streaming_tests)} ({success_count/len(streaming_tests)*100:.1f}%)")
        print(f"  总输出块数: {total_chunks}")
        print(f"  平均响应时间: {avg_time:.2f}s")
        
        print("\n" + "=" * 60)

def main():
    print("\n" + "=" * 60)
    print("类脑AI系统 - 复杂问题真实模型评估")
    print("=" * 60)
    print("\n⚠️  使用真实Qwen模型进行复杂问题评估")
    print("   重点测试流式输出能力\n")
    
    evaluator = StreamingEvaluator(MODEL_PATH)
    
    if not evaluator.load_model():
        print("模型加载失败")
        return
    
    results = evaluator.run_evaluation()
    evaluator.print_summary(results)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    result_file = os.path.join(OUTPUT_PATH, f"complex_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {result_file}")

if __name__ == "__main__":
    main()
