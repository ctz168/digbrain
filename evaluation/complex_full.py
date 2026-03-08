#!/usr/bin/env python3.13
"""
类脑AI系统 - 完整评估（非流式）
"""

import os
import sys
import json
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/home/z/my-project/brain-like-ai/models/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "/home/z/my-project/brain-like-ai/evaluation/results"

# 复杂测试问题
COMPLEX_QUESTIONS = [
    {
        "id": "c001",
        "category": "复杂推理",
        "q": "一个房间里有3个开关，分别控制另一个房间里的3盏灯。你只能进入有灯的房间一次。如何确定哪个开关控制哪盏灯？",
        "keywords": ["温度", "热", "摸", "等待"]
    },
    {
        "id": "c002",
        "category": "复杂数学",
        "q": "一个水池有两个进水管和一个出水管。甲管单独注满需要6小时，乙管单独注满需要8小时，丙管单独放空需要12小时。如果三管同时开放，多少小时能注满水池？",
        "keywords": ["4", "四小时"]
    },
    {
        "id": "c003",
        "category": "复杂编程",
        "q": "请解释什么是死锁？死锁产生的四个必要条件是什么？",
        "keywords": ["互斥", "占有", "循环", "等待"]
    },
    {
        "id": "c004",
        "category": "深度知识",
        "q": "请解释量子纠缠的基本原理。",
        "keywords": ["量子", "纠缠", "叠加"]
    },
    {
        "id": "c005",
        "category": "复杂推理",
        "q": "有12个球，其中11个重量相同，1个重量不同。用天平最少称几次能找出这个球？",
        "keywords": ["3", "三次"]
    },
    {
        "id": "c006",
        "category": "复杂编程",
        "q": "请解释TCP三次握手的过程。为什么需要三次而不是两次？",
        "keywords": ["SYN", "ACK", "连接"]
    },
    {
        "id": "c007",
        "category": "深度知识",
        "q": "请解释CAP定理的含义。",
        "keywords": ["一致性", "可用性", "分区"]
    },
    {
        "id": "c008",
        "category": "创造性",
        "q": "请写一个关于人工智能的短篇故事，100字左右。",
        "keywords": []
    }
]

def main():
    print("=" * 60)
    print("复杂问题完整评估")
    print("=" * 60)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n加载模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        local_files_only=True,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()
    
    print("✅ 模型加载完成\n")
    
    results = []
    category_scores = {}
    
    for q in COMPLEX_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"[{q['category']}] 问题: {q['q'][:50]}...")
        print(f"{'='*60}")
        
        # 生成回答
        text = f"<|im_start|>user\n{q['q']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt")
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        elapsed = time.time() - start_time
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        print(f"\n回答: {response[:300]}...")
        print(f"\n耗时: {elapsed:.2f}s")
        
        # 评估
        matched = [k for k in q['keywords'] if k.lower() in response.lower()] if q['keywords'] else []
        score = len(matched) / len(q['keywords']) * 100 if q['keywords'] else 80
        is_correct = score >= 50 if q['keywords'] else True
        
        status = "✅" if is_correct else "❌"
        print(f"评估: {status} 匹配: {matched} 得分: {score:.0f}%")
        
        results.append({
            "id": q['id'],
            "category": q['category'],
            "question": q['q'],
            "response": response,
            "time": elapsed,
            "matched_keywords": matched,
            "score": score,
            "correct": is_correct
        })
        
        # 按类别统计
        if q['category'] not in category_scores:
            category_scores[q['category']] = {"correct": 0, "total": 0}
        category_scores[q['category']]['total'] += 1
        if is_correct:
            category_scores[q['category']]['correct'] += 1
    
    # 保存结果
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    result_file = os.path.join(OUTPUT_PATH, f"complex_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen2.5-0.5B-Instruct",
            "results": results,
            "category_scores": category_scores
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n结果已保存: {result_file}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    
    total_correct = sum(1 for r in results if r['correct'])
    total = len(results)
    
    print("\n按类别:")
    for cat, scores in category_scores.items():
        acc = scores['correct'] / scores['total'] * 100
        print(f"  {cat}: {scores['correct']}/{scores['total']} ({acc:.0f}%)")
    
    print(f"\n综合得分: {total_correct}/{total} ({total_correct/total*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
