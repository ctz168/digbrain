#!/usr/bin/env python3.13
"""
类脑AI系统 - 流式输出测试
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

# 测试问题
TEST_QUESTIONS = [
    {
        "id": "stream_001",
        "q": "请解释什么是死锁，以及死锁产生的四个必要条件。",
        "keywords": ["互斥", "占有", "循环", "等待"]
    },
    {
        "id": "stream_002", 
        "q": "请解释TCP三次握手的过程。",
        "keywords": ["SYN", "ACK", "连接"]
    },
    {
        "id": "stream_003",
        "q": "请解释量子纠缠的基本原理。",
        "keywords": ["量子", "纠缠", "叠加"]
    }
]

def main():
    print("=" * 60)
    print("流式输出测试")
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
    
    for q in TEST_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"问题: {q['q']}")
        print(f"{'='*60}")
        
        # 流式生成
        text = f"<|im_start|>user\n{q['q']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt")
        
        print("\n📤 流式输出:")
        start_time = time.time()
        chunks = []
        response = ""
        
        with torch.no_grad():
            for i in range(100):  # 最多100个token
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                next_token_id = logits.argmax(dim=-1, keepdim=True)
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                next_token = tokenizer.decode(next_token_id[0])
                chunks.append(next_token)
                response += next_token
                
                # 实时显示
                print(next_token, end='', flush=True)
                
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id], dim=-1)
        
        elapsed = time.time() - start_time
        print()
        
        # 统计
        print(f"\n📊 统计:")
        print(f"   输出块数: {len(chunks)}")
        print(f"   总字符数: {len(response)}")
        print(f"   总时间: {elapsed:.2f}s")
        print(f"   速度: {len(response)/elapsed:.1f} 字符/秒")
        
        # 评估
        matched = [k for k in q['keywords'] if k.lower() in response.lower()]
        score = len(matched) / len(q['keywords']) * 100
        
        print(f"\n📝 评估:")
        print(f"   匹配关键词: {matched}")
        print(f"   得分: {score:.0f}%")
        
        results.append({
            "id": q['id'],
            "question": q['q'],
            "response": response[:300],
            "chunk_count": len(chunks),
            "streaming_time": elapsed,
            "chars_per_second": len(response)/elapsed if elapsed > 0 else 0,
            "matched_keywords": matched,
            "score": score
        })
    
    # 保存结果
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    result_file = os.path.join(OUTPUT_PATH, f"streaming_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen2.5-0.5B-Instruct",
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n结果已保存: {result_file}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("流式输出测试汇总")
    print("=" * 60)
    
    total_chunks = sum(r['chunk_count'] for r in results)
    avg_speed = sum(r['chars_per_second'] for r in results) / len(results)
    avg_score = sum(r['score'] for r in results) / len(results)
    
    print(f"  总输出块数: {total_chunks}")
    print(f"  平均速度: {avg_speed:.1f} 字符/秒")
    print(f"  平均得分: {avg_score:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
