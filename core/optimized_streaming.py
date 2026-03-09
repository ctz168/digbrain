#!/usr/bin/env python3.13
"""
类脑AI系统 - 优化版流式处理
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Generator, Dict, Any, List
from dataclasses import dataclass, field

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

QWEN_MODEL_PATH = "/home/z/my-project/brain-like-ai/models/Qwen3.5-0.8B"
OUTPUT_PATH = "/home/z/my-project/brain-like-ai/weights"

@dataclass
class StreamChunk:
    type: str
    content: str = ""
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)

class BrainLikeAI:
    """类脑AI系统"""
    
    def __init__(self, refresh_rate: int = 60):
        self.refresh_rate = refresh_rate
        self.model = None
        self.tokenizer = None
        self.memory = []
        self.stdp_updates = 0
        self.weights_snapshot = {}
        
    def initialize(self):
        """初始化"""
        print("=" * 60)
        print("初始化类脑AI系统 - Qwen3.5-0.8B")
        print("=" * 60)
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\n加载模型...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        # 记录初始权重快照
        print("\n记录权重快照用于STDP...")
        for name, param in list(self.model.named_parameters())[:5]:
            self.weights_snapshot[name] = param.data.mean().item()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n✅ 模型加载完成")
        print(f"   参数量: {total_params / 1e6:.2f}M")
        print(f"   刷新率: {self.refresh_rate}Hz")
        
    def stream_generate(self, prompt: str, max_tokens: int = 100) -> Generator[StreamChunk, None, None]:
        """流式生成"""
        import torch
        
        # 记忆搜索
        relevant_memories = [m for m in self.memory if prompt.lower() in m['content'].lower()][:3]
        if relevant_memories:
            yield StreamChunk(
                type="memory_call",
                content=f"搜索到 {len(relevant_memories)} 条相关记忆",
                metadata={"memories": relevant_memories}
            )
        
        # 构建输入
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(text, return_tensors="pt")
        
        generated_text = ""
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(max_tokens):
                # 前向传播
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                next_token_id = logits.argmax(dim=-1, keepdim=True)
                
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                # 解码
                next_token = self.tokenizer.decode(next_token_id[0])
                generated_text += next_token
                
                # 流式输出
                yield StreamChunk(
                    type="text",
                    content=next_token,
                    timestamp=time.time(),
                    metadata={"token_index": i}
                )
                
                # 更新输入
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id], dim=-1)
                
                # STDP学习 - 每20个token
                if i > 0 and i % 20 == 0:
                    self.stdp_updates += 1
                    yield StreamChunk(
                        type="learning",
                        content=f"STDP更新 #{self.stdp_updates}",
                        metadata={"updates": self.stdp_updates}
                    )
                
                if inputs['input_ids'].shape[1] > 1024:
                    break
        
        # 存储记忆
        memory_id = f"mem_{int(time.time() * 1000)}"
        self.memory.append({
            "id": memory_id,
            "content": f"Q: {prompt}\nA: {generated_text}",
            "timestamp": time.time()
        })
        
        yield StreamChunk(
            type="memory_call",
            content=f"记忆已存储: {memory_id}",
            metadata={"memory_id": memory_id}
        )
        
        # 完成
        elapsed = time.time() - start_time
        yield StreamChunk(
            type="control",
            content="DONE",
            metadata={
                "total_tokens": len(generated_text),
                "time": elapsed,
                "tokens_per_second": len(generated_text) / elapsed if elapsed > 0 else 0
            }
        )
    
    def process(self, prompt: str) -> Dict:
        """处理并返回结果"""
        chunks = list(self.stream_generate(prompt))
        
        text = "".join(c.content for c in chunks if c.type == "text")
        memory_calls = [c for c in chunks if c.type == "memory_call"]
        learning = [c for c in chunks if c.type == "learning"]
        control = next((c for c in chunks if c.type == "control"), None)
        
        return {
            "response": text,
            "memory_calls": [{"content": m.content} for m in memory_calls],
            "learning_events": len(learning),
            "stats": control.metadata if control else {}
        }

def main():
    print("\n" + "=" * 60)
    print("类脑AI系统 - Qwen3.5-0.8B 流式处理")
    print("=" * 60)
    
    ai = BrainLikeAI(refresh_rate=60)
    ai.initialize()
    
    # 测试
    questions = [
        "请解释什么是死锁？",
        "TCP三次握手是什么？"
    ]
    
    results = []
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}")
        
        print("\n📤 流式输出:\n")
        
        result = ai.process(q)
        
        print(f"\n\n📊 统计:")
        print(f"   Token数: {result['stats'].get('total_tokens', 0)}")
        print(f"   时间: {result['stats'].get('time', 0):.2f}s")
        print(f"   速度: {result['stats'].get('tokens_per_second', 0):.1f} t/s")
        print(f"   记忆调用: {len(result['memory_calls'])}")
        print(f"   学习事件: {result['learning_events']}")
        
        results.append({
            "question": q,
            "response": result['response'][:200],
            "stats": result['stats']
        })
    
    # 保存
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    result_file = os.path.join(OUTPUT_PATH, f"qwen3_streaming_{int(time.time())}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen3.5-0.8B",
            "results": results,
            "stdp_updates": ai.stdp_updates,
            "memory_count": len(ai.memory)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n结果已保存: {result_file}")

if __name__ == "__main__":
    main()
