#!/usr/bin/env python3.13
"""
类脑AI系统 - 优化版流式处理
"""

import os
import sys
import json
import time
import math
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

class SimpleSTDP:
    """简化版STDP"""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.updates = 0
        self.ltp = 0
        self.ltd = 0
        self.history = []
        
    def update(self, activation: float, reward: float = 0):
        if activation > 0.5:
            self.ltp += 1
        else:
            self.ltd += 1
        self.updates += 1
        self.history.append({'act': activation, 'reward': reward})
        
    def stats(self):
        return {'updates': self.updates, 'ltp': self.ltp, 'ltd': self.ltd}

class SimpleMemory:
    """简化版记忆"""
    def __init__(self):
        self.memories = []
        self.searches = 0
        
    def search(self, query):
        self.searches += 1
        return [m for m in self.memories if query.lower() in m['content'].lower()][:3]
    
    def store(self, content):
        self.memories.append({'id': f"mem_{len(self.memories)}", 'content': content, 'time': time.time()})
        return self.memories[-1]['id']
    
    def stats(self):
        return {'count': len(self.memories), 'searches': self.searches}

class BrainLikeAI:
    """类脑AI"""
    
    def __init__(self, refresh_rate=60):
        self.refresh_rate = refresh_rate
        self.model = None
        self.tokenizer = None
        self.stdp = SimpleSTDP()
        self.memory = SimpleMemory()
        self.initialized = False
        
    def init(self):
        print("=" * 60)
        print("初始化 Qwen3.5-0.8B")
        print("=" * 60)
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_PATH, local_files_only=True, trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_PATH, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        self.model.eval()
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ 模型加载完成 ({params/1e6:.1f}M 参数)")
        self.initialized = True
        
    def stream(self, prompt: str, max_tokens: int = 100) -> Generator[StreamChunk, None, None]:
        import torch
        
        if not self.initialized:
            self.init()
        
        # 记忆搜索
        mems = self.memory.search(prompt)
        if mems:
            yield StreamChunk("memory_call", f"找到 {len(mems)} 条记忆", metadata={"memories": mems})
        
        # 生成
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(text, return_tensors="pt")
        
        output_text = ""
        start = time.time()
        
        with torch.no_grad():
            for i in range(max_tokens):
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                next_id = logits.argmax(dim=-1, keepdim=True)
                
                if next_id.item() == self.tokenizer.eos_token_id:
                    break
                
                token = self.tokenizer.decode(next_id[0])
                output_text += token
                
                # 流式输出
                yield StreamChunk("text", token, time.time(), {"idx": i})
                
                # STDP
                act = float(logits.max())
                self.stdp.update(act)
                
                if i % 20 == 0 and i > 0:
                    yield StreamChunk("learning", f"STDP #{self.stdp.updates}", metadata=self.stdp.stats())
                
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_id], dim=-1)
                
                if inputs['input_ids'].shape[1] > 1024:
                    break
        
        # 存储记忆
        mid = self.memory.store(f"Q:{prompt}\nA:{output_text}")
        yield StreamChunk("memory_call", f"存储记忆: {mid}")
        
        # 完成
        elapsed = time.time() - start
        yield StreamChunk("control", "DONE", metadata={
            "tokens": len(output_text),
            "time": elapsed,
            "speed": len(output_text)/elapsed if elapsed > 0 else 0
        })
    
    def process(self, prompt: str) -> Dict:
        chunks = list(self.stream(prompt))
        text = "".join(c.content for c in chunks if c.type == "text")
        ctrl = next((c for c in chunks if c.type == "control"), None)
        return {
            "response": text,
            "stats": ctrl.metadata if ctrl else {},
            "stdp": self.stdp.stats(),
            "memory": self.memory.stats()
        }

def main():
    print("\n" + "=" * 60)
    print("类脑AI系统 - Qwen3.5-0.8B + STDP + 记忆")
    print("=" * 60)
    
    ai = BrainLikeAI(60)
    
    questions = [
        "请解释什么是死锁？",
        "TCP三次握手是什么？",
        "量子纠缠是什么？"
    ]
    
    results = []
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}\n")
        
        print("📤 流式输出:\n")
        
        r = ai.process(q)
        
        print(f"\n\n📊 统计:")
        print(f"   Tokens: {r['stats'].get('tokens', 0)}")
        print(f"   时间: {r['stats'].get('time', 0):.2f}s")
        print(f"   速度: {r['stats'].get('speed', 0):.1f} t/s")
        print(f"   STDP更新: {r['stdp']['updates']}")
        print(f"   记忆数: {r['memory']['count']}")
        
        results.append({
            "question": q,
            "response": r['response'][:200],
            "stats": r['stats']
        })
    
    # 保存
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    f = os.path.join(OUTPUT_PATH, f"result_{int(time.time())}.json")
    with open(f, 'w', encoding='utf-8') as fp:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen3.5-0.8B",
            "results": results,
            "stdp": ai.stdp.stats(),
            "memory": ai.memory.stats()
        }, fp, ensure_ascii=False, indent=2)
    
    print(f"\n\n结果已保存: {f}")

if __name__ == "__main__":
    main()
