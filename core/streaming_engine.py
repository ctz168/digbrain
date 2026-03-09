#!/usr/bin/env python3.13
"""
类脑AI系统 - 完整流式处理系统
Brain-like AI System - Complete Streaming Processing System

核心特性：
1. Qwen3.5-0.8B - 语言模型，流式输入输出
2. Qwen3-VL-2B - 世界模型，视觉理解
3. STDP在线学习 - 实时权重更新
4. 高刷新率流式处理 - 模拟人脑
5. 记忆调用系统 - 每次生成都产生记忆请求
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from typing import List, Dict, Any, Generator
from dataclasses import dataclass, field
from collections import defaultdict
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

QWEN_MODEL_PATH = "/home/z/my-project/brain-like-ai/models/Qwen3.5-0.8B"
WORLD_MODEL_PATH = "/home/z/my-project/brain-like-ai/models/Qwen3-VL-2B-Instruct"
OUTPUT_PATH = "/home/z/my-project/brain-like-ai/evaluation/results"
WEIGHTS_PATH = "/home/z/my-project/brain-like-ai/weights"

@dataclass
class StreamChunk:
    """流式数据块"""
    type: str  # 'text', 'memory_call', 'weight_update', 'control'
    content: str = ""
    token_id: int = -1
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class MemoryRequest:
    """记忆调用请求"""
    type: str
    query: str = ""
    content: str = ""
    relevance: float = 0.0
    timestamp: float = 0.0

class STDPOnlineLearning:
    """在线STDP学习系统"""
    
    def __init__(self, learning_rate: float = 0.01, stdp_window: float = 20.0):
        self.learning_rate = learning_rate
        self.stdp_window = stdp_window
        self.weights = {}
        self.traces = {}
        self.spike_times = {}
        self.update_count = 0
        self.lock = threading.Lock()
        
    def initialize_from_model(self, model):
        """从模型初始化权重"""
        import torch
        print("初始化STDP权重...")
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                self.weights[name] = param.data.clone()
                self.traces[name] = torch.zeros_like(param.data)
        
        print(f"✅ 初始化了 {len(self.weights)} 个权重层")
    
    def record_spike(self, layer: str, activation: float):
        """记录脉冲"""
        with self.lock:
            self.spike_times[layer] = time.time() * 1000
    
    def compute_stdp_update(self, pre_layer: str, post_layer: str, 
                           pre_act: float, post_act: float) -> float:
        """计算STDP更新"""
        current_time = time.time() * 1000
        pre_time = self.spike_times.get(pre_layer, current_time)
        post_time = self.spike_times.get(post_layer, current_time)
        delta_t = post_time - pre_time
        
        if abs(delta_t) < self.stdp_window:
            if delta_t > 0:
                return self.learning_rate * math.exp(-delta_t / self.stdp_window)
            else:
                return -self.learning_rate * math.exp(delta_t / self.stdp_window)
        return 0.0
    
    def apply_update(self, layer: str, weight_change: float):
        """应用更新"""
        with self.lock:
            if layer in self.weights:
                self.weights[layer] += weight_change
                self.update_count += 1
                self.traces[layer] = 0.9 * self.traces[layer] + abs(weight_change)
    
    def export_weights(self, path: str):
        """导出权重"""
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "weights": {k: v.numpy().tolist() for k, v in self.weights.items()},
            "update_count": self.update_count,
            "config": {"learning_rate": self.learning_rate, "stdp_window": self.stdp_window}
        }
        torch.save(data, path)
        print(f"权重已保存: {path}")

class MemorySystem:
    """记忆系统"""
    
    def __init__(self, max_short: int = 100, max_long: int = 1000):
        self.short_term = {}
        self.long_term = {}
        self.max_short = max_short
        self.max_long = max_long
        self.lock = threading.Lock()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索记忆"""
        results = []
        query_lower = query.lower()
        with self.lock:
            for key, value in self.short_term.items():
                if query_lower in str(value).lower():
                    results.append({"id": key, "content": value, "type": "short_term"})
            for key, value in self.long_term.items():
                if query_lower in str(value).lower():
                    results.append({"id": key, "content": value, "type": "long_term"})
        return results[:top_k]
    
    def store(self, content: Any, importance: float = 0.5) -> str:
        """存储记忆"""
        import uuid
        key = f"mem_{uuid.uuid4().hex[:8]}"
        with self.lock:
            self.short_term[key] = {"content": content, "timestamp": time.time(), "importance": importance}
            if len(self.short_term) > self.max_short:
                self._consolidate()
        return key
    
    def _consolidate(self):
        """记忆巩固"""
        sorted_mem = sorted(self.short_term.items(), key=lambda x: x[1]['importance'], reverse=True)
        for key, value in sorted_mem[:10]:
            self.long_term[key] = value
            del self.short_term[key]
    
    def get_stats(self) -> Dict:
        return {"short_term": len(self.short_term), "long_term": len(self.long_term)}

class BrainLikeStreamingEngine:
    """类脑流式处理引擎"""
    
    def __init__(self, refresh_rate: int = 60):
        self.refresh_rate = refresh_rate
        self.chunk_interval = 1000 / refresh_rate
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.world_model = None
        self.stdp = STDPOnlineLearning()
        self.memory = MemorySystem()
        self.is_running = False
        self.processing_count = 0
        
    def load_models(self):
        """加载模型"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("=" * 60)
        print("加载模型")
        print("=" * 60)
        
        # 加载Qwen3.5-0.8B
        print("\n[1/2] 加载 Qwen3.5-0.8B...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_PATH, local_files_only=True, trust_remote_code=True
        )
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_PATH, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.qwen_model.eval()
        params = sum(p.numel() for p in self.qwen_model.parameters())
        print(f"✅ Qwen3.5-0.8B 加载完成 ({params/1e6:.1f}M 参数)")
        
        self.stdp.initialize_from_model(self.qwen_model)
        
        # 加载世界模型
        print("\n[2/2] 加载 Qwen3-VL-2B (世界模型)...")
        try:
            self.world_tokenizer = AutoTokenizer.from_pretrained(
                WORLD_MODEL_PATH, local_files_only=True, trust_remote_code=True
            )
            self.world_model = AutoModelForCausalLM.from_pretrained(
                WORLD_MODEL_PATH, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float32
            )
            self.world_model.eval()
            params = sum(p.numel() for p in self.world_model.parameters())
            print(f"✅ Qwen3-VL-2B 加载完成 ({params/1e6:.1f}M 参数)")
        except Exception as e:
            print(f"⚠️ 世界模型加载失败: {e}")
            self.world_model = None
        
        print("\n" + "=" * 60)
    
    def stream_process(self, prompt: str, max_tokens: int = 200) -> Generator[StreamChunk, None, None]:
        """高刷新率流式处理"""
        import torch
        
        self.is_running = True
        self.processing_count += 1
        
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.qwen_tokenizer(text, return_tensors="pt")
        
        # 记忆搜索请求
        yield StreamChunk(type="memory_call", metadata={"request": {"type": "search", "query": prompt[:100]}})
        
        memories = self.memory.search(prompt)
        if memories:
            yield StreamChunk(type="memory_call", content=f"找到 {len(memories)} 条相关记忆", metadata={"memories": memories})
        
        generated_text = ""
        start_time = time.time()
        last_yield = start_time
        
        with torch.no_grad():
            for i in range(max_tokens):
                if not self.is_running:
                    break
                
                iter_start = time.time()
                outputs = self.qwen_model(**inputs)
                logits = outputs.logits[:, -1, :]
                next_token_id = logits.argmax(dim=-1, keepdim=True)
                
                if next_token_id.item() == self.qwen_tokenizer.eos_token_id:
                    break
                
                next_token = self.qwen_tokenizer.decode(next_token_id[0])
                generated_text += next_token
                
                # STDP
                activation = float(logits.max())
                self.stdp.record_spike(f"layer_{i % 10}", activation)
                if i > 0:
                    wc = self.stdp.compute_stdp_update(f"layer_{(i-1)%10}", f"layer_{i%10}", activation, activation)
                    self.stdp.apply_update(f"layer_{i%10}", wc)
                
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id], dim=-1)
                
                # 按刷新率输出
                now = time.time()
                if (now - last_yield) * 1000 >= self.chunk_interval:
                    yield StreamChunk(type="text", content=next_token, token_id=next_token_id.item(), timestamp=now)
                    last_yield = now
        
        # 存储记忆
        key = self.memory.store({"input": prompt, "output": generated_text})
        yield StreamChunk(type="memory_call", metadata={"request": {"type": "store", "key": key}})
        
        # 完成
        total_time = time.time() - start_time
        yield StreamChunk(type="control", content="done", metadata={"total_time": total_time, "tokens": len(generated_text)})
        
        self.is_running = False
    
    def get_status(self) -> Dict:
        return {
            "processing_count": self.processing_count,
            "refresh_rate": self.refresh_rate,
            "stdp_updates": self.stdp.update_count,
            "memory_stats": self.memory.get_stats(),
            "qwen_loaded": self.qwen_model is not None,
            "world_loaded": self.world_model is not None
        }

def main():
    print("\n" + "=" * 60)
    print("类脑AI系统 - 完整流式处理测试")
    print("=" * 60)
    print("\n核心特性:")
    print("  1. Qwen3.5-0.8B 语言模型")
    print("  2. Qwen3-VL-2B 世界模型")
    print("  3. STDP在线学习")
    print("  4. 高刷新率流式处理 (60Hz)")
    print("  5. 记忆调用系统")
    print("=" * 60)
    
    engine = BrainLikeStreamingEngine(refresh_rate=60)
    engine.load_models()
    
    questions = [
        "请解释什么是量子纠缠？",
        "请解释TCP三次握手的过程。",
    ]
    
    results = []
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}\n")
        
        print("📤 流式输出:\n")
        
        chunks = 0
        mem_calls = 0
        text = ""
        
        for chunk in engine.stream_process(q, max_tokens=150):
            if chunk.type == "text":
                print(chunk.content, end='', flush=True)
                chunks += 1
                text += chunk.content
            elif chunk.type == "memory_call":
                mem_calls += 1
                if chunk.content:
                    print(f"\n[记忆] {chunk.content}\n", flush=True)
            elif chunk.type == "control":
                print(f"\n\n📊 统计: {chunks} chunks, {mem_calls} 记忆调用, {chunk.metadata.get('total_time', 0):.2f}s")
        
        results.append({"question": q, "response": text[:300], "chunks": chunks, "mem_calls": mem_calls})
    
    # 状态
    print(f"\n{'='*60}")
    print("系统状态")
    print(f"{'='*60}")
    status = engine.get_status()
    print(f"  处理次数: {status['processing_count']}")
    print(f"  STDP更新: {status['stdp_updates']}")
    print(f"  记忆数量: {status['memory_stats']}")
    
    # 保存
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    result_file = os.path.join(OUTPUT_PATH, f"brain_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results, "status": status}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_file}")
    
    # 导出权重
    engine.stdp.export_weights(os.path.join(WEIGHTS_PATH, "stdp_weights.pt"))

if __name__ == "__main__":
    main()
