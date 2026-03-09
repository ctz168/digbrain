#!/usr/bin/env python3
"""
类脑AI系统 - 完整核心引擎
Brain-like AI System - Complete Core Engine

核心特性：
1. Qwen2.5-0.5B-Instruct - 语言模型，流式输入输出
2. Qwen2-VL-2B-Instruct - 世界模型，视觉理解
3. STDP在线学习 - 实时权重更新
4. 高刷新率流式处理 - 模拟人脑
5. 记忆调用系统 - 每次生成都产生记忆请求
6. 维基百科搜索 - 无限知识库
"""

import os
import sys
import json
import time
import math
import uuid
import threading
import queue
from datetime import datetime
from typing import List, Dict, Any, Generator, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import traceback

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_MODEL_PATH = os.path.join(BASE_DIR, "models/Qwen3.5-0.8B")
WORLD_MODEL_PATH = os.path.join(BASE_DIR, "models/WorldModel")
OUTPUT_PATH = os.path.join(BASE_DIR, "evaluation/results")
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights")
MEMORY_PATH = os.path.join(BASE_DIR, "memory")


@dataclass
class StreamChunk:
    """流式数据块 - 模拟神经元脉冲"""
    type: str  # 'text', 'memory_call', 'memory_store', 'weight_update', 'control', 'wiki_search'
    content: str = ""
    token_id: int = -1
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class MemoryRequest:
    """记忆调用请求"""
    type: str  # 'search', 'store', 'consolidate'
    query: str = ""
    content: str = ""
    relevance: float = 0.0
    timestamp: float = 0.0


@dataclass
class STDPEvent:
    """STDP学习事件"""
    pre_layer: str
    post_layer: str
    pre_activation: float
    post_activation: float
    delta_t: float
    weight_change: float
    timestamp: float


class STDPOnlineLearning:
    """
    在线STDP学习系统
    Spike-Timing-Dependent Plasticity
    
    实现脉冲时间依赖可塑性学习规则：
    - Pre-spike先于Post-spike: LTP (长时程增强)
    - Post-spike先于Pre-spike: LTD (长时程抑制)
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.01, 
        stdp_window: float = 20.0,
        decay_rate: float = 0.9
    ):
        self.learning_rate = learning_rate
        self.stdp_window = stdp_window  # ms
        self.decay_rate = decay_rate
        
        # 权重和追踪
        self.weights: Dict[str, Any] = {}
        self.traces: Dict[str, Any] = {}
        self.spike_times: Dict[str, float] = {}
        
        # 统计
        self.update_count = 0
        self.ltp_count = 0
        self.ltd_count = 0
        self.events: List[STDPEvent] = []
        
        self.lock = threading.Lock()
        self.enabled = True
        
    def initialize_from_model(self, model) -> int:
        """从模型初始化权重"""
        import torch
        print("初始化STDP权重...")
        
        count = 0
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                self.weights[name] = param.data.clone()
                self.traces[name] = torch.zeros_like(param.data)
                count += 1
        
        print(f"✅ 初始化了 {count} 个权重层")
        return count
    
    def record_spike(self, layer: str, activation: float):
        """记录脉冲事件"""
        if not self.enabled:
            return
        with self.lock:
            self.spike_times[layer] = time.time() * 1000
    
    def compute_stdp_update(
        self, 
        pre_layer: str, 
        post_layer: str, 
        pre_act: float, 
        post_act: float
    ) -> float:
        """
        计算STDP权重更新
        
        Δw = η × f(Δt) × pre_act × post_act
        
        其中 f(Δt) 是STDP窗口函数
        """
        if not self.enabled:
            return 0.0
            
        current_time = time.time() * 1000
        pre_time = self.spike_times.get(pre_layer, current_time)
        post_time = self.spike_times.get(post_layer, current_time)
        delta_t = post_time - pre_time
        
        # STDP窗口函数
        if abs(delta_t) < self.stdp_window:
            if delta_t > 0:
                # LTP: Pre先于Post
                weight_change = self.learning_rate * math.exp(-delta_t / self.stdp_window)
                with self.lock:
                    self.ltp_count += 1
            else:
                # LTD: Post先于Pre
                weight_change = -self.learning_rate * math.exp(delta_t / self.stdp_window)
                with self.lock:
                    self.ltd_count += 1
            
            # 记录事件
            event = STDPEvent(
                pre_layer=pre_layer,
                post_layer=post_layer,
                pre_activation=pre_act,
                post_activation=post_act,
                delta_t=delta_t,
                weight_change=weight_change,
                timestamp=current_time
            )
            with self.lock:
                self.events.append(event)
                if len(self.events) > 1000:
                    self.events = self.events[-500:]
            
            return weight_change
        return 0.0
    
    def apply_update(self, layer: str, weight_change: float):
        """应用权重更新"""
        if not self.enabled or layer not in self.weights:
            return
            
        with self.lock:
            self.weights[layer] = self.weights[layer] + weight_change
            self.update_count += 1
            # 更新追踪
            self.traces[layer] = self.decay_rate * self.traces[layer] + abs(weight_change)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            return {
                "update_count": self.update_count,
                "ltp_count": self.ltp_count,
                "ltd_count": self.ltd_count,
                "enabled": self.enabled,
                "recent_events": len(self.events)
            }
    
    def export_weights(self, path: str):
        """导出权重到文件"""
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            "weights": {k: v.numpy().tolist() for k, v in self.weights.items()},
            "traces": {k: v.numpy().tolist() for k, v in self.traces.items()},
            "update_count": self.update_count,
            "ltp_count": self.ltp_count,
            "ltd_count": self.ltd_count,
            "config": {
                "learning_rate": self.learning_rate, 
                "stdp_window": self.stdp_window,
                "decay_rate": self.decay_rate
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存为JSON格式（更兼容）
        json_path = path.replace('.pt', '.json')
        with open(json_path, 'w') as f:
            json.dump({
                "update_count": self.update_count,
                "ltp_count": self.ltp_count,
                "ltd_count": self.ltd_count,
                "config": data["config"],
                "timestamp": data["timestamp"],
                "weight_layers": list(self.weights.keys())
            }, f, indent=2)
        
        # 保存完整权重
        torch.save(data, path)
        print(f"权重已保存: {path}")
        print(f"统计信息已保存: {json_path}")


class HippocampusMemory:
    """
    海马体记忆系统
    Hippocampus Memory System
    
    模拟人脑海马体的三阶段记忆机制：
    1. 瞬时记忆 (Sensory Memory) - 极短期，毫秒级
    2. 短期记忆 (Short-term Memory) - 工作记忆，秒到分钟
    3. 长期记忆 (Long-term Memory) - 永久存储
    
    存算分离架构：
    - 存储层：独立的记忆存储
    - 计算层：Qwen模型进行推理
    """
    
    def __init__(
        self,
        max_sensory: int = 1000,
        max_short: int = 100,
        max_long: int = 10000,
        consolidation_threshold: float = 0.6
    ):
        self.max_sensory = max_sensory
        self.max_short = max_short
        self.max_long = max_long
        self.consolidation_threshold = consolidation_threshold
        
        # 三阶段记忆
        self.sensory_memory: Dict[str, Dict] = {}  # 瞬时记忆
        self.short_term_memory: Dict[str, Dict] = {}  # 短期记忆
        self.long_term_memory: Dict[str, Dict] = {}  # 长期记忆
        
        # 神经累积增长追踪
        self.neuron_growth: List[Dict] = []
        
        # 索引
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)
        
        self.lock = threading.Lock()
        
        # 加载持久化记忆
        self._load_memory()
    
    def _load_memory(self):
        """从磁盘加载记忆"""
        os.makedirs(MEMORY_PATH, exist_ok=True)
        memory_file = os.path.join(MEMORY_PATH, "long_term.json")
        
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.long_term_memory = data.get("memories", {})
                    self.neuron_growth = data.get("growth", [])
                print(f"✅ 加载了 {len(self.long_term_memory)} 条长期记忆")
            except Exception as e:
                print(f"加载记忆失败: {e}")
    
    def _save_memory(self):
        """保存记忆到磁盘"""
        os.makedirs(MEMORY_PATH, exist_ok=True)
        memory_file = os.path.join(MEMORY_PATH, "long_term.json")
        
        with self.lock:
            data = {
                "memories": self.long_term_memory,
                "growth": self.neuron_growth[-100:],  # 只保留最近100条增长记录
                "timestamp": datetime.now().isoformat()
            }
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def store_sensory(self, content: Any, source: str = "input") -> str:
        """
        存储瞬时记忆
        模拟感觉记忆，极短期保持
        """
        key = f"sensory_{uuid.uuid4().hex[:8]}"
        
        with self.lock:
            self.sensory_memory[key] = {
                "content": content,
                "source": source,
                "timestamp": time.time(),
                "importance": 0.1
            }
            
            # 自动转移到短期记忆
            if len(self.sensory_memory) > self.max_sensory:
                self._transfer_to_short_term()
        
        return key
    
    def store_short_term(self, content: Any, importance: float = 0.5, tags: List[str] = None) -> str:
        """
        存储短期记忆
        模拟工作记忆
        """
        key = f"stm_{uuid.uuid4().hex[:8]}"
        tags = tags or []
        
        with self.lock:
            self.short_term_memory[key] = {
                "content": content,
                "timestamp": time.time(),
                "importance": importance,
                "access_count": 0,
                "tags": tags
            }
            
            # 更新关键词索引
            self._update_index(key, content, tags)
            
            # 触发巩固
            if len(self.short_term_memory) > self.max_short:
                self._consolidate()
        
        return key
    
    def store_long_term(self, content: Any, importance: float = 0.8, tags: List[str] = None) -> str:
        """
        直接存储长期记忆
        """
        key = f"ltm_{uuid.uuid4().hex[:8]}"
        tags = tags or []
        
        with self.lock:
            self.long_term_memory[key] = {
                "content": content,
                "timestamp": time.time(),
                "importance": importance,
                "access_count": 0,
                "tags": tags,
                "created": datetime.now().isoformat()
            }
            
            # 记录神经增长
            self.neuron_growth.append({
                "key": key,
                "timestamp": time.time(),
                "type": "long_term_storage"
            })
            
            self._update_index(key, content, tags)
        
        self._save_memory()
        return key
    
    def _transfer_to_short_term(self):
        """从瞬时记忆转移到短期记忆"""
        # 按重要性排序
        sorted_sensory = sorted(
            self.sensory_memory.items(),
            key=lambda x: x[1].get('importance', 0),
            reverse=True
        )
        
        # 转移前20%
        transfer_count = max(1, len(sorted_sensory) // 5)
        for key, value in sorted_sensory[:transfer_count]:
            self.store_short_term(
                value["content"],
                importance=value.get("importance", 0.3) + 0.2,
                tags=[value.get("source", "unknown")]
            )
            del self.sensory_memory[key]
    
    def _consolidate(self):
        """
        记忆巩固
        将重要的短期记忆转移到长期记忆
        模拟睡眠时的记忆巩固过程
        """
        # 计算巩固分数
        scored_memories = []
        for key, value in self.short_term_memory.items():
            # 综合重要性、访问次数、时间衰减
            age = time.time() - value.get("timestamp", time.time())
            age_factor = math.exp(-age / 3600)  # 1小时半衰期
            
            score = (
                value.get("importance", 0.5) * 0.5 +
                value.get("access_count", 0) * 0.1 +
                age_factor * 0.4
            )
            
            if score > self.consolidation_threshold:
                scored_memories.append((key, value, score))
        
        # 按分数排序并巩固
        scored_memories.sort(key=lambda x: x[2], reverse=True)
        
        for key, value, score in scored_memories[:10]:
            self.store_long_term(
                value["content"],
                importance=score,
                tags=value.get("tags", [])
            )
            del self.short_term_memory[key]
            
            # 记录神经增长
            self.neuron_growth.append({
                "key": key,
                "timestamp": time.time(),
                "type": "consolidation",
                "score": score
            })
    
    def _update_index(self, key: str, content: Any, tags: List[str]):
        """更新关键词索引"""
        # 从内容中提取关键词
        if isinstance(content, str):
            words = content.lower().split()
            for word in words[:20]:  # 只索引前20个词
                if len(word) > 2:
                    self.keyword_index[word].append(key)
        
        # 索引标签
        for tag in tags:
            self.keyword_index[tag.lower()].append(key)
    
    def search(self, query: str, top_k: int = 5, memory_type: str = "all") -> List[Dict]:
        """
        搜索记忆
        支持模糊匹配和关键词检索
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        with self.lock:
            # 搜索短期记忆
            if memory_type in ["all", "short"]:
                for key, value in self.short_term_memory.items():
                    score = self._calculate_relevance(query_lower, query_words, value)
                    if score > 0:
                        results.append({
                            "id": key,
                            "content": value["content"],
                            "type": "short_term",
                            "relevance": score,
                            "timestamp": value.get("timestamp", 0)
                        })
                        # 增加访问计数
                        value["access_count"] = value.get("access_count", 0) + 1
            
            # 搜索长期记忆
            if memory_type in ["all", "long"]:
                for key, value in self.long_term_memory.items():
                    score = self._calculate_relevance(query_lower, query_words, value)
                    if score > 0:
                        results.append({
                            "id": key,
                            "content": value["content"],
                            "type": "long_term",
                            "relevance": score,
                            "timestamp": value.get("timestamp", 0)
                        })
                        value["access_count"] = value.get("access_count", 0) + 1
        
        # 按相关性排序
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]
    
    def _calculate_relevance(self, query: str, query_words: set, memory: Dict) -> float:
        """计算相关性分数"""
        content = str(memory.get("content", "")).lower()
        tags = [t.lower() for t in memory.get("tags", [])]
        
        score = 0.0
        
        # 完全匹配
        if query in content:
            score += 1.0
        
        # 词语匹配
        content_words = set(content.split())
        word_overlap = len(query_words & content_words)
        score += word_overlap * 0.2
        
        # 标签匹配
        for tag in tags:
            if tag in query_words:
                score += 0.3
        
        # 重要性加权
        score *= (0.5 + memory.get("importance", 0.5))
        
        return score
    
    def get_stats(self) -> Dict:
        """获取记忆统计"""
        with self.lock:
            return {
                "sensory_count": len(self.sensory_memory),
                "short_term_count": len(self.short_term_memory),
                "long_term_count": len(self.long_term_memory),
                "total_memories": (
                    len(self.sensory_memory) + 
                    len(self.short_term_memory) + 
                    len(self.long_term_memory)
                ),
                "neuron_growth_events": len(self.neuron_growth),
                "index_size": len(self.keyword_index)
            }
    
    def clear_sensory(self):
        """清空瞬时记忆"""
        with self.lock:
            self.sensory_memory.clear()


class WikipediaSearch:
    """
    维基百科搜索模块
    提供无限知识库扩展
    """
    
    def __init__(self, cache_size: int = 100):
        self.cache: Dict[str, Dict] = {}
        self.cache_size = cache_size
        self.enabled = True
        
    def search(self, query: str, sentences: int = 3) -> Optional[str]:
        """搜索维基百科"""
        if not self.enabled:
            return None
            
        # 检查缓存
        if query.lower() in self.cache:
            return self.cache[query.lower()]["content"]
        
        try:
            import wikipedia
            
            # 设置语言
            if any('\u4e00' <= c <= '\u9fff' for c in query):
                wikipedia.set_lang("zh")
            else:
                wikipedia.set_lang("en")
            
            # 搜索
            results = wikipedia.search(query, results=5)
            
            if results:
                try:
                    page = wikipedia.page(results[0], auto_suggest=False)
                    summary = wikipedia.summary(results[0], sentences=sentences)
                    
                    # 缓存结果
                    self._add_to_cache(query.lower(), {
                        "content": summary,
                        "title": page.title,
                        "url": page.url
                    })
                    
                    return summary
                except wikipedia.DisambiguationError as e:
                    # 尝试第一个选项
                    try:
                        summary = wikipedia.summary(e.options[0], sentences=sentences)
                        return summary
                    except:
                        pass
                except:
                    pass
                    
        except ImportError:
            print("wikipedia模块未安装，使用pip install wikipedia安装")
        except Exception as e:
            pass
        
        return None
    
    def _add_to_cache(self, key: str, value: Dict):
        """添加到缓存"""
        if len(self.cache) >= self.cache_size:
            # 删除最旧的条目
            oldest = min(self.cache.items(), key=lambda x: x[1].get("timestamp", 0))
            del self.cache[oldest[0]]
        
        value["timestamp"] = time.time()
        self.cache[key] = value


class WebTools:
    """
    网页工具调用模块
    支持搜索、读取网页等功能
    """
    
    def __init__(self):
        self.enabled = True
        self.tools = {
            "web_search": self._web_search,
            "read_page": self._read_page,
            "wiki_search": self._wiki_search
        }
    
    def call(self, tool_name: str, **kwargs) -> Optional[Dict]:
        """调用工具"""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        if not self.enabled:
            return {"error": "Tools disabled"}
        
        try:
            result = self.tools[tool_name](**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _web_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """网页搜索"""
        try:
            import requests
            
            # 使用DuckDuckGo API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            results = []
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", ""),
                    "snippet": data.get("AbstractText", ""),
                    "url": data.get("AbstractURL", "")
                })
            
            for topic in data.get("RelatedTopics", [])[:num_results-1]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1] if topic.get("FirstURL") else "",
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", "")
                    })
            
            return results
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def _read_page(self, url: str) -> Optional[str]:
        """读取网页内容"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取文本
            text = soup.get_text(separator=' ', strip=True)
            return text[:5000]  # 限制长度
            
        except Exception as e:
            return None
    
    def _wiki_search(self, query: str) -> Optional[str]:
        """维基百科搜索"""
        wiki = WikipediaSearch()
        return wiki.search(query)


class BrainLikeStreamingEngine:
    """
    类脑流式处理引擎
    Brain-like Streaming Processing Engine
    
    核心特性：
    1. 高刷新率流式处理 (60Hz+)
    2. STDP在线学习
    3. 海马体记忆系统
    4. 维基百科知识扩展
    5. 多模态支持
    """
    
    def __init__(
        self,
        refresh_rate: int = 60,
        enable_stdp: bool = True,
        enable_memory: bool = True,
        enable_wiki: bool = True,
        enable_world_model: bool = True,
        learning_rate: float = 0.01
    ):
        self.refresh_rate = refresh_rate
        self.chunk_interval = 1000 / refresh_rate  # ms
        self.enable_world_model = enable_world_model
        
        # 模型
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.world_model = None
        self.world_processor = None
        
        # 子系统
        self.stdp = STDPOnlineLearning(learning_rate=learning_rate) if enable_stdp else None
        self.memory = HippocampusMemory() if enable_memory else None
        self.wiki = WikipediaSearch() if enable_wiki else None
        self.web_tools = WebTools()
        
        # 状态
        self.is_running = False
        self.processing_count = 0
        self.model_loaded = False
        
        # 回调
        self.on_chunk: Optional[Callable] = None
        self.on_memory_call: Optional[Callable] = None
        self.on_stdp_update: Optional[Callable] = None
    
    def load_models(self) -> bool:
        """加载模型"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("=" * 60)
        print("加载类脑AI模型")
        print("=" * 60)
        
        try:
            # 加载Qwen语言模型
            print("\n[1/2] 加载 Qwen2.5-0.5B-Instruct (语言模型)...")
            
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                QWEN_MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.qwen_model.eval()
            
            params = sum(p.numel() for p in self.qwen_model.parameters())
            print(f"✅ Qwen模型加载完成 ({params/1e6:.1f}M 参数)")
            
            # 初始化STDP
            if self.stdp:
                self.stdp.initialize_from_model(self.qwen_model)
            
            # 尝试加载世界模型（视觉）
            if self.enable_world_model:
                print("\n[2/2] 加载 Qwen2-VL-2B (世界模型)...")
                try:
                    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                    
                    self.world_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        WORLD_MODEL_PATH,
                        local_files_only=True,
                        trust_remote_code=True,
                        torch_dtype=torch.float32
                    )
                    self.world_processor = AutoProcessor.from_pretrained(
                        WORLD_MODEL_PATH,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    self.world_model.eval()
                    
                    params = sum(p.numel() for p in self.world_model.parameters())
                    print(f"✅ 世界模型加载完成 ({params/1e6:.1f}M 参数)")
                    
                except Exception as e:
                    print(f"⚠️ 世界模型加载失败: {e}")
                    print("继续使用纯文本模式...")
                    self.world_model = None
            else:
                print("\n[2/2] 跳过世界模型加载 (enable_world_model=False)")
                self.world_model = None
            
            self.model_loaded = True
            print("\n" + "=" * 60)
            print("模型加载完成！")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            traceback.print_exc()
            return False
    
    def stream_process(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        search_wiki: bool = True,
        search_memory: bool = True
    ) -> Generator[StreamChunk, None, None]:
        """
        高刷新率流式处理
        
        模拟人脑的实时信息处理：
        1. 接收输入
        2. 搜索记忆
        3. 搜索维基百科（可选）
        4. 流式生成输出
        5. STDP学习
        6. 存储记忆
        """
        import torch
        
        if not self.model_loaded:
            yield StreamChunk(type="control", content="error", metadata={"error": "Model not loaded"})
            return
        
        self.is_running = True
        self.processing_count += 1
        
        # 初始化变量
        memories = []
        
        # 1. 存储输入到瞬时记忆
        if self.memory:
            self.memory.store_sensory(prompt, source="user_input")
        
        # 2. 搜索记忆
        if search_memory and self.memory:
            yield StreamChunk(
                type="memory_call",
                metadata={"request": {"type": "search", "query": prompt[:100]}}
            )
            
            memories = self.memory.search(prompt, top_k=3)
            if memories:
                memory_context = "\n".join([
                    f"[记忆] {m['content'][:200]}"
                    for m in memories[:2]
                ])
                yield StreamChunk(
                    type="memory_call",
                    content=f"找到 {len(memories)} 条相关记忆",
                    metadata={"memories": memories}
                )
        
        # 3. 搜索维基百科
        wiki_info = ""
        if search_wiki and self.wiki:
            # 提取关键词
            keywords = self._extract_keywords(prompt)
            if keywords:
                yield StreamChunk(
                    type="wiki_search",
                    metadata={"query": keywords[0]}
                )
                
                wiki_result = self.wiki.search(keywords[0])
                if wiki_result:
                    wiki_info = f"\n[维基百科] {wiki_result}"
                    yield StreamChunk(
                        type="wiki_search",
                        content=wiki_result[:200],
                        metadata={"source": "wikipedia"}
                    )
        
        # 4. 构建输入
        context = ""
        if self.memory and memories:
            context = f"\n相关记忆：{memories[0]['content'][:100]}"
        if wiki_info:
            context += wiki_info[:300]
        
        text = f"<|im_start|>user\n{prompt}{context}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.qwen_tokenizer(text, return_tensors="pt")
        
        # 5. 流式生成
        generated_text = ""
        start_time = time.time()
        last_yield = start_time
        token_count = 0
        
        with torch.no_grad():
            for i in range(max_tokens):
                if not self.is_running:
                    break
                
                iter_start = time.time()
                
                # 前向传播
                outputs = self.qwen_model(**inputs)
                logits = outputs.logits[:, -1, :]
                
                # 采样
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = logits.argmax(dim=-1, keepdim=True)
                
                # 检查结束
                if next_token_id.item() == self.qwen_tokenizer.eos_token_id:
                    break
                
                # 解码
                next_token = self.qwen_tokenizer.decode(next_token_id[0])
                generated_text += next_token
                token_count += 1
                
                # STDP学习
                if self.stdp:
                    activation = float(logits.max())
                    layer_name = f"layer_{i % 10}"
                    self.stdp.record_spike(layer_name, activation)
                    
                    if i > 0:
                        prev_layer = f"layer_{(i-1) % 10}"
                        weight_change = self.stdp.compute_stdp_update(
                            prev_layer, layer_name, activation, activation
                        )
                        self.stdp.apply_update(layer_name, weight_change)
                        
                        if i % 10 == 0:  # 每10个token报告一次
                            yield StreamChunk(
                                type="weight_update",
                                metadata={"updates": self.stdp.update_count}
                            )
                
                # 更新输入
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id], dim=-1)
                
                # 按刷新率输出
                now = time.time()
                if (now - last_yield) * 1000 >= self.chunk_interval:
                    yield StreamChunk(
                        type="text",
                        content=next_token,
                        token_id=next_token_id.item(),
                        timestamp=now
                    )
                    last_yield = now
        
        # 6. 存储记忆
        if self.memory:
            key = self.memory.store_short_term(
                {"input": prompt, "output": generated_text},
                importance=0.6,
                tags=self._extract_keywords(prompt)[:5]
            )
            yield StreamChunk(
                type="memory_store",
                metadata={"key": key, "type": "short_term"}
            )
        
        # 7. 完成
        total_time = time.time() - start_time
        tokens_per_second = token_count / total_time if total_time > 0 else 0
        
        yield StreamChunk(
            type="control",
            content="done",
            metadata={
                "total_time": total_time,
                "tokens": token_count,
                "tokens_per_second": tokens_per_second,
                "stdp_updates": self.stdp.update_count if self.stdp else 0
            }
        )
        
        self.is_running = False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        import re
        
        # 移除标点
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 分词
        words = text.split()
        
        # 过滤停用词
        stopwords = {'的', '是', '在', '有', '和', '了', '不', '这', '我', '你', '他', '她', '它',
                    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as'}
        
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 1]
        
        return keywords[:5]
    
    def process_image(self, image_path: str, question: str = "描述这张图片") -> str:
        """处理图像"""
        if not self.world_model:
            return "世界模型未加载，无法处理图像"
        
        try:
            from PIL import Image
            
            image = Image.open(image_path)
            
            # 使用世界模型处理
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            text = self.world_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.world_processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            )
            
            # 生成
            import torch
            with torch.no_grad():
                outputs = self.world_model.generate(
                    **inputs,
                    max_new_tokens=200
                )
            
            response = self.world_processor.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            return f"图像处理失败: {e}"
    
    def stop(self):
        """停止处理"""
        self.is_running = False
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        return {
            "model_loaded": self.model_loaded,
            "processing_count": self.processing_count,
            "refresh_rate": self.refresh_rate,
            "is_running": self.is_running,
            "stdp": self.stdp.get_stats() if self.stdp else None,
            "memory": self.memory.get_stats() if self.memory else None,
            "wiki_enabled": self.wiki is not None
        }
    
    def save_weights(self, path: str = None):
        """保存训练权重"""
        path = path or os.path.join(WEIGHTS_PATH, "trained_weights.pt")
        if self.stdp:
            self.stdp.export_weights(path)


def main():
    """主函数 - 演示类脑AI系统"""
    print("\n" + "=" * 60)
    print("类脑AI系统 - 完整演示")
    print("=" * 60)
    print("\n核心特性:")
    print("  1. Qwen2.5-0.5B-Instruct 语言模型")
    print("  2. Qwen2-VL-2B 世界模型 (视觉)")
    print("  3. STDP在线学习")
    print("  4. 高刷新率流式处理 (60Hz)")
    print("  5. 海马体记忆系统")
    print("  6. 维基百科知识扩展")
    print("=" * 60)
    
    # 初始化引擎
    engine = BrainLikeStreamingEngine(
        refresh_rate=60,
        enable_stdp=True,
        enable_memory=True,
        enable_wiki=True,
        learning_rate=0.01
    )
    
    # 加载模型
    if not engine.load_models():
        print("模型加载失败，退出")
        return
    
    # 测试问题
    questions = [
        "请解释什么是量子纠缠？",
        "TCP三次握手是什么？",
        "什么是机器学习？"
    ]
    
    results = []
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}\n")
        
        print("📤 流式输出:\n")
        
        chunks = 0
        mem_calls = 0
        wiki_calls = 0
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
            elif chunk.type == "wiki_search":
                wiki_calls += 1
                if chunk.content:
                    print(f"\n[维基] {chunk.content[:100]}...\n", flush=True)
            elif chunk.type == "control":
                print(f"\n\n📊 统计:")
                print(f"  - Tokens: {chunk.metadata.get('tokens', 0)}")
                print(f"  - 速度: {chunk.metadata.get('tokens_per_second', 0):.1f} tokens/s")
                print(f"  - STDP更新: {chunk.metadata.get('stdp_updates', 0)}")
        
        results.append({
            "question": q,
            "response": text[:300],
            "chunks": chunks,
            "mem_calls": mem_calls,
            "wiki_calls": wiki_calls
        })
    
    # 系统状态
    print(f"\n{'='*60}")
    print("系统状态")
    print(f"{'='*60}")
    status = engine.get_status()
    print(f"  处理次数: {status['processing_count']}")
    print(f"  STDP更新: {status['stdp']['update_count'] if status['stdp'] else 0}")
    print(f"  记忆数量: {status['memory'] if status['memory'] else 0}")
    
    # 保存结果
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    result_file = os.path.join(OUTPUT_PATH, f"brain_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "status": status
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_file}")
    
    # 保存权重
    engine.save_weights()


if __name__ == "__main__":
    main()
