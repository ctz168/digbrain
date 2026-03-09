"""
海马体记忆系统
Hippocampal Memory System

实现存算分离的记忆系统，模拟海马体的记忆机制
参考DeepSeek论文框架
"""

import time
import threading
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import os


@dataclass
class Memory:
    """记忆单元"""
    id: str
    content: str
    memory_type: str  # 'short_term', 'long_term', 'working'
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    embeddings: Optional[List[float]] = None


@dataclass
class MemoryRequest:
    """记忆请求"""
    request_type: str  # 'search', 'store', 'consolidate', 'forget'
    query: str = ""
    content: str = ""
    memory_id: str = ""
    relevance_threshold: float = 0.5
    top_k: int = 5
    timestamp: float = field(default_factory=time.time)


class HippocampalMemory:
    """
    海马体记忆系统
    
    特点：
    1. 三阶段记忆：短期记忆 → 长期记忆 → 工作记忆
    2. 神经累积增长：记忆强度随访问增加
    3. 存算分离：存储层与计算层分离
    4. 按需搜索：基于相关性的记忆检索
    """
    
    def __init__(
        self,
        max_short_term: int = 100,
        max_long_term: int = 10000,
        max_working: int = 10,
        consolidation_threshold: float = 0.6,
        decay_rate: float = 0.01,
        storage_path: Optional[str] = None
    ):
        """
        初始化海马体记忆系统
        
        Args:
            max_short_term: 短期记忆最大容量
            max_long_term: 长期记忆最大容量
            max_working: 工作记忆最大容量
            consolidation_threshold: 记忆巩固阈值
            decay_rate: 记忆衰减率
            storage_path: 存储路径（存算分离）
        """
        # 记忆容量配置
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.max_working = max_working
        self.consolidation_threshold = consolidation_threshold
        self.decay_rate = decay_rate
        
        # 存储路径（存算分离）
        self.storage_path = storage_path or "./memory_storage"
        
        # 三阶段记忆存储
        self.short_term: deque = deque(maxlen=max_short_term)
        self.long_term: List[Memory] = []
        self.working: List[Memory] = []
        
        # 记忆索引（用于快速检索）
        self.index: Dict[str, Memory] = {}
        
        # 请求队列
        self.requests: List[MemoryRequest] = []
        
        # 统计信息
        self.stats = {
            'search_count': 0,
            'store_count': 0,
            'consolidate_count': 0,
            'forget_count': 0,
            'total_access': 0
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 初始化存储
        self._init_storage()
    
    def _init_storage(self) -> None:
        """初始化存储层"""
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 尝试加载已有记忆
        index_path = os.path.join(self.storage_path, "memory_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    for item in data.get('long_term', []):
                        mem = Memory(**item)
                        self.long_term.append(mem)
                        self.index[mem.id] = mem
                print(f"加载了 {len(self.long_term)} 条长期记忆")
            except Exception as e:
                print(f"加载记忆失败: {e}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_types: Optional[List[str]] = None
    ) -> List[Tuple[Memory, float]]:
        """
        搜索记忆
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            memory_types: 记忆类型过滤
            
        Returns:
            [(Memory, relevance), ...]
        """
        self.stats['search_count'] += 1
        
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        memory_types = memory_types or ['short_term', 'long_term', 'working']
        
        with self.lock:
            # 搜索短期记忆
            if 'short_term' in memory_types:
                for mem in self.short_term:
                    relevance = self._compute_relevance(query_words, mem)
                    if relevance > 0:
                        results.append((mem, relevance))
            
            # 搜索长期记忆
            if 'long_term' in memory_types:
                for mem in self.long_term:
                    relevance = self._compute_relevance(query_words, mem)
                    if relevance > 0:
                        results.append((mem, relevance))
            
            # 搜索工作记忆
            if 'working' in memory_types:
                for mem in self.working:
                    relevance = self._compute_relevance(query_words, mem)
                    if relevance > 0:
                        results.append((mem, relevance))
        
        # 按相关性排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 更新访问计数
        for mem, _ in results[:top_k]:
            mem.access_count += 1
            mem.last_access = time.time()
            self.stats['total_access'] += 1
        
        # 记录请求
        self.requests.append(MemoryRequest(
            request_type='search',
            query=query,
            top_k=top_k
        ))
        
        return results[:top_k]
    
    def store(
        self,
        content: str,
        memory_type: str = 'short_term',
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        存储记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性
            metadata: 元数据
            
        Returns:
            记忆ID
        """
        self.stats['store_count'] += 1
        
        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {}
        )
        
        with self.lock:
            # 根据类型存储
            if memory_type == 'short_term':
                self.short_term.append(memory)
            elif memory_type == 'long_term':
                self.long_term.append(memory)
            elif memory_type == 'working':
                self.working.append(memory)
                if len(self.working) > self.max_working:
                    self.working.pop(0)
            
            # 更新索引
            self.index[memory_id] = memory
        
        # 记录请求
        self.requests.append(MemoryRequest(
            request_type='store',
            content=content[:100],
            memory_id=memory_id
        ))
        
        # 检查是否需要巩固
        if len(self.short_term) >= self.max_short_term * 0.8:
            self._consolidate()
        
        return memory_id
    
    def _consolidate(self) -> int:
        """
        记忆巩固：短期记忆 → 长期记忆
        
        Returns:
            巩固的记忆数量
        """
        self.stats['consolidate_count'] += 1
        
        consolidated = 0
        to_remove = []
        
        with self.lock:
            for mem in list(self.short_term):
                # 计算记忆强度
                strength = self._compute_strength(mem)
                
                if strength >= self.consolidation_threshold:
                    # 转移到长期记忆
                    mem.memory_type = 'long_term'
                    self.long_term.append(mem)
                    to_remove.append(mem)
                    consolidated += 1
        
        # 从短期记忆中移除
        for mem in to_remove:
            try:
                self.short_term.remove(mem)
            except ValueError:
                pass
        
        # 保存到存储层
        if consolidated > 0:
            self._save_to_storage()
        
        # 记录请求
        self.requests.append(MemoryRequest(
            request_type='consolidate',
            content=f"巩固了 {consolidated} 条记忆"
        ))
        
        return consolidated
    
    def _compute_relevance(self, query_words: set, memory: Memory) -> float:
        """
        计算相关性
        
        Args:
            query_words: 查询词集合
            memory: 记忆
            
        Returns:
            相关性分数
        """
        content_lower = memory.content.lower()
        content_words = set(content_lower.split())
        
        if not query_words:
            return 0.0
        
        # 词重叠
        overlap = len(query_words & content_words)
        relevance = overlap / len(query_words)
        
        # 重要性加权
        relevance *= (0.5 + 0.5 * memory.importance)
        
        # 访问频率加权
        relevance *= (1 + 0.1 * min(memory.access_count, 10))
        
        return relevance
    
    def _compute_strength(self, memory: Memory) -> float:
        """
        计算记忆强度
        
        Args:
            memory: 记忆
            
        Returns:
            强度值
        """
        # 基础强度
        strength = memory.importance
        
        # 访问频率加成
        strength += 0.05 * min(memory.access_count, 20)
        
        # 时间衰减
        age = time.time() - memory.timestamp
        decay = self.decay_rate * (age / 3600)  # 每小时衰减
        strength *= max(0.1, 1 - decay)
        
        return strength
    
    def _save_to_storage(self) -> None:
        """保存到存储层（存算分离）"""
        index_path = os.path.join(self.storage_path, "memory_index.json")
        
        data = {
            'long_term': [
                {
                    'id': mem.id,
                    'content': mem.content,
                    'memory_type': mem.memory_type,
                    'importance': mem.importance,
                    'timestamp': mem.timestamp,
                    'access_count': mem.access_count,
                    'metadata': mem.metadata
                }
                for mem in self.long_term
            ],
            'stats': self.stats
        }
        
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def forget(self, memory_id: str) -> bool:
        """
        遗忘记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否成功
        """
        self.stats['forget_count'] += 1
        
        with self.lock:
            if memory_id in self.index:
                memory = self.index[memory_id]
                
                # 从对应存储中移除
                if memory.memory_type == 'short_term':
                    try:
                        self.short_term.remove(memory)
                    except ValueError:
                        pass
                elif memory.memory_type == 'long_term':
                    try:
                        self.long_term.remove(memory)
                    except ValueError:
                        pass
                elif memory.memory_type == 'working':
                    try:
                        self.working.remove(memory)
                    except ValueError:
                        pass
                
                del self.index[memory_id]
                return True
        
        return False
    
    def get_working_memory(self) -> List[Memory]:
        """获取工作记忆"""
        return list(self.working)
    
    def update_working_memory(self, memories: List[Memory]) -> None:
        """更新工作记忆"""
        with self.lock:
            self.working = memories[:self.max_working]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            return {
                **self.stats,
                'short_term_count': len(self.short_term),
                'long_term_count': len(self.long_term),
                'working_count': len(self.working),
                'total_memories': len(self.index),
                'requests_count': len(self.requests)
            }
    
    def clear(self, memory_type: Optional[str] = None) -> int:
        """
        清除记忆
        
        Args:
            memory_type: 记忆类型（None表示全部）
            
        Returns:
            清除的数量
        """
        count = 0
        
        with self.lock:
            if memory_type is None or memory_type == 'short_term':
                count += len(self.short_term)
                self.short_term.clear()
            
            if memory_type is None or memory_type == 'long_term':
                count += len(self.long_term)
                self.long_term.clear()
            
            if memory_type is None or memory_type == 'working':
                count += len(self.working)
                self.working.clear()
            
            if memory_type is None:
                self.index.clear()
        
        return count
