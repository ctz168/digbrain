"""
STDP在线学习模块
Spike-Timing-Dependent Plasticity Online Learning Module

实现脉冲时间依赖可塑性在线学习
"""

import math
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class STDPEvent:
    """STDP学习事件"""
    layer_name: str
    pre_activation: float
    post_activation: float
    weight_change: float
    delta_t: float
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)


class STDPOnlineLearning:
    """
    脉冲时间依赖可塑性在线学习系统
    Spike-Timing-Dependent Plasticity Online Learning System
    
    STDP规则：
    - 如果pre在post之前激活（Δt > 0）：LTP（长时程增强）
    - 如果post在pre之前激活（Δt < 0）：LTD（长时程抑制）
    
    Δw = η × exp(-|Δt|/τ) × pre_act × post_act
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        stdp_window: float = 20.0,  # ms
        ltp_rate: float = 0.1,
        ltd_rate: float = 0.12,
        reward_modulation: float = 0.5
    ):
        """
        初始化STDP学习系统
        
        Args:
            learning_rate: 学习率
            stdp_window: STDP时间窗口（毫秒）
            ltp_rate: 长时程增强率
            ltd_rate: 长时程抑制率
            reward_modulation: 奖励调制因子
        """
        self.learning_rate = learning_rate
        self.stdp_window = stdp_window
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.reward_modulation = reward_modulation
        
        # 权重存储
        self.weights: Dict[str, Any] = {}
        self.weight_deltas: Dict[str, float] = {}
        
        # 脉冲时间记录
        self.spike_times: Dict[str, Dict] = {}
        
        # 资格迹（Eligibility Trace）
        self.eligibility_traces: Dict[str, np.ndarray] = {}
        
        # 学习事件历史
        self.events: List[STDPEvent] = []
        
        # 统计信息
        self.total_updates = 0
        self.ltp_count = 0
        self.ltd_count = 0
        self.total_reward = 0.0
        
        # 线程锁
        self.lock = threading.Lock()
        
    def initialize_from_model(self, model) -> int:
        """
        从模型初始化权重
        
        Args:
            model: PyTorch模型
            
        Returns:
            初始化的权重层数量
        """
        import torch
        
        print("初始化STDP权重系统...")
        
        count = 0
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                self.weights[name] = {
                    'data': param.data.clone(),
                    'shape': list(param.shape),
                    'mean': float(param.data.mean()),
                    'std': float(param.data.std()),
                    'min': float(param.data.min()),
                    'max': float(param.data.max())
                }
                self.eligibility_traces[name] = np.zeros(param.shape)
                self.weight_deltas[name] = 0.0
                count += 1
                
        print(f"✅ 初始化了 {count} 个权重层")
        return count
    
    def record_spike(self, layer: str, activation: float) -> None:
        """
        记录脉冲（激活）
        
        Args:
            layer: 层名称
            activation: 激活值
        """
        with self.lock:
            self.spike_times[layer] = {
                'time': time.time() * 1000,  # 转换为毫秒
                'activation': activation
            }
    
    def compute_stdp_update(
        self,
        pre_layer: str,
        post_layer: str,
        pre_activation: Optional[float] = None,
        post_activation: Optional[float] = None
    ) -> float:
        """
        计算STDP权重更新
        
        Args:
            pre_layer: 前一层名称
            post_layer: 后一层名称
            pre_activation: 前一层激活值（可选）
            post_activation: 后一层激活值（可选）
            
        Returns:
            权重变化量
        """
        # 获取脉冲时间
        pre_spike = self.spike_times.get(pre_layer)
        post_spike = self.spike_times.get(post_layer)
        
        if pre_spike is None or post_spike is None:
            return 0.0
        
        # 获取激活值
        pre_act = pre_activation if pre_activation is not None else pre_spike['activation']
        post_act = post_activation if post_activation is not None else post_spike['activation']
        
        # 计算时间差
        delta_t = post_spike['time'] - pre_spike['time']
        
        # 检查是否在时间窗口内
        if abs(delta_t) > self.stdp_window:
            return 0.0
        
        # 计算STDP更新
        if delta_t > 0:
            # LTP: 后激活在前激活之后 -> 增强
            weight_change = self.ltp_rate * math.exp(-delta_t / self.stdp_window)
            with self.lock:
                self.ltp_count += 1
        else:
            # LTD: 前激活在后激活之后 -> 抑制
            weight_change = -self.ltd_rate * math.exp(delta_t / self.stdp_window)
            with self.lock:
                self.ltd_count += 1
        
        # 应用激活值调制
        weight_change *= pre_act * post_act * self.learning_rate
        
        return weight_change
    
    def update_weights(
        self,
        layer: str,
        weight_change: float,
        reward: float = 0.0
    ) -> None:
        """
        更新权重
        
        Args:
            layer: 层名称
            weight_change: 权重变化量
            reward: 奖励信号
        """
        if layer not in self.weights:
            return
        
        # 应用奖励调制
        if reward != 0:
            weight_change *= (1 + self.reward_modulation * reward)
            with self.lock:
                self.total_reward += reward
        
        # 更新权重
        with self.lock:
            self.weights[layer]['mean'] += weight_change * 0.001
            self.weight_deltas[layer] = weight_change
            self.total_updates += 1
        
        # 记录事件
        event = STDPEvent(
            layer_name=layer,
            pre_activation=0.0,
            post_activation=0.0,
            weight_change=weight_change,
            delta_t=0.0,
            reward=reward
        )
        self.events.append(event)
    
    def apply_to_model(self, model) -> int:
        """
        将权重更新应用到模型
        
        Args:
            model: PyTorch模型
            
        Returns:
            更新的层数量
        """
        import torch
        
        count = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.weights and name in self.weight_deltas:
                    # 应用小的扰动
                    delta = self.weight_deltas[name]
                    noise = torch.randn_like(param.data) * abs(delta) * 0.0001
                    param.data.add_(noise)
                    count += 1
        
        return count
    
    def compute_reward(self, text: str, metrics: Optional[Dict] = None) -> float:
        """
        计算奖励信号
        
        Args:
            text: 生成的文本
            metrics: 额外指标
            
        Returns:
            奖励值 [-1, 1]
        """
        reward = 0.0
        
        # 长度奖励
        if len(text) > 30:
            reward += 0.1
        if len(text) > 60:
            reward += 0.1
        if len(text) > 100:
            reward += 0.1
        
        # 结构奖励
        if '。' in text or '.' in text:
            reward += 0.1
        if '因为' in text or '所以' in text:
            reward += 0.15
        if '首先' in text or '然后' in text or '最后' in text:
            reward += 0.1
        
        # 多样性奖励
        unique_chars = len(set(text))
        if unique_chars > len(text) * 0.3:
            reward += 0.1
        
        # 质量惩罚
        if text.count('的') > len(text) * 0.1:
            reward -= 0.1
        if text.count('是') > len(text) * 0.1:
            reward -= 0.05
        
        # 额外指标
        if metrics:
            if metrics.get('correct', False):
                reward += 0.2
            if metrics.get('relevance', 0) > 0.8:
                reward += 0.1
        
        return max(-1.0, min(1.0, reward))
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        with self.lock:
            return {
                'total_updates': self.total_updates,
                'ltp_count': self.ltp_count,
                'ltd_count': self.ltd_count,
                'ltp_ratio': self.ltp_count / max(1, self.total_updates),
                'ltd_ratio': self.ltd_count / max(1, self.total_updates),
                'total_reward': self.total_reward,
                'avg_reward': self.total_reward / max(1, self.total_updates),
                'total_layers': len(self.weights),
                'events_count': len(self.events)
            }
    
    def export_weights(self, path: str) -> None:
        """
        导出权重
        
        Args:
            path: 保存路径
        """
        import json
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        export_data = {
            'config': {
                'learning_rate': self.learning_rate,
                'stdp_window': self.stdp_window,
                'ltp_rate': self.ltp_rate,
                'ltd_rate': self.ltd_rate
            },
            'stats': self.get_stats(),
            'weights': {
                name: {
                    'mean': w['mean'],
                    'std': w['std'],
                    'shape': w['shape']
                }
                for name, w in self.weights.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"权重已导出到: {path}")
    
    def reset(self) -> None:
        """重置学习状态"""
        with self.lock:
            self.spike_times.clear()
            self.events.clear()
            self.total_updates = 0
            self.ltp_count = 0
            self.ltd_count = 0
            self.total_reward = 0.0
