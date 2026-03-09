#!/usr/bin/env python3
"""
训练回调系统
Training Callback System

支持：
1. Early Stopping - 早停机制
2. Learning Rate Scheduling - 学习率调度
3. Model Checkpointing - 模型检查点
4. 日志记录
5. 自定义回调
"""

import os
import json
import time
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class TrainingState:
    """训练状态"""
    epoch: int = 0
    global_step: int = 0
    current_loss: float = 0.0
    best_loss: float = float('inf')
    best_metric: float = 0.0
    learning_rate: float = 0.01
    epochs_completed: int = 0
    batches_completed: int = 0
    start_time: float = 0.0
    last_save_time: float = 0.0
    history: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'current_loss': self.current_loss,
            'best_loss': self.best_loss,
            'best_metric': self.best_metric,
            'learning_rate': self.learning_rate,
            'epochs_completed': self.epochs_completed,
            'batches_completed': self.batches_completed,
            'start_time': self.start_time,
            'last_save_time': self.last_save_time,
            'history': self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingState':
        """从字典创建"""
        return cls(**data)


class Callback(ABC):
    """回调基类"""
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.trainer = None
    
    def set_trainer(self, trainer):
        """设置训练器引用"""
        self.trainer = trainer
    
    def on_train_begin(self, state: TrainingState) -> None:
        """训练开始时调用"""
        pass
    
    def on_train_end(self, state: TrainingState) -> None:
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, state: TrainingState) -> None:
        """epoch开始时调用"""
        pass
    
    def on_epoch_end(self, state: TrainingState) -> None:
        """epoch结束时调用"""
        pass
    
    def on_batch_begin(self, state: TrainingState) -> None:
        """batch开始时调用"""
        pass
    
    def on_batch_end(self, state: TrainingState) -> None:
        """batch结束时调用"""
        pass
    
    def on_evaluate(self, state: TrainingState, metrics: Dict) -> None:
        """评估时调用"""
        pass


class EarlyStopping(Callback):
    """
    早停机制
    
    当监控的指标在patience个epoch内没有改善时停止训练
    """
    
    def __init__(
        self,
        monitor: str = 'loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        初始化早停
        
        Args:
            monitor: 监控的指标名称
            patience: 容忍的epoch数
            min_delta: 最小改善量
            mode: 'min'或'max'
            restore_best_weights: 是否恢复最佳权重
            verbose: 是否打印信息
        """
        super().__init__("early_stopping")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def on_train_begin(self, state: TrainingState) -> None:
        """重置状态"""
        self.wait = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.stopped_epoch = 0
        self.stop_training = False
    
    def on_epoch_end(self, state: TrainingState) -> None:
        """检查是否应该停止"""
        current = self._get_current_value(state)
        
        if current is None:
            return
        
        if self._is_improvement(current):
            self.best_value = current
            self.best_epoch = state.epoch
            self.wait = 0
            
            # 保存最佳权重
            if self.restore_best_weights and self.trainer:
                self.best_weights = self._get_weights()
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stop_training = True
                self.stopped_epoch = state.epoch
                
                if self.verbose:
                    print(f"\n[EarlyStopping] 在 epoch {self.stopped_epoch} 停止训练")
                    print(f"[EarlyStopping] 最佳 epoch: {self.best_epoch}, 最佳 {self.monitor}: {self.best_value:.4f}")
    
    def _get_current_value(self, state: TrainingState) -> Optional[float]:
        """获取当前监控值"""
        if self.monitor == 'loss':
            return state.current_loss
        elif self.monitor == 'best_loss':
            return state.best_loss
        elif self.monitor == 'best_metric':
            return state.best_metric
        elif state.history:
            last_record = state.history[-1]
            return last_record.get(self.monitor)
        return None
    
    def _is_improvement(self, current: float) -> bool:
        """判断是否改善"""
        if self.mode == 'min':
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta
    
    def _get_weights(self) -> Dict:
        """获取当前权重"""
        if hasattr(self.trainer, 'model') and self.trainer.model:
            import torch
            return {
                k: v.cpu().clone()
                for k, v in self.trainer.model.state_dict().items()
            }
        return None
    
    def restore_weights(self) -> bool:
        """恢复最佳权重"""
        if self.best_weights and hasattr(self.trainer, 'model') and self.trainer.model:
            import torch
            self.trainer.model.load_state_dict({
                k: v.to(next(self.trainer.model.parameters()).device)
                for k, v in self.best_weights.items()
            })
            return True
        return False


class LearningRateScheduler(Callback):
    """
    学习率调度器
    
    支持多种调度策略：
    - constant: 恒定学习率
    - linear: 线性衰减
    - cosine: 余弦退火
    - exponential: 指数衰减
    - warmup: 预热
    - step: 阶梯衰减
    """
    
    def __init__(
        self,
        initial_lr: float = 0.01,
        schedule_type: str = 'constant',
        warmup_steps: int = 0,
        warmup_factor: float = 0.1,
        decay_steps: int = 1000,
        decay_rate: float = 0.1,
        min_lr: float = 1e-6,
        verbose: bool = True
    ):
        """
        初始化学习率调度器
        
        Args:
            initial_lr: 初始学习率
            schedule_type: 调度类型
            warmup_steps: 预热步数
            warmup_factor: 预热因子
            decay_steps: 衰减步数
            decay_rate: 衰减率
            min_lr: 最小学习率
            verbose: 是否打印信息
        """
        super().__init__("lr_scheduler")
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.current_lr = initial_lr
        self._step_count = 0
    
    def on_train_begin(self, state: TrainingState) -> None:
        """初始化"""
        self._step_count = 0
        self.current_lr = self.initial_lr
        state.learning_rate = self.current_lr
    
    def on_batch_end(self, state: TrainingState) -> None:
        """更新学习率"""
        self._step_count += 1
        
        # 计算新的学习率
        new_lr = self._compute_lr(self._step_count)
        
        if abs(new_lr - self.current_lr) > 1e-8:
            self.current_lr = new_lr
            state.learning_rate = self.current_lr
            
            # 更新优化器
            if self.trainer and hasattr(self.trainer, 'optimizer') and self.trainer.optimizer:
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                
                if self.verbose and self._step_count % 100 == 0:
                    print(f"[LRScheduler] Step {self._step_count}, LR: {self.current_lr:.6f}")
    
    def _compute_lr(self, step: int) -> float:
        """计算学习率"""
        # 预热阶段
        if step < self.warmup_steps:
            warmup_lr = self.initial_lr * (
                self.warmup_factor + (1 - self.warmup_factor) * step / self.warmup_steps
            )
            return max(warmup_lr, self.min_lr)
        
        # 预热后的步数
        step_after_warmup = step - self.warmup_steps
        
        if self.schedule_type == 'constant':
            lr = self.initial_lr
        
        elif self.schedule_type == 'linear':
            lr = self.initial_lr * (1 - step_after_warmup / self.decay_steps)
        
        elif self.schedule_type == 'cosine':
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + math.cos(math.pi * step_after_warmup / self.decay_steps)
            )
        
        elif self.schedule_type == 'exponential':
            lr = self.initial_lr * (self.decay_rate ** (step_after_warmup / self.decay_steps))
        
        elif self.schedule_type == 'step':
            # 每decay_steps步衰减一次
            num_decays = step_after_warmup // self.decay_steps
            lr = self.initial_lr * (self.decay_rate ** num_decays)
        
        elif self.schedule_type == 'polynomial':
            lr = (self.initial_lr - self.min_lr) * (
                1 - step_after_warmup / self.decay_steps
            ) ** 2 + self.min_lr
        
        else:
            lr = self.initial_lr
        
        return max(lr, self.min_lr)
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr


class ModelCheckpoint(Callback):
    """
    模型检查点
    
    支持按epoch或按metric保存模型
    """
    
    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        monitor: str = 'loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        save_every_n_epochs: int = 1,
        max_checkpoints: int = 5,
        filename_prefix: str = 'checkpoint',
        verbose: bool = True
    ):
        """
        初始化检查点
        
        Args:
            checkpoint_dir: 检查点目录
            monitor: 监控指标
            mode: 'min'或'max'
            save_best_only: 是否只保存最佳模型
            save_weights_only: 是否只保存权重
            save_every_n_epochs: 每n个epoch保存一次
            max_checkpoints: 最大检查点数量
            filename_prefix: 文件名前缀
            verbose: 是否打印信息
        """
        super().__init__("model_checkpoint")
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_every_n_epochs = save_every_n_epochs
        self.max_checkpoints = max_checkpoints
        self.filename_prefix = filename_prefix
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.saved_checkpoints: List[str] = []
    
    def on_train_begin(self, state: TrainingState) -> None:
        """创建检查点目录"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.saved_checkpoints = []
    
    def on_epoch_end(self, state: TrainingState) -> None:
        """保存检查点"""
        current = self._get_current_value(state)
        
        if current is None:
            return
        
        should_save = False
        
        if self._is_best(current):
            self.best_value = current
            should_save = True
        elif not self.save_best_only:
            if state.epoch % self.save_every_n_epochs == 0:
                should_save = True
        
        if should_save:
            self._save_checkpoint(state, current)
    
    def on_train_end(self, state: TrainingState) -> None:
        """训练结束时保存最终检查点"""
        if not self.save_best_only:
            self._save_checkpoint(state, state.current_loss, is_final=True)
    
    def _get_current_value(self, state: TrainingState) -> Optional[float]:
        """获取当前监控值"""
        if self.monitor == 'loss':
            return state.current_loss
        elif self.monitor == 'best_loss':
            return state.best_loss
        elif self.monitor == 'best_metric':
            return state.best_metric
        elif state.history:
            last_record = state.history[-1]
            return last_record.get(self.monitor)
        return None
    
    def _is_best(self, current: float) -> bool:
        """判断是否最佳"""
        if self.mode == 'min':
            return current < self.best_value
        else:
            return current > self.best_value
    
    def _save_checkpoint(self, state: TrainingState, metric_value: float, is_final: bool = False):
        """保存检查点"""
        import torch
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if is_final:
            filename = f"{self.filename_prefix}_final_{timestamp}.pt"
        else:
            filename = f"{self.filename_prefix}_epoch{state.epoch}_{timestamp}.pt"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # 准备保存数据
        checkpoint_data = {
            'epoch': state.epoch,
            'global_step': state.global_step,
            'best_loss': state.best_loss,
            'best_metric': state.best_metric,
            'current_loss': state.current_loss,
            'learning_rate': state.learning_rate,
            'timestamp': timestamp,
            'metric_value': metric_value,
            'training_state': state.to_dict()
        }
        
        # 保存模型
        if self.trainer and hasattr(self.trainer, 'model') and self.trainer.model:
            if self.save_weights_only:
                checkpoint_data['model_state_dict'] = self.trainer.model.state_dict()
            else:
                checkpoint_data['model'] = self.trainer.model
            
            if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer:
                checkpoint_data['optimizer_state_dict'] = self.trainer.optimizer.state_dict()
        
        # 保存
        torch.save(checkpoint_data, filepath)
        self.saved_checkpoints.append(filepath)
        
        if self.verbose:
            print(f"[Checkpoint] 保存检查点: {filepath}")
            print(f"[Checkpoint] Epoch {state.epoch}, {self.monitor}: {metric_value:.4f}")
        
        # 清理旧的检查点
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        if len(self.saved_checkpoints) > self.max_checkpoints:
            num_to_remove = len(self.saved_checkpoints) - self.max_checkpoints
            for old_checkpoint in self.saved_checkpoints[:num_to_remove]:
                try:
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                        if self.verbose:
                            print(f"[Checkpoint] 删除旧检查点: {old_checkpoint}")
                except Exception as e:
                    print(f"[Checkpoint] 删除失败: {e}")
            
            self.saved_checkpoints = self.saved_checkpoints[num_to_remove:]


class TrainingLogger(Callback):
    """
    训练日志记录器
    
    支持控制台输出和文件记录
    """
    
    def __init__(
        self,
        log_dir: str = './logs',
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_every_n_steps: int = 10,
        include_memory: bool = True,
        include_time: bool = True
    ):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            log_to_file: 是否记录到文件
            log_to_console: 是否输出到控制台
            log_every_n_steps: 每n步记录一次
            include_memory: 是否包含内存信息
            include_time: 是否包含时间信息
        """
        super().__init__("training_logger")
        self.log_dir = log_dir
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_every_n_steps = log_every_n_steps
        self.include_memory = include_memory
        self.include_time = include_time
        
        self.log_file = None
        self.start_time = 0.0
    
    def on_train_begin(self, state: TrainingState) -> None:
        """创建日志文件"""
        if self.log_to_file:
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(self.log_dir, f'training_{timestamp}.log')
            self.log_file = open(log_path, 'w', encoding='utf-8')
        
        self.start_time = time.time()
        self._log("=" * 60)
        self._log("训练开始")
        self._log(f"开始时间: {datetime.now().isoformat()}")
        self._log("=" * 60)
    
    def on_train_end(self, state: TrainingState) -> None:
        """关闭日志文件"""
        total_time = time.time() - self.start_time
        
        self._log("=" * 60)
        self._log("训练结束")
        self._log(f"结束时间: {datetime.now().isoformat()}")
        self._log(f"总训练时间: {total_time:.2f}秒")
        self._log(f"完成epochs: {state.epochs_completed}")
        self._log(f"完成batches: {state.batches_completed}")
        self._log(f"最佳loss: {state.best_loss:.4f}")
        self._log("=" * 60)
        
        if self.log_file:
            self.log_file.close()
    
    def on_epoch_begin(self, state: TrainingState) -> None:
        """epoch开始"""
        self._log(f"\n--- Epoch {state.epoch} 开始 ---")
    
    def on_epoch_end(self, state: TrainingState) -> None:
        """epoch结束"""
        epoch_time = time.time() - self.start_time
        self._log(f"--- Epoch {state.epoch} 结束 ---")
        self._log(f"  Loss: {state.current_loss:.4f}")
        self._log(f"  Best Loss: {state.best_loss:.4f}")
        self._log(f"  LR: {state.learning_rate:.6f}")
        
        # 记录到历史
        state.history.append({
            'epoch': state.epoch,
            'loss': state.current_loss,
            'best_loss': state.best_loss,
            'lr': state.learning_rate,
            'time': epoch_time
        })
    
    def on_batch_end(self, state: TrainingState) -> None:
        """batch结束"""
        if state.global_step % self.log_every_n_steps == 0:
            elapsed = time.time() - self.start_time
            msg = f"Step {state.global_step}: loss={state.current_loss:.4f}"
            
            if self.include_time:
                msg += f", time={elapsed:.1f}s"
            
            if self.include_memory:
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    msg += f", mem={mem.percent:.1f}%"
                except:
                    pass
            
            self._log(msg)
    
    def _log(self, message: str):
        """记录日志"""
        if self.log_to_console:
            print(message)
        
        if self.log_to_file and self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()


class ProgressTracker(Callback):
    """
    进度追踪器
    
    追踪训练进度并支持恢复
    """
    
    def __init__(
        self,
        save_dir: str = './progress',
        save_every_n_steps: int = 100,
        verbose: bool = True
    ):
        """
        初始化进度追踪器
        
        Args:
            save_dir: 进度保存目录
            save_every_n_steps: 每n步保存一次
            verbose: 是否打印信息
        """
        super().__init__("progress_tracker")
        self.save_dir = save_dir
        self.save_every_n_steps = save_every_n_steps
        self.verbose = verbose
        
        self.progress_file = os.path.join(save_dir, 'training_progress.json')
    
    def on_train_begin(self, state: TrainingState) -> None:
        """创建进度目录"""
        os.makedirs(self.save_dir, exist_ok=True)
        state.start_time = time.time()
    
    def on_batch_end(self, state: TrainingState) -> None:
        """保存进度"""
        if state.global_step % self.save_every_n_steps == 0:
            self._save_progress(state)
    
    def on_epoch_end(self, state: TrainingState) -> None:
        """epoch结束保存进度"""
        self._save_progress(state)
    
    def _save_progress(self, state: TrainingState):
        """保存进度到文件"""
        progress_data = state.to_dict()
        progress_data['last_update'] = datetime.now().isoformat()
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
        
        if self.verbose and state.global_step % (self.save_every_n_steps * 10) == 0:
            print(f"[Progress] 保存进度: Step {state.global_step}, Epoch {state.epoch}")
    
    def load_progress(self) -> Optional[TrainingState]:
        """加载进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                state = TrainingState.from_dict(data)
                
                if self.verbose:
                    print(f"[Progress] 加载进度: Step {state.global_step}, Epoch {state.epoch}")
                
                return state
            except Exception as e:
                print(f"[Progress] 加载失败: {e}")
        
        return None
    
    def clear_progress(self):
        """清除进度"""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            if self.verbose:
                print("[Progress] 进度已清除")


class MetricTracker(Callback):
    """
    指标追踪器
    
    追踪和可视化训练指标
    """
    
    def __init__(
        self,
        metrics: List[str] = None,
        track_ema: bool = True,
        ema_decay: float = 0.9,
        window_size: int = 100
    ):
        """
        初始化指标追踪器
        
        Args:
            metrics: 要追踪的指标列表
            track_ema: 是否追踪指数移动平均
            ema_decay: EMA衰减率
            window_size: 滑动窗口大小
        """
        super().__init__("metric_tracker")
        self.metrics = metrics or ['loss']
        self.track_ema = track_ema
        self.ema_decay = ema_decay
        self.window_size = window_size
        
        self.metric_history: Dict[str, List[float]] = {m: [] for m in self.metrics}
        self.ema_values: Dict[str, float] = {}
        self.recent_values: Dict[str, List[float]] = {m: [] for m in self.metrics}
    
    def on_train_begin(self, state: TrainingState) -> None:
        """重置"""
        self.metric_history = {m: [] for m in self.metrics}
        self.ema_values = {}
        self.recent_values = {m: [] for m in self.metrics}
    
    def on_batch_end(self, state: TrainingState) -> None:
        """更新指标"""
        # 更新loss
        self._update_metric('loss', state.current_loss)
    
    def on_epoch_end(self, state: TrainingState) -> None:
        """记录epoch指标"""
        for metric in self.metrics:
            if metric == 'loss':
                self.metric_history[metric].append(state.current_loss)
    
    def _update_metric(self, metric_name: str, value: float):
        """更新单个指标"""
        # 添加到历史
        self.metric_history[metric_name].append(value)
        
        # 更新EMA
        if self.track_ema:
            if metric_name not in self.ema_values:
                self.ema_values[metric_name] = value
            else:
                self.ema_values[metric_name] = (
                    self.ema_decay * self.ema_values[metric_name] +
                    (1 - self.ema_decay) * value
                )
        
        # 更新滑动窗口
        self.recent_values[metric_name].append(value)
        if len(self.recent_values[metric_name]) > self.window_size:
            self.recent_values[metric_name].pop(0)
    
    def get_ema(self, metric_name: str) -> float:
        """获取EMA值"""
        return self.ema_values.get(metric_name, 0.0)
    
    def get_recent_avg(self, metric_name: str) -> float:
        """获取最近N个值的平均"""
        recent = self.recent_values.get(metric_name, [])
        return sum(recent) / len(recent) if recent else 0.0
    
    def get_trend(self, metric_name: str) -> str:
        """获取趋势"""
        recent = self.recent_values.get(metric_name, [])
        if len(recent) < 10:
            return "unknown"
        
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        if second_half < first_half * 0.95:
            return "decreasing"
        elif second_half > first_half * 1.05:
            return "increasing"
        else:
            return "stable"
    
    def get_summary(self) -> Dict:
        """获取摘要"""
        summary = {}
        for metric in self.metrics:
            history = self.metric_history.get(metric, [])
            if history:
                summary[metric] = {
                    'current': history[-1],
                    'min': min(history),
                    'max': max(history),
                    'ema': self.get_ema(metric),
                    'recent_avg': self.get_recent_avg(metric),
                    'trend': self.get_trend(metric)
                }
        return summary


class CallbackHandler:
    """
    回调处理器
    
    管理多个回调
    """
    
    def __init__(self, callbacks: List[Callback] = None):
        """
        初始化回调处理器
        
        Args:
            callbacks: 回调列表
        """
        self.callbacks: List[Callback] = callbacks or []
        self._lock = threading.Lock()
    
    def add_callback(self, callback: Callback):
        """添加回调"""
        with self._lock:
            self.callbacks.append(callback)
    
    def remove_callback(self, name: str):
        """移除回调"""
        with self._lock:
            self.callbacks = [cb for cb in self.callbacks if cb.name != name]
    
    def set_trainer(self, trainer):
        """设置训练器引用"""
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def on_train_begin(self, state: TrainingState):
        """训练开始"""
        for callback in self.callbacks:
            callback.on_train_begin(state)
    
    def on_train_end(self, state: TrainingState):
        """训练结束"""
        for callback in self.callbacks:
            callback.on_train_end(state)
    
    def on_epoch_begin(self, state: TrainingState):
        """epoch开始"""
        for callback in self.callbacks:
            callback.on_epoch_begin(state)
    
    def on_epoch_end(self, state: TrainingState):
        """epoch结束"""
        for callback in self.callbacks:
            callback.on_epoch_end(state)
    
    def on_batch_begin(self, state: TrainingState):
        """batch开始"""
        for callback in self.callbacks:
            callback.on_batch_begin(state)
    
    def on_batch_end(self, state: TrainingState):
        """batch结束"""
        for callback in self.callbacks:
            callback.on_batch_end(state)
    
    def on_evaluate(self, state: TrainingState, metrics: Dict):
        """评估"""
        for callback in self.callbacks:
            callback.on_evaluate(state, metrics)
    
    def should_stop(self) -> bool:
        """检查是否应该停止训练"""
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping) and callback.stop_training:
                return True
        return False


def create_default_callbacks(
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    early_stopping_patience: int = 5,
    lr_schedule_type: str = 'cosine',
    initial_lr: float = 0.01
) -> List[Callback]:
    """
    创建默认回调集合
    
    Args:
        checkpoint_dir: 检查点目录
        log_dir: 日志目录
        early_stopping_patience: 早停耐心值
        lr_schedule_type: 学习率调度类型
        initial_lr: 初始学习率
        
    Returns:
        回调列表
    """
    callbacks = [
        TrainingLogger(log_dir=log_dir),
        EarlyStopping(patience=early_stopping_patience),
        LearningRateScheduler(initial_lr=initial_lr, schedule_type=lr_schedule_type),
        ModelCheckpoint(checkpoint_dir=checkpoint_dir),
        ProgressTracker(),
        MetricTracker()
    ]
    
    return callbacks


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("回调系统测试")
    print("=" * 60)
    
    # 创建测试状态
    state = TrainingState()
    
    # 测试学习率调度器
    print("\n[1] 学习率调度器测试...")
    lr_scheduler = LearningRateScheduler(
        initial_lr=0.01,
        schedule_type='cosine',
        warmup_steps=100,
        decay_steps=1000
    )
    
    test_steps = [0, 50, 100, 200, 500, 1000]
    for step in test_steps:
        lr = lr_scheduler._compute_lr(step)
        print(f"   Step {step}: LR = {lr:.6f}")
    
    # 测试早停
    print("\n[2] 早停机制测试...")
    early_stop = EarlyStopping(patience=3, monitor='loss')
    
    losses = [1.0, 0.9, 0.85, 0.84, 0.83, 0.84, 0.85, 0.86, 0.87]
    for i, loss in enumerate(losses):
        state.epoch = i + 1
        state.current_loss = loss
        early_stop.on_epoch_end(state)
        print(f"   Epoch {i+1}: loss={loss:.2f}, wait={early_stop.wait}")
        if early_stop.stop_training:
            print(f"   -> 早停触发!")
            break
    
    # 测试指标追踪
    print("\n[3] 指标追踪器测试...")
    tracker = MetricTracker()
    
    import random
    for i in range(20):
        loss = 1.0 - i * 0.03 + random.uniform(-0.02, 0.02)
        state.current_loss = max(0.1, loss)
        tracker.on_batch_end(state)
    
    summary = tracker.get_summary()
    print(f"   当前loss: {summary['loss']['current']:.4f}")
    print(f"   EMA: {summary['loss']['ema']:.4f}")
    print(f"   趋势: {summary['loss']['trend']}")
    
    # 测试回调处理器
    print("\n[4] 回调处理器测试...")
    handler = CallbackHandler(create_default_callbacks())
    print(f"   已注册 {len(handler.callbacks)} 个回调")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
