#!/usr/bin/env python3
"""
离线训练模块 - 完整版 v2.0
Offline Training Module - Complete Version

支持：
1. 各模块单独训练（记忆模块、STDP模块、语言模块）
2. 综合多线程训练模块
3. 训练进度保存和恢复
4. 训练日志和可视化
5. 回调系统集成
6. 数据加载器集成
"""

import os
import sys
import json
import time
import threading
import multiprocessing
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Generator, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback
import math

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_MODEL_PATH = os.path.join(BASE_DIR, "models/Qwen3.5-0.8B")
OUTPUT_PATH = os.path.join(BASE_DIR, "weights")
DATA_PATH = os.path.join(BASE_DIR, "training_data")
LOG_PATH = os.path.join(BASE_DIR, "logs")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints")

# 导入回调模块
from .callbacks import (
    Callback, CallbackHandler, TrainingState,
    EarlyStopping, LearningRateScheduler, ModelCheckpoint,
    TrainingLogger, ProgressTracker, MetricTracker,
    create_default_callbacks
)

# 导入数据加载器
from .data_loader import (
    DataLoader, DataLoaderConfig, DataSample,
    MemoryTrainingDataset, STDPTrainingDataset, LanguageTrainingDataset
)


@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 0.01
    epochs: int = 10
    batch_size: int = 4
    max_length: int = 512
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 200
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    lr_schedule_type: str = 'cosine'
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingConfig':
        return cls(**data)


class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or LOG_PATH
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.loss_history: List[float] = []
        self.lr_history: List[float] = []
        self.epoch_times: List[float] = []
        self.timestamps: List[float] = []
    
    def record(self, loss: float, lr: float):
        """记录训练数据"""
        self.loss_history.append(loss)
        self.lr_history.append(lr)
        self.timestamps.append(time.time())
    
    def record_epoch_time(self, epoch_time: float):
        """记录epoch时间"""
        self.epoch_times.append(epoch_time)
    
    def plot_training_curves(self, save_path: str = None):
        """绘制训练曲线"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Loss曲线
            if self.loss_history:
                ax1 = axes[0, 0]
                ax1.plot(self.loss_history, 'b-', label='Training Loss', linewidth=1.5)
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss Curve')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 学习率曲线
            if self.lr_history:
                ax2 = axes[0, 1]
                ax2.plot(self.lr_history, 'g-', label='Learning Rate', linewidth=1.5)
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate Schedule')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Loss分布直方图
            if self.loss_history:
                ax3 = axes[1, 0]
                ax3.hist(self.loss_history, bins=30, color='blue', alpha=0.7, edgecolor='black')
                ax3.set_xlabel('Loss')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Loss Distribution')
                ax3.grid(True, alpha=0.3)
            
            # Epoch时间
            if self.epoch_times:
                ax4 = axes[1, 1]
                epochs = range(1, len(self.epoch_times) + 1)
                ax4.bar(epochs, self.epoch_times, color='orange', alpha=0.7)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Time (seconds)')
                ax4.set_title('Training Time per Epoch')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = save_path or os.path.join(self.output_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            print(f"[Visualizer] 训练曲线已保存: {save_path}")
            return save_path
            
        except ImportError:
            print("[Visualizer] matplotlib未安装，跳过绘图")
            return None
        except Exception as e:
            print(f"[Visualizer] 绘图失败: {e}")
            return None
    
    def generate_text_report(self, save_path: str = None) -> str:
        """生成文本报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("训练可视化报告")
        lines.append("=" * 60)
        lines.append(f"生成时间: {datetime.now().isoformat()}")
        lines.append("")
        
        if self.loss_history:
            lines.append("损失统计:")
            lines.append(f"  总步数: {len(self.loss_history)}")
            lines.append(f"  初始损失: {self.loss_history[0]:.4f}")
            lines.append(f"  最终损失: {self.loss_history[-1]:.4f}")
            lines.append(f"  最小损失: {min(self.loss_history):.4f}")
            lines.append(f"  最大损失: {max(self.loss_history):.4f}")
            lines.append(f"  平均损失: {sum(self.loss_history)/len(self.loss_history):.4f}")
            lines.append("")
        
        if self.lr_history:
            lines.append("学习率统计:")
            lines.append(f"  初始学习率: {self.lr_history[0]:.6f}")
            lines.append(f"  最终学习率: {self.lr_history[-1]:.6f}")
            lines.append(f"  最小学习率: {min(self.lr_history):.6f}")
            lines.append(f"  最大学习率: {max(self.lr_history):.6f}")
            lines.append("")
        
        if self.epoch_times:
            lines.append("时间统计:")
            lines.append(f"  Epoch数量: {len(self.epoch_times)}")
            lines.append(f"  总时间: {sum(self.epoch_times):.2f}秒")
            lines.append(f"  平均每epoch: {sum(self.epoch_times)/len(self.epoch_times):.2f}秒")
            lines.append(f"  最快epoch: {min(self.epoch_times):.2f}秒")
            lines.append(f"  最慢epoch: {max(self.epoch_times):.2f}秒")
            lines.append("")
        
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"[Visualizer] 文本报告已保存: {save_path}")
        
        return report
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'loss_history': self.loss_history,
            'lr_history': self.lr_history,
            'epoch_times': self.epoch_times,
            'timestamps': self.timestamps
        }
    
    def from_dict(self, data: Dict):
        """从字典恢复"""
        self.loss_history = data.get('loss_history', [])
        self.lr_history = data.get('lr_history', [])
        self.epoch_times = data.get('epoch_times', [])
        self.timestamps = data.get('timestamps', [])


class ModuleTrainer:
    """单个模块训练器"""
    
    def __init__(
        self,
        module_name: str,
        model_path: str,
        output_path: str,
        config: TrainingConfig = None
    ):
        self.module_name = module_name
        self.model_path = model_path
        self.output_path = output_path
        self.config = config or TrainingConfig()
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.is_training = False
        
        # 状态和回调
        self.state = TrainingState()
        self.callback_handler = CallbackHandler()
        self.visualizer = TrainingVisualizer(output_path)
        
        # 数据加载器
        self.data_loader = DataLoader()
        self.train_data: List[DataSample] = []
        self.eval_data: List[DataSample] = []
    
    def setup_callbacks(self, callbacks: List[Callback] = None):
        """设置回调"""
        if callbacks is None:
            callbacks = create_default_callbacks(
                checkpoint_dir=os.path.join(CHECKPOINT_PATH, self.module_name),
                log_dir=os.path.join(LOG_PATH, self.module_name),
                early_stopping_patience=self.config.early_stopping_patience,
                lr_schedule_type=self.config.lr_schedule_type,
                initial_lr=self.config.learning_rate
            )
        
        self.callback_handler = CallbackHandler(callbacks)
        self.callback_handler.set_trainer(self)
    
    def load_model(self) -> bool:
        """加载模型"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"[{self.module_name}] 加载模型: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            
            # 设置优化器
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay
            )
            
            print(f"[{self.module_name}] ✅ 模型加载完成")
            return True
            
        except Exception as e:
            print(f"[{self.module_name}] ❌ 加载失败: {e}")
            traceback.print_exc()
            return False
    
    def load_data(self, data_type: str = None, data_path: str = None):
        """加载训练数据"""
        data_type = data_type or self.module_name
        
        try:
            dataset = self.data_loader.load_builtin(data_type)
            train, val, test = dataset.split(
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1
            )
            
            self.train_data = train
            self.eval_data = val
            
            print(f"[{self.module_name}] 加载数据: 训练{len(train)}, 验证{len(val)}")
            
        except Exception as e:
            print(f"[{self.module_name}] 加载数据失败: {e}")
            # 使用默认数据
            self.train_data = [DataSample(text=f"训练样本 {i}", source="default") for i in range(10)]
            self.eval_data = [DataSample(text=f"验证样本 {i}", source="default") for i in range(3)]
    
    def prepare_batch(self, samples: List[DataSample]) -> Dict:
        """准备批次数据"""
        import torch
        
        texts = [s.text for s in samples]
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        
        return inputs
    
    def train_epoch(self, epoch: int) -> Dict:
        """训练一个epoch"""
        import torch
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        epoch_start = time.time()
        
        # 打乱数据
        import random
        random.shuffle(self.train_data)
        
        for i in range(0, len(self.train_data), self.config.batch_size):
            if not self.is_training:
                break
            
            batch_samples = self.train_data[i:i + self.config.batch_size]
            if not batch_samples:
                continue
            
            # 回调：batch开始
            self.callback_handler.on_batch_begin(self.state)
            
            # 准备输入
            inputs = self.prepare_batch(batch_samples)
            
            # 前向传播
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 更新状态
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1
            
            self.state.current_loss = loss_value
            self.state.global_step += 1
            self.state.batches_completed += 1
            
            # 记录可视化数据
            self.visualizer.record(loss_value, self.state.learning_rate)
            
            # 回调：batch结束
            self.callback_handler.on_batch_end(self.state)
            
            # 检查是否应该停止
            if self.callback_handler.should_stop():
                break
            
            if (i // self.config.batch_size + 1) % 5 == 0:
                avg_loss = total_loss / num_batches
                print(f"[{self.module_name}] Epoch {epoch}, Batch {num_batches}, Loss: {loss_value:.4f}, Avg: {avg_loss:.4f}")
        
        epoch_time = time.time() - epoch_start
        self.visualizer.record_epoch_time(epoch_time)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "batches": num_batches,
            "time": epoch_time
        }
    
    def evaluate(self) -> Dict:
        """评估模型"""
        import torch
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(self.eval_data), self.config.batch_size):
                batch_samples = self.eval_data[i:i + self.config.batch_size]
                if not batch_samples:
                    continue
                
                inputs = self.prepare_batch(batch_samples)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {"avg_loss": avg_loss, "batches": num_batches}
    
    def train(
        self,
        train_data: List[Dict] = None,
        eval_data: List[Dict] = None,
        callbacks: List[Callback] = None,
        resume_from: str = None
    ) -> Dict:
        """
        完整训练流程
        
        Args:
            train_data: 训练数据
            eval_data: 验证数据
            callbacks: 回调列表
            resume_from: 恢复训练的路径
        """
        # 加载模型
        if not self.load_model():
            return {"error": "模型加载失败"}
        
        # 加载数据
        if train_data:
            self.train_data = [DataSample(text=d.get('text', ''), metadata=d) for d in train_data]
        else:
            self.load_data()
        
        if eval_data:
            self.eval_data = [DataSample(text=d.get('text', ''), metadata=d) for d in eval_data]
        
        # 设置回调
        self.setup_callbacks(callbacks)
        
        # 恢复训练
        if resume_from:
            self._load_checkpoint(resume_from)
        
        self.is_training = True
        
        results = {
            "module": self.module_name,
            "start_time": datetime.now().isoformat(),
            "epochs": [],
            "final_loss": 0,
            "config": self.config.to_dict()
        }
        
        print(f"\n[{self.module_name}] 开始训练，共 {self.config.epochs} 个epoch")
        
        # 回调：训练开始
        self.callback_handler.on_train_begin(self.state)
        
        for epoch in range(1, self.config.epochs + 1):
            if not self.is_training:
                print(f"[{self.module_name}] 训练被中断")
                break
            
            self.state.epoch = epoch
            
            # 回调：epoch开始
            self.callback_handler.on_epoch_begin(self.state)
            
            # 训练一个epoch
            epoch_result = self.train_epoch(epoch)
            results["epochs"].append(epoch_result)
            results["final_loss"] = epoch_result["avg_loss"]
            
            self.state.current_loss = epoch_result["avg_loss"]
            self.state.epochs_completed += 1
            
            if epoch_result["avg_loss"] < self.state.best_loss:
                self.state.best_loss = epoch_result["avg_loss"]
            
            # 回调：epoch结束
            self.callback_handler.on_epoch_end(self.state)
            
            # 评估
            if self.eval_data and epoch % 2 == 0:
                eval_result = self.evaluate()
                print(f"[{self.module_name}] 评估 Loss: {eval_result['avg_loss']:.4f}")
                
                # 回调：评估
                self.callback_handler.on_evaluate(self.state, eval_result)
            
            # 检查是否应该停止
            if self.callback_handler.should_stop():
                break
        
        results["end_time"] = datetime.now().isoformat()
        self.is_training = False
        
        # 回调：训练结束
        self.callback_handler.on_train_end(self.state)
        
        # 生成可视化
        self.visualizer.plot_training_curves()
        self.visualizer.generate_text_report()
        
        return results
    
    def save_checkpoint(self, path: str = None):
        """保存检查点"""
        import torch
        
        path = path or os.path.join(CHECKPOINT_PATH, self.module_name, f"checkpoint_epoch{self.state.epoch}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_loss": self.state.best_loss,
            "current_loss": self.state.current_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "visualizer": self.visualizer.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        torch.save(checkpoint, path)
        print(f"[{self.module_name}] 检查点已保存: {path}")
    
    def _load_checkpoint(self, path: str) -> bool:
        """加载检查点"""
        import torch
        
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            self.state.epoch = checkpoint.get("epoch", 0)
            self.state.global_step = checkpoint.get("global_step", 0)
            self.state.best_loss = checkpoint.get("best_loss", float('inf'))
            self.state.current_loss = checkpoint.get("current_loss", 0)
            
            if "model_state_dict" in checkpoint and self.model:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            
            if "optimizer_state_dict" in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if "visualizer" in checkpoint:
                self.visualizer.from_dict(checkpoint["visualizer"])
            
            print(f"[{self.module_name}] 检查点已加载: {path}")
            print(f"[{self.module_name}] 恢复到 Epoch {self.state.epoch}, Step {self.state.global_step}")
            
            return True
            
        except Exception as e:
            print(f"[{self.module_name}] 加载检查点失败: {e}")
            return False
    
    def save_weights(self, path: str = None):
        """保存权重"""
        import torch
        
        path = path or os.path.join(self.output_path, f"{self.module_name}_weights.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "training_state": self.state.to_dict(),
            "visualizer": self.visualizer.to_dict()
        }, path)
        
        print(f"[{self.module_name}] 权重已保存: {path}")
    
    def stop(self):
        """停止训练"""
        self.is_training = False


class MemoryModuleTrainer(ModuleTrainer):
    """记忆模块训练器"""
    
    def __init__(self, model_path: str, output_path: str, config: TrainingConfig = None):
        super().__init__("memory", model_path, output_path, config)
        self.memory_patterns = []
    
    def load_data(self, data_type: str = None, data_path: str = None):
        """加载记忆训练数据"""
        super().load_data("memory", data_path)


class STDPModuleTrainer(ModuleTrainer):
    """STDP模块训练器"""
    
    def __init__(self, model_path: str, output_path: str, config: TrainingConfig = None):
        super().__init__("stdp", model_path, output_path, config)
        self.stdp_events = []
    
    def load_data(self, data_type: str = None, data_path: str = None):
        """加载STDP训练数据"""
        super().load_data("stdp", data_path)


class LanguageModuleTrainer(ModuleTrainer):
    """语言模块训练器"""
    
    def __init__(self, model_path: str, output_path: str, config: TrainingConfig = None):
        super().__init__("language", model_path, output_path, config)
    
    def load_data(self, data_type: str = None, data_path: str = None):
        """加载语言训练数据"""
        super().load_data("language", data_path)


class MultiThreadTrainer:
    """
    多线程综合训练器
    支持并行训练多个模块
    """
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        num_workers: int = None
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.num_workers = num_workers or min(4, multiprocessing.cpu_count())
        
        self.trainers: Dict[str, ModuleTrainer] = {}
        self.results: Dict[str, Dict] = {}
        self.is_training = False
        
        self._lock = threading.Lock()
    
    def setup_trainers(self, config: TrainingConfig = None):
        """设置各模块训练器"""
        config = config or TrainingConfig()
        
        self.trainers = {
            "memory": MemoryModuleTrainer(self.model_path, self.output_path, config),
            "stdp": STDPModuleTrainer(self.model_path, self.output_path, config),
            "language": LanguageModuleTrainer(self.model_path, self.output_path, config),
        }
    
    def train_module(self, module_name: str, epochs: int = 5, callbacks: List[Callback] = None) -> Dict:
        """训练单个模块"""
        trainer = self.trainers.get(module_name)
        if not trainer:
            return {"error": f"Unknown module: {module_name}"}
        
        trainer.config.epochs = epochs
        return trainer.train(callbacks=callbacks)
    
    def train_all_parallel(self, epochs: int = 5) -> Dict:
        """并行训练所有模块"""
        self.is_training = True
        self.setup_trainers()
        
        print(f"\n{'='*60}")
        print(f"开始多线程综合训练 (workers: {self.num_workers})")
        print(f"{'='*60}\n")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "modules": {},
            "parallel": True,
            "num_workers": self.num_workers
        }
        
        def train_worker(module_name: str) -> Tuple[str, Dict]:
            try:
                result = self.train_module(module_name, epochs)
                return module_name, result
            except Exception as e:
                return module_name, {"error": str(e), "traceback": traceback.format_exc()}
        
        # 使用线程池并行训练
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(train_worker, name): name
                for name in self.trainers.keys()
            }
            
            for future in as_completed(futures):
                module_name, result = future.result()
                results["modules"][module_name] = result
                print(f"\n✅ {module_name} 训练完成")
        
        results["end_time"] = datetime.now().isoformat()
        self.is_training = False
        
        return results
    
    def train_all_sequential(self, epochs: int = 5) -> Dict:
        """顺序训练所有模块"""
        self.is_training = True
        self.setup_trainers()
        
        print(f"\n{'='*60}")
        print("开始顺序综合训练")
        print(f"{'='*60}\n")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "modules": {},
            "parallel": False
        }
        
        for name in self.trainers.keys():
            if not self.is_training:
                break
            
            print(f"\n{'='*40}")
            print(f"训练模块: {name}")
            print(f"{'='*40}")
            
            result = self.train_module(name, epochs)
            results["modules"][name] = result
        
        results["end_time"] = datetime.now().isoformat()
        self.is_training = False
        
        return results
    
    def save_all_weights(self):
        """保存所有模块权重"""
        for name, trainer in self.trainers.items():
            trainer.save_weights()
    
    def stop(self):
        """停止所有训练"""
        self.is_training = False
        for trainer in self.trainers.values():
            trainer.stop()


class OfflineTrainer:
    """
    离线训练主类
    整合所有训练功能
    """
    
    def __init__(
        self,
        model_path: str = None,
        output_path: str = None,
        learning_rate: float = 0.01,
        epochs: int = 10
    ):
        self.model_path = model_path or QWEN_MODEL_PATH
        self.output_path = output_path or OUTPUT_PATH
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.config = TrainingConfig(
            learning_rate=learning_rate,
            epochs=epochs
        )
        
        self.multi_trainer = MultiThreadTrainer(
            self.model_path,
            self.output_path
        )
        
        # 确保目录存在
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(LOG_PATH, exist_ok=True)
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
    def train_module(self, module_name: str, epochs: int = None, callbacks: List[Callback] = None) -> Dict:
        """
        训练单个模块
        
        Args:
            module_name: 模块名称 (memory, stdp, language)
            epochs: 训练轮数
            callbacks: 回调列表
        """
        epochs = epochs or self.epochs
        self.multi_trainer.setup_trainers(self.config)
        return self.multi_trainer.train_module(module_name, epochs, callbacks)
    
    def train_all(self, parallel: bool = True, epochs: int = None) -> Dict:
        """
        综合训练所有模块
        
        Args:
            parallel: 是否并行训练
            epochs: 训练轮数
        """
        epochs = epochs or self.epochs
        
        if parallel:
            return self.multi_trainer.train_all_parallel(epochs)
        else:
            return self.multi_trainer.train_all_sequential(epochs)
    
    def save_weights(self):
        """保存权重"""
        self.multi_trainer.save_all_weights()
    
    def export_training_report(self, results: Dict, path: str = None):
        """导出训练报告"""
        path = path or os.path.join(self.output_path, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"训练报告已保存: {path}")
        return path
    
    def resume_training(self, checkpoint_path: str, module_name: str = None) -> Dict:
        """
        从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点路径
            module_name: 模块名称（如果指定则只恢复该模块）
        """
        if module_name:
            trainer = self._create_trainer(module_name)
            return trainer.train(resume_from=checkpoint_path)
        else:
            # 尝试从目录恢复所有模块
            results = {}
            for name in ["memory", "stdp", "language"]:
                checkpoint_file = os.path.join(checkpoint_path, name, "checkpoint_latest.pt")
                if os.path.exists(checkpoint_file):
                    trainer = self._create_trainer(name)
                    results[name] = trainer.train(resume_from=checkpoint_file)
            return results
    
    def _create_trainer(self, module_name: str) -> ModuleTrainer:
        """创建模块训练器"""
        trainer_classes = {
            "memory": MemoryModuleTrainer,
            "stdp": STDPModuleTrainer,
            "language": LanguageModuleTrainer,
        }
        
        trainer_class = trainer_classes.get(module_name, ModuleTrainer)
        return trainer_class(self.model_path, self.output_path, self.config)


def main():
    """主函数 - 演示离线训练"""
    print("\n" + "=" * 60)
    print("类脑AI系统 - 离线训练模块 v2.0")
    print("=" * 60)
    
    trainer = OfflineTrainer(
        learning_rate=0.01,
        epochs=3  # 演示用，减少epoch数
    )
    
    # 选择训练模式
    print("\n训练模式:")
    print("  1. 单独训练记忆模块")
    print("  2. 单独训练STDP模块")
    print("  3. 单独训练语言模块")
    print("  4. 并行综合训练")
    print("  5. 顺序综合训练")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        print("\n训练记忆模块...")
        result = trainer.train_module("memory", epochs=3)
    elif choice == "2":
        print("\n训练STDP模块...")
        result = trainer.train_module("stdp", epochs=3)
    elif choice == "3":
        print("\n训练语言模块...")
        result = trainer.train_module("language", epochs=3)
    elif choice == "4":
        print("\n并行综合训练...")
        result = trainer.train_all(parallel=True, epochs=2)
    elif choice == "5":
        print("\n顺序综合训练...")
        result = trainer.train_all(parallel=False, epochs=2)
    else:
        print("无效选择，执行顺序综合训练...")
        result = trainer.train_all(parallel=False, epochs=2)
    
    # 保存结果
    trainer.export_training_report(result)
    trainer.save_weights()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
