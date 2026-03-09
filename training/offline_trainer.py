#!/usr/bin/env python3
"""
离线训练模块 - 完整版
Offline Training Module - Complete Version

支持：
1. 各模块单独训练
2. 多线程综合训练
3. 记忆模块训练
4. STDP模块训练
5. 权重导出和加载
"""

import os
import sys
import json
import time
import threading
import multiprocessing
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_MODEL_PATH = os.path.join(BASE_DIR, "models/Qwen3.5-0.8B")
OUTPUT_PATH = os.path.join(BASE_DIR, "weights")
DATA_PATH = os.path.join(BASE_DIR, "training_data")


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
        self.training_history = []
        self.is_training = False
        
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
            return False
    
    def train_epoch(
        self,
        train_data: List[Dict],
        epoch: int
    ) -> Dict:
        """训练一个epoch"""
        import torch
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for i, batch in enumerate(train_data):
            # 准备输入
            text = batch.get("text", "")
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            )
            
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
            
            total_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % 10 == 0:
                print(f"[{self.module_name}] Epoch {epoch}, Batch {i+1}/{len(train_data)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {"epoch": epoch, "avg_loss": avg_loss, "batches": num_batches}
    
    def train(
        self,
        train_data: List[Dict],
        eval_data: List[Dict] = None,
        progress_callback: Callable = None
    ) -> Dict:
        """完整训练流程"""
        if not self.load_model():
            return {"error": "模型加载失败"}
        
        self.is_training = True
        results = {
            "module": self.module_name,
            "start_time": datetime.now().isoformat(),
            "epochs": [],
            "final_loss": 0
        }
        
        print(f"\n[{self.module_name}] 开始训练，共 {self.config.epochs} 个epoch")
        
        for epoch in range(1, self.config.epochs + 1):
            if not self.is_training:
                print(f"[{self.module_name}] 训练被中断")
                break
            
            epoch_result = self.train_epoch(train_data, epoch)
            results["epochs"].append(epoch_result)
            results["final_loss"] = epoch_result["avg_loss"]
            
            if progress_callback:
                progress_callback(epoch, self.config.epochs, epoch_result)
            
            # 评估
            if eval_data and epoch % 2 == 0:
                eval_result = self.evaluate(eval_data)
                print(f"[{self.module_name}] 评估 Loss: {eval_result['avg_loss']:.4f}")
        
        results["end_time"] = datetime.now().isoformat()
        self.is_training = False
        
        return results
    
    def evaluate(self, eval_data: List[Dict]) -> Dict:
        """评估模型"""
        import torch
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_data:
                text = batch.get("text", "")
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True
                )
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return {"avg_loss": total_loss / num_batches if num_batches > 0 else 0}
    
    def save_weights(self, path: str = None):
        """保存权重"""
        import torch
        
        path = path or os.path.join(self.output_path, f"{self.module_name}_weights.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "config": self.config.__dict__
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
    
    def generate_memory_training_data(self) -> List[Dict]:
        """生成记忆训练数据"""
        training_data = [
            {"text": "记忆存储：用户询问了关于人工智能的问题，回答是人工智能是模拟人类智能的技术。"},
            {"text": "记忆检索：搜索关键词'机器学习'，找到3条相关记忆。"},
            {"text": "记忆巩固：将短期记忆转移到长期记忆，重要性分数0.8。"},
            {"text": "记忆关联：建立'神经网络'与'深度学习'之间的关联。"},
            {"text": "记忆遗忘：移除重要性低于0.3的记忆条目。"},
        ]
        return training_data
    
    def train(self, epochs: int = 5) -> Dict:
        """训练记忆模块"""
        train_data = self.generate_memory_training_data()
        self.config.epochs = epochs
        return super().train(train_data)


class STDPModuleTrainer(ModuleTrainer):
    """STDP模块训练器"""
    
    def __init__(self, model_path: str, output_path: str, config: TrainingConfig = None):
        super().__init__("stdp", model_path, output_path, config)
        self.stdp_events = []
    
    def generate_stdp_training_data(self) -> List[Dict]:
        """生成STDP训练数据"""
        training_data = [
            {"text": "STDP学习：pre-spike先于post-spike 10ms，产生LTP，权重增加0.01。"},
            {"text": "STDP学习：post-spike先于pre-spike 5ms，产生LTD，权重减少0.005。"},
            {"text": "STDP窗口：时间差在20ms窗口内，应用学习规则。"},
            {"text": "权重更新：layer_0权重矩阵更新，追踪值衰减0.9。"},
            {"text": "脉冲记录：记录神经元激活时间戳，用于STDP计算。"},
        ]
        return training_data
    
    def train(self, epochs: int = 5) -> Dict:
        """训练STDP模块"""
        train_data = self.generate_stdp_training_data()
        self.config.epochs = epochs
        return super().train(train_data)


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
        
    def setup_trainers(self):
        """设置各模块训练器"""
        config = TrainingConfig()
        
        self.trainers = {
            "memory": MemoryModuleTrainer(self.model_path, self.output_path, config),
            "stdp": STDPModuleTrainer(self.model_path, self.output_path, config),
            "language": ModuleTrainer("language", self.model_path, self.output_path, config),
            "reasoning": ModuleTrainer("reasoning", self.model_path, self.output_path, config)
        }
    
    def train_module(self, module_name: str, epochs: int = 5) -> Dict:
        """训练单个模块"""
        trainer = self.trainers.get(module_name)
        if not trainer:
            return {"error": f"Unknown module: {module_name}"}
        
        return trainer.train(epochs=epochs)
    
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
            "parallel": True
        }
        
        def train_worker(module_name: str):
            try:
                result = self.train_module(module_name, epochs)
                return module_name, result
            except Exception as e:
                return module_name, {"error": str(e)}
        
        # 使用线程池并行训练
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(train_worker, name): name
                for name in self.trainers.keys()
            }
            
            for future in futures:
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
        
        os.makedirs(self.output_path, exist_ok=True)
    
    def train_module(self, module_name: str, epochs: int = None) -> Dict:
        """
        训练单个模块
        
        Args:
            module_name: 模块名称 (memory, stdp, language, reasoning)
            epochs: 训练轮数
        """
        epochs = epochs or self.epochs
        self.multi_trainer.setup_trainers()
        return self.multi_trainer.train_module(module_name, epochs)
    
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
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"训练报告已保存: {path}")


def main():
    """主函数 - 演示离线训练"""
    print("\n" + "=" * 60)
    print("类脑AI系统 - 离线训练模块")
    print("=" * 60)
    
    trainer = OfflineTrainer(
        learning_rate=0.01,
        epochs=3  # 演示用，减少epoch数
    )
    
    # 选择训练模式
    print("\n训练模式:")
    print("  1. 单独训练记忆模块")
    print("  2. 单独训练STDP模块")
    print("  3. 并行综合训练")
    print("  4. 顺序综合训练")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == "1":
        print("\n训练记忆模块...")
        result = trainer.train_module("memory", epochs=3)
    elif choice == "2":
        print("\n训练STDP模块...")
        result = trainer.train_module("stdp", epochs=3)
    elif choice == "3":
        print("\n并行综合训练...")
        result = trainer.train_all(parallel=True, epochs=2)
    elif choice == "4":
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
