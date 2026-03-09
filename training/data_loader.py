#!/usr/bin/env python3
"""
训练数据加载器
Training Data Loader

支持：
1. 多种数据格式（JSON, JSONL, CSV, TXT）
2. 自定义数据集
3. 数据预处理和增强
4. 批量加载和迭代
"""

import os
import json
import csv
import random
import re
from typing import Dict, List, Any, Optional, Callable, Iterator, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading


@dataclass
class DataSample:
    """数据样本"""
    text: str
    metadata: Dict = field(default_factory=dict)
    source: str = ""
    label: Optional[str] = None
    importance: float = 0.5


@dataclass
class DataLoaderConfig:
    """数据加载器配置"""
    batch_size: int = 4
    shuffle: bool = True
    max_length: int = 512
    min_length: int = 10
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


class BaseDataset(ABC):
    """数据集基类"""
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.samples: List[DataSample] = []
        self._lock = threading.Lock()
    
    @abstractmethod
    def load(self, path: str) -> int:
        """加载数据"""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[DataSample]:
        for sample in self.samples:
            yield sample
    
    def get_batch(self, batch_size: int) -> List[DataSample]:
        """获取一批数据"""
        with self._lock:
            return random.sample(self.samples, min(batch_size, len(self.samples)))
    
    def shuffle(self, seed: int = None):
        """打乱数据"""
        with self._lock:
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.samples)
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> tuple:
        """划分数据集"""
        random.seed(seed)
        samples_copy = self.samples.copy()
        random.shuffle(samples_copy)
        
        n = len(samples_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_set = samples_copy[:train_end]
        val_set = samples_copy[train_end:val_end]
        test_set = samples_copy[val_end:]
        
        return train_set, val_set, test_set


class JSONDataset(BaseDataset):
    """JSON格式数据集"""
    
    def __init__(self):
        super().__init__("json")
    
    def load(self, path: str) -> int:
        """加载JSON数据"""
        count = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    sample = self._parse_item(item, path)
                    if sample:
                        self.samples.append(sample)
                        count += 1
            elif isinstance(data, dict):
                sample = self._parse_item(data, path)
                if sample:
                    self.samples.append(sample)
                    count += 1
            
            print(f"[JSONDataset] 加载了 {count} 个样本 from {path}")
            return count
            
        except Exception as e:
            print(f"[JSONDataset] 加载失败: {e}")
            return 0
    
    def _parse_item(self, item: Dict, source: str) -> Optional[DataSample]:
        """解析数据项"""
        # 支持多种字段名
        text = item.get('text') or item.get('content') or item.get('input') or item.get('prompt', '')
        if not text:
            return None
        
        return DataSample(
            text=text,
            metadata=item.get('metadata', {}),
            source=source,
            label=item.get('label') or item.get('output'),
            importance=item.get('importance', 0.5)
        )


class JSONLDataset(BaseDataset):
    """JSONL格式数据集"""
    
    def __init__(self):
        super().__init__("jsonl")
    
    def load(self, path: str) -> int:
        """加载JSONL数据（每行一个JSON对象）"""
        count = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        sample = self._parse_item(item, path)
                        if sample:
                            self.samples.append(sample)
                            count += 1
                    except json.JSONDecodeError:
                        continue
            
            print(f"[JSONLDataset] 加载了 {count} 个样本 from {path}")
            return count
            
        except Exception as e:
            print(f"[JSONLDataset] 加载失败: {e}")
            return 0
    
    def _parse_item(self, item: Dict, source: str) -> Optional[DataSample]:
        """解析数据项"""
        text = item.get('text') or item.get('content') or item.get('input', '')
        if not text:
            return None
        
        return DataSample(
            text=text,
            metadata=item.get('metadata', {}),
            source=source,
            label=item.get('label') or item.get('output'),
            importance=item.get('importance', 0.5)
        )


class CSVDataset(BaseDataset):
    """CSV格式数据集"""
    
    def __init__(self, text_column: str = 'text'):
        super().__init__("csv")
        self.text_column = text_column
    
    def load(self, path: str) -> int:
        """加载CSV数据"""
        count = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get(self.text_column, '')
                    if not text:
                        # 尝试其他常见列名
                        for col in ['content', 'input', 'prompt', 'text', 'sentence']:
                            if col in row:
                                text = row[col]
                                break
                    
                    if text:
                        self.samples.append(DataSample(
                            text=text,
                            metadata=dict(row),
                            source=path,
                            label=row.get('label') or row.get('output'),
                            importance=0.5
                        ))
                        count += 1
            
            print(f"[CSVDataset] 加载了 {count} 个样本 from {path}")
            return count
            
        except Exception as e:
            print(f"[CSVDataset] 加载失败: {e}")
            return 0


class TextDataset(BaseDataset):
    """纯文本数据集"""
    
    def __init__(self, split_by: str = "paragraph"):
        super().__init__("text")
        self.split_by = split_by
    
    def load(self, path: str) -> int:
        """加载纯文本数据"""
        count = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按指定方式分割
            if self.split_by == "paragraph":
                chunks = re.split(r'\n\s*\n', content)
            elif self.split_by == "line":
                chunks = content.split('\n')
            elif self.split_by == "sentence":
                chunks = re.split(r'[.!?。！？]', content)
            else:
                chunks = [content]
            
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) > 10:  # 过滤太短的片段
                    self.samples.append(DataSample(
                        text=chunk,
                        source=path
                    ))
                    count += 1
            
            print(f"[TextDataset] 加载了 {count} 个样本 from {path}")
            return count
            
        except Exception as e:
            print(f"[TextDataset] 加载失败: {e}")
            return 0


class MemoryTrainingDataset(BaseDataset):
    """记忆模块训练数据集"""
    
    def __init__(self):
        super().__init__("memory")
    
    def load(self, path: str = None) -> int:
        """加载记忆训练数据"""
        # 生成默认的记忆训练数据
        default_data = [
            {"text": "记忆存储：用户询问了关于人工智能的问题，回答是人工智能是模拟人类智能的技术。", "importance": 0.8},
            {"text": "记忆检索：搜索关键词'机器学习'，找到3条相关记忆，按相关性排序返回。", "importance": 0.7},
            {"text": "记忆巩固：将短期记忆转移到长期记忆，重要性分数超过阈值0.6。", "importance": 0.8},
            {"text": "记忆关联：建立'神经网络'与'深度学习'之间的关联，关联强度0.85。", "importance": 0.6},
            {"text": "记忆遗忘：移除重要性低于0.3的记忆条目，释放存储空间。", "importance": 0.5},
            {"text": "记忆检索成功：用户询问'什么是量子计算'，从长期记忆中找到相关条目。", "importance": 0.9},
            {"text": "记忆更新：更新现有记忆条目的访问计数和时间戳，增加记忆强度。", "importance": 0.7},
            {"text": "记忆压缩：将相似的记忆条目合并，保留最完整的信息。", "importance": 0.6},
            {"text": "记忆索引：为新记忆建立关键词索引，提高检索效率。", "importance": 0.5},
            {"text": "记忆权重调整：根据反馈调整记忆的重要性分数，优化检索结果。", "importance": 0.7},
            {"text": "短期记忆存储：临时存储当前对话信息，保持时间约30秒。", "importance": 0.6},
            {"text": "工作记忆激活：加载相关记忆到工作记忆区，准备用于当前推理。", "importance": 0.8},
            {"text": "记忆验证：验证记忆内容的准确性和时效性，标记过时信息。", "importance": 0.7},
            {"text": "记忆关联网络：构建记忆之间的关联图，支持关联检索。", "importance": 0.6},
            {"text": "记忆优先级排序：根据多个因素对记忆进行优先级排序。", "importance": 0.5},
        ]
        
        count = 0
        
        # 加载默认数据
        for item in default_data:
            self.samples.append(DataSample(
                text=item["text"],
                metadata={"type": "memory_operation"},
                source="generated",
                importance=item["importance"]
            ))
            count += 1
        
        # 如果指定了路径，加载额外数据
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    text = item.get('text', '')
                    if text:
                        self.samples.append(DataSample(
                            text=text,
                            metadata=item.get('metadata', {}),
                            source=path,
                            importance=item.get('importance', 0.5)
                        ))
                        count += 1
            except Exception as e:
                print(f"[MemoryDataset] 加载外部数据失败: {e}")
        
        print(f"[MemoryDataset] 总共加载了 {count} 个样本")
        return count


class STDPTrainingDataset(BaseDataset):
    """STDP模块训练数据集"""
    
    def __init__(self):
        super().__init__("stdp")
    
    def load(self, path: str = None) -> int:
        """加载STDP训练数据"""
        default_data = [
            {"text": "STDP学习：pre-spike先于post-spike 10ms，产生LTP，权重增加0.01。", "importance": 0.8},
            {"text": "STDP学习：post-spike先于pre-spike 5ms，产生LTD，权重减少0.005。", "importance": 0.8},
            {"text": "STDP窗口：时间差在20ms窗口内，应用学习规则进行权重更新。", "importance": 0.7},
            {"text": "权重更新：layer_0权重矩阵更新，追踪值衰减系数0.9。", "importance": 0.6},
            {"text": "脉冲记录：记录神经元激活时间戳，用于STDP时间窗口计算。", "importance": 0.5},
            {"text": "LTP增强：高频刺激导致突触强度增加，权重变化为正。", "importance": 0.9},
            {"text": "LTD抑制：低频刺激或逆序激活导致突触强度减弱。", "importance": 0.9},
            {"text": "学习率调节：根据奖励信号动态调整学习率。", "importance": 0.7},
            {"text": "权重裁剪：将权重限制在合理范围内，防止过大或过小。", "importance": 0.6},
            {"text": "资格迹更新：衰减旧的资格迹，添加新的激活记录。", "importance": 0.5},
            {"text": "奖励调制：基于任务奖励调整STDP学习强度。", "importance": 0.8},
            {"text": "突触可塑性：实现Hebbian学习规则，加强同时激活的连接。", "importance": 0.9},
            {"text": "时间窗口函数：指数衰减函数决定权重更新幅度。", "importance": 0.7},
            {"text": "多巴胺调制：奖励预测误差影响学习率变化。", "importance": 0.6},
            {"text": "突触竞争：有限资源导致突触之间的竞争性增长。", "importance": 0.5},
        ]
        
        count = 0
        
        for item in default_data:
            self.samples.append(DataSample(
                text=item["text"],
                metadata={"type": "stdp_operation"},
                source="generated",
                importance=item["importance"]
            ))
            count += 1
        
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    text = item.get('text', '')
                    if text:
                        self.samples.append(DataSample(
                            text=text,
                            metadata=item.get('metadata', {}),
                            source=path,
                            importance=item.get('importance', 0.5)
                        ))
                        count += 1
            except Exception as e:
                print(f"[STDPAataset] 加载外部数据失败: {e}")
        
        print(f"[STDPAataset] 总共加载了 {count} 个样本")
        return count


class LanguageTrainingDataset(BaseDataset):
    """语言模块训练数据集"""
    
    def __init__(self):
        super().__init__("language")
    
    def load(self, path: str = None) -> int:
        """加载语言训练数据"""
        default_data = [
            {"text": "问题：什么是人工智能？回答：人工智能是模拟人类智能的技术领域。", "importance": 0.8},
            {"text": "问题：解释机器学习。回答：机器学习是让计算机从数据中学习的技术。", "importance": 0.7},
            {"text": "问题：什么是深度学习？回答：深度学习是使用神经网络的机器学习方法。", "importance": 0.8},
            {"text": "任务：将以下句子翻译成英文。今天天气很好。答案：The weather is nice today.", "importance": 0.6},
            {"text": "任务：总结以下文章。人工智能正在改变世界...总结：AI技术推动社会变革。", "importance": 0.7},
            {"text": "问题：TCP三次握手是什么？回答：TCP建立连接需要三次通信确认。", "importance": 0.8},
            {"text": "问题：什么是量子纠缠？回答：量子纠缠是量子力学中的特殊关联现象。", "importance": 0.9},
            {"text": "任务：完成以下句子。人工智能的发展将...答案：推动科技和社会进步。", "importance": 0.6},
            {"text": "问题：解释区块链。回答：区块链是分布式账本技术，具有去中心化特点。", "importance": 0.7},
            {"text": "任务：改写以下句子。机器学习很有用。改写：机器学习技术具有广泛应用价值。", "importance": 0.5},
            {"text": "问题：什么是神经网络？回答：神经网络是模拟人脑结构的计算模型。", "importance": 0.8},
            {"text": "任务：生成标题。文章：人工智能技术在医疗领域的应用...标题：AI医疗应用前景广阔。", "importance": 0.6},
            {"text": "问题：解释云计算。回答：云计算是通过网络提供计算资源的服务模式。", "importance": 0.7},
            {"text": "任务：回答问题。为什么要学习编程？回答：编程是解决问题和创造价值的重要技能。", "importance": 0.6},
            {"text": "问题：什么是大数据？回答：大数据是指无法用传统方法处理的海量数据集。", "importance": 0.7},
        ]
        
        count = 0
        
        for item in default_data:
            self.samples.append(DataSample(
                text=item["text"],
                metadata={"type": "language_task"},
                source="generated",
                importance=item["importance"]
            ))
            count += 1
        
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    text = item.get('text', '')
                    if text:
                        self.samples.append(DataSample(
                            text=text,
                            metadata=item.get('metadata', {}),
                            source=path,
                            importance=item.get('importance', 0.5)
                        ))
                        count += 1
            except Exception as e:
                print(f"[LanguageDataset] 加载外部数据失败: {e}")
        
        print(f"[LanguageDataset] 总共加载了 {count} 个样本")
        return count


class DataAugmentor:
    """数据增强器"""
    
    def __init__(self):
        self.augmentations: List[Callable] = []
    
    def add_augmentation(self, func: Callable):
        """添加增强函数"""
        self.augmentations.append(func)
    
    def augment(self, sample: DataSample) -> List[DataSample]:
        """对样本进行增强"""
        results = [sample]
        
        for aug_func in self.augmentations:
            try:
                augmented = aug_func(sample)
                if isinstance(augmented, list):
                    results.extend(augmented)
                elif isinstance(augmented, DataSample):
                    results.append(augmented)
            except Exception as e:
                print(f"[Augmentor] 增强失败: {e}")
        
        return results
    
    @staticmethod
    def random_delete(sample: DataSample, delete_prob: float = 0.1) -> DataSample:
        """随机删除词语"""
        words = sample.text.split()
        new_words = [w for w in words if random.random() > delete_prob]
        if not new_words:
            new_words = words
        
        return DataSample(
            text=' '.join(new_words),
            metadata={**sample.metadata, 'augmentation': 'random_delete'},
            source=sample.source,
            importance=sample.importance
        )
    
    @staticmethod
    def random_swap(sample: DataSample, swap_prob: float = 0.1) -> DataSample:
        """随机交换词语"""
        words = sample.text.split()
        if len(words) < 2:
            return sample
        
        for _ in range(int(len(words) * swap_prob)):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        
        return DataSample(
            text=' '.join(words),
            metadata={**sample.metadata, 'augmentation': 'random_swap'},
            source=sample.source,
            importance=sample.importance
        )


class DataLoader:
    """
    统一数据加载器
    
    支持：
    1. 多种数据格式
    2. 批量加载
    3. 数据增强
    4. 迭代器接口
    """
    
    def __init__(self, config: DataLoaderConfig = None):
        self.config = config or DataLoaderConfig()
        self.datasets: Dict[str, BaseDataset] = {}
        self.augmentor = DataAugmentor()
        self._lock = threading.Lock()
        
        # 注册数据集类型
        self._dataset_types = {
            'json': JSONDataset,
            'jsonl': JSONLDataset,
            'csv': CSVDataset,
            'txt': TextDataset,
            'text': TextDataset,
            'memory': MemoryTrainingDataset,
            'stdp': STDPTrainingDataset,
            'language': LanguageTrainingDataset,
        }
    
    def register_dataset_type(self, name: str, dataset_class: type):
        """注册自定义数据集类型"""
        self._dataset_types[name.lower()] = dataset_class
    
    def load(
        self,
        path: str,
        dataset_type: str = None,
        name: str = None
    ) -> BaseDataset:
        """
        加载数据集
        
        Args:
            path: 数据文件路径
            dataset_type: 数据集类型（自动检测如果未指定）
            name: 数据集名称
            
        Returns:
            加载的数据集
        """
        # 自动检测类型
        if dataset_type is None:
            dataset_type = self._detect_type(path)
        
        # 创建数据集
        dataset_class = self._dataset_types.get(dataset_type.lower())
        if dataset_class is None:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        dataset = dataset_class()
        dataset.load(path)
        
        # 存储
        name = name or os.path.basename(path)
        self.datasets[name] = dataset
        
        return dataset
    
    def load_builtin(self, dataset_type: str, name: str = None) -> BaseDataset:
        """
        加载内置数据集
        
        Args:
            dataset_type: 数据集类型（memory, stdp, language）
            name: 数据集名称
            
        Returns:
            加载的数据集
        """
        if dataset_type.lower() not in ['memory', 'stdp', 'language']:
            raise ValueError(f"Unknown builtin dataset: {dataset_type}")
        
        dataset_class = self._dataset_types[dataset_type.lower()]
        dataset = dataset_class()
        dataset.load()
        
        name = name or dataset_type
        self.datasets[name] = dataset
        
        return dataset
    
    def _detect_type(self, path: str) -> str:
        """自动检测文件类型"""
        ext = os.path.splitext(path)[1].lower()
        
        type_map = {
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.csv': 'csv',
            '.txt': 'text',
        }
        
        return type_map.get(ext, 'text')
    
    def get_dataset(self, name: str) -> Optional[BaseDataset]:
        """获取数据集"""
        return self.datasets.get(name)
    
    def merge_datasets(self, names: List[str], new_name: str) -> BaseDataset:
        """合并多个数据集"""
        merged = BaseDataset("merged")
        merged.samples = []
        
        for name in names:
            dataset = self.datasets.get(name)
            if dataset:
                merged.samples.extend(dataset.samples)
        
        self.datasets[new_name] = merged
        return merged
    
    def create_batch_iterator(
        self,
        dataset_name: str,
        batch_size: int = None,
        shuffle: bool = None
    ) -> Iterator[List[DataSample]]:
        """
        创建批量迭代器
        
        Args:
            dataset_name: 数据集名称
            batch_size: 批量大小
            shuffle: 是否打乱
            
        Yields:
            批量数据
        """
        dataset = self.datasets.get(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        batch_size = batch_size or self.config.batch_size
        shuffle = shuffle if shuffle is not None else self.config.shuffle
        
        if shuffle:
            dataset.shuffle(self.config.seed)
        
        samples = dataset.samples.copy()
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            yield batch
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_datasets': len(self.datasets),
            'datasets': {}
        }
        
        for name, dataset in self.datasets.items():
            stats['datasets'][name] = {
                'name': dataset.name,
                'samples': len(dataset)
            }
        
        return stats


def create_training_data(
    output_path: str,
    dataset_type: str = 'memory',
    num_samples: int = 100
) -> str:
    """
    创建训练数据文件
    
    Args:
        output_path: 输出路径
        dataset_type: 数据集类型
        num_samples: 样本数量
        
    Returns:
        输出文件路径
    """
    loader = DataLoader()
    dataset = loader.load_builtin(dataset_type)
    
    # 确保有足够的样本
    while len(dataset.samples) < num_samples:
        # 复制并修改现有样本
        for sample in dataset.samples[:min(10, len(dataset.samples))]:
            new_sample = DataSample(
                text=sample.text + f" [变体{len(dataset.samples)}]",
                metadata=sample.metadata,
                source="augmented",
                importance=sample.importance
            )
            dataset.samples.append(new_sample)
            if len(dataset.samples) >= num_samples:
                break
    
    # 保存到文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = [
        {
            'text': s.text,
            'metadata': s.metadata,
            'importance': s.importance
        }
        for s in dataset.samples[:num_samples]
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"创建了 {len(data)} 个训练样本: {output_path}")
    return output_path


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("训练数据加载器测试")
    print("=" * 60)
    
    # 测试内置数据集
    loader = DataLoader()
    
    print("\n[1] 测试记忆训练数据集...")
    memory_dataset = loader.load_builtin('memory')
    print(f"   样本数: {len(memory_dataset)}")
    if memory_dataset.samples:
        print(f"   示例: {memory_dataset.samples[0].text[:50]}...")
    
    print("\n[2] 测试STDP训练数据集...")
    stdp_dataset = loader.load_builtin('stdp')
    print(f"   样本数: {len(stdp_dataset)}")
    
    print("\n[3] 测试批量迭代器...")
    batch_iter = loader.create_batch_iterator('memory', batch_size=3)
    for i, batch in enumerate(batch_iter):
        print(f"   Batch {i+1}: {len(batch)} 样本")
        if i >= 2:
            break
    
    print("\n[4] 数据集划分测试...")
    train, val, test = memory_dataset.split()
    print(f"   训练集: {len(train)}")
    print(f"   验证集: {len(val)}")
    print(f"   测试集: {len(test)}")
    
    print("\n[5] 统计信息...")
    stats = loader.get_stats()
    print(f"   {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
