"""
类脑AI系统 - 核心模块
Brain-like AI System - Core Module
"""

from .streaming_engine import BrainLikeAI, StreamChunk
from .stdp_learning import STDPOnlineLearning
from .memory_system import HippocampalMemory
from .multimodal import MultimodalProcessor

__version__ = "1.0.0"
__all__ = [
    "BrainLikeAI",
    "StreamChunk", 
    "STDPOnlineLearning",
    "HippocampalMemory",
    "MultimodalProcessor"
]
