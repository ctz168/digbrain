"""
类脑AI核心模块
"""

from .streaming_engine import BrainLikeStreamingEngine as BrainLikeAI, StreamChunk
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
