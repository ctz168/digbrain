"""
多模态处理模块
Multimodal Processing Module

支持文本、图像、视频的统一处理
支持多种模型配置（Qwen2.5-0.5B / Qwen3.5-0.8B）
"""

import os
import time
import base64
from typing import Generator, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from io import BytesIO

# 导入模型配置
try:
    from .brain_engine import WORLD_MODEL_CONFIG, MODELS_DIR
    DEFAULT_MODEL_PATH = WORLD_MODEL_CONFIG.local_path
except ImportError:
    # 回退配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "WorldModel")


@dataclass
class MultimodalInput:
    """多模态输入"""
    input_type: str  # 'text', 'image', 'video', 'audio'
    content: Any
    metadata: Dict = None


@dataclass
class MultimodalOutput:
    """多模态输出"""
    output_type: str
    content: Any
    confidence: float = 0.0
    metadata: Dict = None


class MultimodalProcessor:
    """
    多模态处理器
    
    支持：
    1. 文本处理
    2. 图像理解（需要世界模型）
    3. 视频流处理（逐帧）
    4. 音频处理
    """
    
    def __init__(self, model_path: str = None):
        """
        初始化多模态处理器
        
        Args:
            model_path: 模型路径（默认使用配置中的世界模型路径）
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = None
        self.processor = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """初始化模型"""
        try:
            # 尝试加载Qwen-VL模型
            print("初始化多模态处理器...")
            print(f"模型路径: {self.model_path}")
            
            # 检查是否有模型
            if self.model_path and os.path.exists(self.model_path):
                # 检查关键文件
                config_path = os.path.join(self.model_path, "config.json")
                if os.path.exists(config_path):
                    print(f"✅ 找到多模态模型配置")
                    self.initialized = True
                    print("✅ 多模态处理器初始化完成")
                    return True
                else:
                    print("⚠️ 多模态模型配置不完整，使用基础模式")
                    self.initialized = True
                    return True
            else:
                print("⚠️ 多模态模型未找到，使用基础模式")
                print(f"   提示: 运行 python scripts/download_qwen.py --world 下载世界模型")
                self.initialized = True
                return True
                
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def process_text(self, text: str) -> MultimodalOutput:
        """
        处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            处理结果
        """
        return MultimodalOutput(
            output_type='text',
            content=text,
            confidence=1.0,
            metadata={'length': len(text)}
        )
    
    def process_image(
        self,
        image_input: Any,
        prompt: str = "描述这张图片"
    ) -> MultimodalOutput:
        """
        处理图像
        
        Args:
            image_input: 图像输入（路径、URL或PIL Image）
            prompt: 提示词
            
        Returns:
            处理结果
        """
        try:
            # 加载图像
            if isinstance(image_input, str):
                if os.path.exists(image_input):
                    from PIL import Image
                    image = Image.open(image_input)
                else:
                    # 假设是base64
                    image_data = base64.b64decode(image_input)
                    from PIL import Image
                    image = Image.open(BytesIO(image_data))
            else:
                image = image_input
            
            # 处理图像
            # 这里应该调用实际的视觉模型
            description = f"[图像分析] 尺寸: {image.size}, 模式: {image.mode}"
            
            return MultimodalOutput(
                output_type='image_description',
                content=description,
                confidence=0.8,
                metadata={'size': image.size, 'mode': image.mode}
            )
            
        except Exception as e:
            return MultimodalOutput(
                output_type='error',
                content=str(e),
                confidence=0.0
            )
    
    def process_video_stream(
        self,
        video_input: Any,
        prompt: str = "分析视频内容",
        frame_interval: int = 30
    ) -> Generator[MultimodalOutput, None, None]:
        """
        流式处理视频（逐帧）
        
        Args:
            video_input: 视频输入
            prompt: 提示词
            frame_interval: 帧间隔
            
        Yields:
            每帧的处理结果
        """
        try:
            import cv2
            
            # 打开视频
            if isinstance(video_input, str):
                cap = cv2.VideoCapture(video_input)
            else:
                cap = video_input
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 按间隔处理帧
                if frame_count % frame_interval == 0:
                    # 转换为RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    from PIL import Image
                    image = Image.fromarray(frame_rgb)
                    
                    # 处理帧
                    yield MultimodalOutput(
                        output_type='video_frame',
                        content=f"帧 {frame_count}: {image.size}",
                        confidence=0.7,
                        metadata={
                            'frame_number': frame_count,
                            'size': image.size
                        }
                    )
            
            cap.release()
            
            # 最终总结
            yield MultimodalOutput(
                output_type='video_summary',
                content=f"视频处理完成，共 {frame_count} 帧",
                confidence=0.9,
                metadata={'total_frames': frame_count}
            )
            
        except Exception as e:
            yield MultimodalOutput(
                output_type='error',
                content=str(e),
                confidence=0.0
            )
    
    def process_audio(
        self,
        audio_input: Any,
        prompt: str = "转录音频"
    ) -> MultimodalOutput:
        """
        处理音频
        
        Args:
            audio_input: 音频输入
            prompt: 提示词
            
        Returns:
            处理结果
        """
        # 基础实现
        return MultimodalOutput(
            output_type='audio_transcription',
            content="[音频处理]",
            confidence=0.5,
            metadata={'note': '需要音频模型支持'}
        )
    
    def process(
        self,
        input_data: MultimodalInput
    ) -> MultimodalOutput:
        """
        统一处理接口
        
        Args:
            input_data: 多模态输入
            
        Returns:
            处理结果
        """
        if not self.initialized:
            self.initialize()
        
        if input_data.input_type == 'text':
            return self.process_text(input_data.content)
        elif input_data.input_type == 'image':
            return self.process_image(input_data.content)
        elif input_data.input_type == 'video':
            # 返回第一帧结果
            for result in self.process_video_stream(input_data.content):
                return result
        elif input_data.input_type == 'audio':
            return self.process_audio(input_data.content)
        else:
            return MultimodalOutput(
                output_type='error',
                content=f"不支持的输入类型: {input_data.input_type}",
                confidence=0.0
            )
    
    def batch_process(
        self,
        inputs: List[MultimodalInput]
    ) -> List[MultimodalOutput]:
        """
        批量处理
        
        Args:
            inputs: 输入列表
            
        Returns:
            结果列表
        """
        return [self.process(inp) for inp in inputs]
