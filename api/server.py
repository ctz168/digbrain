#!/usr/bin/env python3
"""
API服务器模块 - 完整版
API Server Module - Complete Version

提供RESTful API和WebSocket接口：
- /chat - 对话接口（支持流式）
- /status - 系统状态
- /memory - 记忆操作（搜索、存储、统计）
- /train - 训练接口
- /evaluate - 评估接口
- WebSocket实时通信
"""

import os
import sys
import json
import time
import asyncio
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator, Callable
from dataclasses import dataclass, asdict
import uuid
import hashlib

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class APIResponse:
    """统一API响应格式"""
    success: bool
    data: Any = None
    error: str = None
    code: int = 200
    timestamp: str = None
    
    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "code": self.code,
            "timestamp": self.timestamp
        }


class TrainingManager:
    """训练管理器"""
    
    def __init__(self):
        self.active_trainings: Dict[str, Dict] = {}
        self.training_history: List[Dict] = []
        self.lock = threading.Lock()
    
    def start_training(self, config: Dict) -> str:
        """启动训练任务"""
        training_id = f"train_{uuid.uuid4().hex[:8]}"
        
        with self.lock:
            self.active_trainings[training_id] = {
                "id": training_id,
                "status": "running",
                "progress": 0,
                "config": config,
                "start_time": datetime.now().isoformat(),
                "metrics": {},
                "logs": []
            }
        
        # 异步执行训练
        thread = threading.Thread(
            target=self._run_training,
            args=(training_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return training_id
    
    def _run_training(self, training_id: str, config: Dict):
        """执行训练"""
        try:
            from training.offline_trainer import OfflineTrainer
            
            module = config.get("module", "all")
            epochs = config.get("epochs", 5)
            parallel = config.get("parallel", True)
            
            trainer = OfflineTrainer(
                learning_rate=config.get("learning_rate", 0.01),
                epochs=epochs
            )
            
            # 更新进度
            total_epochs = epochs
            for epoch in range(1, total_epochs + 1):
                with self.lock:
                    if training_id in self.active_trainings:
                        self.active_trainings[training_id]["progress"] = epoch / total_epochs * 100
                        self.active_trainings[training_id]["logs"].append(
                            f"Epoch {epoch}/{total_epochs} 完成"
                        )
                
                time.sleep(0.5)  # 模拟训练时间
            
            # 执行实际训练
            if module == "all":
                result = trainer.train_all(parallel=parallel, epochs=epochs)
            else:
                result = trainer.train_module(module, epochs=epochs)
            
            # 更新完成状态
            with self.lock:
                if training_id in self.active_trainings:
                    self.active_trainings[training_id]["status"] = "completed"
                    self.active_trainings[training_id]["progress"] = 100
                    self.active_trainings[training_id]["end_time"] = datetime.now().isoformat()
                    self.active_trainings[training_id]["result"] = result
                    
                    self.training_history.append(self.active_trainings[training_id])
            
        except Exception as e:
            with self.lock:
                if training_id in self.active_trainings:
                    self.active_trainings[training_id]["status"] = "failed"
                    self.active_trainings[training_id]["error"] = str(e)
    
    def get_training_status(self, training_id: str) -> Optional[Dict]:
        """获取训练状态"""
        with self.lock:
            if training_id in self.active_trainings:
                return self.active_trainings[training_id].copy()
            # 检查历史
            for record in self.training_history:
                if record["id"] == training_id:
                    return record.copy()
        return None
    
    def stop_training(self, training_id: str) -> bool:
        """停止训练"""
        with self.lock:
            if training_id in self.active_trainings:
                self.active_trainings[training_id]["status"] = "stopped"
                return True
        return False


class EvaluationManager:
    """评估管理器"""
    
    def __init__(self):
        self.evaluation_history: List[Dict] = []
        self.active_evaluations: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    def run_evaluation(self, config: Dict) -> str:
        """运行评估"""
        eval_id = f"eval_{uuid.uuid4().hex[:8]}"
        
        with self.lock:
            self.active_evaluations[eval_id] = {
                "id": eval_id,
                "status": "running",
                "config": config,
                "start_time": datetime.now().isoformat(),
                "progress": 0
            }
        
        # 异步执行评估
        thread = threading.Thread(
            target=self._run_evaluation,
            args=(eval_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return eval_id
    
    def _run_evaluation(self, eval_id: str, config: Dict):
        """执行评估"""
        try:
            from evaluation.benchmark import BrainBenchmark
            
            benchmark = BrainBenchmark()
            
            # 运行评估
            results = {}
            test_types = config.get("tests", ["math", "code", "knowledge", "reasoning"])
            
            total_tests = len(test_types)
            for i, test_type in enumerate(test_types):
                # 更新进度
                with self.lock:
                    if eval_id in self.active_evaluations:
                        self.active_evaluations[eval_id]["progress"] = (i / total_tests) * 100
                
                # 执行测试
                if hasattr(benchmark, f"test_{test_type}"):
                    test_func = getattr(benchmark, f"test_{test_type}")
                    results[test_type] = test_func()
                else:
                    results[test_type] = {"score": 0, "error": "Unknown test type"}
            
            # 完成评估
            with self.lock:
                if eval_id in self.active_evaluations:
                    self.active_evaluations[eval_id]["status"] = "completed"
                    self.active_evaluations[eval_id]["progress"] = 100
                    self.active_evaluations[eval_id]["results"] = results
                    self.active_evaluations[eval_id]["end_time"] = datetime.now().isoformat()
                    
                    # 计算总分
                    total_score = sum(
                        r.get("score", 0) for r in results.values() if isinstance(r, dict)
                    )
                    self.active_evaluations[eval_id]["total_score"] = total_score / len(results)
                    
                    self.evaluation_history.append(self.active_evaluations[eval_id])
        
        except Exception as e:
            with self.lock:
                if eval_id in self.active_evaluations:
                    self.active_evaluations[eval_id]["status"] = "failed"
                    self.active_evaluations[eval_id]["error"] = str(e)
    
    def get_evaluation_status(self, eval_id: str) -> Optional[Dict]:
        """获取评估状态"""
        with self.lock:
            if eval_id in self.active_evaluations:
                return self.active_evaluations[eval_id].copy()
            for record in self.evaluation_history:
                if record["id"] == eval_id:
                    return record.copy()
        return None


class BrainAPIServer:
    """
    类脑AI API服务器 - 完整版
    支持HTTP REST和WebSocket
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        refresh_rate: int = 60
    ):
        self.host = host
        self.port = port
        self.refresh_rate = refresh_rate
        
        self.engine = None
        self.server = None
        self.is_running = False
        
        # 会话管理
        self.sessions: Dict[str, Dict] = {}
        
        # 训练和评估管理
        self.training_manager = TrainingManager()
        self.evaluation_manager = EvaluationManager()
        
        # WebSocket连接
        self.ws_clients: List[Any] = []
        
        # API统计
        self.stats = {
            "total_requests": 0,
            "chat_requests": 0,
            "train_requests": 0,
            "eval_requests": 0,
            "errors": 0
        }
    
    def initialize(self) -> bool:
        """初始化引擎"""
        print("=" * 60)
        print("初始化类脑AI引擎...")
        print("=" * 60)
        
        try:
            from core.brain_engine import BrainLikeStreamingEngine
            
            self.engine = BrainLikeStreamingEngine(
                refresh_rate=self.refresh_rate,
                enable_stdp=True,
                enable_memory=True,
                enable_wiki=True
            )
            
            success = self.engine.load_models()
            if success:
                print("✅ 引擎初始化成功")
            return success
            
        except Exception as e:
            print(f"❌ 引擎初始化失败: {e}")
            traceback.print_exc()
            return False
    
    def create_session(self, user_id: str = None) -> str:
        """创建新会话"""
        session_id = uuid.uuid4().hex
        
        self.sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "created": datetime.now().isoformat(),
            "messages": [],
            "memory_context": [],
            "settings": {
                "enable_stdp": True,
                "enable_memory": True,
                "enable_wiki": True,
                "temperature": 0.7,
                "max_tokens": 300
            }
        }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def chat(self, message: str, session_id: str = None, stream: bool = True, **kwargs) -> Dict:
        """
        对话接口
        
        Args:
            message: 用户消息
            session_id: 会话ID
            stream: 是否流式输出
            **kwargs: 额外参数
            
        Returns:
            响应结果
        """
        self.stats["total_requests"] += 1
        self.stats["chat_requests"] += 1
        
        if not self.engine:
            return {"error": "Engine not initialized", "code": 500}
        
        # 创建或获取会话
        if not session_id or session_id not in self.sessions:
            session_id = self.create_session()
        
        session = self.sessions[session_id]
        settings = session.get("settings", {})
        
        # 合并设置
        enable_stdp = kwargs.get("enable_stdp", settings.get("enable_stdp", True))
        enable_memory = kwargs.get("enable_memory", settings.get("enable_memory", True))
        enable_wiki = kwargs.get("enable_wiki", settings.get("enable_wiki", True))
        max_tokens = kwargs.get("max_tokens", settings.get("max_tokens", 300))
        temperature = kwargs.get("temperature", settings.get("temperature", 0.7))
        
        # 记录用户消息
        session["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        if stream:
            return {
                "session_id": session_id,
                "stream": True,
                "message": "Use WebSocket or SSE for streaming"
            }
        else:
            # 非流式处理
            full_response = ""
            metadata = {}
            
            for chunk in self.engine.stream_process(
                message, 
                max_tokens=max_tokens,
                temperature=temperature,
                search_wiki=enable_wiki,
                search_memory=enable_memory
            ):
                if chunk.type == "text":
                    full_response += chunk.content
                elif chunk.type == "control" and chunk.content == "done":
                    metadata = chunk.metadata
            
            # 保存响应
            session["messages"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata
            })
            
            return {
                "session_id": session_id,
                "response": full_response,
                "stream": False,
                "metadata": metadata
            }
    
    def stream_chat(self, message: str, session_id: str = None, **kwargs) -> Generator[Dict, None, None]:
        """流式对话"""
        if not self.engine:
            yield {"type": "error", "content": "Engine not initialized"}
            return
        
        # 创建或获取会话
        if not session_id or session_id not in self.sessions:
            session_id = self.create_session()
        
        session = self.sessions[session_id]
        settings = session.get("settings", {})
        
        # 合并设置
        enable_stdp = kwargs.get("enable_stdp", settings.get("enable_stdp", True))
        enable_memory = kwargs.get("enable_memory", settings.get("enable_memory", True))
        enable_wiki = kwargs.get("enable_wiki", settings.get("enable_wiki", True))
        max_tokens = kwargs.get("max_tokens", settings.get("max_tokens", 300))
        temperature = kwargs.get("temperature", settings.get("temperature", 0.7))
        
        # 记录用户消息
        session["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        full_response = ""
        
        for chunk in self.engine.stream_process(
            message, 
            max_tokens=max_tokens,
            temperature=temperature,
            search_wiki=enable_wiki,
            search_memory=enable_memory
        ):
            data = {
                "session_id": session_id,
                "type": chunk.type,
                "content": chunk.content,
                "timestamp": chunk.timestamp,
                "metadata": chunk.metadata
            }
            
            if chunk.type == "text":
                full_response += chunk.content
            
            yield data
        
        # 保存响应
        session["messages"].append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        status = {
            "status": "running" if self.is_running else "ready",
            "uptime": None,
            "api_version": "2.0.0"
        }
        
        if self.engine:
            status["engine"] = self.engine.get_status()
        
        status["sessions"] = len(self.sessions)
        status["stats"] = self.stats.copy()
        status["active_trainings"] = len(self.training_manager.active_trainings)
        status["active_evaluations"] = len(self.evaluation_manager.active_evaluations)
        
        return status
    
    # ================== 记忆操作 ==================
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        if not self.engine or not self.engine.memory:
            return {"error": "Memory not available"}
        
        return self.engine.memory.get_stats()
    
    def search_memory(self, query: str, top_k: int = 5, memory_type: str = "all") -> List[Dict]:
        """搜索记忆"""
        if not self.engine or not self.engine.memory:
            return []
        
        return self.engine.memory.search(query, top_k=top_k, memory_type=memory_type)
    
    def store_memory(self, content: str, importance: float = 0.5, tags: List[str] = None) -> Dict:
        """存储记忆"""
        if not self.engine or not self.engine.memory:
            return {"error": "Memory not available"}
        
        key = self.engine.memory.store_short_term(
            content, 
            importance=importance, 
            tags=tags or []
        )
        
        return {
            "success": True,
            "key": key,
            "type": "short_term"
        }
    
    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """获取单条记忆"""
        if not self.engine or not self.engine.memory:
            return None
        
        with self.engine.memory.lock:
            # 搜索短期记忆
            if memory_id in self.engine.memory.short_term_memory:
                mem = self.engine.memory.short_term_memory[memory_id]
                return {
                    "id": memory_id,
                    "content": mem.get("content"),
                    "type": "short_term",
                    "importance": mem.get("importance"),
                    "timestamp": mem.get("timestamp")
                }
            
            # 搜索长期记忆
            if memory_id in self.engine.memory.long_term_memory:
                mem = self.engine.memory.long_term_memory[memory_id]
                return {
                    "id": memory_id,
                    "content": mem.get("content"),
                    "type": "long_term",
                    "importance": mem.get("importance"),
                    "timestamp": mem.get("timestamp")
                }
        
        return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        if not self.engine or not self.engine.memory:
            return False
        
        return self.engine.memory.forget(memory_id)
    
    def clear_memory(self, memory_type: str = None) -> Dict:
        """清除记忆"""
        if not self.engine or not self.engine.memory:
            return {"error": "Memory not available"}
        
        count = self.engine.memory.clear(memory_type)
        return {"success": True, "cleared": count}
    
    # ================== 训练接口 ==================
    
    def start_training(self, config: Dict) -> Dict:
        """启动训练"""
        self.stats["train_requests"] += 1
        training_id = self.training_manager.start_training(config)
        
        return {
            "success": True,
            "training_id": training_id,
            "message": f"Training started with ID: {training_id}"
        }
    
    def get_training_status(self, training_id: str) -> Optional[Dict]:
        """获取训练状态"""
        return self.training_manager.get_training_status(training_id)
    
    def stop_training(self, training_id: str) -> Dict:
        """停止训练"""
        success = self.training_manager.stop_training(training_id)
        return {
            "success": success,
            "message": "Training stopped" if success else "Training not found"
        }
    
    def get_training_history(self) -> List[Dict]:
        """获取训练历史"""
        return self.training_manager.training_history.copy()
    
    # ================== 评估接口 ==================
    
    def start_evaluation(self, config: Dict) -> Dict:
        """启动评估"""
        self.stats["eval_requests"] += 1
        eval_id = self.evaluation_manager.run_evaluation(config)
        
        return {
            "success": True,
            "evaluation_id": eval_id,
            "message": f"Evaluation started with ID: {eval_id}"
        }
    
    def get_evaluation_status(self, eval_id: str) -> Optional[Dict]:
        """获取评估状态"""
        return self.evaluation_manager.get_evaluation_status(eval_id)
    
    def get_evaluation_history(self) -> List[Dict]:
        """获取评估历史"""
        return self.evaluation_manager.evaluation_history.copy()
    
    # ================== WebSocket支持 ==================
    
    async def handle_websocket(self, websocket, path: str):
        """处理WebSocket连接"""
        self.ws_clients.append(websocket)
        
        try:
            # 发送欢迎消息
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "Connected to Brain-like AI WebSocket",
                "timestamp": datetime.now().isoformat()
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self._handle_ws_message(data)
                    
                    if isinstance(response, list):
                        for chunk in response:
                            await websocket.send(json.dumps(chunk))
                    else:
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                    
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.ws_clients.remove(websocket)
    
    async def _handle_ws_message(self, data: Dict) -> Any:
        """处理WebSocket消息"""
        msg_type = data.get("type", "unknown")
        
        if msg_type == "chat":
            # 流式聊天
            message = data.get("message", "")
            session_id = data.get("session_id")
            
            results = []
            for chunk in self.stream_chat(message, session_id, **data):
                results.append(chunk)
            return results
        
        elif msg_type == "ping":
            return {"type": "pong", "timestamp": datetime.now().isoformat()}
        
        elif msg_type == "status":
            return {"type": "status", "data": self.get_status()}
        
        else:
            return {"type": "error", "message": f"Unknown message type: {msg_type}"}
    
    def broadcast_to_ws(self, message: Dict):
        """广播消息到所有WebSocket客户端"""
        for client in self.ws_clients:
            try:
                asyncio.create_task(client.send(json.dumps(message)))
            except:
                pass
    
    # ================== Flask服务器 ==================
    
    def run_flask(self):
        """运行Flask服务器"""
        try:
            from flask import Flask, request, jsonify, Response, send_from_directory
            from flask_cors import CORS
        except ImportError:
            print("Flask未安装，使用简单服务器")
            self.run_simple()
            return
        
        app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "web"))
        CORS(app, resources={
            r"/*": {
                "origins": "*",
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
        
        server = self
        
        # ================== API信息 ==================
        
        @app.route("/")
        def index():
            return jsonify({
                "name": "Brain-like AI API",
                "version": "2.0.0",
                "description": "类脑AI系统RESTful API",
                "endpoints": {
                    "chat": "/chat - POST - 对话接口",
                    "status": "/status - GET - 系统状态",
                    "memory": {
                        "stats": "/memory/stats - GET - 记忆统计",
                        "search": "/memory/search - POST - 搜索记忆",
                        "store": "/memory - POST - 存储记忆",
                        "get": "/memory/<id> - GET - 获取记忆",
                        "delete": "/memory/<id> - DELETE - 删除记忆"
                    },
                    "train": {
                        "start": "/train - POST - 开始训练",
                        "status": "/train/<id> - GET - 训练状态",
                        "stop": "/train/<id>/stop - POST - 停止训练",
                        "history": "/train/history - GET - 训练历史"
                    },
                    "evaluate": {
                        "start": "/evaluate - POST - 开始评估",
                        "status": "/evaluate/<id> - GET - 评估状态",
                        "history": "/evaluate/history - GET - 评估历史"
                    },
                    "session": {
                        "new": "/session - POST - 创建会话",
                        "get": "/session/<id> - GET - 获取会话",
                        "delete": "/session/<id> - DELETE - 删除会话"
                    },
                    "docs": "/docs - GET - API文档"
                }
            })
        
        # ================== 状态接口 ==================
        
        @app.route("/status", methods=["GET"])
        def status():
            return jsonify(server.get_status())
        
        # ================== 对话接口 ==================
        
        @app.route("/chat", methods=["POST", "OPTIONS"])
        def chat():
            if request.method == "OPTIONS":
                return jsonify({})
            
            data = request.json or {}
            message = data.get("message", "")
            session_id = data.get("session_id")
            stream = data.get("stream", True)
            
            if not message:
                return jsonify(APIResponse(
                    success=False, 
                    error="Message required", 
                    code=400
                ).to_dict()), 400
            
            if stream:
                def generate():
                    for chunk in server.stream_chat(message, session_id, **data):
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
                return Response(
                    generate(),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive"
                    }
                )
            else:
                result = server.chat(message, session_id, stream=False, **data)
                return jsonify(APIResponse(success=True, data=result).to_dict())
        
        # ================== 记忆接口 ==================
        
        @app.route("/memory/stats", methods=["GET"])
        def memory_stats():
            result = server.get_memory_stats()
            return jsonify(APIResponse(success=True, data=result).to_dict())
        
        @app.route("/memory/search", methods=["POST"])
        def memory_search():
            data = request.json or {}
            query = data.get("query", "")
            top_k = data.get("top_k", 5)
            memory_type = data.get("memory_type", "all")
            
            results = server.search_memory(query, top_k, memory_type)
            return jsonify(APIResponse(
                success=True, 
                data={"results": results, "count": len(results)}
            ).to_dict())
        
        @app.route("/memory", methods=["POST"])
        def store_memory():
            data = request.json or {}
            content = data.get("content", "")
            
            if not content:
                return jsonify(APIResponse(
                    success=False, 
                    error="Content required", 
                    code=400
                ).to_dict()), 400
            
            result = server.store_memory(
                content,
                importance=data.get("importance", 0.5),
                tags=data.get("tags", [])
            )
            return jsonify(APIResponse(success=True, data=result).to_dict())
        
        @app.route("/memory/<memory_id>", methods=["GET"])
        def get_memory(memory_id):
            result = server.get_memory(memory_id)
            if result:
                return jsonify(APIResponse(success=True, data=result).to_dict())
            return jsonify(APIResponse(
                success=False, 
                error="Memory not found", 
                code=404
            ).to_dict()), 404
        
        @app.route("/memory/<memory_id>", methods=["DELETE"])
        def delete_memory(memory_id):
            success = server.delete_memory(memory_id)
            return jsonify(APIResponse(
                success=success, 
                data={"deleted": memory_id} if success else None,
                error="Memory not found" if not success else None,
                code=200 if success else 404
            ).to_dict())
        
        @app.route("/memory/clear", methods=["POST"])
        def clear_memory():
            data = request.json or {}
            result = server.clear_memory(data.get("memory_type"))
            return jsonify(APIResponse(success=True, data=result).to_dict())
        
        # ================== 训练接口 ==================
        
        @app.route("/train", methods=["POST"])
        def start_training():
            data = request.json or {}
            result = server.start_training(data)
            return jsonify(APIResponse(success=True, data=result).to_dict())
        
        @app.route("/train/<training_id>", methods=["GET"])
        def get_training_status(training_id):
            result = server.get_training_status(training_id)
            if result:
                return jsonify(APIResponse(success=True, data=result).to_dict())
            return jsonify(APIResponse(
                success=False, 
                error="Training not found", 
                code=404
            ).to_dict()), 404
        
        @app.route("/train/<training_id>/stop", methods=["POST"])
        def stop_training(training_id):
            result = server.stop_training(training_id)
            return jsonify(APIResponse(success=result["success"], data=result).to_dict())
        
        @app.route("/train/history", methods=["GET"])
        def training_history():
            result = server.get_training_history()
            return jsonify(APIResponse(
                success=True, 
                data={"history": result, "count": len(result)}
            ).to_dict())
        
        # ================== 评估接口 ==================
        
        @app.route("/evaluate", methods=["POST"])
        def start_evaluation():
            data = request.json or {}
            result = server.start_evaluation(data)
            return jsonify(APIResponse(success=True, data=result).to_dict())
        
        @app.route("/evaluate/<eval_id>", methods=["GET"])
        def get_evaluation_status(eval_id):
            result = server.get_evaluation_status(eval_id)
            if result:
                return jsonify(APIResponse(success=True, data=result).to_dict())
            return jsonify(APIResponse(
                success=False, 
                error="Evaluation not found", 
                code=404
            ).to_dict()), 404
        
        @app.route("/evaluate/history", methods=["GET"])
        def evaluation_history():
            result = server.get_evaluation_history()
            return jsonify(APIResponse(
                success=True, 
                data={"history": result, "count": len(result)}
            ).to_dict())
        
        # ================== 会话接口 ==================
        
        @app.route("/session", methods=["POST"])
        def create_session():
            data = request.json or {}
            session_id = server.create_session(data.get("user_id"))
            return jsonify(APIResponse(
                success=True, 
                data={"session_id": session_id}
            ).to_dict())
        
        @app.route("/session/<session_id>", methods=["GET"])
        def get_session(session_id):
            session = server.get_session(session_id)
            if session:
                return jsonify(APIResponse(success=True, data=session).to_dict())
            return jsonify(APIResponse(
                success=False, 
                error="Session not found", 
                code=404
            ).to_dict()), 404
        
        @app.route("/session/<session_id>", methods=["DELETE"])
        def delete_session(session_id):
            if session_id in server.sessions:
                del server.sessions[session_id]
                return jsonify(APIResponse(success=True).to_dict())
            return jsonify(APIResponse(
                success=False, 
                error="Session not found", 
                code=404
            ).to_dict()), 404
        
        @app.route("/session/<session_id>/settings", methods=["PUT"])
        def update_session_settings(session_id):
            if session_id not in server.sessions:
                return jsonify(APIResponse(
                    success=False, 
                    error="Session not found", 
                    code=404
                ).to_dict()), 404
            
            data = request.json or {}
            server.sessions[session_id]["settings"].update(data)
            return jsonify(APIResponse(
                success=True, 
                data=server.sessions[session_id]["settings"]
            ).to_dict())
        
        # ================== API文档 ==================
        
        @app.route("/docs", methods=["GET"])
        def api_docs():
            return jsonify(server.get_openapi_spec())
        
        # ================== 静态文件 ==================
        
        @app.route("/web/<path:filename>")
        def serve_static(filename):
            return send_from_directory(os.path.join(BASE_DIR, "web"), filename)
        
        # ================== 错误处理 ==================
        
        @app.errorhandler(404)
        def not_found(e):
            return jsonify(APIResponse(
                success=False, 
                error="Not found", 
                code=404
            ).to_dict()), 404
        
        @app.errorhandler(500)
        def server_error(e):
            server.stats["errors"] += 1
            return jsonify(APIResponse(
                success=False, 
                error="Internal server error", 
                code=500
            ).to_dict()), 500
        
        # ================== 启动 ==================
        
        print(f"\n启动Flask服务器: http://{self.host}:{self.port}")
        print("\nAPI端点:")
        print("  GET  /                   - API信息")
        print("  GET  /status             - 系统状态")
        print("  POST /chat               - 对话 (支持SSE流式)")
        print("  GET  /memory/stats       - 记忆统计")
        print("  POST /memory/search      - 搜索记忆")
        print("  POST /memory             - 存储记忆")
        print("  GET  /memory/<id>        - 获取记忆")
        print("  DELETE /memory/<id>      - 删除记忆")
        print("  POST /train              - 开始训练")
        print("  GET  /train/<id>         - 训练状态")
        print("  POST /evaluate           - 开始评估")
        print("  GET  /evaluate/<id>      - 评估状态")
        print("  GET  /docs               - API文档")
        
        self.is_running = True
        app.run(host=self.host, port=self.port, threaded=True)
    
    # ================== 简单HTTP服务器 ==================
    
    def run_simple(self):
        """运行简单的HTTP服务器"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
        
        server = self
        
        class Handler(BaseHTTPRequestHandler):
            def do_OPTIONS(self):
                """处理CORS预检请求"""
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
            
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                
                if path == "/":
                    self.send_json({
                        "name": "Brain-like AI API",
                        "version": "2.0.0",
                        "endpoints": ["/status", "/chat", "/memory", "/train", "/evaluate"]
                    })
                elif path == "/status":
                    self.send_json(server.get_status())
                elif path == "/memory/stats":
                    self.send_json(server.get_memory_stats())
                elif path.startswith("/train/"):
                    training_id = path.split("/")[-1]
                    result = server.get_training_status(training_id)
                    if result:
                        self.send_json(result)
                    else:
                        self.send_json({"error": "Training not found"}, 404)
                elif path.startswith("/evaluate/"):
                    eval_id = path.split("/")[-1]
                    result = server.get_evaluation_status(eval_id)
                    if result:
                        self.send_json(result)
                    else:
                        self.send_json({"error": "Evaluation not found"}, 404)
                elif path == "/docs":
                    self.send_json(server.get_openapi_spec())
                else:
                    self.send_error(404)
            
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                try:
                    data = json.loads(body) if body else {}
                except:
                    data = {}
                
                path = self.path.split("?")[0]
                
                if path == "/chat":
                    message = data.get("message", "")
                    if not message:
                        self.send_json({"error": "Message required"}, 400)
                        return
                    
                    # 检查是否流式
                    if data.get("stream", True):
                        self.send_stream_response(message, data)
                    else:
                        result = server.chat(message, data.get("session_id"), stream=False, **data)
                        self.send_json(result)
                
                elif path == "/memory/search":
                    query = data.get("query", "")
                    results = server.search_memory(query, data.get("top_k", 5))
                    self.send_json({"results": results})
                
                elif path == "/memory":
                    content = data.get("content", "")
                    if not content:
                        self.send_json({"error": "Content required"}, 400)
                        return
                    result = server.store_memory(
                        content,
                        importance=data.get("importance", 0.5),
                        tags=data.get("tags", [])
                    )
                    self.send_json(result)
                
                elif path == "/memory/clear":
                    result = server.clear_memory(data.get("memory_type"))
                    self.send_json(result)
                
                elif path == "/train":
                    result = server.start_training(data)
                    self.send_json(result)
                
                elif path.startswith("/train/") and path.endswith("/stop"):
                    training_id = path.split("/")[-2]
                    result = server.stop_training(training_id)
                    self.send_json(result)
                
                elif path == "/evaluate":
                    result = server.start_evaluation(data)
                    self.send_json(result)
                
                elif path == "/session":
                    session_id = server.create_session(data.get("user_id"))
                    self.send_json({"session_id": session_id})
                
                else:
                    self.send_error(404)
            
            def do_DELETE(self):
                path = self.path.split("?")[0]
                
                if path.startswith("/memory/"):
                    memory_id = path.split("/")[-1]
                    success = server.delete_memory(memory_id)
                    if success:
                        self.send_json({"success": True, "deleted": memory_id})
                    else:
                        self.send_json({"error": "Memory not found"}, 404)
                
                elif path.startswith("/session/"):
                    session_id = path.split("/")[-1]
                    if session_id in server.sessions:
                        del server.sessions[session_id]
                        self.send_json({"success": True})
                    else:
                        self.send_json({"error": "Session not found"}, 404)
                
                else:
                    self.send_error(404)
            
            def send_json(self, data, code=200):
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            
            def send_stream_response(self, message: str, data: Dict):
                """发送流式响应"""
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                
                try:
                    for chunk in server.stream_chat(
                        message, 
                        data.get("session_id"),
                        **data
                    ):
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                except Exception as e:
                    error_data = {"type": "error", "content": str(e)}
                    self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode())
            
            def log_message(self, format, *args):
                print(f"[API] {args[0]}")
        
        self.server = HTTPServer((self.host, self.port), Handler)
        
        print(f"\n启动简单HTTP服务器: http://{self.host}:{self.port}")
        print("API端点:")
        print("  GET  /              - API信息")
        print("  GET  /status        - 系统状态")
        print("  POST /chat          - 对话")
        print("  GET  /memory/stats  - 记忆统计")
        print("  POST /memory/search - 记忆搜索")
        print("  POST /memory        - 存储记忆")
        print("  POST /train         - 开始训练")
        print("  POST /evaluate      - 开始评估")
        print("  GET  /docs          - API文档")
        
        self.is_running = True
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器停止")
        finally:
            self.is_running = False
    
    def get_openapi_spec(self) -> Dict:
        """获取OpenAPI规范"""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Brain-like AI API",
                "version": "2.0.0",
                "description": "类脑AI系统RESTful API - 支持流式对话、记忆管理、训练和评估"
            },
            "servers": [
                {"url": f"http://{self.host}:{self.port}", "description": "本地服务器"}
            ],
            "paths": {
                "/chat": {
                    "post": {
                        "summary": "对话接口",
                        "description": "发送消息并获取AI响应，支持流式输出",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "message": {"type": "string", "description": "用户消息"},
                                            "session_id": {"type": "string", "description": "会话ID"},
                                            "stream": {"type": "boolean", "default": True},
                                            "enable_memory": {"type": "boolean", "default": True},
                                            "enable_wiki": {"type": "boolean", "default": True},
                                            "max_tokens": {"type": "integer", "default": 300},
                                            "temperature": {"type": "number", "default": 0.7}
                                        },
                                        "required": ["message"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {"description": "成功响应"},
                            "400": {"description": "请求错误"}
                        }
                    }
                },
                "/status": {
                    "get": {
                        "summary": "获取系统状态",
                        "responses": {"200": {"description": "系统状态信息"}}
                    }
                },
                "/memory/search": {
                    "post": {
                        "summary": "搜索记忆",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string"},
                                            "top_k": {"type": "integer", "default": 5}
                                        },
                                        "required": ["query"]
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "搜索结果"}}
                    }
                },
                "/train": {
                    "post": {
                        "summary": "启动训练",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "module": {"type": "string", "enum": ["memory", "stdp", "all"]},
                                            "epochs": {"type": "integer", "default": 10},
                                            "learning_rate": {"type": "number", "default": 0.01},
                                            "parallel": {"type": "boolean", "default": True}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "训练任务ID"}}
                    }
                },
                "/evaluate": {
                    "post": {
                        "summary": "启动评估",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "tests": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "default": ["math", "code", "knowledge", "reasoning"]
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "评估任务ID"}}
                    }
                }
            }
        }
    
    def start(self, use_flask: bool = False):
        """启动服务器"""
        if not self.initialize():
            print("初始化失败")
            return
        
        if use_flask:
            try:
                self.run_flask()
            except ImportError:
                print("Flask未安装，使用简单服务器")
                self.run_simple()
        else:
            self.run_simple()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="类脑AI API服务器 v2.0")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--flask", action="store_true", help="使用Flask服务器")
    parser.add_argument("--refresh-rate", type=int, default=60, help="刷新率")
    
    args = parser.parse_args()
    
    server = BrainAPIServer(
        host=args.host,
        port=args.port,
        refresh_rate=args.refresh_rate
    )
    
    server.start(use_flask=args.flask)


if __name__ == "__main__":
    main()
