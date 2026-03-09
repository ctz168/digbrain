#!/usr/bin/env python3
"""
API服务器模块
API Server Module

提供RESTful API和WebSocket接口
"""

import os
import sys
import json
import time
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, asdict

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.brain_engine import BrainLikeStreamingEngine, StreamChunk


class BrainAPIServer:
    """
    类脑AI API服务器
    支持HTTP和WebSocket
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
        
        self.engine: Optional[BrainLikeStreamingEngine] = None
        self.server = None
        self.is_running = False
        
        # 会话管理
        self.sessions: Dict[str, Dict] = {}
    
    def initialize(self) -> bool:
        """初始化引擎"""
        print("初始化类脑AI引擎...")
        
        self.engine = BrainLikeStreamingEngine(
            refresh_rate=self.refresh_rate,
            enable_stdp=True,
            enable_memory=True,
            enable_wiki=True
        )
        
        return self.engine.load_models()
    
    def create_session(self) -> str:
        """创建新会话"""
        import uuid
        session_id = uuid.uuid4().hex
        
        self.sessions[session_id] = {
            "created": datetime.now().isoformat(),
            "messages": [],
            "memory_context": []
        }
        
        return session_id
    
    def chat(self, message: str, session_id: str = None, stream: bool = True) -> Dict:
        """
        对话接口
        
        Args:
            message: 用户消息
            session_id: 会话ID
            stream: 是否流式输出
            
        Returns:
            响应结果
        """
        if not self.engine:
            return {"error": "Engine not initialized"}
        
        # 创建或获取会话
        if not session_id or session_id not in self.sessions:
            session_id = self.create_session()
        
        session = self.sessions[session_id]
        session["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 处理
        if stream:
            return {
                "session_id": session_id,
                "stream": True,
                "message": "Use WebSocket for streaming"
            }
        else:
            # 非流式处理
            full_response = ""
            for chunk in self.engine.stream_process(message, max_tokens=200):
                if chunk.type == "text":
                    full_response += chunk.content
            
            session["messages"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "session_id": session_id,
                "response": full_response,
                "stream": False
            }
    
    def stream_chat(self, message: str, session_id: str = None) -> Generator[Dict, None, None]:
        """流式对话"""
        if not self.engine:
            yield {"error": "Engine not initialized"}
            return
        
        # 创建或获取会话
        if not session_id or session_id not in self.sessions:
            session_id = self.create_session()
        
        session = self.sessions[session_id]
        session["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        full_response = ""
        
        for chunk in self.engine.stream_process(message, max_tokens=300):
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
        if not self.engine:
            return {"status": "not_initialized"}
        
        return {
            "status": "running" if self.is_running else "ready",
            "engine": self.engine.get_status(),
            "sessions": len(self.sessions)
        }
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        if not self.engine or not self.engine.memory:
            return {"error": "Memory not available"}
        
        return self.engine.memory.get_stats()
    
    def search_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索记忆"""
        if not self.engine or not self.engine.memory:
            return []
        
        return self.engine.memory.search(query, top_k=top_k)
    
    def run_flask(self):
        """运行Flask服务器"""
        from flask import Flask, request, jsonify, Response
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        @app.route("/")
        def index():
            return jsonify({
                "name": "Brain-like AI API",
                "version": "1.0.0",
                "status": "running"
            })
        
        @app.route("/status", methods=["GET"])
        def status():
            return jsonify(self.get_status())
        
        @app.route("/chat", methods=["POST"])
        def chat():
            data = request.json
            message = data.get("message", "")
            session_id = data.get("session_id")
            stream = data.get("stream", True)
            
            if not message:
                return jsonify({"error": "Message required"}), 400
            
            if stream:
                def generate():
                    for chunk in self.stream_chat(message, session_id):
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
                return Response(
                    generate(),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                return jsonify(self.chat(message, session_id, stream=False))
        
        @app.route("/memory/stats", methods=["GET"])
        def memory_stats():
            return jsonify(self.get_memory_stats())
        
        @app.route("/memory/search", methods=["POST"])
        def memory_search():
            data = request.json
            query = data.get("query", "")
            top_k = data.get("top_k", 5)
            
            results = self.search_memory(query, top_k)
            return jsonify({"results": results})
        
        @app.route("/session/new", methods=["POST"])
        def new_session():
            session_id = self.create_session()
            return jsonify({"session_id": session_id})
        
        print(f"\n启动Flask服务器: http://{self.host}:{self.port}")
        app.run(host=self.host, port=self.port, threaded=True)
    
    def run_simple(self):
        """运行简单的HTTP服务器"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
        
        server = self
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                
                if parsed.path == "/":
                    self.send_json({
                        "name": "Brain-like AI API",
                        "version": "1.0.0",
                        "endpoints": ["/status", "/chat", "/memory/stats", "/memory/search"]
                    })
                elif parsed.path == "/status":
                    self.send_json(server.get_status())
                elif parsed.path == "/memory/stats":
                    self.send_json(server.get_memory_stats())
                else:
                    self.send_error(404)
            
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                try:
                    data = json.loads(body) if body else {}
                except:
                    data = {}
                
                if self.path == "/chat":
                    message = data.get("message", "")
                    if not message:
                        self.send_json({"error": "Message required"}, 400)
                        return
                    
                    # 非流式响应
                    result = server.chat(message, data.get("session_id"), stream=False)
                    self.send_json(result)
                
                elif self.path == "/memory/search":
                    query = data.get("query", "")
                    results = server.search_memory(query, data.get("top_k", 5))
                    self.send_json({"results": results})
                
                else:
                    self.send_error(404)
            
            def send_json(self, data, code=200):
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            
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
        
        self.is_running = True
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器停止")
        finally:
            self.is_running = False
    
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
    
    parser = argparse.ArgumentParser(description="类脑AI API服务器")
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
