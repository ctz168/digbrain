"""
API服务器
API Server

提供RESTful API和WebSocket支持
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BrainAPIServer:
    """
    类脑AI API服务器
    """
    
    def __init__(self, model_path: str, port: int = 8000):
        """
        初始化服务器
        
        Args:
            model_path: 模型路径
            port: 端口
        """
        self.model_path = model_path
        self.port = port
        self.ai = None
        self.app = None
        
    def initialize(self):
        """初始化"""
        from core.streaming_engine import BrainLikeAI
        
        print("初始化API服务器...")
        
        self.ai = BrainLikeAI(
            model_path=self.model_path,
            refresh_rate=60
        )
        self.ai.initialize()
        
        print("✅ API服务器初始化完成")
    
    def create_app(self):
        """创建FastAPI应用"""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import StreamingResponse
            from pydantic import BaseModel
            
            app = FastAPI(
                title="类脑AI API",
                description="Brain-like AI System API",
                version="1.0.0"
            )
            
            # CORS
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # 请求模型
            class ChatRequest(BaseModel):
                message: str
                stream: bool = False
                max_tokens: int = 200
            
            class ChatResponse(BaseModel):
                response: str
                stats: Dict[str, Any]
            
            @app.get("/")
            async def root():
                return {
                    "name": "类脑AI API",
                    "version": "1.0.0",
                    "status": "running"
                }
            
            @app.get("/health")
            async def health():
                return {"status": "healthy", "timestamp": datetime.now().isoformat()}
            
            @app.post("/chat", response_model=ChatResponse)
            async def chat(request: ChatRequest):
                """对话接口"""
                if not self.ai:
                    raise HTTPException(status_code=503, detail="AI not initialized")
                
                result = self.ai.process(request.message)
                
                return ChatResponse(
                    response=result["response"],
                    stats=result.get("stats", {})
                )
            
            @app.post("/chat/stream")
            async def chat_stream(request: ChatRequest):
                """流式对话接口"""
                if not self.ai:
                    raise HTTPException(status_code=503, detail="AI not initialized")
                
                async def generate():
                    for chunk in self.ai.stream_chat(request.message):
                        data = {
                            "type": chunk.type,
                            "content": chunk.content,
                            "timestamp": chunk.timestamp
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream"
                )
            
            @app.get("/stats")
            async def stats():
                """获取统计信息"""
                if not self.ai:
                    raise HTTPException(status_code=503, detail="AI not initialized")
                
                return self.ai.get_stats()
            
            @app.get("/memory/stats")
            async def memory_stats():
                """获取记忆统计"""
                if not self.ai:
                    raise HTTPException(status_code=503, detail="AI not initialized")
                
                return self.ai.memory.get_stats()
            
            self.app = app
            return app
            
        except ImportError:
            print("FastAPI未安装，使用简化服务器")
            return None
    
    def run(self):
        """运行服务器"""
        if self.app:
            import uvicorn
            uvicorn.run(self.app, host="0.0.0.0", port=self.port)
        else:
            self._run_simple_server()
    
    def _run_simple_server(self):
        """运行简化服务器"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
        
        ai = self.ai
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                
                if parsed.path == "/":
                    self._send_json({"name": "类脑AI API", "version": "1.0.0"})
                elif parsed.path == "/health":
                    self._send_json({"status": "healthy"})
                elif parsed.path == "/stats":
                    self._send_json(ai.get_stats() if ai else {})
                else:
                    self.send_error(404)
            
            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)
                
                if parsed.path == "/chat":
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length)
                    data = json.loads(body)
                    
                    message = data.get("message", "")
                    result = ai.process(message) if ai else {"response": "", "stats": {}}
                    
                    self._send_json({
                        "response": result.get("response", ""),
                        "stats": result.get("stats", {})
                    })
                else:
                    self.send_error(404)
            
            def _send_json(self, data):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
        
        server = HTTPServer(("0.0.0.0", self.port), Handler)
        print(f"简化API服务器运行在端口 {self.port}")
        server.serve_forever()


def run_server(port: int = 8000, model_path: str = "./models/Qwen3.5-0.8B"):
    """
    运行API服务器
    
    Args:
        port: 端口
        model_path: 模型路径
    """
    server = BrainAPIServer(model_path=model_path, port=port)
    server.initialize()
    server.create_app()
    server.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="./models/Qwen3.5-0.8B")
    args = parser.parse_args()
    
    run_server(port=args.port, model_path=args.model)
