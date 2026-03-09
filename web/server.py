#!/usr/bin/env python3
"""
Web服务器模块
Web Server Module

提供静态文件服务和API接口
"""

import os
import sys
import json
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import traceback

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")
sys.path.insert(0, BASE_DIR)


class BrainWebHandler(BaseHTTPRequestHandler):
    """Web请求处理器"""
    
    engine = None
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")
    
    def do_GET(self):
        """处理GET请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/" or path == "/index.html":
            self.serve_file("index.html", "text/html")
        elif path == "/status":
            self.handle_status()
        elif path == "/memory/stats":
            self.handle_memory_stats()
        elif path.startswith("/static/"):
            self.serve_static(path[8:])
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """处理POST请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        
        try:
            data = json.loads(body)
        except:
            data = {}
        
        if path == "/chat":
            self.handle_chat(data)
        elif path == "/memory/search":
            self.handle_memory_search(data)
        elif path == "/session/new":
            self.handle_new_session()
        else:
            self.send_error(404, "Not Found")
    
    def serve_file(self, filename, content_type):
        """提供静态文件"""
        filepath = os.path.join(WEB_DIR, filename)
        
        if os.path.exists(filepath):
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File Not Found")
    
    def serve_static(self, filename):
        """提供静态资源"""
        mime_types = {
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon'
        }
        
        ext = os.path.splitext(filename)[1]
        content_type = mime_types.get(ext, 'application/octet-stream')
        self.serve_file(filename, content_type)
    
    def send_json(self, data, code=200):
        """发送JSON响应"""
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def handle_status(self):
        """处理状态请求"""
        if self.engine:
            status = self.engine.get_status()
            self.send_json({"status": "running", "engine": status})
        else:
            self.send_json({"status": "not_initialized", "engine": None})
    
    def handle_memory_stats(self):
        """处理记忆统计请求"""
        if self.engine and self.engine.memory:
            stats = self.engine.memory.get_stats()
            self.send_json(stats)
        else:
            self.send_json({"error": "Memory not available"}, 400)
    
    def handle_memory_search(self, data):
        """处理记忆搜索请求"""
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        
        if self.engine and self.engine.memory:
            results = self.engine.memory.search(query, top_k=top_k)
            self.send_json({"results": results})
        else:
            self.send_json({"results": []})
    
    def handle_new_session(self):
        """创建新会话"""
        import uuid
        session_id = uuid.uuid4().hex
        self.send_json({"session_id": session_id})
    
    def handle_chat(self, data):
        """处理对话请求"""
        message = data.get("message", "")
        stream = data.get("stream", True)
        
        if not message:
            self.send_json({"error": "Message required"}, 400)
            return
        
        if not self.engine:
            self.send_json({"error": "Engine not initialized"}, 500)
            return
        
        if stream:
            # 流式响应
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            
            try:
                for chunk in self.engine.stream_process(
                    message,
                    max_tokens=300,
                    search_wiki=data.get("enable_wiki", True),
                    search_memory=data.get("enable_memory", True)
                ):
                    chunk_data = {
                        "type": chunk.type,
                        "content": chunk.content,
                        "timestamp": chunk.timestamp,
                        "metadata": chunk.metadata
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
            except Exception as e:
                error_data = {"type": "error", "content": str(e)}
                self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode())
        else:
            # 非流式响应
            full_response = ""
            for chunk in self.engine.stream_process(message, max_tokens=300):
                if chunk.type == "text":
                    full_response += chunk.content
            
            self.send_json({
                "response": full_response,
                "stream": False
            })


class BrainWebServer:
    """类脑AI Web服务器"""
    
    def __init__(self, host="0.0.0.0", port=8000, refresh_rate=60):
        self.host = host
        self.port = port
        self.refresh_rate = refresh_rate
        self.server = None
        self.engine = None
    
    def initialize(self):
        """初始化引擎"""
        print("=" * 60)
        print("初始化类脑AI系统")
        print("=" * 60)
        
        try:
            from core.brain_engine import BrainLikeStreamingEngine
            
            self.engine = BrainLikeStreamingEngine(
                refresh_rate=self.refresh_rate,
                enable_stdp=True,
                enable_memory=True,
                enable_wiki=True,
                enable_world_model=False  # 节省内存
            )
            
            if not self.engine.load_models():
                print("❌ 模型加载失败")
                return False
            
            # 设置全局引擎
            BrainWebHandler.engine = self.engine
            
            print("\n✅ 初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            traceback.print_exc()
            return False
    
    def start(self):
        """启动服务器"""
        if not self.initialize():
            print("初始化失败，启动无引擎模式")
        
        self.server = HTTPServer((self.host, self.port), BrainWebHandler)
        
        print("\n" + "=" * 60)
        print(f"类脑AI Web服务器启动")
        print("=" * 60)
        print(f"\n访问地址: http://{self.host}:{self.port}")
        print(f"本地访问: http://localhost:{self.port}")
        print("\nAPI端点:")
        print("  GET  /              - Web界面")
        print("  GET  /status        - 系统状态")
        print("  POST /chat          - 对话 (流式)")
        print("  GET  /memory/stats  - 记忆统计")
        print("  POST /memory/search - 记忆搜索")
        print("\n按 Ctrl+C 停止服务器")
        print("=" * 60)
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n\n服务器停止")
        finally:
            if self.engine:
                self.engine.save_weights()
            self.server.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="类脑AI Web服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--refresh-rate", type=int, default=60, help="刷新率")
    
    args = parser.parse_args()
    
    server = BrainWebServer(
        host=args.host,
        port=args.port,
        refresh_rate=args.refresh_rate
    )
    
    server.start()


if __name__ == "__main__":
    main()
