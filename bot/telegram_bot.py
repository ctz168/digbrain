#!/usr/bin/env python3
"""
Telegram Bot 服务
Telegram Bot Service

绑定类脑AI系统到Telegram Bot
"""

import os
import sys
import json
import asyncio
import threading
from datetime import datetime
from typing import Optional, Dict, List

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Telegram Bot Token
BOT_TOKEN = "8508628625:AAF7aReEUT6lZszahWid7qN5_b-R3QoYz-g"


class TelegramBot:
    """
    Telegram Bot 服务类
    """
    
    def __init__(self, token: str):
        self.token = token
        self.engine = None
        self.api_url = f"https://api.telegram.org/bot{token}"
        self.offset = 0
        self.sessions: Dict[int, Dict] = {}
        
    def initialize(self) -> bool:
        """初始化引擎"""
        print("=" * 60)
        print("初始化类脑AI系统")
        print("=" * 60)
        
        try:
            from core.brain_engine import BrainLikeStreamingEngine
            
            self.engine = BrainLikeStreamingEngine(
                refresh_rate=60,
                enable_stdp=True,
                enable_memory=True,
                enable_wiki=True,
                enable_world_model=False
            )
            
            if not self.engine.load_models():
                print("❌ 模型加载失败")
                return False
            
            print("\n✅ 初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    def get_updates(self, timeout: int = 30) -> List[Dict]:
        """获取更新"""
        import requests
        
        try:
            response = requests.get(
                f"{self.api_url}/getUpdates",
                params={
                    "offset": self.offset,
                    "timeout": timeout
                },
                timeout=timeout + 10
            )
            
            data = response.json()
            
            if data.get("ok"):
                updates = data.get("result", [])
                if updates:
                    self.offset = updates[-1]["update_id"] + 1
                return updates
            
        except Exception as e:
            print(f"获取更新失败: {e}")
        
        return []
    
    def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
        """发送消息"""
        import requests
        
        try:
            # 分段发送长消息
            max_length = 4000
            messages = []
            
            if len(text) > max_length:
                for i in range(0, len(text), max_length):
                    messages.append(text[i:i+max_length])
            else:
                messages.append(text)
            
            for msg in messages:
                response = requests.post(
                    f"{self.api_url}/sendMessage",
                    data={
                        "chat_id": chat_id,
                        "text": msg,
                        "parse_mode": parse_mode
                    },
                    timeout=30
                )
                
                if not response.json().get("ok"):
                    print(f"发送失败: {response.json()}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"发送消息失败: {e}")
            return False
    
    def send_typing(self, chat_id: int):
        """发送正在输入状态"""
        import requests
        
        try:
            requests.post(
                f"{self.api_url}/sendChatAction",
                data={
                    "chat_id": chat_id,
                    "action": "typing"
                },
                timeout=10
            )
        except:
            pass
    
    def process_message(self, chat_id: int, user_id: int, text: str, username: str = None) -> str:
        """处理消息"""
        # 获取或创建会话
        if chat_id not in self.sessions:
            self.sessions[chat_id] = {
                "created": datetime.now().isoformat(),
                "messages": []
            }
        
        session = self.sessions[chat_id]
        
        # 处理命令
        if text.startswith("/"):
            return self.handle_command(chat_id, text)
        
        # 使用AI引擎处理
        if self.engine:
            response_text = ""
            
            for chunk in self.engine.stream_process(
                text,
                max_tokens=300,
                search_wiki=True,
                search_memory=True
            ):
                if chunk.type == "text":
                    response_text += chunk.content
            
            return response_text if response_text else "抱歉，我无法生成回复。"
        else:
            return "AI引擎未初始化，请稍后再试。"
    
    def handle_command(self, chat_id: int, command: str) -> str:
        """处理命令"""
        cmd = command.lower().strip()
        
        if cmd == "/start":
            return """🧠 <b>类脑AI系统</b>

欢迎使用类脑AI系统！我是基于人脑架构设计的AI助手。

<b>核心特性：</b>
• 高刷新流式处理 (60Hz)
• STDP在线学习
• 海马体记忆系统
• 维基百科知识扩展

<b>命令列表：</b>
/start - 开始对话
/help - 帮助信息
/status - 系统状态
/memory - 记忆统计
/clear - 清空会话

直接发送消息即可开始对话！"""
        
        elif cmd == "/help":
            return """📖 <b>使用帮助</b>

<b>基本用法：</b>
直接发送消息，我会用类脑AI系统为您生成回复。

<b>特色功能：</b>
• 支持中英文对话
• 自动搜索记忆
• 维基百科知识检索
• 在线学习优化

<b>命令：</b>
/start - 开始
/help - 帮助
/status - 状态
/memory - 记忆
/clear - 清空"""
        
        elif cmd == "/status":
            if self.engine:
                status = self.engine.get_status()
                return f"""📊 <b>系统状态</b>

• 模型状态: {'✅ 已加载' if status['model_loaded'] else '❌ 未加载'}
• 处理次数: {status['processing_count']}
• STDP更新: {status['stdp']['update_count'] if status['stdp'] else 0}
• 记忆数量: {status['memory']['total_memories'] if status['memory'] else 0}
• 刷新率: {status['refresh_rate']} Hz"""
            else:
                return "❌ 引擎未初始化"
        
        elif cmd == "/memory":
            if self.engine and self.engine.memory:
                stats = self.engine.memory.get_stats()
                return f"""🧠 <b>记忆系统</b>

• 瞬时记忆: {stats['sensory_count']}
• 短期记忆: {stats['short_term_count']}
• 长期记忆: {stats['long_term_count']}
• 总计: {stats['total_memories']}
• 神经增长: {stats['neuron_growth_events']}"""
            else:
                return "❌ 记忆系统未启用"
        
        elif cmd == "/clear":
            if chat_id in self.sessions:
                self.sessions[chat_id]["messages"] = []
            return "✅ 会话已清空"
        
        else:
            return f"未知命令: {command}\n使用 /help 查看可用命令。"
    
    def run(self):
        """运行Bot"""
        print("\n" + "=" * 60)
        print("Telegram Bot 启动中...")
        print("=" * 60)
        
        if not self.initialize():
            print("初始化失败，退出")
            return
        
        print(f"\n✅ Bot 已启动")
        print("等待消息... (按 Ctrl+C 停止)")
        print("=" * 60)
        
        try:
            while True:
                updates = self.get_updates()
                
                for update in updates:
                    if "message" in update:
                        message = update["message"]
                        chat_id = message["chat"]["id"]
                        user_id = message["from"]["id"]
                        text = message.get("text", "")
                        username = message["from"].get("username", "User")
                        
                        if text:
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {username}: {text[:50]}...")
                            
                            # 发送正在输入状态
                            self.send_typing(chat_id)
                            
                            # 处理消息
                            response = self.process_message(chat_id, user_id, text, username)
                            
                            # 发送回复
                            self.send_message(chat_id, response)
                            
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] 已回复")
                
                # 短暂休眠
                import time
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nBot 停止")
        finally:
            if self.engine:
                self.engine.save_weights()


class AsyncTelegramBot:
    """
    异步Telegram Bot (使用python-telegram-bot库)
    推荐使用此版本，性能更好
    """
    
    def __init__(self, token: str):
        self.token = token
        self.engine = None
        self.application = None
    
    def initialize(self) -> bool:
        """初始化引擎"""
        print("=" * 60)
        print("初始化类脑AI系统")
        print("=" * 60)
        
        try:
            from core.brain_engine import BrainLikeStreamingEngine
            
            self.engine = BrainLikeStreamingEngine(
                refresh_rate=60,
                enable_stdp=True,
                enable_memory=True,
                enable_wiki=True,
                enable_world_model=False
            )
            
            if not self.engine.load_models():
                print("❌ 模型加载失败")
                return False
            
            print("\n✅ 初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    async def start(self, update, context):
        """处理 /start 命令"""
        welcome = """🧠 <b>类脑AI系统</b>

欢迎使用类脑AI系统！我是基于人脑架构设计的AI助手。

<b>核心特性：</b>
• 高刷新流式处理 (60Hz)
• STDP在线学习
• 海马体记忆系统
• 维基百科知识扩展

直接发送消息即可开始对话！"""
        await update.message.reply_text(welcome, parse_mode="HTML")
    
    async def help(self, update, context):
        """处理 /help 命令"""
        help_text = """📖 <b>使用帮助</b>

直接发送消息，我会用类脑AI系统为您生成回复。

<b>命令：</b>
/start - 开始
/help - 帮助
/status - 状态
/memory - 记忆"""
        await update.message.reply_text(help_text, parse_mode="HTML")
    
    async def status(self, update, context):
        """处理 /status 命令"""
        if self.engine:
            status = self.engine.get_status()
            text = f"""📊 <b>系统状态</b>

• 模型: {'✅' if status['model_loaded'] else '❌'}
• 处理: {status['processing_count']}
• STDP: {status['stdp']['update_count'] if status['stdp'] else 0}
• 记忆: {status['memory']['total_memories'] if status['memory'] else 0}"""
        else:
            text = "❌ 引擎未初始化"
        
        await update.message.reply_text(text, parse_mode="HTML")
    
    async def memory(self, update, context):
        """处理 /memory 命令"""
        if self.engine and self.engine.memory:
            stats = self.engine.memory.get_stats()
            text = f"""🧠 <b>记忆系统</b>

• 短期: {stats['short_term_count']}
• 长期: {stats['long_term_count']}
• 总计: {stats['total_memories']}"""
        else:
            text = "❌ 记忆系统未启用"
        
        await update.message.reply_text(text, parse_mode="HTML")
    
    async def handle_message(self, update, context):
        """处理消息"""
        if not self.engine:
            await update.message.reply_text("AI引擎未初始化")
            return
        
        user_message = update.message.text
        chat_id = update.effective_chat.id
        
        # 发送正在输入状态
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # 处理消息
        response_text = ""
        for chunk in self.engine.stream_process(
            user_message,
            max_tokens=300,
            search_wiki=True,
            search_memory=True
        ):
            if chunk.type == "text":
                response_text += chunk.content
        
        if not response_text:
            response_text = "抱歉，我无法生成回复。"
        
        await update.message.reply_text(response_text)
    
    def run(self):
        """运行Bot"""
        try:
            from telegram import ApplicationBuilder
            from telegram.ext import CommandHandler, MessageHandler, filters
            
            print("\n" + "=" * 60)
            print("Telegram Bot 启动中 (异步模式)")
            print("=" * 60)
            
            if not self.initialize():
                print("初始化失败，退出")
                return
            
            # 创建应用
            self.application = ApplicationBuilder().token(self.token).build()
            
            # 添加处理器
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("help", self.help))
            self.application.add_handler(CommandHandler("status", self.status))
            self.application.add_handler(CommandHandler("memory", self.memory))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
            print(f"\n✅ Bot 已启动")
            print("等待消息... (按 Ctrl+C 停止)")
            print("=" * 60)
            
            # 运行
            self.application.run_polling()
            
        except ImportError:
            print("python-telegram-bot 未安装，使用简单模式")
            # 回退到简单模式
            simple_bot = TelegramBot(self.token)
            simple_bot.run()
        except Exception as e:
            print(f"启动失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Telegram Bot 服务")
    parser.add_argument("--token", default=BOT_TOKEN, help="Bot Token")
    parser.add_argument("--async", dest="use_async", action="store_true", help="使用异步模式")
    
    args = parser.parse_args()
    
    if args.use_async:
        bot = AsyncTelegramBot(args.token)
    else:
        bot = TelegramBot(args.token)
    
    bot.run()


if __name__ == "__main__":
    main()
