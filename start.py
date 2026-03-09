#!/usr/bin/env python3
"""
类脑AI系统 - 一键启动脚本
Brain-like AI System - Quick Start Script

功能：
1. 检查依赖
2. 下载模型
3. 启动API服务器
4. 启动Web界面
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 需要Python 3.8+，当前版本:", sys.version)
        return False
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """检查依赖"""
    print("\n检查依赖...")
    
    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("flask", "Flask"),
        ("numpy", "NumPy"),
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - 未安装")
            missing.append(module)
    
    if missing:
        print(f"\n缺少依赖，请运行: pip install {' '.join(missing)}")
        return False
    return True


def download_models():
    """下载模型"""
    print("\n检查模型...")
    
    model_dir = BASE_DIR / "models"
    qwen_path = model_dir / "Qwen2.5-0.5B-Instruct"
    
    if qwen_path.exists() and (qwen_path / "config.json").exists():
        print("  ✅ 模型已存在")
        return True
    
    print("  📥 模型未找到，开始下载...")
    download_script = BASE_DIR / "scripts" / "download_qwen.py"
    
    if download_script.exists():
        subprocess.run([sys.executable, str(download_script), "-m", "Qwen2.5-0.5B-Instruct"])
        return True
    else:
        print("  ⚠️ 下载脚本不存在，请手动下载模型")
        return False


def start_api_server(port=8000, host="0.0.0.0"):
    """启动API服务器"""
    print(f"\n启动API服务器: http://{host}:{port}")
    
    os.chdir(BASE_DIR)
    
    # 使用gunicorn或Flask
    try:
        import gunicorn
        subprocess.run([
            "gunicorn", 
            "-w", "4",
            "-b", f"{host}:{port}",
            "api.server:app"
        ])
    except:
        # 直接运行Flask
        from api.server import BrainAPIServer
        server = BrainAPIServer(host=host, port=port)
        server.initialize()
        server.run_flask()


def start_web_server(port=8000, host="0.0.0.0"):
    """启动Web服务器"""
    print(f"\n启动Web服务器: http://{host}:{port}")
    
    os.chdir(BASE_DIR)
    
    from web.server import BrainWebServer
    server = BrainWebServer(host=host, port=port)
    server.start()


def run_interactive():
    """运行交互模式"""
    print("\n启动交互模式...")
    
    os.chdir(BASE_DIR)
    
    from main import interactive_mode
    from core.brain_engine import BrainLikeStreamingEngine
    
    engine = BrainLikeStreamingEngine(
        refresh_rate=60,
        enable_stdp=True,
        enable_memory=True,
        enable_wiki=True
    )
    
    if not engine.load_models():
        print("模型加载失败")
        return
    
    interactive_mode(engine)


def run_demo():
    """运行演示模式"""
    print("\n启动演示模式...")
    
    os.chdir(BASE_DIR)
    
    from main import demo_mode
    from core.brain_engine import BrainLikeStreamingEngine
    
    engine = BrainLikeStreamingEngine(
        refresh_rate=60,
        enable_stdp=True,
        enable_memory=True,
        enable_wiki=True
    )
    
    if not engine.load_models():
        print("模型加载失败")
        return
    
    demo_mode(engine)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="类脑AI系统 - Brain-like AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python start.py                  # 启动API服务器
  python start.py --web            # 启动Web界面
  python start.py --interactive    # 交互模式
  python start.py --demo           # 演示模式
  python start.py --download       # 只下载模型
        """
    )
    
    parser.add_argument("--web", "-w", action="store_true", help="启动Web服务器")
    parser.add_argument("--api", "-a", action="store_true", help="启动API服务器")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--demo", "-d", action="store_true", help="演示模式")
    parser.add_argument("--download", action="store_true", help="下载模型")
    parser.add_argument("--port", "-p", type=int, default=8000, help="服务器端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--skip-check", action="store_true", help="跳过依赖检查")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 类脑AI系统 | Brain-like AI System")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        return 1
    
    # 检查依赖
    if not args.skip_check and not check_dependencies():
        return 1
    
    # 下载模型
    if args.download:
        download_models()
        return 0
    
    # 默认启动API服务器
    if args.web:
        start_web_server(args.port, args.host)
    elif args.interactive:
        run_interactive()
    elif args.demo:
        run_demo()
    else:
        # 启动API服务器
        start_api_server(args.port, args.host)
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
