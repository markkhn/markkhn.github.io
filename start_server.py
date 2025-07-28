#!/usr/bin/env python3
"""
启动脚本 - 在正确的目录启动HTTP服务器
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    
    print("🚀 启动个人主页服务器")
    print("=" * 40)
    print(f"📁 项目目录: {project_root}")
    print(f"🌐 服务器地址: http://localhost:8000")
    print(f"📝 个人主页: http://localhost:8000/")
    print(f"📚 博客页面: http://localhost:8000/blog/")
    print()
    print("💡 提示:")
    print("- 按 Ctrl+C 停止服务器")
    print("- 确保在项目根目录运行此脚本")
    print()
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    # 启动HTTP服务器
    try:
        subprocess.run([sys.executable, "-m", "http.server", "8000"], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")

if __name__ == "__main__":
    main() 