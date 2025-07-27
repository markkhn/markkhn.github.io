#!/usr/bin/env python3
"""
博客文章管理脚本
用于添加新的博客文章到 posts.json 文件
"""

import json
import os
import sys
from datetime import datetime

def load_posts_data():
    """加载现有的文章数据"""
    try:
        with open('posts.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"posts": []}

def save_posts_data(data):
    """保存文章数据到文件"""
    with open('posts.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_markdown_template(filename, title):
    """创建Markdown文件模板"""
    template = f"""## Introduction

这里是你的博客文章内容。注意不要在Markdown文件开头写标题，因为标题会从博客数据中自动获取。

## 主要内容

### 1. 第一个要点

这里是第一个要点的详细内容。

### 2. 第二个要点

这里是第二个要点的详细内容。

## 代码示例

```python
def example_function():
    """
    这是一个示例函数
    """
    print("Hello, World!")
    return "Success"
```

## 结论

这里是文章的结论部分。

---

*这是文章的结尾说明。*
"""
    
    with open(f'posts/{filename}', 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✅ 已创建Markdown文件: posts/{filename}")

def add_new_post():
    """添加新博客文章"""
    print("📝 添加新博客文章")
    print("=" * 50)
    
    # 获取用户输入
    post_id = input("文章ID (例如: my-new-post): ").strip()
    title = input("文章标题: ").strip()
    excerpt = input("文章摘要: ").strip()
    tags_input = input("标签 (用逗号分隔，例如: AI,Technology,Research): ").strip()
    
    # 处理标签
    tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
    
    # 生成文件名
    filename = f"{post_id}.md"
    
    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # 创建新文章对象
    new_post = {
        "id": post_id,
        "title": title,
        "date": current_date,
        "tags": tags,
        "filename": filename,
        "excerpt": excerpt
    }
    
    # 加载现有数据
    data = load_posts_data()
    
    # 检查ID是否已存在
    existing_ids = [post["id"] for post in data["posts"]]
    if post_id in existing_ids:
        print(f"❌ 错误: 文章ID '{post_id}' 已存在")
        return
    
    # 添加新文章
    data["posts"].append(new_post)
    
    # 保存数据
    save_posts_data(data)
    
    # 创建Markdown文件
    create_markdown_template(filename, title)
    
    print("\n✅ 新文章已成功添加!")
    print(f"📄 文章ID: {post_id}")
    print(f"📝 标题: {title}")
    print(f"🏷️ 标签: {', '.join(tags)}")
    print(f"📁 Markdown文件: posts/{filename}")
    print(f"📊 数据文件: posts.json")
    print("\n💡 提示:")
    print("1. 编辑 posts/{filename} 添加文章内容")
    print("2. 如果需要新标签，记得在 blog/index.html 中添加")
    print("3. 访问 http://localhost:8001/blog/ 查看效果")

def list_posts():
    """列出所有文章"""
    data = load_posts_data()
    print("📚 当前所有文章:")
    print("=" * 50)
    
    for i, post in enumerate(data["posts"], 1):
        print(f"{i}. {post['title']}")
        print(f"   ID: {post['id']}")
        print(f"   日期: {post['date']}")
        print(f"   标签: {', '.join(post['tags'])}")
        print(f"   文件: {post['filename']}")
        print()

def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "list":
            list_posts()
        elif command == "add":
            add_new_post()
        else:
            print("❌ 未知命令")
            print("用法: python add_post.py [list|add]")
    else:
        print("📝 博客文章管理工具")
        print("=" * 30)
        print("1. list - 列出所有文章")
        print("2. add  - 添加新文章")
        print()
        choice = input("请选择操作 (1/2): ").strip()
        
        if choice == "1":
            list_posts()
        elif choice == "2":
            add_new_post()
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main() 