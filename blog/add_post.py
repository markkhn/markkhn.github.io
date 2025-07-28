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

def load_tags_data():
    """加载标签数据"""
    try:
        with open('tags.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  警告: tags.json 文件不存在，将使用空标签列表")
        return {"tags": []}

def save_tags_data(data):
    """保存标签数据"""
    with open('tags.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_tag_counts():
    """更新标签使用计数"""
    posts_data = load_posts_data()
    tags_data = load_tags_data()
    
    if not posts_data or not tags_data:
        return False
    
    # 统计所有文章中的标签使用情况
    all_tags = []
    for post in posts_data["posts"]:
        all_tags.extend(post["tags"])
    
    from collections import Counter
    tag_counts = Counter(all_tags)
    
    # 更新标签计数
    for tag in tags_data["tags"]:
        tag["count"] = tag_counts.get(tag["name"], 0)
    
    save_tags_data(tags_data)
    return True

def create_markdown_template(filename, title):
    """创建Markdown文件模板"""
    template = """## Introduction

这里是你的博客文章内容。注意不要在Markdown文件开头写标题，因为标题会从博客数据中自动获取。

## 主要内容

### 1. 第一个要点

这里是第一个要点的详细内容。

### 2. 第二个要点

这里是第二个要点的详细内容。

## 代码示例

```python
def example_function():
    \"\"\"
    This is an example function
    \"\"\"
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

def get_available_tags():
    """获取可用标签列表"""
    tags_data = load_tags_data()
    return [tag["name"] for tag in tags_data["tags"]]

def add_new_post():
    """添加新博客文章"""
    print("📝 添加新博客文章")
    print("=" * 50)
    
    # 获取用户输入
    post_id = input("文章ID (例如: my-new-post): ").strip()
    title = input("文章标题: ").strip()
    excerpt = input("文章摘要: ").strip()
    
    # 显示可用标签
    available_tags = get_available_tags()
    if available_tags:
        print(f"\n🏷️ 可用标签: {', '.join(available_tags)}")
        print("💡 提示: 可以输入新标签名，系统会自动创建")
    
    tags_input = input("标签 (用逗号分隔，例如: AI,Technology,Research): ").strip()
    
    # 处理标签
    tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
    
    # 检查并创建新标签
    tags_data = load_tags_data()
    for tag_name in tags:
        # 检查标签是否已存在
        existing_tag_names = [tag["name"] for tag in tags_data["tags"]]
        if tag_name not in existing_tag_names:
            print(f"🆕 发现新标签: {tag_name}")
            tag_description = input(f"请输入标签 '{tag_name}' 的描述: ").strip()
            tag_color = input(f"请输入标签 '{tag_name}' 的颜色 (十六进制，如 #007bff): ").strip()
            if not tag_color:
                tag_color = "#6c757d"  # 默认灰色
            
            # 创建新标签
            new_tag = {
                "name": tag_name,
                "description": tag_description,
                "color": tag_color,
                "count": 0
            }
            tags_data["tags"].append(new_tag)
            save_tags_data(tags_data)
            print(f"✅ 新标签 '{tag_name}' 已创建")
    
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
    
    # 更新标签计数
    update_tag_counts()
    
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
    print("2. 访问 http://localhost:8001/blog/ 查看效果")
    print("3. 使用 'python add_tag.py html' 生成HTML标签代码")

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