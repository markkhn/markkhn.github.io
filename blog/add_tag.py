#!/usr/bin/env python3
"""
标签管理脚本
用于管理独立的标签系统
"""

import json
import re
import sys
from collections import Counter

def load_tags_data():
    """加载标签数据"""
    try:
        with open('tags.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 错误: tags.json 文件不存在")
        return None

def save_tags_data(data):
    """保存标签数据"""
    with open('tags.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_posts_data():
    """加载文章数据"""
    try:
        with open('posts.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 错误: posts.json 文件不存在")
        return None

def save_posts_data(data):
    """保存文章数据"""
    with open('posts.json', 'w', encoding='utf-8') as f:
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
    tag_counts = Counter(all_tags)
    
    # 更新标签计数
    for tag in tags_data["tags"]:
        tag["count"] = tag_counts.get(tag["name"], 0)
    
    save_tags_data(tags_data)
    return True

def get_all_tags():
    """获取所有标签"""
    data = load_tags_data()
    if not data:
        return []
    return data["tags"]

def list_all_tags():
    """列出所有标签"""
    tags = get_all_tags()
    if tags:
        print("🏷️ 当前所有标签:")
        print("=" * 50)
        print(f"{'标签名':<15} {'描述':<25} {'使用次数':<8} {'颜色'}")
        print("-" * 50)
        for tag in sorted(tags, key=lambda x: x["name"]):
            count_str = f"{tag['count']} 次" if tag['count'] > 0 else "未使用"
            print(f"{tag['name']:<15} {tag['description']:<25} {count_str:<8} {tag['color']}")
    else:
        print("📝 当前没有标签")

def list_posts_with_tags():
    """列出所有文章及其标签"""
    data = load_posts_data()
    if not data:
        return
    
    print("📚 文章及其标签:")
    print("=" * 50)
    for post in data["posts"]:
        print(f"📄 {post['title']}")
        print(f"   ID: {post['id']}")
        print(f"   🏷️ 标签: {', '.join(post['tags'])}")
        print()

def generate_html_tag():
    """生成HTML标签代码"""
    tags = get_all_tags()
    if not tags:
        print("❌ 没有可用的标签")
        return
    
    print("📝 在 blog/index.html 中添加以下代码:")
    print("=" * 60)
    print("找到侧边栏的Categories部分，添加:")
    print()
    
    for tag in tags:
        if tag["count"] > 0:  # 只显示有文章的标签
            print(f'<span class="badge tag-badge me-1 mb-1 filter-tag" data-tag="{tag["name"]}">{tag["name"]}</span>')
    
    print()
    print("💡 提示:")
    print("- 只显示有文章的标签")
    print("- 确保 data-tag 属性与标签名一致")
    print("- 标签名区分大小写")

def add_new_tag():
    """添加新标签的交互式流程"""
    print("🏷️ 添加新标签")
    print("=" * 30)
    
    # 显示现有标签
    current_tags = get_all_tags()
    if current_tags:
        print("当前标签:", ", ".join([tag["name"] for tag in current_tags]))
        print()
    
    # 获取新标签信息
    new_tag_name = input("新标签名: ").strip()
    if not new_tag_name:
        print("❌ 标签名不能为空")
        return
    
    # 检查标签是否已存在
    existing_tags = [tag["name"] for tag in current_tags]
    if new_tag_name in existing_tags:
        print(f"⚠️  标签 '{new_tag_name}' 已存在")
        return
    
    new_tag_description = input("标签描述: ").strip()
    new_tag_color = input("标签颜色 (十六进制，如 #007bff): ").strip()
    if not new_tag_color:
        new_tag_color = "#6c757d"  # 默认灰色
    
    # 创建新标签
    new_tag = {
        "name": new_tag_name,
        "description": new_tag_description,
        "color": new_tag_color,
        "count": 0
    }
    
    # 添加到标签文件
    tags_data = load_tags_data()
    if not tags_data:
        tags_data = {"tags": []}
    
    tags_data["tags"].append(new_tag)
    save_tags_data(tags_data)
    
    print(f"✅ 标签 '{new_tag_name}' 已成功添加!")
    print("📝 记得在 blog/index.html 中添加对应的HTML标签")

def remove_tag():
    """删除标签"""
    print("🗑️ 删除标签")
    print("=" * 20)
    
    tags = get_all_tags()
    if not tags:
        print("📝 当前没有标签")
        return
    
    print("当前标签:")
    for i, tag in enumerate(tags, 1):
        count_str = f"({tag['count']} 次使用)" if tag['count'] > 0 else "(未使用)"
        print(f"{i}. {tag['name']} - {tag['description']} {count_str}")
    
    try:
        choice = int(input("\n选择要删除的标签编号: ")) - 1
        if 0 <= choice < len(tags):
            tag_to_remove = tags[choice]
            
            if tag_to_remove["count"] > 0:
                print(f"⚠️  警告: 标签 '{tag_to_remove['name']}' 正在被 {tag_to_remove['count']} 篇文章使用")
                confirm = input("确定要删除吗? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("❌ 取消删除")
                    return
            
            # 从标签文件中删除
            tags_data = load_tags_data()
            tags_data["tags"].pop(choice)
            save_tags_data(tags_data)
            
            # 从所有文章中删除该标签
            posts_data = load_posts_data()
            if posts_data:
                for post in posts_data["posts"]:
                    if tag_to_remove["name"] in post["tags"]:
                        post["tags"].remove(tag_to_remove["name"])
                save_posts_data(posts_data)
            
            print(f"✅ 标签 '{tag_to_remove['name']}' 已删除")
        else:
            print("❌ 无效的选择")
    except ValueError:
        print("❌ 请输入有效的数字")

def sync_tags():
    """同步标签和文章数据"""
    print("🔄 同步标签和文章数据")
    print("=" * 30)
    
    if update_tag_counts():
        print("✅ 标签计数已更新")
    else:
        print("❌ 同步失败")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "list":
            list_all_tags()
        elif command == "posts":
            list_posts_with_tags()
        elif command == "html":
            generate_html_tag()
        elif command == "add":
            add_new_tag()
        elif command == "remove":
            remove_tag()
        elif command == "sync":
            sync_tags()
        else:
            print("❌ 未知命令")
            print("用法: python add_tag.py [list|posts|html|add|remove|sync]")
    else:
        print("🏷️ 标签管理工具")
        print("=" * 20)
        print("1. list   - 列出所有标签")
        print("2. posts  - 列出文章及其标签")
        print("3. html   - 生成HTML标签代码")
        print("4. add    - 添加新标签")
        print("5. remove - 删除标签")
        print("6. sync   - 同步标签计数")
        print()
        choice = input("请选择操作 (1-6): ").strip()
        
        if choice == "1":
            list_all_tags()
        elif choice == "2":
            list_posts_with_tags()
        elif choice == "3":
            generate_html_tag()
        elif choice == "4":
            add_new_tag()
        elif choice == "5":
            remove_tag()
        elif choice == "6":
            sync_tags()
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main() 