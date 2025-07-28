# 🚀 博客系统快速使用指南

## 📝 添加新博客文章的简单方法

### 方法1: 使用管理脚本（推荐）

```bash
# 进入博客目录
cd blog

# 添加新文章
python add_post.py add

# 按提示输入信息：
# - 文章ID: my-new-article
# - 标题: My New Article
# - 摘要: This is a brief description...
# - 标签: AI,Technology,Research
```

### 方法2: 手动编辑

1. **创建Markdown文件**
   ```bash
   touch blog/posts/my-new-article.md
   ```

2. **编辑 posts.json**
   ```json
   {
     "posts": [
       {
         "id": "my-new-article",
         "title": "My New Article",
         "date": "2024-01-25",
         "tags": ["AI", "Technology"],
         "filename": "my-new-article.md",
         "excerpt": "Brief description..."
       }
     ]
   }
   ```

3. **编写文章内容**
   ```markdown
   ## Introduction
   
   你的文章内容...
   
   ## 主要内容
   
   ### 要点1
   
   内容...
   ```

## 🛠️ 管理命令

### 列出所有文章
```bash
cd blog
python add_post.py list
```

### 添加新文章
```bash
cd blog
python add_post.py add
```

### 交互式菜单
```bash
cd blog
python add_post.py
```

## 📁 文件结构

```
blog/
├── posts.json          # 文章数据（自动加载）
├── add_post.py         # 管理脚本
├── index.html          # 博客主页
├── post.html           # 文章页面
└── posts/              # Markdown文件
    ├── my-article.md
    └── another-article.md
```

## 🎯 优势

1. **集中管理**: 所有文章数据在 `posts.json` 中
2. **自动同步**: 无需手动更新多个文件
3. **简单操作**: 使用脚本一键添加文章
4. **错误检查**: 自动检查ID重复等问题
5. **模板生成**: 自动创建Markdown模板

## 💡 提示

- 文章ID要唯一
- 标签用逗号分隔
- Markdown文件不要写标题
- 新标签需要在 `index.html` 中添加

## 🏷️ 添加新标签

### 1. 在博客主页添加标签按钮
编辑 `blog/index.html`，在侧边栏的Categories部分添加：
```html
<span class="badge tag-badge me-1 mb-1 filter-tag" data-tag="新标签">新标签</span>
```

### 2. 在文章中使用新标签
在 `blog/posts.json` 中为文章添加新标签：
```json
{
  "posts": [
    {
      "id": "your-post",
      "title": "Your Post Title",
      "date": "2024-01-25",
      "tags": ["AI", "新标签", "Technology"],
      "filename": "your-post.md",
      "excerpt": "Your excerpt..."
    }
  ]
}
```

### 3. 标签命名规范
- 使用描述性名称（如 "Machine Learning" 而不是 "ML"）
- 保持大小写一致
- 避免特殊字符，只使用空格

---

*现在添加新文章只需要一个命令！* 