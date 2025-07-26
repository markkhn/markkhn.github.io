# 博客使用指南

## 快速开始

### 1. 写新博客文章

1. 在项目根目录创建一个新的 `.md` 文件，例如 `my_new_post.md`
2. 在文件开头添加 YAML front matter：

```markdown
---
title: "我的新博客文章"
date: "2024-01-16"
excerpt: "这是文章的简短描述"
tags: "AI, 技术, 研究"
---

# 我的新博客文章

这里是你的博客内容...
```

3. 运行以下命令生成HTML：

```bash
python blog_manager.py my_new_post.md
```

### 2. 批量处理所有Markdown文件

```bash
python blog_manager.py
```

## YAML Front Matter 说明

每个Markdown文件开头都需要包含以下元数据：

```yaml
---
title: "文章标题"           # 必需：文章标题
date: "2024-01-16"        # 必需：发布日期
excerpt: "文章摘要"        # 可选：文章摘要，用于博客列表显示
tags: "标签1, 标签2"      # 可选：文章标签，用逗号分隔
---
```

## Markdown 语法支持

### 标题
```markdown
# 一级标题
## 二级标题
### 三级标题
```

### 文本格式
```markdown
**粗体文本**
*斜体文本*
`行内代码`
```

### 代码块
```markdown
```python
def hello_world():
    print("Hello, World!")
```
```

### 列表
```markdown
- 无序列表项1
- 无序列表项2

1. 有序列表项1
2. 有序列表项2
```

### 链接和图片
```markdown
[链接文本](URL)
![图片描述](图片URL)
```

## 文件结构

```
markkhn.github.io/
├── index.html              # 主页
├── blog/
│   ├── index.html         # 博客列表页
│   ├── posts.json         # 博客文章元数据
│   └── *.html            # 生成的博客文章HTML
├── blog_manager.py        # 博客管理脚本
└── *.md                  # 你的Markdown博客文章
```

## 工作流程

### 日常写作流程

1. **创建新文章**：
   ```bash
   # 创建新的Markdown文件
   touch my_new_post.md
   ```

2. **编辑文章**：
   - 添加YAML front matter
   - 用Markdown写内容
   - 保存文件

3. **生成HTML**：
   ```bash
   python blog_manager.py my_new_post.md
   ```

4. **提交到GitHub**：
   ```bash
   git add .
   git commit -m "Add new blog post: my_new_post"
   git push
   ```

### 批量更新

如果你有多个Markdown文件需要处理：

```bash
python blog_manager.py
```

这会处理当前目录下的所有 `.md` 文件。

## 自定义样式

### 修改博客样式

博客文章的样式在 `blog_manager.py` 中的 `create_html_template` 函数里定义。你可以修改CSS来改变：

- 字体和颜色
- 布局和间距
- 响应式设计

### 修改主页样式

主页样式在 `index.html` 的 `<style>` 标签中定义。

## 故障排除

### 常见问题

1. **图片路径错误**：
   - 确保图片文件存在
   - 使用正确的相对路径

2. **HTML生成失败**：
   - 检查Markdown语法
   - 确保YAML front matter格式正确

3. **博客列表不显示**：
   - 检查 `posts.json` 文件格式
   - 确保JavaScript能正确加载

### 调试技巧

1. **查看生成的HTML**：
   ```bash
   cat blog/your_post.html
   ```

2. **检查posts.json**：
   ```bash
   cat blog/posts.json
   ```

3. **本地测试**：
   ```bash
   python -m http.server 8000
   # 然后在浏览器中访问 http://localhost:8000
   ```

## 高级功能

### 添加新功能

如果你想添加新功能（如评论系统、搜索功能等），可以：

1. 修改 `blog/index.html` 添加新功能
2. 更新 `blog_manager.py` 生成相应的HTML
3. 根据需要添加新的CSS和JavaScript

### 自定义模板

你可以修改 `blog_manager.py` 中的 `create_html_template` 函数来自定义博客文章的HTML模板。

## 联系和支持

如果你遇到问题或需要帮助，可以：

- 检查GitHub Issues
- 查看项目文档
- 联系维护者

---

**提示**：建议定期备份你的Markdown文件，因为它们是你的原始内容。 