# 博客写作指南

## 📝 正确的博客文章格式

### ✅ 正确的格式

**Markdown文件内容：**
```markdown
## Introduction

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
```

**博客数据配置：**
```javascript
{
    id: 'your-post-id',
    title: '你的文章标题',  // 标题在这里定义
    date: '2024-01-25',
    tags: ['AI', 'Technology', 'Your Tag'],
    filename: 'your-post-filename.md',
    excerpt: '文章摘要...'
}
```

### ❌ 错误的格式

**不要在Markdown文件开头写标题：**
```markdown
# 文章标题  <- 不要这样做！

## Introduction
...
```

## 🎯 为什么这样设计？

1. **避免重复**：标题在页面中只显示一次
2. **统一管理**：所有文章信息集中在JavaScript数据中
3. **易于维护**：修改标题只需要改一个地方
4. **SEO友好**：页面标题结构更清晰

## 📋 写作步骤

### 1. 创建Markdown文件
```bash
touch blog/posts/your-article.md
```

### 2. 编写内容（不要写标题）
```markdown
## Introduction

你的文章内容...

## 主要内容

### 要点1

内容...

## 结论

总结...
```

### 3. 更新博客数据
在 `blog/index.html` 和 `blog/post.html` 中添加：
```javascript
{
    id: 'your-article',
    title: '你的文章标题',  // 标题在这里
    date: '2024-01-25',
    tags: ['AI', 'Technology'],
    filename: 'your-article.md',
    excerpt: '文章摘要...'
}
```

### 4. 添加标签（可选）
如果需要新标签，在侧边栏添加：
```html
<span class="badge tag-badge me-1 mb-1 filter-tag" data-tag="新标签">新标签</span>
```

## 🎨 Markdown功能支持

- **标题**：`##`、`###`、`####`
- **粗体**：`**text**`
- **斜体**：`*text*`
- **代码**：`` `code` `` 和 ``` ``` ```
- **列表**：`-` 和 `1.`
- **链接**：`[text](url)`
- **图片**：`![alt](url)`
- **引用**：`> text`

## 💡 提示

1. **保持简洁**：Markdown文件专注于内容
2. **使用模板**：参考 `template.md` 文件
3. **测试预览**：写完文章后测试显示效果
4. **标签一致**：确保文章标签与侧边栏标签一致

---

*遵循这个指南，你的博客文章将显示得更加清晰和专业！* 