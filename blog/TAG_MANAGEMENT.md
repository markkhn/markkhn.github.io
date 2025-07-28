# 标签管理系统

## 概述

新的标签管理系统将标签和文章管理分离，使用独立的 `tags.json` 文件来管理所有标签，确保标签数据的一致性和完整性。

## 文件结构

```
blog/
├── tags.json          # 标签数据文件
├── posts.json         # 文章数据文件
├── add_tag.py         # 标签管理脚本
├── add_post.py        # 文章管理脚本
└── posts/             # Markdown文章文件夹
```

## 标签数据格式

`tags.json` 文件格式：
```json
{
  "tags": [
    {
      "name": "AI",
      "description": "Artificial Intelligence",
      "color": "#007bff",
      "count": 4
    }
  ]
}
```

字段说明：
- `name`: 标签名称
- `description`: 标签描述
- `color`: 标签颜色（十六进制）
- `count`: 使用该标签的文章数量

## 管理命令

### 标签管理 (`add_tag.py`)

```bash
# 列出所有标签
python add_tag.py list

# 列出文章及其标签
python add_tag.py posts

# 生成HTML标签代码
python add_tag.py html

# 添加新标签
python add_tag.py add

# 删除标签
python add_tag.py remove

# 同步标签计数
python add_tag.py sync
```

### 文章管理 (`add_post.py`)

```bash
# 列出所有文章
python add_post.py list

# 添加新文章
python add_post.py add
```

## 工作流程

### 添加新文章
1. 运行 `python add_post.py add`
2. 输入文章信息
3. 系统会自动：
   - 检查标签是否存在
   - 创建新标签（如果需要）
   - 更新标签计数
   - 创建Markdown文件

### 添加新标签
1. 运行 `python add_tag.py add`
2. 输入标签信息
3. 系统会自动更新标签文件

### 同步数据
运行 `python add_tag.py sync` 来同步标签计数

## 优势

1. **数据一致性**: 标签数据集中管理，避免不一致
2. **自动计数**: 自动统计标签使用次数
3. **完整信息**: 每个标签包含描述、颜色等元数据
4. **易于维护**: 分离的脚本便于维护和扩展
5. **HTML生成**: 自动生成HTML标签代码

## 注意事项

1. 删除标签时会同时从所有文章中移除该标签
2. 标签名称区分大小写
3. 建议定期运行 `sync` 命令保持数据同步
4. 新标签需要手动添加到 `blog/index.html` 中（可使用 `html` 命令生成代码） 