# 博客分类功能说明

## 🏷️ 新增功能

### 1. 标签分类系统

博客系统现在支持按标签分类浏览文章：

- **动态标签统计**：显示每个标签下的文章数量
- **一键筛选**：点击标签即可筛选相关文章
- **智能推荐**：文章页面显示基于标签相似度的相关文章

### 2. 功能特点

#### 博客主页 (`blog/index.html`)
- **分类标签**：侧边栏显示所有可用标签
- **文章计数**：每个标签显示文章数量，如 "AI (3)"
- **实时筛选**：点击标签立即筛选文章
- **视觉反馈**：选中的标签高亮显示

#### 文章页面 (`blog/post.html`)
- **当前文章标签**：显示当前文章的所有标签
- **智能相关文章**：基于标签相似度推荐相关文章
- **标签关联提示**：显示相关文章的共同标签

### 3. 使用方法

#### 浏览分类
1. 访问博客主页
2. 在侧边栏点击任意标签
3. 页面会只显示包含该标签的文章
4. 点击 "All" 显示所有文章

#### 添加新标签
1. 在 `blog/index.html` 的侧边栏添加新标签：
```html
<span class="badge tag-badge me-1 mb-1 filter-tag" data-tag="新标签">新标签</span>
```

2. 在文章数据中添加标签：
```javascript
{
    id: 'your-post-id',
    title: 'Your Post Title',
    date: '2024-01-25',
    tags: ['AI', '新标签', 'Technology'],
    filename: 'your-post.md',
    excerpt: 'Your excerpt...'
}
```

### 4. 技术实现

#### 筛选算法
- 使用 `data-tags` 属性存储文章标签
- JavaScript 实时筛选显示/隐藏文章
- CSS 过渡动画提供流畅体验

#### 相似度计算
- 基于标签重叠度计算文章相似度
- 公式：`相似度 = 共同标签数 / max(文章A标签数, 文章B标签数)`
- 按相似度排序推荐相关文章

#### 标签统计
- 自动统计每个标签的文章数量
- 动态更新标签显示格式：`标签名 (数量)`

### 5. 支持的标签

当前支持的标签：
- **All**：显示所有文章
- **AI**：人工智能相关
- **Technology**：技术相关
- **Research**：研究相关
- **Deep Learning**：深度学习
- **Computer Vision**：计算机视觉
- **Healthcare**：医疗保健
- **Radar**：雷达技术

### 6. 自定义样式

可以通过修改CSS来自定义标签样式：

```css
.tag-badge {
    background-color: #e9ecef;
    color: #2c3e50;
    font-size: 0.8em;
    cursor: pointer;
    transition: all 0.3s ease;
}

.tag-badge:hover {
    background-color: #007bff;
    color: white;
    transform: translateY(-1px);
}

.tag-badge.active {
    background-color: #007bff;
    color: white;
}
```

### 7. 扩展功能

未来可以添加的功能：
- **多标签筛选**：同时选择多个标签
- **标签搜索**：搜索特定标签
- **标签云**：可视化标签分布
- **标签页面**：专门的标签详情页面
- **RSS订阅**：按标签订阅RSS源

---

*这个分类系统让博客内容更有组织性，用户可以快速找到感兴趣的主题。* 