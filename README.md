# Markkhn's Personal Website

My personal website and blog built with static HTML and CSS.

## Features

- **Personal Profile**: Information about my background, education, and research interests
- **Blog System**: Static blog with Markdown support
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Clean and professional design using Bootstrap

## Blog Usage

### Writing New Blog Posts

1. **Create a Markdown file** in the root directory with `.md` extension
2. **Add YAML front matter** at the top of your markdown file:

```markdown
---
title: "Your Blog Post Title"
date: "2024-01-15"
excerpt: "A brief description of your post"
tags: "AI, Technology, Research"
---

# Your Blog Post Content

Your markdown content here...
```

3. **Run the blog manager** to convert your markdown to HTML:

```bash
python blog_manager.py your_post.md
```

Or to process all markdown files:

```bash
python blog_manager.py
```

### Blog Post Features

- **YAML Front Matter**: Add metadata like title, date, excerpt, and tags
- **Markdown Support**: Write in markdown with support for:
  - Headers (# ## ###)
  - Bold (**text**) and italic (*text*)
  - Code blocks (```) and inline code (`code`)
  - Lists (* and -)
  - Links and images

### File Structure

```
markkhn.github.io/
├── index.html              # Main homepage
├── blog/
│   ├── index.html         # Blog listing page
│   ├── posts.json         # Blog posts metadata
│   └── *.html            # Individual blog posts
├── assets/
│   └── images/
│       └── Profile_photo.png
├── blog_manager.py        # Python script to manage blog posts
└── *.md                  # Your markdown blog posts
```

## Development

### Local Development

1. Clone the repository
2. Open `index.html` in your browser
3. Write markdown files for blog posts
4. Run `python blog_manager.py` to generate HTML
5. Commit and push to GitHub

### Adding New Blog Posts

1. Create a new `.md` file in the root directory
2. Add YAML front matter with metadata
3. Write your content in markdown
4. Run `python blog_manager.py filename.md`
5. Commit the generated HTML file and updated `posts.json`

## Technologies Used

- **HTML5**: Semantic markup
- **CSS3**: Styling and responsive design
- **Bootstrap 5**: UI framework
- **JavaScript**: Dynamic content loading
- **Python**: Blog management script

## Contact

- Email: markkhn@sjtu.edu.cn
- GitHub: [markkhn](https://github.com/markkhn)