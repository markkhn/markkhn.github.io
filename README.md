# Markkhn's Personal Homepage

A personal homepage with a Markdown-based blog system.

## 🚀 Quick Start

### View the Site
```bash
# Start local server
python3 -m http.server 8000

# Visit
http://localhost:8000
```

### Add New Blog Post
```bash
cd blog
python add_post.py add
```

## 📁 Project Structure

```
markkhn.github.io/
├── index.html              # Main homepage
├── assets/
│   └── images/
│       └── Profile_photo.png
└── blog/                   # Blog system
    ├── index.html          # Blog homepage
    ├── post.html           # Blog post viewer
    ├── posts.json          # Blog posts data
    ├── add_post.py         # Post management script
    ├── QUICK_START.md      # Quick start guide
    ├── README.md           # Blog documentation
    └── posts/              # Markdown files
        ├── first-blog-post.md
        ├── deep-learning-radar.md
        ├── ai-in-healthcare.md
        └── ai-music-generation.md
```

## ✨ Features

- **Personal Profile**: Professional homepage with education and research info
- **Markdown Blog**: Write posts in Markdown with code highlighting
- **Tag Categories**: Filter posts by tags (AI, Technology, Research, etc.)
- **Responsive Design**: Works on all devices
- **Easy Management**: Simple script to add new posts

## 🛠️ Blog Management

### List All Posts
```bash
cd blog
python add_post.py list
```

### Add New Post
```bash
cd blog
python add_post.py add
```

### Manual Setup
1. Create `.md` file in `blog/posts/`
2. Add entry to `blog/posts.json`
3. Update tags in `blog/index.html` if needed

## 📚 Documentation

- [Blog Quick Start](blog/QUICK_START.md) - How to add posts
- [Blog Documentation](blog/README.md) - Complete blog system guide

## 🎯 Current Posts

- **My First Blog Post** - Introduction to AI and research
- **Deep Learning in Radar Perception** - Technical implementation
- **AI in Healthcare** - Current trends and applications
- **AI in Music Generation** - From theory to practice

## 🔧 Technologies

- HTML5 + CSS3 + JavaScript
- Bootstrap 5 (UI framework)
- Marked.js (Markdown parser)
- Highlight.js (Code syntax highlighting)

---

*Built with ❤️ for sharing knowledge and research.*