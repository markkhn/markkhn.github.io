#!/usr/bin/env python3
"""
Blog Manager for Markkhn's GitHub Pages
Converts Markdown files to HTML and updates posts.json
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path

class BlogManager:
    def __init__(self, blog_dir="blog", posts_file="blog/posts.json"):
        self.blog_dir = Path(blog_dir)
        self.posts_file = Path(posts_file)
        self.posts = self.load_posts()
    
    def load_posts(self):
        """Load existing posts from JSON file"""
        if self.posts_file.exists():
            with open(self.posts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_posts(self):
        """Save posts to JSON file"""
        with open(self.posts_file, 'w', encoding='utf-8') as f:
            json.dump(self.posts, f, indent=4, ensure_ascii=False)
    
    def extract_metadata(self, content):
        """Extract metadata from markdown content"""
        metadata = {}
        lines = content.split('\n')
        
        # Look for YAML front matter
        if lines and lines[0].strip() == '---':
            i = 1
            while i < len(lines) and lines[i].strip() != '---':
                line = lines[i].strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                i += 1
            content = '\n'.join(lines[i+1:])
        
        return metadata, content
    
    def markdown_to_html(self, markdown_content):
        """Convert markdown to HTML (basic conversion)"""
        html = markdown_content
        
        # Headers (process in reverse order to avoid conflicts)
        html = re.sub(r'^### (.*$)', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*$)', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*$)', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # Bold and italic
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
        # Code blocks
        html = re.sub(r'```(\w+)?\n(.*?)```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
        html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)
        
        # Lists
        html = re.sub(r'^\* (.*$)', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^- (.*$)', r'<li>\1</li>', html, flags=re.MULTILINE)
        
        # Wrap lists in ul tags
        html = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', html, flags=re.DOTALL)
        html = re.sub(r'</ul>\s*<ul>', '', html)
        
        # Paragraphs (only for text not already in tags)
        lines = html.split('\n')
        processed_lines = []
        in_tag = False
        
        for line in lines:
            if line.strip() == '':
                processed_lines.append('')
                continue
                
            if re.match(r'^<(h[1-6]|p|ul|ol|li|pre|code)', line.strip()):
                processed_lines.append(line)
            else:
                processed_lines.append(f'<p>{line}</p>')
        
        html = '\n'.join(processed_lines)
        
        # Clean up empty paragraphs and extra whitespace
        html = re.sub(r'<p></p>', '', html)
        html = re.sub(r'\n\s*\n', '\n', html)
        
        return html
    
    def create_html_template(self, title, date, content, tags=None):
        """Create HTML template for blog post"""
        if tags is None:
            tags = []
        
        tags_html = ''.join([f'<span class="ms-3">🏷️ {tag}</span>' for tag in tags])
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Markkhn's Blog</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Custom CSS -->
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }}
        .blog-content {{
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section-title {{
            color: #2c3e50;
            border-bottom: 2px solid #007bff;
            padding-bottom: 0.5em;
        }}
        .back-link {{
            color: #007bff;
            text-decoration: none;
        }}
        .back-link:hover {{
            text-decoration: underline;
        }}
        .blog-meta {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .blog-content h2 {{
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }}
        .blog-content p {{
            line-height: 1.6;
            margin-bottom: 1rem;
        }}
        .blog-content code {{
            background-color: #f8f9fa;
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
        }}
        .blog-content pre {{
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
        }}
        .blog-content ul, .blog-content ol {{
            margin-bottom: 1rem;
        }}
        .blog-content li {{
            margin-bottom: 0.5rem;
        }}
    </style>
</head>
<body class="py-4">
    <div class="container">
        <!-- Header -->
        <header class="mb-4">
            <a href="index.html" class="back-link mb-3 d-inline-block">← Back to Blog</a>
        </header>

        <!-- Blog Content -->
        <article class="blog-content p-4">
            <header class="mb-4">
                <h1 class="section-title">{title}</h1>
                <div class="blog-meta">
                    <span>📅 {date}</span>
                    {tags_html}
                </div>
            </header>

            <div class="blog-content">
                {content}
                
                <hr class="my-4">
                <p class="text-muted">
                    <small>
                        Thanks for reading! If you found this post interesting, 
                        feel free to share it or leave a comment below.
                    </small>
                </p>
            </div>
        </article>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''
    
    def process_markdown_file(self, markdown_file):
        """Process a markdown file and convert it to HTML"""
        markdown_path = Path(markdown_file)
        
        if not markdown_path.exists():
            print(f"Error: File {markdown_file} not found")
            return
        
        # Read markdown content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata
        metadata, markdown_content = self.extract_metadata(content)
        
        # Get title from metadata or filename
        title = metadata.get('title', markdown_path.stem.replace('_', ' ').title())
        date = metadata.get('date', datetime.now().strftime('%Y-%m-%d'))
        tags = metadata.get('tags', '').split(',') if metadata.get('tags') else []
        tags = [tag.strip().strip('"') for tag in tags if tag.strip()]
        
        # Clean up title and date (remove quotes if present)
        title = title.strip('"')
        date = date.strip('"')
        
        # Convert markdown to HTML
        html_content = self.markdown_to_html(markdown_content)
        
        # Create HTML file
        html_filename = f"{markdown_path.stem}.html"
        html_path = self.blog_dir / html_filename
        
        # Generate HTML template
        html_template = self.create_html_template(title, date, html_content, tags)
        
        # Write HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        # Update posts.json
        excerpt = metadata.get('excerpt', markdown_content[:200] + '...')
        excerpt = excerpt.strip('"') if excerpt else markdown_content[:200] + '...'
        
        post_info = {
            "title": title,
            "date": date,
            "excerpt": excerpt,
            "filename": html_filename,
            "tags": tags
        }
        
        # Remove existing post with same filename if exists
        self.posts = [post for post in self.posts if post.get('filename') != html_filename]
        
        # Add new post at the beginning
        self.posts.insert(0, post_info)
        
        # Save posts
        self.save_posts()
        
        print(f"✅ Processed {markdown_file} -> {html_filename}")
        print(f"   Title: {title}")
        print(f"   Date: {date}")
        print(f"   Tags: {', '.join(tags) if tags else 'None'}")
    
    def process_all_markdown_files(self):
        """Process all markdown files in the current directory"""
        markdown_files = list(Path('.').glob('*.md'))
        
        if not markdown_files:
            print("No markdown files found in current directory")
            return
        
        print(f"Found {len(markdown_files)} markdown file(s):")
        for file in markdown_files:
            print(f"  - {file}")
        
        print("\nProcessing files...")
        for file in markdown_files:
            self.process_markdown_file(file)
        
        print(f"\n✅ All files processed! Blog updated successfully.")

def main():
    """Main function"""
    manager = BlogManager()
    
    import sys
    if len(sys.argv) > 1:
        # Process specific file
        markdown_file = sys.argv[1]
        manager.process_markdown_file(markdown_file)
    else:
        # Process all markdown files
        manager.process_all_markdown_files()

if __name__ == "__main__":
    main() 