# Blog System Documentation

This is a Markdown-based blog system for the personal homepage.

## Features

- **Markdown Support**: Write blog posts in Markdown format
- **Code Highlighting**: Automatic syntax highlighting for code blocks
- **Responsive Design**: Works on all devices
- **Easy to Maintain**: Simple file structure

## File Structure

```
blog/
├── index.html          # Blog homepage
├── post.html           # Blog post viewer
├── posts/              # Markdown files directory
│   ├── first-blog-post.md
│   └── deep-learning-radar.md
└── README.md           # This file
```

## How to Add New Blog Posts

### 1. Create Markdown File

Create a new `.md` file in the `posts/` directory:

```markdown
# Your Blog Post Title

Your content here...

## Section 1

Content...

## Section 2

Content...

## Code Example

```python
def example():
    print("Hello, World!")
```

## Conclusion

Your conclusion...
```

### 2. Update Blog Posts Data

Edit the `blogPosts` array in both `index.html` and `post.html`:

```javascript
const blogPosts = [
    // ... existing posts ...
    {
        id: 'your-post-id',
        title: 'Your Blog Post Title',
        date: '2024-01-25',
        tags: ['AI', 'Technology', 'Your Tag'],
        filename: 'your-post-filename.md',
        excerpt: 'Brief description of your post...'
    }
];
```

### 3. File Naming Convention

- Use kebab-case for filenames: `my-blog-post.md`
- Use descriptive IDs: `my-blog-post`
- Keep filenames and IDs consistent

## Markdown Features Supported

- **Headers**: `#`, `##`, `###`
- **Bold**: `**text**`
- **Italic**: `*text*`
- **Code**: `` `code` `` and ``` ``` ```
- **Lists**: `-` and `1.`
- **Links**: `[text](url)`
- **Images**: `![alt](url)`
- **Blockquotes**: `> text`

## Code Highlighting

The system supports syntax highlighting for many languages:

```python
def example():
    return "Python code"
```

```javascript
function example() {
    return "JavaScript code";
}
```

```bash
echo "Shell commands"
```

## Customization

### Styling

Edit the CSS in the HTML files to customize:
- Colors
- Fonts
- Layout
- Spacing

### Adding Features

To add new features:
1. Modify the JavaScript in `index.html` and `post.html`
2. Update the CSS as needed
3. Test thoroughly

## Tips for Writing Blog Posts

1. **Use Clear Headers**: Structure your content with proper headers
2. **Include Code Examples**: Use syntax highlighting for code
3. **Add Tags**: Use relevant tags for categorization
4. **Write Excerpts**: Provide brief descriptions for the blog list
5. **Use Images**: Include relevant images when helpful
6. **Keep it Readable**: Use proper formatting and spacing

## Troubleshooting

### Post Not Loading
- Check that the markdown file exists in `posts/`
- Verify the filename matches the `filename` field in the blog posts data
- Check browser console for errors

### Code Not Highlighting
- Ensure the language is specified in the code block
- Check that highlight.js is loading properly

### Styling Issues
- Clear browser cache
- Check CSS for conflicts
- Verify Bootstrap is loading

## Future Enhancements

Potential improvements:
- Search functionality
- Category filtering
- Comment system
- Social sharing
- RSS feed
- Pagination for many posts 