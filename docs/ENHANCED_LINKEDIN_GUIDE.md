# Enhanced LinkedIn Crawler - Complete Guide

## 🚀 New Features

The enhanced LinkedIn crawler now supports:
- ✅ **Google OAuth Login** - Automatic login with Google credentials
- ✅ **Manual Login Support** - Fallback to manual login
- ✅ **Feed Crawling** - Extract posts from user's feed
- ✅ **Batch Processing** - Save multiple posts at once
- ✅ **Database Integration** - All posts saved to `articles.json`

## 📋 How It Works

### 1. **Login Methods**
```python
from linkedin_enhanced import EnhancedLinkedInCrawler

# Option 1: Google OAuth (automatic)
crawler = EnhancedLinkedInCrawler(
    email="your_email@gmail.com", 
    password="your_password"
)

# Option 2: Manual login (recommended for testing)
crawler = EnhancedLinkedInCrawler()  # Will prompt for manual login
```

### 2. **Feed Crawling Process**
1. **Login** - Google OAuth or manual
2. **Navigate to Feed** - Goes to LinkedIn feed
3. **Scroll & Load** - Scrolls to load more posts
4. **Extract Posts** - Finds post containers and extracts data
5. **Save to Database** - Stores all posts as `ArticleDocument`

### 3. **Data Extracted per Post**
```json
{
  "content": {
    "Title": "Post by Author Name",
    "Subtitle": "By Author Name", 
    "Content": "Full post text content...",
    "language": "en"
  },
  "link": "https://www.linkedin.com/posts/...",
  "platform": "www.linkedin.com",
  "author_id": "user_id",
  "author_full_name": "User Name"
}
```

## 🛠️ Usage Examples

### **Example 1: Basic Feed Crawling**
```python
from linkedin_enhanced import EnhancedLinkedInCrawler
from domain.documents import UserDocument

# Create user
user = UserDocument(
    id="ronel_solomon",
    full_name="Ronel Solomon", 
    email="ronel@example.com"
)

# Initialize crawler (manual login)
crawler = EnhancedLinkedInCrawler()

# Crawl feed and save up to 20 posts
success = crawler.crawl_feed_and_save(user=user, max_posts=20)

if success:
    print("✅ Feed crawling completed!")
```

### **Example 2: Google OAuth Login**
```python
# With Google credentials
crawler = EnhancedLinkedInCrawler(
    email="your_email@gmail.com",
    password="your_app_password"  # Use app password for 2FA
)

# Crawl feed
crawler.crawl_feed_and_save(user=user, max_posts=10)
```

### **Example 3: Integration with Existing System**
```python
from crawler import CrawlerDispatcher

# The enhanced crawler is now integrated
dispatcher = CrawlerDispatcher.build().register_linkedin()

# For individual posts (backward compatible)
crawler = dispatcher.get_crawler("https://www.linkedin.com/posts/...")
crawler.extract(link="post_url", user=user)

# For feed crawling (new feature)
enhanced_crawler = EnhancedLinkedInCrawler()
enhanced_crawler.crawl_feed_and_save(user=user, max_posts=15)
```

## 🔧 Testing

### **Run the Test Script**
```bash
python test_enhanced_linkedin.py
```

This will:
1. Open LinkedIn login page
2. Wait for manual login (60 seconds)
3. Navigate to your feed
4. Extract up to 10 posts
5. Save them to database
6. Show results

### **Check Results**
```python
from domain.documents import ArticleDocument

# Get all LinkedIn articles
articles = ArticleDocument.all()
linkedin_articles = [a for a in articles if a.platform == "www.linkedin.com"]

print(f"Found {len(linkedin_articles)} LinkedIn articles:")
for article in linkedin_articles:
    print(f"- {article.content.get('Title', 'N/A')}")
```

## 🔐 Authentication Options

### **Google OAuth (Recommended for Production)**
- ✅ Fully automated
- ✅ Handles 2FA (with app passwords)
- ⚠️ Requires Google account linked to LinkedIn

### **Manual Login (Recommended for Testing)**
- ✅ Works with any login method
- ✅ Handles all edge cases
- ⚠️ Requires manual intervention

### **Cookie-based Authentication (Advanced)**
```python
# Load cookies from file
crawler = EnhancedLinkedInCrawler()
crawler.load_cookies("linkedin_cookies.json")
```

## 📊 Database Integration

### **Automatic Saving**
All extracted posts are automatically saved to:
- **File**: `/Users/ronel/Downloads/llm twin/data/articles.json`
- **Model**: `ArticleDocument`
- **Deduplication**: Checks for existing posts by URL

### **Query Saved Posts**
```python
# Get recent LinkedIn posts
from domain.documents import ArticleDocument

articles = ArticleDocument.all()
linkedin_posts = [a for a in articles if a.platform == "www.linkedin.com"]

# Sort by creation date
linkedin_posts.sort(key=lambda x: x.created_at, reverse=True)

for post in linkedin_posts[:5]:
    print(f"Title: {post.content.get('Title')}")
    print(f"Content: {post.content.get('Content', '')[:100]}...")
    print(f"Link: {post.link}")
    print("---")
```

## ⚠️ Important Notes

### **Rate Limiting**
- LinkedIn may rate-limit automated access
- Built-in delays between actions
- Recommended: Limit to 20-50 posts per session

### **2FA Handling**
- For Google OAuth: Use app-specific passwords
- For manual login: Complete 2FA in browser when prompted

### **Content Visibility**
- Only extracts publicly visible posts
- Private posts require proper authentication
- Some content may be restricted by LinkedIn

## 🚀 Quick Start

1. **Test with Manual Login**:
   ```bash
   python test_enhanced_linkedin.py
   ```

2. **Check Database**:
   ```python
   from domain.documents import ArticleDocument
   print(f"LinkedIn posts: {len([a for a in ArticleDocument.all() if a.platform == 'www.linkedin.com'])}")
   ```

3. **Integrate into Pipeline**:
   ```python
   # Already integrated in crawl_links.py
   dispatcher = CrawlerDispatcher.build().register_linkedin()
   ```

The enhanced LinkedIn crawler is now **production-ready** with full feed crawling capabilities!
