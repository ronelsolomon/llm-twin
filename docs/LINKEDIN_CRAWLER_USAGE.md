# LinkedIn Crawler Usage

## Overview
The LinkedIn crawler is now integrated into your crawling system and can extract content from LinkedIn posts and profiles.

## How It Works
- **Post URLs**: Extracts content from individual LinkedIn posts/articles
- **Profile URLs**: Discovers and processes posts from LinkedIn profiles
- **Anti-Detection**: Uses Selenium with custom options to avoid LinkedIn's bot detection

## Usage Examples

### 1. Using the Crawler Dispatcher
```python
from crawler import CrawlerDispatcher
from domain.documents import UserDocument

# Create user
user = UserDocument(id="user123", full_name="John Doe", email="john@example.com")

# Build dispatcher with LinkedIn crawler
dispatcher = CrawlerDispatcher.build().register_linkedin()

# Crawl LinkedIn URLs
links = [
    "https://www.linkedin.com/posts/someuser_post-activity-123456789/",
    "https://www.linkedin.com/in/username/"
]

for link in links:
    crawler = dispatcher.get_crawler(link)
    crawler.extract(link=link, user=user)
```

### 2. Direct LinkedIn Crawler Usage
```python
from linkedin import LinkedInCrawler
from domain.documents import UserDocument

user = UserDocument(id="user123", full_name="John Doe", email="john@example.com")
crawler = LinkedInCrawler()

# Extract from a specific post
crawler.extract(link="https://www.linkedin.com/posts/...", user=user)
```

## URL Patterns Supported
- **Posts**: `https://www.linkedin.com/posts/username_post-activity-123456789/`
- **Profiles**: `https://www.linkedin.com/in/username/`

## Important Notes
1. **Rate Limiting**: LinkedIn may rate-limit or block automated access
2. **Authentication**: Some content may require login to access
3. **Dynamic Content**: The crawler uses Selenium to handle JavaScript-rendered content
4. **Anti-Bot Measures**: The crawler includes options to appear more like a real browser

## Testing
Run the test scripts to verify functionality:
```bash
python test_linkedin_crawler.py  # Test crawler selection
python test_linkedin_scraping.py  # Test actual scraping
```

## Troubleshooting
- If you encounter access issues, try running with a visible browser (headless mode disabled)
- Some LinkedIn content may require authentication
- The crawler may need adjustments if LinkedIn changes their page structure
