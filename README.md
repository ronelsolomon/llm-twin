# LLM Twin

A comprehensive web scraping and data collection system for various platforms including Medium, GitHub, and LinkedIn.

## Project Structure

```
llm-twin/
├── src/
│   ├── crawlers/           # Platform-specific crawlers
│   │   ├── base.py        # Abstract base crawler
│   │   ├── selenium_base.py # Selenium-based crawler with anti-detection
│   │   ├── medium.py      # Medium article scraper
│   │   ├── github.py      # GitHub repository scraper
│   │   └── linkedin.py    # LinkedIn content scraper
│   ├── domain/            # Data models and documents
│   │   └── documents.py   # NoSQL document classes
│   └── pipelines/         # Data processing pipelines
│       └── digital_data_etl.py
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── data/                  # Data storage (JSON files)
├── debug/                 # Debug files and HTML dumps
└── requirements.txt       # Python dependencies
```

## Features

### Crawlers
- **MediumCrawler**: Scrapes Medium articles and profiles with anti-detection measures
- **GithubCrawler**: Clones and analyzes GitHub repositories
- **LinkedInCrawler**: Extracts content from LinkedIn profiles and posts

### Base Infrastructure
- **BaseCrawler**: Abstract base class for all crawlers
- **BaseSeleniumCrawler**: Selenium-based crawler with anti-bot detection
- **NoSQLBaseDocument**: File-based NoSQL document system

### Data Models
- **UserDocument**: User profile information
- **ArticleDocument**: Article content and metadata
- **RepositoryDocument**: Repository information and analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-twin
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.domain.documents import UserDocument
from src.crawlers.medium import MediumCrawler

# Create a user
user = UserDocument(
    id="user_001",
    full_name="Ronel Solomon"
)

# Initialize crawler
crawler = MediumCrawler()

# Scrape Medium profile
crawler.extract("https://medium.com/@ronelsolomon", user=user)
```

### Running Tests

```bash
# Run individual test files
python tests/test_medium_crawler.py
python tests/test_github_crawler.py
python tests/test_linkedin_crawler.py
```

### Data Storage

All data is stored in JSON files under the `data/` directory:
- `data/users.json` - User profiles
- `data/articles.json` - Article content
- `data/repositories.json` - Repository information

## Anti-Detection Features

The Selenium-based crawlers include:
- Realistic user agents
- Automation detection bypass
- Custom browser options
- Rate limiting and delays

## Platform-Specific Notes

### Medium
- Handles Cloudflare protection
- Supports both profile and article pages
- Automatic duplicate detection

### GitHub
- Clones repositories temporarily
- Ignores specified file patterns
- Extracts repository metadata

### LinkedIn
- Requires manual authentication for full access
- Handles profile and post pages
- Anti-detection measures

## Development

### Adding New Crawlers

1. Inherit from `BaseCrawler` or `BaseSeleniumCrawler`
2. Implement the `extract` method
3. Define the appropriate document model
4. Add to `src/crawlers/__init__.py`

### Testing

Test files are organized in the `tests/` directory:
- Unit tests for individual components
- Integration tests for full workflows
- Manual testing scripts for debugging

## Troubleshooting

### Common Issues

1. **Cloudflare Blocks**: Use manual bypass or proxy rotation
2. **LinkedIn Authentication**: Manual login may be required
3. **Rate Limiting**: Add delays and respect robots.txt
4. **Selenium Issues**: Check ChromeDriver compatibility

### Debug Files

Debug HTML files are stored in `debug/` directory for troubleshooting scraping issues.

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

[Add your license information here]
