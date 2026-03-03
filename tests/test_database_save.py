#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from domain.documents import UserDocument, ArticleDocument
from crawler import CrawlerDispatcher

# Test database saving with LinkedIn crawler
def test_database_save():
    logger.info("Testing LinkedIn crawler database saving...")
    
    # Create test user
    user = UserDocument(
        id="test_user",
        full_name="Test User",
        email="test@example.com"
    )
    
    # Check current database state
    logger.info("Current articles in database:")
    try:
        articles = ArticleDocument.all()
        logger.info(f"Found {len(articles)} articles")
        for article in articles[-3:]:  # Show last 3
            logger.info(f"  - {article.link} ({article.platform})")
    except Exception as e:
        logger.error(f"Error reading database: {e}")
    
    # Test with a sample LinkedIn URL (may not work due to auth)
    test_url = "https://www.linkedin.com/posts/williamhgates_a-lot-of-people-are-asking-what-i-think-activity-7161234567890123456/"
    
    dispatcher = CrawlerDispatcher.build().register_linkedin()
    crawler = dispatcher.get_crawler(test_url)
    
    try:
        logger.info(f"Testing with: {test_url}")
        crawler.extract(link=test_url, user=user)
        logger.success("✓ Extraction completed")
        
        # Check if it was saved
        saved_article = ArticleDocument.find(link=test_url)
        if saved_article:
            logger.success("✅ Article successfully saved to database!")
            logger.info(f"  Title: {saved_article.content.get('Title', 'N/A')}")
            logger.info(f"  Platform: {saved_article.platform}")
        else:
            logger.warning("❌ Article was not saved (likely due to authentication)")
            
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
    
    finally:
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()

if __name__ == "__main__":
    test_database_save()
