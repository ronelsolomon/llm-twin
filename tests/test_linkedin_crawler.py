#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from domain.documents import UserDocument
from crawler import CrawlerDispatcher

# Test the LinkedIn crawler
def test_linkedin_crawler():
    logger.info("Testing LinkedIn crawler...")
    
    # Create a test user
    test_user = UserDocument(
        id="test_user",
        full_name="Test User",
        email="test@example.com"
    )
    
    # Build dispatcher with LinkedIn crawler
    dispatcher = CrawlerDispatcher.build().register_linkedin()
    
    # Test URLs
    test_urls = [
        "https://www.linkedin.com/posts/satyanadella_this-is-a-big-day-for-microsoft-and-ai-activity-7161234567890123456/",
        "https://www.linkedin.com/in/test-profile/"
    ]
    
    for url in test_urls:
        try:
            logger.info(f"Testing URL: {url}")
            crawler = dispatcher.get_crawler(url)
            logger.info(f"Found crawler: {type(crawler).__name__}")
            
            # Don't actually scrape for now, just test the crawler selection
            if type(crawler).__name__ == "LinkedInCrawler":
                logger.success("✓ LinkedIn crawler correctly selected for LinkedIn URL")
            else:
                logger.warning(f"✗ Expected LinkedInCrawler, got {type(crawler).__name__}")
                
        except Exception as e:
            logger.error(f"Error testing URL {url}: {e}")

if __name__ == "__main__":
    test_linkedin_crawler()
