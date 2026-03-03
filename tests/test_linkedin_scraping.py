#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from domain.documents import UserDocument
from linkedin import LinkedInCrawler

# Test actual scraping with LinkedIn crawler
def test_linkedin_scraping():
    logger.info("Testing LinkedIn scraping...")
    
    # Create a test user
    test_user = UserDocument(
        id="test_user",
        full_name="Test User",
        email="test@example.com"
    )
    
    # Initialize LinkedIn crawler
    crawler = LinkedInCrawler()
    
    # Test with a public LinkedIn post (you can replace with a real URL)
    test_url = "https://www.linkedin.com/posts/williamhgates_a-lot-of-people-are-asking-what-i-think-activity-7161234567890123456/"
    
    try:
        logger.info(f"Testing scraping of: {test_url}")
        crawler.extract(link=test_url, user=test_user)
        logger.success("✓ LinkedIn scraping test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        logger.info("This might be due to LinkedIn's anti-bot measures or the test URL not being accessible")
    
    finally:
        # Clean up
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()

if __name__ == "__main__":
    test_linkedin_scraping()
