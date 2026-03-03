#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from domain.documents import UserDocument
from crawler import CrawlerDispatcher

# Test crawling your LinkedIn profile
def test_my_linkedin_profile():
    logger.info("Testing LinkedIn crawler with your profile...")
    
    # Create user with your information
    user = UserDocument(
        id="ronel_solomon",
        full_name="Ronel Solomon",
        email="ronel@example.com"  # Replace with your email if needed
    )
    
    # Build dispatcher with all crawlers
    dispatcher = CrawlerDispatcher.build().register_linkedin().register_medium().register_github()
    
    # Your LinkedIn profile URL
    profile_url = "https://www.linkedin.com/in/ronel-solomon/"
    
    try:
        logger.info(f"Starting to crawl your LinkedIn profile: {profile_url}")
        
        # Get the appropriate crawler
        crawler = dispatcher.get_crawler(profile_url)
        logger.info(f"Selected crawler: {type(crawler).__name__}")
        
        # Extract content
        crawler.extract(link=profile_url, user=user)
        
        logger.success("✓ Successfully crawled your LinkedIn profile!")
        
    except Exception as e:
        logger.error(f"Error crawling your LinkedIn profile: {e}")
        logger.info("This might be due to LinkedIn's privacy settings or anti-bot measures")
        logger.info("You may need to log in to LinkedIn in the browser that Selenium opens")
    
    finally:
        # Clean up any open browsers
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()

if __name__ == "__main__":
    test_my_linkedin_profile()
