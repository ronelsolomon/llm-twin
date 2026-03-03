#!/usr/bin/env python3
"""
Manual test script for Medium crawler with Cloudflare bypass
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from domain.documents import UserDocument
from mediumcrawl import MediumCrawler
from loguru import logger

def test_medium_crawler_manual():
    """Test the Medium crawler with manual Cloudflare bypass"""
    
    # Create a test user
    user = UserDocument(
        id="test_user_001",
        full_name="Ronel Solomon"
    )
    
    # Initialize the crawler
    crawler = MediumCrawler()
    
    # Test URL from the specified Medium profile
    url = "https://medium.com/@ronelsolomon"
    
    logger.info(f"Please manually bypass Cloudflare in the browser window that will open")
    logger.info(f"Once you see the Medium profile page, press Enter to continue...")
    
    try:
        # Navigate to the page
        crawler.driver.get(url)
        
        # Wait for manual intervention
        input("Press Enter after you've bypassed Cloudflare...")
        
        # Now try to extract content
        crawler.extract(url, user=user)
        logger.success(f"Successfully processed: {url}")
        
    except Exception as e:
        logger.error(f"Failed to process {url}: {e}")
    
    finally:
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()
    
    logger.info("Test completed")

if __name__ == "__main__":
    test_medium_crawler_manual()
