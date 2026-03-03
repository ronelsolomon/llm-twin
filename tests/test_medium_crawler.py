#!/usr/bin/env python3
"""
Test script for Medium crawler
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.documents import UserDocument
from src.crawlers.medium import MediumCrawler
from loguru import logger
import time

def test_medium_crawler():
    """Test the Medium crawler with a specific Medium profile"""
    
    # Create a test user
    user = UserDocument(
        id="test_user_001",
        full_name="Ronel Solomon"
    )
    
    # Initialize the crawler
    crawler = MediumCrawler()
    
    # Test URL from the specified Medium profile
    test_urls = [
        "https://medium.com/@ronelsolomon",
        # You can add specific article URLs here if needed
    ]
    
    for url in test_urls:
        logger.info(f"Testing URL: {url}")
        try:
            crawler.extract(url, user=user)
            logger.success(f"Successfully processed: {url}")
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            continue
    
    logger.info("Test completed")

if __name__ == "__main__":
    test_medium_crawler()
