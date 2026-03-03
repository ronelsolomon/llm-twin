#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from domain.documents import UserDocument, ArticleDocument
from linkedin_enhanced import EnhancedLinkedInCrawler

# Test the enhanced LinkedIn crawler with feed crawling
def test_enhanced_linkedin():
    logger.info("Testing Enhanced LinkedIn Crawler with Feed Crawling...")
    
    # Create user
    user = UserDocument(
        id="ronel_solomon",
        full_name="Ronel Solomon",
        email="ronel@example.com"
    )
    
    # Initialize enhanced crawler
    # Option 1: With credentials (for Google login)
    # crawler = EnhancedLinkedInCrawler(email="your_email@gmail.com", password="your_password")
    
    # Option 2: Manual login (recommended for testing)
    crawler = EnhancedLinkedInCrawler()
    
    try:
        # Crawl feed and save posts
        success = crawler.crawl_feed_and_save(user=user, max_posts=10)
        
        if success:
            logger.success("✓ Feed crawling completed successfully!")
            
            # Check what was saved
            articles = ArticleDocument.all()
            linkedin_articles = [a for a in articles if a.platform == "www.linkedin.com"]
            
            logger.info(f"Total LinkedIn articles in database: {len(linkedin_articles)}")
            
            # Show latest saved articles
            for i, article in enumerate(linkedin_articles[-3:]):
                logger.info(f"Article {i+1}:")
                logger.info(f"  Title: {article.content.get('Title', 'N/A')}")
                logger.info(f"  Link: {article.link}")
                logger.info(f"  Author: {article.author_full_name}")
        else:
            logger.error("❌ Feed crawling failed")
    
    except Exception as e:
        logger.error(f"Error during enhanced crawling: {e}")
    
    finally:
        # Keep browser open for inspection
        logger.info("Browser will remain open for 30 seconds for inspection...")
        import time
        time.sleep(30)
        
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()

if __name__ == "__main__":
    test_enhanced_linkedin()
