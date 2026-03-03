#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from domain.documents import UserDocument, ArticleDocument
from linkedin_enhanced import EnhancedLinkedInCrawler

# Demo script for LinkedIn feed crawling
def linkedin_feed_demo():
    """
    Demo: Crawl LinkedIn feed and save posts to database
    
    Instructions:
    1. Run this script
    2. A browser window will open to LinkedIn login
    3. Log in manually (you have 60 seconds)
    4. The crawler will then automatically:
       - Navigate to your feed
       - Scroll to load posts
       - Extract post content
       - Save everything to the database
    """
    
    logger.info("🚀 LinkedIn Feed Crawler Demo")
    logger.info("=" * 50)
    
    # Create user
    user = UserDocument(
        id="ronel_solomon",
        full_name="Ronel Solomon",
        email="ronel@example.com"
    )
    
    # Show current database state
    logger.info("📊 Current Database State:")
    try:
        all_articles = ArticleDocument.all()
        linkedin_articles = [a for a in all_articles if a.platform == "www.linkedin.com"]
        logger.info(f"   Total articles: {len(all_articles)}")
        logger.info(f"   LinkedIn articles: {len(linkedin_articles)}")
    except Exception as e:
        logger.error(f"Error reading database: {e}")
    
    logger.info("\n🔐 Starting LinkedIn Feed Crawler...")
    logger.info("⏱️  You have 60 seconds to log in when the browser opens...")
    
    # Initialize crawler
    crawler = EnhancedLinkedInCrawler()
    
    try:
        # Crawl feed and save posts
        success = crawler.crawl_feed_and_save(user=user, max_posts=15)
        
        if success:
            logger.success("\n✅ SUCCESS! Feed crawling completed!")
            
            # Show results
            logger.info("\n📈 Results:")
            all_articles = ArticleDocument.all()
            new_linkedin_articles = [a for a in all_articles if a.platform == "www.linkedin.com"]
            
            logger.info(f"   Total LinkedIn articles now: {len(new_linkedin_articles)}")
            
            # Show latest articles
            logger.info("\n📝 Latest LinkedIn Posts:")
            for i, article in enumerate(new_linkedin_articles[-3:]):
                title = article.content.get('Title', 'No Title')
                content = article.content.get('Content', 'No Content')
                logger.info(f"   {i+1}. {title}")
                logger.info(f"      Content: {content[:100]}...")
                logger.info(f"      Link: {article.link}")
                logger.info("")
            
            logger.success("🎉 All posts have been saved to data/articles.json!")
            
        else:
            logger.error("\n❌ Feed crawling failed")
            logger.info("💡 Tips:")
            logger.info("   - Make sure to complete the login within 60 seconds")
            logger.info("   - Check if you're logged into LinkedIn successfully")
            logger.info("   - Try running the script again")
    
    except Exception as e:
        logger.error(f"\n💥 Error during crawling: {e}")
        logger.info("💡 This might be due to:")
        logger.info("   - Network issues")
        logger.info("   - LinkedIn rate limiting")
        logger.info("   - Page structure changes")
    
    finally:
        # Keep browser open for a bit longer for inspection
        logger.info("\n🔍 Browser will remain open for 10 seconds...")
        import time
        time.sleep(10)
        
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()
        
        logger.info("✨ Demo completed!")

if __name__ == "__main__":
    linkedin_feed_demo()
