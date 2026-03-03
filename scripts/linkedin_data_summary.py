#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from loguru import logger

def show_linkedin_data_summary():
    """
    Show summary of LinkedIn data created from Profile.pdf
    """
    logger.info("📊 LinkedIn Data Summary (from Profile.pdf)")
    logger.info("=" * 50)
    
    try:
        # Read articles from database
        with open('data/articles.json', 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        # Filter LinkedIn posts
        linkedin_posts = [a for a in articles if a.get('platform') == 'www.linkedin.com']
        
        logger.info(f"🔢 Total LinkedIn posts: {len(linkedin_posts)}")
        logger.info(f"📄 Total articles in database: {len(articles)}")
        
        # Show categories of posts
        categories = {}
        for post in linkedin_posts:
            title = post.get('content', {}).get('Title', 'Unknown')
            if 'University' in title:
                categories.setdefault('Education', 0)
                categories['Education'] += 1
            elif any(company in title for company in ['Cisco', 'KPMG', 'Colt', 'Visa']):
                categories.setdefault('Work Experience', 0)
                categories['Work Experience'] += 1
            elif 'Skills' in title or 'ML Engineer' in title:
                categories.setdefault('Professional Info', 0)
                categories['Professional Info'] += 1
            else:
                categories.setdefault('Other', 0)
                categories['Other'] += 1
        
        logger.info("\n📂 Post Categories:")
        for category, count in categories.items():
            logger.info(f"   {category}: {count}")
        
        # Show detailed posts
        logger.info("\n📝 Detailed Posts:")
        for i, post in enumerate(linkedin_posts, 1):
            title = post.get('content', {}).get('Title', 'No Title')
            content = post.get('content', {}).get('Content', 'No Content')
            link = post.get('link', 'No Link')
            
            logger.info(f"\n{i}. {title}")
            logger.info(f"   📎 {link}")
            logger.info(f"   📄 {content[:100]}...")
        
        # Show markdown file info
        logger.info("\n📄 Generated Files:")
        logger.info("   ✅ data/Profile.pdf (original)")
        logger.info("   ✅ data/Profile.md (markdown conversion)")
        logger.info("   ✅ data/articles.json (database with LinkedIn posts)")
        
        logger.success("\n🎉 LinkedIn data successfully created from Profile.pdf!")
        
    except Exception as e:
        logger.error(f"❌ Error showing summary: {e}")

if __name__ == "__main__":
    show_linkedin_data_summary()
