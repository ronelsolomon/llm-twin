#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from linkedin import LinkedInCrawler
from domain.documents import UserDocument

# Guide for authenticated LinkedIn crawling
def linkedin_with_auth_guide():
    """
    LinkedIn requires authentication to view most profile content and posts.
    Here are the options to make the LinkedIn crawler work effectively:
    """
    
    logger.info("LinkedIn Authentication Options:")
    logger.info("")
    logger.info("1. Manual Login (Recommended for testing):")
    logger.info("   - The crawler will open a visible browser")
    logger.info("   - Manually log into LinkedIn when the browser opens")
    logger.info("   - The crawler will then have access to your profile")
    logger.info("")
    logger.info("2. Cookie-based Authentication:")
    logger.info("   - Export your LinkedIn cookies and load them")
    logger.info("   - More complex but allows automated access")
    logger.info("")
    logger.info("3. LinkedIn API (Enterprise):")
    logger.info("   - Use official LinkedIn API for production use")
    logger.info("   - Requires LinkedIn developer account and approval")
    logger.info("")
    
    # Demo of manual login approach
    logger.info("Testing manual login approach...")
    
    user = UserDocument(
        id="ronel_solomon",
        full_name="Ronel Solomon",
        email="ronel@example.com"
    )
    
    # Create crawler with visible browser
    crawler = LinkedInCrawler()
    
    profile_url = "https://www.linkedin.com/in/ronel-solomon/"
    
    try:
        logger.info("Opening browser... Please manually log into LinkedIn.")
        logger.info("You have 30 seconds to log in before the script continues...")
        
        crawler.driver.get(profile_url)
        
        # Wait for user to manually log in
        import time
        time.sleep(30)
        
        # Check if we're past the auth wall
        current_url = crawler.driver.current_url
        if "authwall" not in current_url and "signup" not in current_url:
            logger.success("✓ Successfully past authentication wall!")
            
            # Now try to extract content
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(crawler.driver.page_source, "html.parser")
            
            # Look for posts
            post_links = soup.select("a[href*='/posts/'], a[href*='/activity/']")
            logger.info(f"Found {len(post_links)} posts after authentication")
            
            # Extract profile info
            title_elem = soup.find("h1")
            if title_elem:
                logger.info(f"Profile name: {title_elem.get_text(strip=True)}")
            
            # Try to extract some posts
            if post_links:
                logger.info("Processing first few posts...")
                for i, link in enumerate(post_links[:3]):
                    href = link.get('href', '')
                    if href.startswith('/'):
                        href = 'https://www.linkedin.com' + href
                    
                    try:
                        logger.info(f"Processing post {i+1}: {href}")
                        crawler.extract(href, user=user)
                    except Exception as e:
                        logger.error(f"Error processing post {i+1}: {e}")
            
        else:
            logger.warning("Still at authentication wall. Please log in and try again.")
            logger.info(f"Current URL: {current_url}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
    
    finally:
        # Keep browser open for inspection
        logger.info("Browser will remain open for 60 seconds for inspection...")
        time.sleep(60)
        
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()

if __name__ == "__main__":
    linkedin_with_auth_guide()
