#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bs4 import BeautifulSoup
import time
from loguru import logger
from linkedin import LinkedInCrawler
from domain.documents import UserDocument

# Debug what LinkedIn page structure we're seeing
def debug_linkedin_page():
    logger.info("Debugging LinkedIn page structure...")
    
    # Create user
    user = UserDocument(
        id="ronel_solomon",
        full_name="Ronel Solomon",
        email="ronel@example.com"
    )
    
    # Initialize crawler
    crawler = LinkedInCrawler()
    
    profile_url = "https://www.linkedin.com/in/ronel-solomon/"
    
    try:
        logger.info(f"Loading page: {profile_url}")
        crawler.driver.get(profile_url)
        time.sleep(5)  # Wait longer for page to load
        
        # Get page source
        page_source = crawler.driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        
        # Save page source for inspection
        with open("linkedin_profile_debug.html", "w", encoding="utf-8") as f:
            f.write(page_source)
        logger.info("Page source saved to linkedin_profile_debug.html")
        
        # Check page title
        title = soup.find("title")
        if title:
            logger.info(f"Page title: {title.get_text()}")
        
        # Check for login/signin indicators
        login_indicators = [
            "sign in", "login", "join now", "sign up", 
            "authentication", "password", "email"
        ]
        
        page_text = soup.get_text().lower()
        for indicator in login_indicators:
            if indicator in page_text:
                logger.warning(f"Found login indicator: {indicator}")
        
        # Look for any potential post/activity containers
        potential_selectors = [
            {"name": "Posts container", "selector": "[data-test-id='posts-container']"},
            {"name": "Activity feed", "selector": ".feed-shared-update-v2"},
            {"name": "Post links", "selector": "a[href*='/posts/']"},
            {"name": "Activity links", "selector": "a[href*='/activity/']"},
            {"name": "Any links with /posts/", "selector": "a[href*='/posts/']"},
            {"name": "Any links with /activity/", "selector": "a[href*='/activity/']"},
        ]
        
        found_elements = []
        for item in potential_selectors:
            elements = soup.select(item["selector"])
            if elements:
                found_elements.append(f"{item['name']}: {len(elements)} found")
                logger.info(f"✓ {item['name']}: {len(elements)} elements found")
            else:
                logger.info(f"✗ {item['name']}: 0 elements found")
        
        # Look for all links that might be posts
        all_links = soup.find_all("a", href=True)
        post_links = [link for link in all_links if '/posts/' in link.get('href', '') or '/activity/' in link.get('href', '')]
        
        if post_links:
            logger.info(f"Found {len(post_links)} potential post links:")
            for i, link in enumerate(post_links[:5]):  # Show first 5
                href = link.get('href', '')
                text = link.get_text(strip=True)[:50]  # First 50 chars
                logger.info(f"  {i+1}. {href} - {text}")
        else:
            logger.warning("No post links found")
        
        # Check if we're being redirected to login
        if "login" in crawler.driver.current_url.lower():
            logger.warning("Redirected to login page")
        
        logger.info(f"Current URL: {crawler.driver.current_url}")
        
    except Exception as e:
        logger.error(f"Error during debug: {e}")
    
    finally:
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()

if __name__ == "__main__":
    debug_linkedin_page()
