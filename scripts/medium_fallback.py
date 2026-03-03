#!/usr/bin/env python3
"""
Fallback Medium scraper using requests for when Selenium fails due to Cloudflare
"""

import requests
import json
from bs4 import BeautifulSoup
from loguru import logger
from domain.documents import UserDocument, ArticleDocument

def scrape_medium_profile_fallback(profile_url: str, user: UserDocument):
    """
    Fallback method to scrape Medium profile using requests
    This may work better for some cases where Selenium is blocked
    """
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        logger.info(f"Attempting to fetch profile with requests: {profile_url}")
        response = requests.get(profile_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Save debug info
            with open("debug_fallback_page.html", "w", encoding="utf-8") as f:
                f.write(str(soup))
            
            # Try to find article links
            article_links = []
            
            # Look for various patterns that might indicate articles
            for link in soup.find_all("a", href=True):
                href = link.get("href")
                if href and "medium.com" in href and "@" in href:
                    if len(href) > len(profile_url):
                        full_link = href if href.startswith("http") else f"https://medium.com{href}"
                        if full_link not in article_links:
                            article_links.append(full_link)
            
            logger.info(f"Found {len(article_links)} potential article links")
            
            # Try to scrape a few articles
            for article_url in article_links[:3]:
                try:
                    scrape_article_fallback(article_url, user)
                except Exception as e:
                    logger.error(f"Error scraping article {article_url}: {e}")
                    
        else:
            logger.error(f"Failed to fetch profile: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error in fallback scraper: {e}")

def scrape_article_fallback(article_url: str, user: UserDocument):
    """Scrape a single Medium article using requests"""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    # Check if article already exists
    existing = ArticleDocument.find(link=article_url)
    if existing:
        logger.info(f"Article already exists: {article_url}")
        return
    
    try:
        logger.info(f"Scraping article: {article_url}")
        response = requests.get(article_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to extract title and content
            title_elem = soup.find("h1")
            title = title_elem.get_text().strip() if title_elem else "No title found"
            
            # Extract main content
            content_elem = soup.find("article") or soup.find("main") or soup.find("div", class_="article")
            content = content_elem.get_text().strip() if content_elem else soup.get_text()
            
            article_data = {
                "Title": title,
                "Content": content[:5000],  # Limit content length
            }
            
            article = ArticleDocument(
                platform="medium",
                content=article_data,
                link=article_url,
                author_id=user.id,
                author_full_name=user.full_name,
            )
            article.save()
            logger.success(f"Saved article: {title}")
            
        else:
            logger.error(f"Failed to fetch article: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error scraping article {article_url}: {e}")

if __name__ == "__main__":
    # Test the fallback
    user = UserDocument(id="test_user_001", full_name="Ronel Solomon")
    scrape_medium_profile_fallback("https://medium.com/@ronelsolomon", user)
