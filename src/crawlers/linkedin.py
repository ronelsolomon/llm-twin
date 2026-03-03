from bs4 import BeautifulSoup
import time
from loguru import logger
from src.domain.documents import ArticleDocument
from src.crawlers.selenium_base import BaseSeleniumCrawler


class LinkedInCrawler(BaseSeleniumCrawler):
    model = ArticleDocument
    
    def set_extra_driver_options(self, options) -> None:
        # LinkedIn often blocks automated browsers, so we need to set some options
        options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
    
    def scroll_page_safely(self, scroll_pause_time=2, max_scrolls=5):
        """Safely scroll down the page to load dynamic content with error handling"""
        try:
            # Try to get scroll height safely
            last_height = self.driver.execute_script("return document.body.scrollHeight")
        except:
            try:
                # Fallback to documentElement
                last_height = self.driver.execute_script("return document.documentElement.scrollHeight")
            except:
                logger.warning("Could not get page scroll height, skipping scroll")
                return
        
        for i in range(max_scrolls):
            try:
                # Scroll to bottom
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Wait to load page
                time.sleep(scroll_pause_time)
                
                # Calculate new scroll height and compare with last scroll height
                try:
                    new_height = self.driver.execute_script("return document.body.scrollHeight")
                except:
                    try:
                        new_height = self.driver.execute_script("return document.documentElement.scrollHeight")
                    except:
                        logger.warning("Could not get new scroll height, stopping scroll")
                        break
                
                if new_height == last_height:
                    break
                last_height = new_height
                
            except Exception as e:
                logger.warning(f"Error during scroll iteration {i+1}: {e}")
                break
        
        logger.info(f"Scrolled page {i+1} times")
    
    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")
            return
        
        logger.info(f"Starting scrapping LinkedIn article: {link}")
        
        try:
            self.driver.get(link)
            
            # Wait for page to load
            time.sleep(3)
            
            # Check for authentication wall
            current_url = self.driver.current_url
            if "authwall" in current_url or "signup" in current_url:
                logger.warning("LinkedIn authentication wall detected. Content may not be accessible without login.")
                logger.info("Current URL: " + current_url)
                
                # Check page title to confirm
                title = self.driver.title.lower()
                if any(word in title for word in ["sign up", "login", "join"]):
                    logger.warning("Redirected to login/signup page. Manual authentication required.")
            
            # Try to scroll safely
            try:
                self.scroll_page_safely()
            except Exception as scroll_error:
                logger.warning(f"Scrolling failed, continuing without scroll: {scroll_error}")
            
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            
            # Check if this is a profile page or an article
            if "/in/" in link and not any(x in link for x in ["/posts/", "/activity/"]):
                # This is a profile page, extract article links and process them
                self._process_profile_page(soup, link, **kwargs)
            else:
                # This is an article or post page, extract content
                self._process_article_page(soup, link, **kwargs)
            
            self.driver.close()
            logger.info(f"Successfully scraped and saved: {link}")
            
        except Exception as e:
            logger.error(f"Error scraping LinkedIn {link}: {e}")
            if hasattr(self, 'driver') and self.driver:
                self.driver.close()
            raise
    
    def _process_article_page(self, soup, link: str, **kwargs):
        """Extract content from a LinkedIn article or post page"""
        user = kwargs["user"]
        
        # Try to extract article content
        title_selectors = [
            "h1.break-words",
            ".share-update-card__title",
            "[data-test-id='post-title']",
            "h1"
        ]
        
        title = None
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                break
        
        # Try to extract content
        content_selectors = [
            ".share-update-card__content",
            "[data-test-id='post-content']",
            ".feed-shared-update-v2__description",
            ".feed-shared-text"
        ]
        
        content = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(strip=True)
                break
        
        if not title and not content:
            logger.warning(f"Could not extract title or content from LinkedIn page: {link}")
            return
        
        # Create article document
        article_data = {
            "Title": title or "LinkedIn Post",
            "Subtitle": "",
            "Content": content or "",
            "language": "en"
        }
        
        from urllib.parse import urlparse
        platform = urlparse(link).netloc
        
        instance = self.model(
            content=article_data,
            link=link,
            platform=platform,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()
        logger.info(f"Successfully saved LinkedIn article: {title or 'Untitled'}")
    
    def _process_profile_page(self, soup, link: str, **kwargs):
        """Extract article links from a LinkedIn profile page"""
        logger.info(f"Processing LinkedIn profile page: {link}")
        
        # Look for post/article links on the profile
        post_links = []
        
        # Common selectors for LinkedIn posts
        post_selectors = [
            "a[href*='/posts/']",
            "a[href*='/activity/']",
            ".feed-shared-update-v2 a"
        ]
        
        for selector in post_selectors:
            links = soup.select(selector)
            for link_elem in links:
                href = link_elem.get('href', '')
                if href and ('/posts/' in href or '/activity/' in href):
                    if href.startswith('/'):
                        href = 'https://www.linkedin.com' + href
                    post_links.append(href)
        
        logger.info(f"Found {len(post_links)} posts on profile")
        
        # Process each post link
        for post_link in post_links[:10]:  # Limit to avoid rate limiting
            try:
                self.extract(post_link, **kwargs)
            except Exception as e:
                logger.error(f"Error processing post {post_link}: {e}")
                continue
