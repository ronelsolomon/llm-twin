from bs4 import BeautifulSoup
import time
from loguru import logger
from domain.documents import ArticleDocument
from application.crawlers.selenium_base import BaseSeleniumCrawler
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


class EnhancedLinkedInCrawler(BaseSeleniumCrawler):
    model = ArticleDocument
    
    def __init__(self, email=None, password=None):
        super().__init__()
        self.email = email
        self.password = password
        self.logged_in = False
    
    def set_extra_driver_options(self, options) -> None:
        # LinkedIn often blocks automated browsers, so we need to set some options
        options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # Keep browser visible for login
        options.add_argument("--start-maximized")
    
    def login_with_google(self, email, password):
        """Log in to LinkedIn using Google OAuth"""
        try:
            logger.info("Starting LinkedIn login with Google...")
            
            # Go to LinkedIn login page
            self.driver.get("https://www.linkedin.com/login")
            time.sleep(3)
            
            # Click "Sign in with Google" button
            google_signin_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'google') or contains(text(), 'Google')]"))
            )
            google_signin_button.click()
            time.sleep(3)
            
            # Switch to Google OAuth window/tab if opened
            if len(self.driver.window_handles) > 1:
                self.driver.switch_to.window(self.driver.window_handles[-1])
            
            # Enter email
            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "identifier"))
            )
            email_field.clear()
            email_field.send_keys(email)
            email_field.send_keys(Keys.RETURN)
            time.sleep(2)
            
            # Enter password
            password_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "password"))
            )
            password_field.clear()
            password_field.send_keys(password)
            password_field.send_keys(Keys.RETURN)
            time.sleep(3)
            
            # Handle 2FA if present (wait for user)
            logger.info("If 2FA is enabled, please complete it manually in the browser...")
            time.sleep(10)
            
            # Switch back to LinkedIn if needed
            if len(self.driver.window_handles) > 1:
                self.driver.switch_to.window(self.driver.window_handles[0])
            
            # Wait for login to complete
            time.sleep(5)
            
            # Verify login success
            if "feed" in self.driver.current_url or "in/" in self.driver.current_url:
                self.logged_in = True
                logger.success("✓ Successfully logged into LinkedIn with Google!")
                return True
            else:
                logger.warning("Login may have failed - checking current page...")
                logger.info(f"Current URL: {self.driver.current_url}")
                return False
                
        except Exception as e:
            logger.error(f"Error during Google login: {e}")
            return False
    
    def manual_login_prompt(self):
        """Prompt user to manually log in"""
        logger.info("Please log in manually in the browser window...")
        logger.info("You have 60 seconds to complete the login process.")
        
        self.driver.get("https://www.linkedin.com/login")
        time.sleep(60)
        
        # Check if logged in
        if "feed" in self.driver.current_url or "in/" in self.driver.current_url:
            self.logged_in = True
            logger.success("✓ Manual login successful!")
            return True
        else:
            logger.warning("Manual login may have failed")
            return False
    
    def navigate_to_feed(self):
        """Navigate to the user's feed"""
        try:
            logger.info("Navigating to LinkedIn feed...")
            
            # Try multiple feed URLs
            feed_urls = [
                "https://www.linkedin.com/feed/",
                "https://www.linkedin.com/"
            ]
            
            for url in feed_urls:
                self.driver.get(url)
                time.sleep(3)
                
                if "feed" in self.driver.current_url:
                    logger.success("✓ Successfully navigated to feed!")
                    return True
            
            logger.warning("Could not navigate to feed")
            return False
            
        except Exception as e:
            logger.error(f"Error navigating to feed: {e}")
            return False
    
    def scroll_feed(self, scroll_count=5):
        """Scroll through the feed to load more posts"""
        logger.info(f"Scrolling feed {scroll_count} times to load posts...")
        
        for i in range(scroll_count):
            try:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                logger.info(f"Scroll {i+1}/{scroll_count} completed")
            except Exception as e:
                logger.warning(f"Error during scroll {i+1}: {e}")
    
    def extract_posts_from_feed(self, max_posts=20):
        """Extract post links and content from the feed"""
        logger.info("Extracting posts from feed...")
        
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        posts_data = []
        
        # Find post containers - try multiple selectors
        post_selectors = [
            ".feed-shared-update-v2",
            "[data-test-id='feed-update']",
            ".occludable-update"
        ]
        
        post_containers = []
        for selector in post_selectors:
            containers = soup.select(selector)
            if containers:
                post_containers = containers
                logger.info(f"Found {len(containers)} posts using selector: {selector}")
                break
        
        if not post_containers:
            logger.warning("No posts found in feed")
            return posts_data
        
        # Extract data from each post
        for i, container in enumerate(post_containers[:max_posts]):
            try:
                post_data = self._extract_post_data(container)
                if post_data:
                    posts_data.append(post_data)
                    logger.info(f"Extracted post {i+1}: {post_data['title'][:50]}...")
            except Exception as e:
                logger.warning(f"Error extracting post {i+1}: {e}")
        
        logger.info(f"Successfully extracted {len(posts_data)} posts from feed")
        return posts_data
    
    def _extract_post_data(self, container):
        """Extract data from a single post container"""
        try:
            # Extract post text/content
            content_selectors = [
                ".feed-shared-text",
                ".feed-shared-update-v2__description",
                "[data-test-id='post-content']"
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = container.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(strip=True)
                    break
            
            # Extract author info
            author_selectors = [
                ".feed-shared-actor__title",
                "[data-test-id='actor-name']"
            ]
            
            author = "Unknown"
            for selector in author_selectors:
                author_elem = container.select_one(selector)
                if author_elem:
                    author = author_elem.get_text(strip=True)
                    break
            
            # Extract post link
            link_elem = container.select_one("a[href*='/posts/'], a[href*='/activity/']")
            post_link = ""
            if link_elem:
                post_link = link_elem.get('href', '')
                if post_link.startswith('/'):
                    post_link = 'https://www.linkedin.com' + post_link
            
            # Extract timestamp
            time_elem = container.select_one("[data-test-id='post-timestamp']")
            timestamp = time_elem.get_text(strip=True) if time_elem else ""
            
            if content or post_link:
                return {
                    'title': f"Post by {author}",
                    'content': content,
                    'author': author,
                    'link': post_link,
                    'timestamp': timestamp,
                    'platform': 'www.linkedin.com'
                }
            
        except Exception as e:
            logger.warning(f"Error extracting post data: {e}")
        
        return None
    
    def crawl_feed_and_save(self, user, max_posts=20):
        """Main method to crawl feed and save posts"""
        logger.info("Starting LinkedIn feed crawl...")
        
        # Login if credentials provided
        if self.email and self.password:
            if not self.login_with_google(self.email, self.password):
                logger.warning("Google login failed, trying manual login...")
                self.manual_login_prompt()
        else:
            self.manual_login_prompt()
        
        if not self.logged_in:
            logger.error("Login failed - cannot crawl feed")
            return False
        
        # Navigate to feed
        if not self.navigate_to_feed():
            logger.error("Cannot navigate to feed")
            return False
        
        # Scroll to load posts
        self.scroll_feed(scroll_count=5)
        
        # Extract posts
        posts = self.extract_posts_from_feed(max_posts=max_posts)
        
        # Save posts to database
        saved_count = 0
        for post in posts:
            try:
                # Check if already exists
                if post['link']:
                    existing = ArticleDocument.find(link=post['link'])
                    if existing:
                        logger.info(f"Post already exists: {post['link']}")
                        continue
                
                # Create article document
                article_data = {
                    "Title": post['title'],
                    "Subtitle": f"By {post['author']}",
                    "Content": post['content'],
                    "language": "en"
                }
                
                instance = self.model(
                    content=article_data,
                    link=post['link'] or f"https://www.linkedin.com/feed/post/{saved_count}",
                    platform=post['platform'],
                    author_id=user.id,
                    author_full_name=user.full_name,
                )
                instance.save()
                saved_count += 1
                logger.info(f"Saved post {saved_count}: {post['title'][:50]}...")
                
            except Exception as e:
                logger.error(f"Error saving post: {e}")
        
        logger.success(f"✓ Successfully saved {saved_count} LinkedIn posts to database!")
        return True
    
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
        """Extract single post (for compatibility with existing system)"""
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
