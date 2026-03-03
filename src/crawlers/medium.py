from bs4 import BeautifulSoup
from loguru import logger
from src.domain.documents import ArticleDocument
from src.crawlers.selenium_base import BaseSeleniumCrawler


class MediumCrawler(BaseSeleniumCrawler):
    model = ArticleDocument
    
    def set_extra_driver_options(self, options) -> None:
        # Comment out headless mode for testing Cloudflare issues
        # options.add_argument(r"--profile-directory=Profile 2")
        
        # For testing, disable headless to see if we can bypass Cloudflare
        pass
    
    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")
            return
        
        logger.info(f"Starting scrapping Medium article: {link}")
        
        try:
            self.driver.get(link)
            self.scroll_page()
            
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            
            # Check if this is a profile page or an article
            if "/@" in link and not link.endswith("/"):
                # This is a profile page, extract article links and process them
                self._process_profile_page(soup, link, **kwargs)
            else:
                # This is an article page, extract content
                self._process_article_page(soup, link, **kwargs)
            
            self.driver.close()
            logger.info(f"Successfully scraped and saved: {link}")
            
        except Exception as e:
            logger.error(f"Error scraping Medium {link}: {e}")
            if hasattr(self, 'driver') and self.driver:
                self.driver.close()
            raise
    
    def _process_article_page(self, soup, link: str, **kwargs):
        """Extract content from a Medium article page"""
        title = soup.find_all("h1", class_="pw-post-title")
        subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")
        
        data = {
            "Title": title[0].string if title else None,
            "Subtitle": subtitle[0].string if subtitle else None,
            "Content": soup.get_text(),
        }
        
        user = kwargs["user"]
        
        instance = self.model(
            platform="medium",
            content=data,
            link=link,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()
        logger.info(f"Successfully scraped and saved article: {link}")
    
    def _process_profile_page(self, soup, profile_link: str, **kwargs):
        """Extract article links from a Medium profile page and process them"""
        # Find article links on the profile page
        article_links = []
        
        # Debug: Save the page source to see what we're getting
        with open("debug_profile_page.html", "w", encoding="utf-8") as f:
            f.write(str(soup))
        logger.info("Saved debug profile page to debug_profile_page.html")
        
        # Look for links that might be articles using multiple selectors
        selectors = [
            "a[href*='/@']",
            "a[href*='medium.com']",
            "article a",
            ".postArticle a",
            "[data-testid='postArticle'] a"
        ]
        
        for selector in selectors:
            try:
                for link in soup.select(selector):
                    href = link.get("href")
                    if href and self._is_article_link(href, profile_link):
                        full_link = self._extract_actual_article_url(href)
                        if not full_link.startswith("http"):
                            full_link = f"https://medium.com{full_link}"
                        if full_link not in article_links:
                            article_links.append(full_link)
                            logger.debug(f"Found article link: {full_link}")
            except Exception as e:
                logger.warning(f"Error with selector {selector}: {e}")
        
        # Also try finding all links and filtering
        all_links = soup.find_all("a", href=True)
        logger.info(f"Total links found on page: {len(all_links)}")
        
        for link in all_links:
            href = link.get("href")
            if href and self._is_article_link(href, profile_link):
                full_link = self._extract_actual_article_url(href)
                if not full_link.startswith("http"):
                    full_link = f"https://medium.com{full_link}"
                if full_link not in article_links:
                    article_links.append(full_link)
                    logger.debug(f"Found article link (fallback): {full_link}")
        
        logger.info(f"Found {len(article_links)} articles on profile page")
        
        # Process each article found
        for article_link in article_links[:5]:  # Limit to first 5 articles for testing
            try:
                # Check if article already exists
                existing = self.model.find(link=article_link)
                if existing is None:
                    # Navigate to article page
                    self.driver.get(article_link)
                    self.scroll_page()
                    article_soup = BeautifulSoup(self.driver.page_source, "html.parser")
                    self._process_article_page(article_soup, article_link, **kwargs)
                else:
                    logger.info(f"Article already exists: {article_link}")
            except Exception as e:
                logger.error(f"Error processing article {article_link}: {e}")
                continue
    
    def _is_article_link(self, href: str, profile_link: str) -> bool:
        """Check if a link is likely an article from this profile"""
        profile_username = profile_link.split("/")[-1].lstrip("@")
        
        # Check if it's a Medium signin redirect with article URL
        if "medium.com/m/signin" in href and "redirect=" in href:
            # Extract the actual article URL from the redirect parameter
            from urllib.parse import unquote, urlparse, parse_qs
            try:
                parsed = urlparse(href)
                redirect_url = unquote(parse_qs(parsed.query)['redirect'][0])
                if f"@{profile_username}" in redirect_url and len(redirect_url) > len(profile_link):
                    return True
            except:
                pass
        
        # Check if it's a direct Medium article link
        elif "medium.com" in href and profile_username in href:
            # Exclude the profile page itself
            if href != profile_link and not href.endswith(f"/{profile_username}"):
                return True
        
        return False
    
    def _extract_actual_article_url(self, href: str) -> str:
        """Extract the actual article URL from a signin redirect"""
        if "medium.com/m/signin" in href and "redirect=" in href:
            from urllib.parse import unquote, urlparse, parse_qs
            try:
                parsed = urlparse(href)
                redirect_url = unquote(parse_qs(parsed.query)['redirect'][0])
                return redirect_url
            except:
                pass
        return href