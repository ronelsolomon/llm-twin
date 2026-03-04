#!/usr/bin/env python3
"""
Standalone script to extract complete data from GitHub repositories and articles
without ZenML dependencies. This script saves raw text content to separate files.
"""
import os
import json
import subprocess
import tempfile
import shutil
import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Try to import selenium for web scraping
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from bs4 import BeautifulSoup
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available. Web scraping will be limited.")

class StandaloneDataExtractor:
    def __init__(self, output_dir: str = "data/raw", cache_dir: str = "data/cache"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache system
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index from file"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {}
    
    def _save_cache_index(self) -> None:
        """Save cache index to file"""
        with open(self.cache_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
    
    def _get_cache_key(self, url: str, content_type: str) -> str:
        """Generate cache key for URL and content type"""
        import hashlib
        key_string = f"{content_type}:{url}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cached(self, url: str, content_type: str, max_age_hours: int = 24) -> bool:
        """Check if content is cached and not too old"""
        cache_key = self._get_cache_key(url, content_type)
        
        if cache_key not in self.cache_index:
            return False
        
        cache_entry = self.cache_index[cache_key]
        cached_time = datetime.fromisoformat(cache_entry.get('cached_at', ''))
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
        
        return age_hours < max_age_hours
    
    def _get_cached_content(self, url: str, content_type: str) -> Optional[Dict[str, Any]]:
        """Get cached content if available"""
        cache_key = self._get_cache_key(url, content_type)
        
        if cache_key not in self.cache_index:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return None
    
    def _cache_content(self, url: str, content_type: str, content: Dict[str, Any]) -> None:
        """Cache content for future use"""
        cache_key = self._get_cache_key(url, content_type)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Add cache metadata
        content['_cache_metadata'] = {
            'url': url,
            'content_type': content_type,
            'cached_at': datetime.now().isoformat(),
            'cache_key': cache_key
        }
        
        # Save content to cache file
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        # Update cache index
        self.cache_index[cache_key] = {
            'url': url,
            'content_type': content_type,
            'cached_at': datetime.now().isoformat(),
            'cache_file': str(cache_file)
        }
        
        self._save_cache_index()
        print(f"💾 Cached {content_type} content: {url}")
    
    def extract_github_repository(self, repo_url: str, use_cache: bool = True) -> Dict[str, Any]:
        """Extract complete content from a GitHub repository"""
        print(f"Extracting GitHub repository: {repo_url}")
        
        # Check cache first
        if use_cache and self._is_cached(repo_url, 'github'):
            cached_content = self._get_cached_content(repo_url, 'github')
            if cached_content:
                print(f"⚡ Using cached content for: {repo_url}")
                # Also save to output directory if not exists
                repo_name = repo_url.rstrip("/").split("/")[-1]
                owner = repo_url.rstrip("/").split("/")[-2]
                output_file = self.output_dir / f"github_{owner}_{repo_name}.json"
                if not output_file.exists():
                    # Remove cache metadata before saving
                    content_to_save = cached_content.copy()
                    if '_cache_metadata' in content_to_save:
                        del content_to_save['_cache_metadata']
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(content_to_save, f, indent=2, ensure_ascii=False)
                    print(f"✅ Saved cached repository to {output_file}")
                return cached_content
        
        repo_name = repo_url.rstrip("/").split("/")[-1]
        owner = repo_url.rstrip("/").split("/")[-2]
        
        # Create temporary directory for cloning
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Clone the repository
            print(f"Cloning repository to {temp_dir}")
            result = subprocess.run(
                ["git", "clone", repo_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to clone repository: {result.stderr}")
            
            # Extract all file contents
            repo_content = {
                "repository_url": repo_url,
                "owner": owner,
                "name": repo_name,
                "extracted_at": datetime.now().isoformat(),
                "files": {}
            }
            
            # Walk through all files
            for root, dirs, files in os.walk(temp_dir):
                # Skip .git directory and other common ignores
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'target', 'build']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, temp_dir)
                    
                    # Skip binary files and very large files
                    if self._should_skip_file(file_path):
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Only include files with meaningful content
                        if len(content.strip()) > 10:
                            repo_content["files"][relative_path] = {
                                "content": content,
                                "size": len(content),
                                "language": self._detect_language(file_path)
                            }
                            
                    except (UnicodeDecodeError, PermissionError, OSError) as e:
                        print(f"Skipping file {relative_path}: {e}")
                        continue
            
            # Cache the result
            self._cache_content(repo_url, 'github', repo_content)
            
            # Save to file
            repo_name = repo_url.rstrip("/").split("/")[-1]
            owner = repo_url.rstrip("/").split("/")[-2]
            output_file = self.output_dir / f"github_{owner}_{repo_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(repo_content, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved repository content to {output_file}")
            print(f"   Files extracted: {len(repo_content['files'])}")
            
            return repo_content
            
        except Exception as e:
            print(f"❌ Error extracting repository {repo_url}: {e}")
            return None
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def extract_article_content(self, article_url: str, use_cache: bool = True) -> Dict[str, Any]:
        """Extract complete content from an article URL"""
        print(f"Extracting article: {article_url}")
        
        # Check cache first
        if use_cache and self._is_cached(article_url, 'article'):
            cached_content = self._get_cached_content(article_url, 'article')
            if cached_content:
                print(f"⚡ Using cached content for: {article_url}")
                # Also save to output directory if not exists
                safe_filename = self._safe_filename(article_url)
                output_file = self.output_dir / f"article_{safe_filename}.json"
                if not output_file.exists():
                    # Remove cache metadata before saving
                    content_to_save = cached_content.copy()
                    if '_cache_metadata' in content_to_save:
                        del content_to_save['_cache_metadata']
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(content_to_save, f, indent=2, ensure_ascii=False)
                    print(f"✅ Saved cached article to {output_file}")
                return cached_content
        
        if not SELENIUM_AVAILABLE:
            return self._extract_article_basic(article_url)
        
        # Use Selenium for JavaScript-heavy sites
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            driver = webdriver.Chrome(options=options)
            
            try:
                driver.get(article_url)
                time.sleep(3)  # Wait for page to load
                
                # Scroll to load dynamic content
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Get page content
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # Extract content based on platform
                if "medium.com" in article_url:
                    content = self._extract_medium_content(soup)
                elif "linkedin.com" in article_url:
                    content = self._extract_linkedin_content(soup)
                elif "substack.com" in article_url:
                    content = self._extract_substack_content(soup)
                else:
                    content = self._extract_generic_content(soup)
                
                article_data = {
                    "url": article_url,
                    "platform": self._detect_platform(article_url),
                    "extracted_at": datetime.now().isoformat(),
                    **content
                }
                
                # Cache the result
                self._cache_content(article_url, 'article', article_data)
                
                # Save to file
                safe_filename = self._safe_filename(article_url)
                output_file = self.output_dir / f"article_{safe_filename}.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(article_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Saved article content to {output_file}")
                return article_data
                
            finally:
                driver.quit()
                
        except Exception as e:
            print(f"❌ Error extracting article {article_url}: {e}")
            # Fallback to basic extraction
            return self._extract_article_basic(article_url)
    
    def _extract_article_basic(self, article_url: str) -> Dict[str, Any]:
        """Basic article extraction without Selenium"""
        try:
            response = self.session.get(article_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            content = self._extract_generic_content(soup)
            
            article_data = {
                "url": article_url,
                "platform": self._detect_platform(article_url),
                "extracted_at": datetime.now().isoformat(),
                "extraction_method": "basic",
                **content
            }
            
            # Cache the result
            self._cache_content(article_url, 'article', article_data)
            
            # Save to file
            safe_filename = self._safe_filename(article_url)
            output_file = self.output_dir / f"article_basic_{safe_filename}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved article content (basic) to {output_file}")
            return article_data
            
        except Exception as e:
            print(f"❌ Error with basic extraction for {article_url}: {e}")
            return None
    
    def _extract_medium_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract content from Medium articles"""
        # Try multiple selectors for Medium content
        title_selectors = [
            "h1.pw-post-title",
            "h1",
            "title"
        ]
        
        content_selectors = [
            "article",
            ".pw-post-body",
            ".postArticle-content",
            "[data-testid='postContent']",
            "main"
        ]
        
        title = self._extract_text_by_selectors(soup, title_selectors)
        content = self._extract_text_by_selectors(soup, content_selectors)
        
        return {
            "title": title,
            "content": content,
            "content_length": len(content) if content else 0
        }
    
    def _extract_linkedin_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract content from LinkedIn articles"""
        title_selectors = [
            "h1.break-words",
            ".share-update-card__title",
            "[data-test-id='post-title']",
            "h1"
        ]
        
        content_selectors = [
            ".share-update-card__content",
            "[data-test-id='post-content']",
            ".feed-shared-update-v2__description",
            "article"
        ]
        
        title = self._extract_text_by_selectors(soup, title_selectors)
        content = self._extract_text_by_selectors(soup, content_selectors)
        
        return {
            "title": title,
            "content": content,
            "content_length": len(content) if content else 0
        }
    
    def _extract_substack_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract content from Substack articles"""
        title_selectors = [
            "h1",
            ".post-title",
            ".title"
        ]
        
        content_selectors = [
            ".body",
            ".post-body",
            "article",
            ".content"
        ]
        
        title = self._extract_text_by_selectors(soup, title_selectors)
        content = self._extract_text_by_selectors(soup, content_selectors)
        
        return {
            "title": title,
            "content": content,
            "content_length": len(content) if content else 0
        }
    
    def _extract_generic_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Generic content extraction for any website"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Try to find title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # Try to find main content
        main_content = None
        for selector in ['main', 'article', '.content', '.post', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            # Fallback to body
            main_content = soup.find('body')
        
        content_text = main_content.get_text(separator='\n', strip=True) if main_content else ""
        
        return {
            "title": title_text,
            "content": content_text,
            "content_length": len(content_text)
        }
    
    def _extract_text_by_selectors(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Extract text using multiple selectors in order of preference"""
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(separator='\n', strip=True)
                if text and len(text.strip()) > 10:  # Only return meaningful content
                    return text
        return ""
    
    def _should_skip_file(self, file_path: str) -> bool:
        """Check if file should be skipped during extraction"""
        file_path = file_path.lower()
        
        # Skip binary files
        binary_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.exe', '.dll', '.so', '.dylib',
            '.mp3', '.mp4', '.avi', '.mov', '.wav',
            '.ttf', '.otf', '.woff', '.woff2'
        }
        
        # Skip large files (>1MB)
        try:
            if os.path.getsize(file_path) > 1024 * 1024:
                return True
        except OSError:
            return True
        
        # Check extension
        for ext in binary_extensions:
            if file_path.endswith(ext):
                return True
        
        return False
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.sql': 'sql',
            '.sh': 'bash',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.md': 'markdown',
            '.txt': 'text',
            '.dockerfile': 'dockerfile'
        }
        
        return language_map.get(ext, 'unknown')
    
    def _detect_platform(self, url: str) -> str:
        """Detect platform from URL"""
        domain = urlparse(url).netloc.lower()
        
        if 'medium.com' in domain:
            return 'medium'
        elif 'linkedin.com' in domain:
            return 'linkedin'
        elif 'substack.com' in domain:
            return 'substack'
        elif 'github.com' in domain:
            return 'github'
        else:
            return 'unknown'
    
    def _safe_filename(self, url: str) -> str:
        """Create safe filename from URL"""
        # Extract meaningful part of URL
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) > 0:
            # Use the last meaningful part
            filename = path_parts[-1]
        else:
            # Use domain as fallback
            filename = parsed.netloc.replace('.', '_')
        
        # Clean filename
        filename = ''.join(c if c.isalnum() or c in '-_' else '_' for c in filename)
        filename = filename[:50]  # Limit length
        
        # Add timestamp to avoid conflicts
        timestamp = int(time.time())
        return f"{filename}_{timestamp}"
    
    def extract_multiple_repositories(self, repo_urls: List[str], use_cache: bool = True) -> List[Dict[str, Any]]:
        """Extract multiple GitHub repositories"""
        results = []
        for url in repo_urls:
            result = self.extract_github_repository(url, use_cache=use_cache)
            if result:
                results.append(result)
            time.sleep(1)  # Rate limiting
        return results
    
    def extract_multiple_articles(self, article_urls: List[str], use_cache: bool = True) -> List[Dict[str, Any]]:
        """Extract multiple articles"""
        results = []
        for url in article_urls:
            result = self.extract_article_content(url, use_cache=use_cache)
            if result:
                results.append(result)
            time.sleep(2)  # Rate limiting
        return results
    
    def clear_cache(self, content_type: Optional[str] = None, max_age_hours: Optional[int] = None) -> None:
        """Clear cache entries
        
        Args:
            content_type: If specified, only clear this type ('github' or 'article')
            max_age_hours: If specified, only clear entries older than this many hours
        """
        keys_to_remove = []
        
        for cache_key, cache_entry in self.cache_index.items():
            # Filter by content type if specified
            if content_type and cache_entry.get('content_type') != content_type:
                continue
            
            # Filter by age if specified
            if max_age_hours:
                cached_time = datetime.fromisoformat(cache_entry.get('cached_at', ''))
                age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                if age_hours < max_age_hours:
                    continue
            
            keys_to_remove.append(cache_key)
            
            # Remove cache file
            cache_file = Path(cache_entry.get('cache_file', ''))
            if cache_file.exists():
                cache_file.unlink()
                print(f"🗑️  Removed cache file: {cache_file}")
        
        # Remove from index
        for key in keys_to_remove:
            del self.cache_index[key]
        
        if keys_to_remove:
            self._save_cache_index()
            print(f"🧹 Cleared {len(keys_to_remove)} cache entries")
        else:
            print("✨ No cache entries to clear")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'total_entries': len(self.cache_index),
            'github_entries': 0,
            'article_entries': 0,
            'cache_size_mb': 0,
            'oldest_entry': None,
            'newest_entry': None
        }
        
        cache_files_size = 0
        oldest_time = datetime.now()
        newest_time = datetime.fromisoformat('1970-01-01T00:00:00')
        
        for cache_entry in self.cache_index.values():
            content_type = cache_entry.get('content_type', '')
            if content_type == 'github':
                stats['github_entries'] += 1
            elif content_type == 'article':
                stats['article_entries'] += 1
            
            # Calculate cache size
            cache_file = Path(cache_entry.get('cache_file', ''))
            if cache_file.exists():
                cache_files_size += cache_file.stat().st_size
            
            # Track oldest and newest
            cached_time = datetime.fromisoformat(cache_entry.get('cached_at', ''))
            if cached_time < oldest_time:
                oldest_time = cached_time
                stats['oldest_entry'] = cache_entry.get('url', '')
            if cached_time > newest_time:
                newest_time = cached_time
                stats['newest_entry'] = cache_entry.get('url', '')
        
        stats['cache_size_mb'] = round(cache_files_size / (1024 * 1024), 2)
        
        return stats


def main():
    """Main function to run the extractor"""
    extractor = StandaloneDataExtractor()
    
    print("=== Standalone Data Extractor ===")
    print("This script will extract complete content from GitHub repositories and articles.")
    print()
    
    # All GitHub URLs from existing data
    all_github_urls = [
        "https://github.com/ronelsolomon/2D-Farming-Game.git",
        "https://github.com/ronelsolomon/2d-game-dev.git",
        "https://github.com/ronelsolomon/2d-game.git",
        "https://github.com/ronelsolomon/AI-Engineer-questions.git",
        "https://github.com/ronelsolomon/AWS-backend.git",
        "https://github.com/ronelsolomon/ETL-pipeline.git",
        "https://github.com/ronelsolomon/FindYoutube.git",
        "https://github.com/ronelsolomon/Graph-k-means.git",
        "https://github.com/ronelsolomon/ai-videos.git",
        "https://github.com/ronelsolomon/aiadds.git",
        "https://github.com/ronelsolomon/air-quality-MLOPs.git",
        "https://github.com/ronelsolomon/aivideo.git",
        "https://github.com/ronelsolomon/aleoex.git",
        "https://github.com/ronelsolomon/assementTemplate.git",
        "https://github.com/ronelsolomon/cebras.git",
        "https://github.com/ronelsolomon/changeAudio.git",
        "https://github.com/ronelsolomon/crawlerx.git",
        "https://github.com/ronelsolomon/csvDashboardRetetion.git",
        "https://github.com/ronelsolomon/curify-gallery.git",
        "https://github.com/ronelsolomon/dev.git",
        "https://github.com/ronelsolomon/dma.git",
        "https://github.com/ronelsolomon/expresive.git",
        "https://github.com/ronelsolomon/file-audio.git",
        "https://github.com/ronelsolomon/filesummarize.git",
        "https://github.com/ronelsolomon/genAI.git",
        "https://github.com/ronelsolomon/healthcare.git",
        "https://github.com/ronelsolomon/hearpython.git",
        "https://github.com/ronelsolomon/interview_questions.git",
        "https://github.com/ronelsolomon/java-project.git",
        "https://github.com/ronelsolomon/keboola-mcp.git"
    ]
    
    # All article URLs from existing data (cleaned and extracted)
    all_article_urls = [
        # LinkedIn posts
        "https://www.linkedin.com/in/ronel-solomon/#education",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-1-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-2-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-16-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-17-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-26-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-27-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-28-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-47-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-63-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-74-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-84-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-92-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-97-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-103-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-111-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-112-20260302/",
        "https://www.linkedin.com/posts/ronel-solomon-profile-update-115-20260302/",
        
        # Medium articles (extracted actual URLs from redirects)
        "https://medium.com/@ronelsolomon/pitch-deck-verification-system-7b691653fff3",
        "https://medium.com/@ronelsolomon/how-i-solved-a-problem-in-microsoft-teams-7fd85dc9bc48",
        "https://medium.com/@ronelsolomon/how-to-utilize-multimodal-data-in-an-etl-pipeline-01db8784c892",
        "https://medium.com/@ronelsolomon/%EF%B8%8F-detecting-sports-fields-from-images-with-ai-28e9136a9941",
        
        # Additional sample URLs for testing
        "https://medium.com/decodingml/an-end-to-end-framework-for-production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f",
        "https://maximelabonne.substack.com/p/uncensor-any-llm-with-abliteration-d30148b7d43e"
    ]
    
    # Sample URLs for testing (smaller subsets)
    sample_github_urls = [
        "https://github.com/ronelsolomon/2D-Farming-Game.git",
        "https://github.com/ronelsolomon/ai-videos.git",
        "https://github.com/ronelsolomon/ETL-pipeline.git"
    ]
    
    sample_article_urls = [
        "https://medium.com/decodingml/an-end-to-end-framework-for-production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f",
        "https://maximelabonne.substack.com/p/uncensor-any-llm-with-abliteration-d30148b7d43e"
    ]

    
    print("Select what to extract:")
    print("1. Sample GitHub repositories (3 repos)")
    print("2. Sample articles (2 articles)")
    print("3. Both samples")
    print("4. ALL GitHub repositories (30 repos)")
    print("5. ALL articles (23 articles)")
    print("6. ALL repositories AND articles")
    print("7. Custom URLs")
    print("8. Cache management")
    
    choice = input("Enter your choice (1-8): ").strip()
    
    if choice == "1":
        print("\nExtracting sample GitHub repositories...")
        extractor.extract_multiple_repositories(sample_github_urls)
        
    elif choice == "2":
        print("\nExtracting sample articles...")
        extractor.extract_multiple_articles(sample_article_urls)
        
    elif choice == "3":
        print("\nExtracting both samples...")
        extractor.extract_multiple_repositories(sample_github_urls)
        extractor.extract_multiple_articles(sample_article_urls)
        
    elif choice == "4":
        print(f"\nExtracting ALL {len(all_github_urls)} GitHub repositories...")
        print("This may take a while, but cached content will be used where available.")
        extractor.extract_multiple_repositories(all_github_urls)
        
    elif choice == "5":
        print(f"\nExtracting ALL {len(all_article_urls)} articles...")
        print("This may take a while, but cached content will be used where available.")
        extractor.extract_multiple_articles(all_article_urls)
        
    elif choice == "6":
        print(f"\nExtracting ALL content ({len(all_github_urls)} repos + {len(all_article_urls)} articles)...")
        print("This will take significant time, but cached content will be used where available.")
        extractor.extract_multiple_repositories(all_github_urls)
        extractor.extract_multiple_articles(all_article_urls)
        
    elif choice == "7":
        print("\nEnter custom URLs (one per line, empty line to finish):")
        
        github_urls = []
        article_urls = []
        
        while True:
            url = input("URL: ").strip()
            if not url:
                break
            
            if 'github.com' in url:
                github_urls.append(url)
            else:
                article_urls.append(url)
        
        if github_urls:
            print(f"\nExtracting {len(github_urls)} GitHub repositories...")
            extractor.extract_multiple_repositories(github_urls)
        
        if article_urls:
            print(f"\nExtracting {len(article_urls)} articles...")
            extractor.extract_multiple_articles(article_urls)
    
    elif choice == "8":
        print("\n=== Cache Management ===")
        stats = extractor.get_cache_stats()
        print(f"Current cache: {stats['total_entries']} entries ({stats['cache_size_mb']} MB)")
        print("Options:")
        print("a. Show cache statistics")
        print("b. Clear all cache")
        print("c. Clear GitHub cache only")
        print("d. Clear article cache only")
        print("e. Clear cache older than 24 hours")
        print("f. Return to main menu")
        
        cache_choice = input("Enter your choice (a-f): ").strip().lower()
        
        if cache_choice == "a":
            print("\n📊 Cache Statistics:")
            print(f"   Total entries: {stats['total_entries']}")
            print(f"   GitHub entries: {stats['github_entries']}")
            print(f"   Article entries: {stats['article_entries']}")
            print(f"   Cache size: {stats['cache_size_mb']} MB")
            if stats['oldest_entry']:
                print(f"   Oldest entry: {stats['oldest_entry']}")
            if stats['newest_entry']:
                print(f"   Newest entry: {stats['newest_entry']}")
        elif cache_choice == "b":
            print("\n🧹 Clearing all cache...")
            extractor.clear_cache()
        elif cache_choice == "c":
            print("\n🧹 Clearing GitHub cache...")
            extractor.clear_cache(content_type='github')
        elif cache_choice == "d":
            print("\n🧹 Clearing article cache...")
            extractor.clear_cache(content_type='article')
        elif cache_choice == "e":
            print("\n🧹 Clearing cache older than 24 hours...")
            extractor.clear_cache(max_age_hours=24)
        elif cache_choice == "f":
            pass
        else:
            print("❌ Invalid choice.")
        
        return
        
    else:
        print("Invalid choice.")
        return
    
    print(f"\n✅ Extraction complete! Check the {extractor.output_dir} directory for results.")


if __name__ == "__main__":
    main()
