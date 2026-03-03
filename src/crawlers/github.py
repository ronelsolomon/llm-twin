import os
import subprocess
import tempfile
import shutil
import re
from urllib.parse import urlparse
from loguru import logger
from src.crawlers.base import BaseCrawler
from src.domain.documents import RepositoryDocument


class GithubCrawler(BaseCrawler):
    model = RepositoryDocument
    
    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__()
        self._ignore = ignore

    def _is_profile_url(self, link: str) -> bool:
        """Check if URL is a GitHub profile URL"""
        parsed = urlparse(link)
        path_parts = parsed.path.strip("/").split("/")
        return len(path_parts) == 1 and parsed.netloc == "github.com"
    
    def _is_repository_url(self, link: str) -> bool:
        """Check if URL is a GitHub repository URL"""
        parsed = urlparse(link)
        path_parts = parsed.path.strip("/").split("/")
        return len(path_parts) >= 2 and parsed.netloc == "github.com"
    
    def _discover_repositories(self, profile_url: str) -> list[str]:
        """Discover all repositories from a GitHub profile page"""
        logger.info(f"Discovering repositories from profile: {profile_url}")
        
        # Use git to list repositories (more reliable than web scraping)
        username = profile_url.rstrip("/").split("/")[-1]
        
        # Try multiple methods to discover repositories
        
        # Method 1: Try GitHub API (without authentication for public repos)
        repos = []
        try:
            import requests
            api_url = f"https://api.github.com/users/{username}/repos"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                repo_data = response.json()
                repos = [repo["clone_url"] for repo in repo_data]
                logger.info(f"Found {len(repos)} repositories via GitHub API")
        except Exception as e:
            logger.warning(f"GitHub API method failed: {e}")
        
        # Method 2: Try common repository names (fallback)
        if not repos:
            common_repos = [
                f"https://github.com/{username}/portfolio",
                f"https://github.com/{username}/website", 
                f"https://github.com/{username}/projects",
                f"https://github.com/{username}/dotfiles",
                f"https://github.com/{username}/setup",
                f"https://github.com/{username}/config",
                f"https://github.com/{username}/scripts",
                f"https://github.com/{username}/tools",
            ]
            
            # Test which ones actually exist
            for repo_url in common_repos:
                try:
                    result = subprocess.run(
                        ["git", "ls-remote", repo_url], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    if result.returncode == 0:
                        repos.append(repo_url)
                except Exception:
                    continue
            
            logger.info(f"Found {len(repos)} repositories via common names")
        
        return repos
    
    def _crawl_repository(self, repo_link: str, user, repo_name: str = None) -> None:
        """Crawl a single repository"""
        # Check if already exists
        old_model = self.model.find(link=repo_link)
        if old_model is not None:
            logger.info(f"Repository already exists in the database: {repo_link}")
            return 

        logger.info(f"Starting scrapping GitHub repository: {repo_link}")
        
        if not repo_name:
            repo_name = repo_link.rstrip("/").split("/")[-1]
        
        local_temp = tempfile.mkdtemp()
        original_dir = os.getcwd()

        try:
            os.chdir(local_temp)
            result = subprocess.run(["git", "clone", repo_link], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to clone repository: {result.stderr}")

            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])
            tree = {}
            for root, _, files in os.walk(repo_path):
                dir_path = root.replace(repo_path, "").lstrip("/")
                
                # Skip directories that start with ignored patterns
                if any(dir_path.startswith(pattern) for pattern in self._ignore):
                    continue
                    
                for file in files:
                    # Skip files that end with ignored patterns
                    if any(file.endswith(pattern) for pattern in self._ignore):
                        continue
                        
                    file_path = os.path.join(dir_path, file)
                    full_file_path = os.path.join(root, file)
                    
                    try:
                        with open(full_file_path, "r", errors="ignore") as f:
                            tree[file_path] = f.read().replace(" ", "")
                    except (UnicodeDecodeError, PermissionError):
                        # Skip binary files or files we can't read
                        continue

            instance = self.model(
                content=tree,
                name=repo_name,
                link=repo_link,
                platform="github",
                author_id=user.id,
                author_full_name=user.full_name,
            )
            instance.save()
            logger.info(f"Successfully saved repository: {repo_name}")

        except Exception as e:
            logger.error(f"Error crawling GitHub repository {repo_link}: {e}")
            raise
        finally:
            os.chdir(original_dir)
            shutil.rmtree(local_temp)

    def extract(self, link: str, **kwargs) -> None:
        user = kwargs["user"]
        
        # Check if repository already exists (for single repository URLs)
        if self._is_repository_url(link):
            old_model = self.model.find(link=link)
            if old_model is not None:
                logger.info(f"Repository already exists in the database: {link}")
                return
        
        if self._is_profile_url(link):
            # Handle profile URL - discover and crawl all repositories
            logger.info(f"Detected profile URL, discovering repositories...")
            repo_urls = self._discover_repositories(link)
            
            if not repo_urls:
                logger.warning(f"No repositories found for profile: {link}")
                return
            
            logger.info(f"Found {len(repo_urls)} repositories to crawl")
            
            for repo_url in repo_urls:
                try:
                    self._crawl_repository(repo_url, user)
                except Exception as e:
                    logger.error(f"Failed to crawl repository {repo_url}: {e}")
                    continue
            
            logger.info(f"Finished crawling profile: {link}")
            
        elif self._is_repository_url(link):
            # Handle single repository URL
            self._crawl_repository(link, user)
            
        else:
            raise ValueError(f"Invalid GitHub URL: {link}")