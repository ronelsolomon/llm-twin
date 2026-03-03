"""
Enhanced ZenML pipeline for gathering raw data into the data warehouse with web crawling
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from zenml import pipeline, step
from loguru import logger

# Import domain documents directly to avoid circular import
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.domain.documents import UserDocument, ArticleDocument, RepositoryDocument
from src.crawlers.medium import MediumCrawler
from src.crawlers.github import GithubCrawler
from src.crawlers.linkedin import LinkedInCrawler


@step
def create_or_get_user(user_full_name: str) -> UserDocument:
    """Create or get user from full name"""
    logger.info(f"Creating or getting user: {user_full_name}")
    
    # Split name into first and last name
    name_parts = user_full_name.split()
    first_name = name_parts[0] if name_parts else ""
    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
    
    # Create user document
    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
    user.full_name = user_full_name
    user.save()
    
    logger.info(f"User created/retrieved: {user.id} - {user.full_name}")
    return user


@step
def crawl_medium_articles(links: List[str], user: UserDocument) -> List[ArticleDocument]:
    """Crawl articles from Medium links"""
    logger.info(f"Crawling {len(links)} Medium links for user: {user.full_name}")
    
    crawler = MediumCrawler()
    crawled_articles = []
    
    for link in links:
        if "medium.com" in link:
            try:
                crawler.extract(link, user=user)
                # Find the created article
                article = ArticleDocument.find(link=link)
                if article:
                    crawled_articles.append(article)
                    logger.info(f"Successfully crawled Medium article: {link}")
            except Exception as e:
                logger.error(f"Failed to crawl Medium article {link}: {e}")
    
    logger.info(f"Successfully crawled {len(crawled_articles)} Medium articles")
    return crawled_articles


@step
def crawl_github_repositories(links: List[str], user: UserDocument) -> List[RepositoryDocument]:
    """Crawl repositories from GitHub links"""
    logger.info(f"Crawling {len(links)} GitHub links for user: {user.full_name}")
    
    crawler = GithubCrawler()
    crawled_repos = []
    
    for link in links:
        if "github.com" in link:
            try:
                crawler.extract(link, user=user)
                # Find the created repository
                repo = RepositoryDocument.find(link=link)
                if repo:
                    crawled_repos.append(repo)
                    logger.info(f"Successfully crawled GitHub repository: {link}")
            except Exception as e:
                logger.error(f"Failed to crawl GitHub repository {link}: {e}")
    
    logger.info(f"Successfully crawled {len(crawled_repos)} GitHub repositories")
    return crawled_repos


@step
def crawl_linkedin_content(links: List[str], user: UserDocument) -> List[ArticleDocument]:
    """Crawl content from LinkedIn links"""
    logger.info(f"Crawling {len(links)} LinkedIn links for user: {user.full_name}")
    
    crawler = LinkedInCrawler()
    crawled_articles = []
    
    for link in links:
        if "linkedin.com" in link:
            try:
                crawler.extract(link, user=user)
                # Find the created article
                article = ArticleDocument.find(link=link)
                if article:
                    crawled_articles.append(article)
                    logger.info(f"Successfully crawled LinkedIn content: {link}")
            except Exception as e:
                logger.error(f"Failed to crawl LinkedIn content {link}: {e}")
    
    logger.info(f"Successfully crawled {len(crawled_articles)} LinkedIn articles")
    return crawled_articles


@step
def crawl_substack_articles(links: List[str], user: UserDocument) -> List[ArticleDocument]:
    """Crawl articles from Substack links (using Medium crawler as fallback)"""
    logger.info(f"Crawling {len(links)} Substack links for user: {user.full_name}")
    
    crawler = MediumCrawler()  # Substack has similar structure to Medium
    crawled_articles = []
    
    for link in links:
        if "substack.com" in link:
            try:
                crawler.extract(link, user=user)
                # Find the created article
                article = ArticleDocument.find(link=link)
                if article:
                    crawled_articles.append(article)
                    logger.info(f"Successfully crawled Substack article: {link}")
            except Exception as e:
                logger.error(f"Failed to crawl Substack article {link}: {e}")
    
    logger.info(f"Successfully crawled {len(crawled_articles)} Substack articles")
    return crawled_articles


@step
def crawl_personal_blog_articles(links: List[str], user: UserDocument) -> List[ArticleDocument]:
    """Crawl articles from personal blog links"""
    logger.info(f"Crawling {len(links)} personal blog links for user: {user.full_name}")
    
    crawler = MediumCrawler()  # Use Medium crawler as general web crawler
    crawled_articles = []
    
    for link in links:
        # Skip known platforms
        if any(platform in link for platform in ["medium.com", "github.com", "linkedin.com", "substack.com"]):
            continue
            
        try:
            crawler.extract(link, user=user)
            # Find the created article
            article = ArticleDocument.find(link=link)
            if article:
                crawled_articles.append(article)
                logger.info(f"Successfully crawled blog article: {link}")
        except Exception as e:
            logger.error(f"Failed to crawl blog article {link}: {e}")
    
    logger.info(f"Successfully crawled {len(crawled_articles)} blog articles")
    return crawled_articles


@step
def generate_crawling_report(
    medium_articles: List[ArticleDocument],
    github_repos: List[RepositoryDocument], 
    linkedin_articles: List[ArticleDocument],
    substack_articles: List[ArticleDocument],
    blog_articles: List[ArticleDocument],
    user: UserDocument,
    total_links: int
) -> Dict[str, Any]:
    """Generate a comprehensive crawling report"""
    
    total_crawled = len(medium_articles) + len(github_repos) + len(linkedin_articles) + len(substack_articles) + len(blog_articles)
    
    report = {
        "user_full_name": user.full_name,
        "user_id": user.id,
        "total_links_provided": total_links,
        "total_items_crawled": total_crawled,
        "success_rate": total_crawled / total_links if total_links > 0 else 0,
        "crawled_links": {
            "medium": {
                "count": len(medium_articles),
                "links": [article.link for article in medium_articles]
            },
            "github": {
                "count": len(github_repos),
                "links": [repo.link for repo in github_repos]
            },
            "linkedin": {
                "count": len(linkedin_articles),
                "links": [article.link for article in linkedin_articles]
            },
            "substack": {
                "count": len(substack_articles),
                "links": [article.link for article in substack_articles]
            },
            "personal_blogs": {
                "count": len(blog_articles),
                "links": [article.link for article in blog_articles]
            }
        }
    }
    
    logger.info(f"Crawling report generated for {user.full_name}:")
    logger.info(f"  Total links: {total_links}")
    logger.info(f"  Successfully crawled: {total_crawled}")
    logger.info(f"  Success rate: {report['success_rate']:.2%}")
    
    return report


@pipeline
def digital_data_etl_pipeline(user_full_name: str, links: List[str]) -> Dict[str, Any]:
    """
    Complete digital data ETL pipeline with web crawling
    
    Args:
        user_full_name: Full name of the user (e.g., "Maxime Labonne")
        links: List of URLs to crawl (Medium, GitHub, LinkedIn, Substack, personal blogs)
    
    Returns:
        Crawling report with statistics and results
    """
    
    # Create or get user
    user = create_or_get_user(user_full_name=user_full_name)
    
    # Crawl content from different platforms
    medium_articles = crawl_medium_articles(links=links, user=user)
    github_repos = crawl_github_repositories(links=links, user=user)
    linkedin_articles = crawl_linkedin_content(links=links, user=user)
    substack_articles = crawl_substack_articles(links=links, user=user)
    blog_articles = crawl_personal_blog_articles(links=links, user=user)
    
    # Generate comprehensive report
    report = generate_crawling_report(
        medium_articles=medium_articles,
        github_repos=github_repos,
        linkedin_articles=linkedin_articles,
        substack_articles=substack_articles,
        blog_articles=blog_articles,
        user=user,
        total_links=len(links)
    )
    
    return report


if __name__ == "__main__":
    # Example usage
    from zenml.client import Client
    
    # Initialize ZenML with error handling
    try:
        Client().create_project("digital_data_etl", delete_if_exists=True)
    except Exception as e:
        print(f"Warning: Could not create ZenML project: {e}")
        print("Continuing without explicit project creation...")
    
    # Example configuration for Maxime Labonne
    maxime_links = [
        # Personal Blog
        "https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html",
        "https://mlabonne.github.io/blog/posts/2024-07-15_The_Rise_of_Agentic_Data_Generation.html",
        # Substack
        "https://maximelabonne.substack.com/p/uncensor-any-llm-with-abliteration-d30148b7d43e",
        "https://maximelabonne.substack.com/p/create-mixtures-of-experts-with-mergekit-11b318c99562",
        "https://maximelabonne.substack.com/p/merge-large-language-models-with-mergekit-2118fb392b54",
    ]
    
    # Run pipeline
    result = digital_data_etl_pipeline(
        user_full_name="Maxime Labonne",
        links=maxime_links
    )
    
    logger.info(f"Pipeline completed: {result}")
