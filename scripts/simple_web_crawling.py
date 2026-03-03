#!/usr/bin/env python3
"""
Simple web crawling pipeline without ZenML decorators
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Import domain documents and crawlers
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.domain.documents import UserDocument, ArticleDocument, RepositoryDocument
from src.crawlers.medium import MediumCrawler
from src.crawlers.github import GithubCrawler
from src.crawlers.linkedin import LinkedInCrawler


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


def simple_digital_data_etl_pipeline(user_full_name: str, links: List[str]) -> Dict[str, Any]:
    """
    Simple digital data ETL pipeline without ZenML
    
    Args:
        user_full_name: Full name of the user (e.g., "Ronel Solomon")
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
    total_crawled = len(medium_articles) + len(github_repos) + len(linkedin_articles) + len(substack_articles) + len(blog_articles)
    
    report = {
        "user_full_name": user.full_name,
        "user_id": user.id,
        "total_links_provided": len(links),
        "total_items_crawled": total_crawled,
        "success_rate": total_crawled / len(links) if len(links) > 0 else 0,
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
    logger.info(f"  Total links: {len(links)}")
    logger.info(f"  Successfully crawled: {total_crawled}")
    logger.info(f"  Success rate: {report['success_rate']:.2%}")
    
    return report


if __name__ == "__main__":
    # Example usage for Ronel Solomon
    ronel_links = [
        # GitHub repositories from existing data
        "https://github.com/ronelsolomon/2D-Farming-Game.git",
        "https://github.com/ronelsolomon/2d-game.git",
        "https://github.com/ronelsolomon/2d-game-dev.git",
        "https://github.com/ronelsolomon/AI-Engineer-questions.git",
        "https://github.com/ronelsolomon/ai-videos.git",
        "https://github.com/ronelsolomon/aiadds.git",
        "https://github.com/ronelsolomon/air-quality-MLOPs.git",
        "https://github.com/ronelsolomon/aivideo.git",
        "https://github.com/ronelsolomon/aleoex.git",
        "https://github.com/ronelsolomon/assementTemplate.git",
        "https://github.com/ronelsolomon/AWS-backend.git",
        "https://github.com/ronelsolomon/cebras.git",
        "https://github.com/ronelsolomon/changeAudio.git",
        "https://github.com/ronelsolomon/crawlerx.git",
        "https://github.com/ronelsolomon/csvDashboardRetetion.git",
        "https://github.com/ronelsolomon/curify-gallery.git",
        "https://github.com/ronelsolomon/dev.git",
        "https://github.com/ronelsolomon/dma.git",
        "https://github.com/ronelsolomon/ETL-pipeline.git",
        "https://github.com/ronelsolomon/expresive.git",
        "https://github.com/ronelsolomon/file-audio.git",
        "https://github.com/ronelsolomon/filesummarize.git",
        "https://github.com/ronelsolomon/FindYoutube.git",
        "https://github.com/ronelsolomon/genAI.git",
        "https://github.com/ronelsolomon/Graph-k-means.git",
        "https://github.com/ronelsolomon/healthcare.git",
        "https://github.com/ronelsolomon/hearpython.git",
        "https://github.com/ronelsolomon/interview_questions.git",
        "https://github.com/ronelsolomon/java-project.git",
        "https://github.com/ronelsolomon/keboola-mcp.git",
    ]
    
    # Run pipeline
    result = simple_digital_data_etl_pipeline(
        user_full_name="Ronel Solomon",
        links=ronel_links
    )
    
    logger.info(f"Pipeline completed: {result}")
    
    # Save report to file
    report_path = Path("crawling_report.json")
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Report saved to {report_path}")
