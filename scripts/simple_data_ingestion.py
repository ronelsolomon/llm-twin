#!/usr/bin/env python3
"""
Simple script to ingest raw data into the data warehouse without ZenML complexity
"""
import sys
from pathlib import Path
import json
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.domain.documents import UserDocument, ArticleDocument, RepositoryDocument


def ingest_articles(data_path: str) -> int:
    """Ingest articles from JSON file into data warehouse"""
    logger.info(f"Ingesting articles from {data_path}")
    
    articles_file = Path(data_path) / "articles.json"
    if not articles_file.exists():
        logger.warning(f"Articles file not found: {articles_file}")
        return 0
    
    with open(articles_file, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    
    ingested_count = 0
    for article_data in articles_data:
        try:
            article = ArticleDocument(**article_data)
            article.save()
            ingested_count += 1
            logger.info(f"Ingested article: {article.id}")
        except Exception as e:
            logger.error(f"Failed to ingest article {article_data.get('id', 'unknown')}: {e}")
    
    return ingested_count


def ingest_repositories(data_path: str) -> int:
    """Ingest repositories from JSON file into data warehouse"""
    logger.info(f"Ingesting repositories from {data_path}")
    
    repos_file = Path(data_path) / "repositories.json"
    if not repos_file.exists():
        logger.warning(f"Repositories file not found: {repos_file}")
        return 0
    
    with open(repos_file, 'r', encoding='utf-8') as f:
        repos_data = json.load(f)
    
    ingested_count = 0
    for repo_data in repos_data:
        try:
            repo = RepositoryDocument(**repo_data)
            repo.save()
            ingested_count += 1
            logger.info(f"Ingested repository: {repo.id}")
        except Exception as e:
            logger.error(f"Failed to ingest repository {repo_data.get('id', 'unknown')}: {e}")
    
    return ingested_count


def ingest_users(data_path: str) -> int:
    """Ingest users from JSON file into data warehouse"""
    logger.info(f"Ingesting users from {data_path}")
    
    users_file = Path(data_path) / "users.json"
    if not users_file.exists():
        logger.warning(f"Users file not found: {users_file}")
        return 0
    
    with open(users_file, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
    
    ingested_count = 0
    for user_data in users_data:
        try:
            user = UserDocument(**user_data)
            user.save()
            ingested_count += 1
            logger.info(f"Ingested user: {user.id}")
        except Exception as e:
            logger.error(f"Failed to ingest user {user_data.get('id', 'unknown')}: {e}")
    
    return ingested_count


def process_linkedin_profile(data_path: str) -> bool:
    """Process LinkedIn profile data"""
    logger.info(f"Processing LinkedIn profile from {data_path}")
    
    profile_file = Path(data_path) / "linkedin_structured_data.json"
    if not profile_file.exists():
        logger.warning(f"LinkedIn structured data file not found: {profile_file}")
        return False
    
    with open(profile_file, 'r', encoding='utf-8') as f:
        profile_data = json.load(f)
    
    # Create or update user from LinkedIn profile
    if profile_data.get('full_name'):
        try:
            user = UserDocument(
                id=profile_data.get('id', 'linkedin_user'),
                full_name=profile_data['full_name']
            )
            user.save()
            logger.info(f"Processed LinkedIn user: {user.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to process LinkedIn profile: {e}")
    
    return False


def main():
    """Run the data ingestion process"""
    
    # Define data path
    data_path = "/Users/ronel/Downloads/llm twin/data"
    
    logger.info(f"Starting data ingestion from: {data_path}")
    
    # Ingest all data types
    articles_count = ingest_articles(data_path)
    repositories_count = ingest_repositories(data_path)
    users_count = ingest_users(data_path)
    linkedin_processed = process_linkedin_profile(data_path)
    
    # Summary
    logger.info("Data ingestion completed!")
    logger.info(f"Articles ingested: {articles_count}")
    logger.info(f"Repositories ingested: {repositories_count}")
    logger.info(f"Users ingested: {users_count}")
    logger.info(f"LinkedIn profile processed: {linkedin_processed}")
    
    return {
        "articles_count": articles_count,
        "repositories_count": repositories_count,
        "users_count": users_count,
        "linkedin_profile_processed": linkedin_processed
    }


if __name__ == "__main__":
    main()
