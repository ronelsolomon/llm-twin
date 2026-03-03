"""
ZenML pipeline for gathering raw data into the data warehouse
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from zenml import pipeline, step
from loguru import logger

# Import domain documents directly to avoid circular import
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.domain.documents import UserDocument, ArticleDocument, RepositoryDocument


@step
def ingest_articles(data_path: str) -> List[ArticleDocument]:
    """Ingest articles from JSON file into data warehouse"""
    logger.info(f"Ingesting articles from {data_path}")
    
    articles_file = Path(data_path) / "articles.json"
    if not articles_file.exists():
        logger.warning(f"Articles file not found: {articles_file}")
        return []
    
    with open(articles_file, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    
    ingested_articles = []
    for article_data in articles_data:
        try:
            article = ArticleDocument(**article_data)
            article.save()
            ingested_articles.append(article)
            logger.info(f"Ingested article: {article.id}")
        except Exception as e:
            logger.error(f"Failed to ingest article {article_data.get('id', 'unknown')}: {e}")
    
    return ingested_articles


@step
def ingest_repositories(data_path: str) -> List[RepositoryDocument]:
    """Ingest repositories from JSON file into data warehouse"""
    logger.info(f"Ingesting repositories from {data_path}")
    
    repos_file = Path(data_path) / "repositories.json"
    if not repos_file.exists():
        logger.warning(f"Repositories file not found: {repos_file}")
        return []
    
    with open(repos_file, 'r', encoding='utf-8') as f:
        repos_data = json.load(f)
    
    ingested_repos = []
    for repo_data in repos_data:
        try:
            repo = RepositoryDocument(**repo_data)
            repo.save()
            ingested_repos.append(repo)
            logger.info(f"Ingested repository: {repo.id}")
        except Exception as e:
            logger.error(f"Failed to ingest repository {repo_data.get('id', 'unknown')}: {e}")
    
    return ingested_repos


@step
def ingest_users(data_path: str) -> List[UserDocument]:
    """Ingest users from JSON file into data warehouse"""
    logger.info(f"Ingesting users from {data_path}")
    
    users_file = Path(data_path) / "users.json"
    if not users_file.exists():
        logger.warning(f"Users file not found: {users_file}")
        return []
    
    with open(users_file, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
    
    ingested_users = []
    for user_data in users_data:
        try:
            user = UserDocument(**user_data)
            user.save()
            ingested_users.append(user)
            logger.info(f"Ingested user: {user.id}")
        except Exception as e:
            logger.error(f"Failed to ingest user {user_data.get('id', 'unknown')}: {e}")
    
    return ingested_users


@step
def process_linkedin_profile(data_path: str) -> Dict[str, Any]:
    """Process LinkedIn profile data"""
    logger.info(f"Processing LinkedIn profile from {data_path}")
    
    profile_file = Path(data_path) / "linkedin_structured_data.json"
    if not profile_file.exists():
        logger.warning(f"LinkedIn structured data file not found: {profile_file}")
        return {}
    
    with open(profile_file, 'r', encoding='utf-8') as f:
        profile_data = json.load(f)
    
    # Create or update user from LinkedIn profile
    if profile_data.get('full_name'):
        # Create user directly using UserDocument
        user = UserDocument(
            id=profile_data.get('id', 'linkedin_user'),
            full_name=profile_data['full_name']
        )
        user.save()
        logger.info(f"Processed LinkedIn user: {user.id}")
    
    return profile_data


@pipeline
def data_warehouse_ingestion_pipeline(data_path: str):
    """Main pipeline for ingesting all raw data into data warehouse"""
    
    # Ingest all data types
    articles = ingest_articles(data_path)
    repositories = ingest_repositories(data_path)
    users = ingest_users(data_path)
    linkedin_profile = process_linkedin_profile(data_path)
    
    return {
        "articles_count": len(articles),
        "repositories_count": len(repositories), 
        "users_count": len(users),
        "linkedin_profile_processed": bool(linkedin_profile)
    }


if __name__ == "__main__":
    # Run the pipeline
    from zenml.client import Client
    
    # Initialize ZenML
    Client().create_project("data_warehouse", delete_if_exists=True)
    
    # Run pipeline
    data_path = "/Users/ronel/Downloads/llm twin/data"
    result = data_warehouse_ingestion_pipeline(data_path)
    
    logger.info(f"Pipeline completed: {result}")
