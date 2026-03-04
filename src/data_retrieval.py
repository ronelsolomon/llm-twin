"""
Data retrieval functions for querying the data warehouse
"""
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from .domain.documents import ArticleDocument, RepositoryDocument, NoSQLBaseDocument


def __fetch_articles(user_id: str) -> List[NoSQLBaseDocument]:
    """Fetch articles for a given user ID"""
    try:
        articles = ArticleDocument.find_all(author_id=user_id)
        logger.info(f"Found {len(articles)} articles for user {user_id}")
        return articles
    except Exception as e:
        logger.error(f"Error fetching articles for user {user_id}: {e}")
        return []


def __fetch_posts(user_id: str) -> List[NoSQLBaseDocument]:
    """Fetch posts for a given user ID (placeholder for future implementation)"""
    try:
        # Placeholder for posts - currently no PostDocument class exists
        logger.info(f"Posts not implemented yet for user {user_id}")
        return []
    except Exception as e:
        logger.error(f"Error fetching posts for user {user_id}: {e}")
        return []


def __fetch_repositories(user_id: str) -> List[NoSQLBaseDocument]:
    """Fetch repositories for a given user ID"""
    try:
        repositories = RepositoryDocument.find_all(author_id=user_id)
        logger.info(f"Found {len(repositories)} repositories for user {user_id}")
        return repositories
    except Exception as e:
        logger.error(f"Error fetching repositories for user {user_id}: {e}")
        return []


def fetch_all_data(user) -> Dict[str, List[NoSQLBaseDocument]]:
    """
    Fetch all data (articles, posts, and repositories) for a given user using ThreadPoolExecutor
    
    Args:
        user: UserDocument instance
        
    Returns:
        Dictionary with 'articles', 'posts', and 'repositories' keys containing lists of documents
    """
    user_id = str(user.id)
    
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_articles, user_id): "articles",
            executor.submit(__fetch_posts, user_id): "posts",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }
        
        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")
                results[query_name] = []
    
    return results
