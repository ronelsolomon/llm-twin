"""
Enums and constants for the LLM Twin project
"""
from enum import StrEnum


class DataCategory(StrEnum):
    """Data categories for different document types"""
    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
