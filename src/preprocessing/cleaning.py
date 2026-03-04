"""
Cleaning dispatcher and handlers for processing different document types
"""
from abc import ABC, abstractmethod
from typing import Union, Generic, TypeVar
import re
from loguru import logger

from ..domain.documents import NoSQLBaseDocument, ArticleDocument, RepositoryDocument
from ..domain.vector_documents import (
    VectorBaseDocument, CleanedDocument, CleanedPostDocument,
    CleanedArticleDocument, CleanedRepositoryDocument
)
from ..domain.enums import DataCategory

# Type variables for Generic typing
DocumentT = TypeVar("DocumentT", bound=NoSQLBaseDocument)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    """Abstract base class for cleaning handlers"""
    
    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        """
        Clean the input data model
        
        Args:
            data_model: Raw document to clean
            
        Returns:
            Cleaned document
        """
        pass
    
    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning utility
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


class PostCleaningHandler(CleaningDataHandler[NoSQLBaseDocument, CleanedPostDocument]):
    """Handler for cleaning post documents"""
    
    def clean(self, data_model: NoSQLBaseDocument) -> CleanedPostDocument:
        """Clean post document"""
        try:
            # Extract content from various possible fields
            content = self._extract_content(data_model)
            cleaned_content = self._clean_text(" #### ".join(data_model.content.values())) if hasattr(data_model, 'content') and isinstance(data_model.content, dict) else self._clean_text(str(data_model))
            
            # Extract platform
            platform = getattr(data_model, 'platform', 'unknown')
            
            # Extract author information
            author_id = getattr(data_model, 'author_id', '')
            author_full_name = getattr(data_model, 'author_full_name', 'Unknown Author')
            
            # Extract image if available
            image = None
            if hasattr(data_model, 'content') and isinstance(data_model.content, dict):
                image = data_model.content.get('image')
            
            cleaned_post = CleanedPostDocument(
                id=data_model.id,
                content=cleaned_content,
                platform=platform,
                author_id=author_id,
                author_full_name=author_full_name,
                image=image
            )
            
            logger.info(f"Post cleaned successfully. Content length: {len(cleaned_content)}")
            return cleaned_post
            
        except Exception as e:
            logger.error(f"Error cleaning post: {e}")
            # Return basic cleaned post on error
            return CleanedPostDocument(
                id=getattr(data_model, 'id', ''),
                content=str(data_model),
                platform='unknown',
                author_id='',
                author_full_name='Unknown'
            )
    
    def _extract_content(self, data_model: NoSQLBaseDocument) -> str:
        """Extract content from post document"""
        if hasattr(data_model, 'content') and isinstance(data_model.content, dict):
            # Try different content fields
            for field in ['text', 'content', 'body', 'message']:
                if field in data_model.content and isinstance(data_model.content[field], str):
                    return data_model.content[field]
            
            # Fallback to all string content
            content_parts = []
            for value in data_model.content.values():
                if isinstance(value, str):
                    content_parts.append(value)
            return ' '.join(content_parts)
        
        return str(data_model.content) if hasattr(data_model, 'content') else str(data_model)


class ArticleCleaningHandler(CleaningDataHandler[NoSQLBaseDocument, CleanedArticleDocument]):
    """Handler for cleaning article documents"""
    
    def clean(self, data_model: NoSQLBaseDocument) -> CleanedArticleDocument:
        """Clean article document"""
        try:
            # Extract valid content from article
            valid_content = []
            if hasattr(data_model, 'content') and isinstance(data_model.content, dict):
                valid_content = [content for content in data_model.content.values() if content]
            
            cleaned_content = self._clean_text(" #### ".join(valid_content))
            
            # Extract link
            link = getattr(data_model, 'link', '')
            
            # Extract platform
            platform = getattr(data_model, 'platform', 'unknown')
            
            # Extract author information
            author_id = getattr(data_model, 'author_id', '')
            author_full_name = getattr(data_model, 'author_full_name', 'Unknown Author')
            
            cleaned_article = CleanedArticleDocument(
                id=data_model.id,
                content=cleaned_content,
                platform=platform,
                link=link,
                author_id=author_id,
                author_full_name=author_full_name
            )
            
            logger.info(f"Article cleaned successfully. Content length: {len(cleaned_content)}")
            return cleaned_article
            
        except Exception as e:
            logger.error(f"Error cleaning article: {e}")
            # Return basic cleaned article on error
            return CleanedArticleDocument(
                id=getattr(data_model, 'id', ''),
                content=str(data_model),
                platform='unknown',
                link='',
                author_id='',
                author_full_name='Unknown'
            )


class RepositoryCleaningHandler(CleaningDataHandler[NoSQLBaseDocument, CleanedRepositoryDocument]):
    """Handler for cleaning repository documents"""
    
    def clean(self, data_model: NoSQLBaseDocument) -> CleanedRepositoryDocument:
        """Clean repository document"""
        try:
            # Extract content from repository
            cleaned_content = self._clean_text(" #### ".join(data_model.content.values())) if hasattr(data_model, 'content') and isinstance(data_model.content, dict) else self._clean_text(str(data_model))
            
            # Extract repository name and link
            name = getattr(data_model, 'name', '')
            link = getattr(data_model, 'link', '')
            
            # Extract platform
            platform = getattr(data_model, 'platform', 'unknown')
            
            # Extract author information
            author_id = getattr(data_model, 'author_id', '')
            author_full_name = getattr(data_model, 'author_full_name', 'Unknown Author')
            
            cleaned_repo = CleanedRepositoryDocument(
                id=data_model.id,
                content=cleaned_content,
                name=name,
                link=link,
                platform=platform,
                author_id=author_id,
                author_full_name=author_full_name
            )
            
            logger.info(f"Repository cleaned successfully. Content length: {len(cleaned_content)}")
            return cleaned_repo
            
        except Exception as e:
            logger.error(f"Error cleaning repository: {e}")
            # Return basic cleaned repository on error
            return CleanedRepositoryDocument(
                id=getattr(data_model, 'id', ''),
                content=str(data_model),
                name='',
                link='',
                platform='unknown',
                author_id='',
                author_full_name='Unknown'
            )


class CleaningHandlerFactory:
    """Factory for creating cleaning handlers based on data category"""
    
    @staticmethod
    def create_handler(data_category: DataCategory) -> CleaningDataHandler:
        """
        Create appropriate cleaning handler for data category
        
        Args:
            data_category: Category of data to clean
            
        Returns:
            Appropriate cleaning handler instance
            
        Raises:
            ValueError: If data category is not supported
        """
        if data_category == DataCategory.POSTS:
            return PostCleaningHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleCleaningHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryCleaningHandler()
        else:
            raise ValueError(f"Unsupported data type: {data_category}")


class CleaningDispatcher:
    """Dispatcher for cleaning documents based on their data category"""
    
    cleaning_factory = CleaningHandlerFactory()
    
    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        """
        Dispatch document to appropriate cleaning handler
        
        Args:
            data_model: Raw document to clean
            
        Returns:
            Cleaned document
        """
        try:
            # Get data category from collection name
            collection_name = data_model.get_collection_name()
            data_category = DataCategory(collection_name)
            
            # Get appropriate handler
            handler = cls.cleaning_factory.create_handler(data_category)
            
            # Clean the document
            clean_model = handler.clean(data_model)
            
            logger.info(
                "Data cleaned successfully.",
                data_category=data_category,
                cleaned_content_len=len(clean_model.content),
            )
            
            return clean_model
            
        except Exception as e:
            logger.error(f"Error in cleaning dispatcher: {e}")
            # Return basic cleaned document as fallback
            return CleanedArticleDocument(
                content=str(data_model),
                link='',
                platform='unknown',
                author_id='',
                author_full_name='Unknown'
            )
