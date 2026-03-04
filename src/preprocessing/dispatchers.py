"""
Dispatcher classes for cleaning, chunking, and embedding documents
"""
import re
from typing import List, Any
from loguru import logger
from ..domain.documents import NoSQLBaseDocument, ArticleDocument, RepositoryDocument


class CleaningDispatcher:
    """Dispatcher for cleaning documents based on their type"""
    
    @staticmethod
    def dispatch(document: NoSQLBaseDocument) -> NoSQLBaseDocument:
        """
        Dispatch document to appropriate cleaning method based on its type
        
        Args:
            document: Document to clean
            
        Returns:
            Cleaned document
        """
        if isinstance(document, ArticleDocument):
            return CleaningDispatcher._clean_article(document)
        elif isinstance(document, RepositoryDocument):
            return CleaningDispatcher._clean_repository(document)
        else:
            logger.warning(f"No cleaning logic for document type: {type(document)}")
            return document
    
    @staticmethod
    def _clean_article(article: ArticleDocument) -> ArticleDocument:
        """Clean article document"""
        try:
            # Clean content
            if isinstance(article.content, dict):
                cleaned_content = {}
                for key, value in article.content.items():
                    if isinstance(value, str):
                        # Basic text cleaning
                        cleaned_value = CleaningDispatcher._clean_text(value)
                        cleaned_content[key] = cleaned_value
                    else:
                        cleaned_content[key] = value
                
                # Create cleaned article with same properties but cleaned content
                cleaned_article = ArticleDocument(
                    content=cleaned_content,
                    link=article.link,
                    platform=article.platform,
                    author_id=article.author_id,
                    author_full_name=article.author_full_name,
                    id=article.id,
                    created_at=article.created_at,
                    updated_at=article.updated_at
                )
                return cleaned_article
            
        except Exception as e:
            logger.error(f"Error cleaning article {article.id}: {e}")
            return article
    
    @staticmethod
    def _clean_repository(repository: RepositoryDocument) -> RepositoryDocument:
        """Clean repository document"""
        try:
            # Clean content
            if isinstance(repository.content, dict):
                cleaned_content = {}
                for key, value in repository.content.items():
                    if isinstance(value, str):
                        # Basic text cleaning for repository content
                        cleaned_value = CleaningDispatcher._clean_text(value)
                        cleaned_content[key] = cleaned_value
                    elif isinstance(value, list):
                        # Clean list items (e.g., topics, languages)
                        cleaned_list = []
                        for item in value:
                            if isinstance(item, str):
                                cleaned_list.append(CleaningDispatcher._clean_text(item))
                            else:
                                cleaned_list.append(item)
                        cleaned_content[key] = cleaned_list
                    else:
                        cleaned_content[key] = value
                
                # Create cleaned repository with same properties but cleaned content
                cleaned_repository = RepositoryDocument(
                    content=cleaned_content,
                    name=repository.name,
                    link=repository.link,
                    platform=repository.platform,
                    author_id=repository.author_id,
                    author_full_name=repository.author_full_name,
                    id=repository.id,
                    created_at=repository.created_at,
                    updated_at=repository.updated_at
                )
                return cleaned_repository
            
        except Exception as e:
            logger.error(f"Error cleaning repository {repository.id}: {e}")
            return repository
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


class ChunkingDispatcher:
    """Dispatcher for chunking documents based on their type"""
    
    @staticmethod
    def dispatch(document: NoSQLBaseDocument) -> List[dict]:
        """
        Dispatch document to appropriate chunking method based on its type
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        if isinstance(document, ArticleDocument):
            return ChunkingDispatcher._chunk_article(document)
        elif isinstance(document, RepositoryDocument):
            return ChunkingDispatcher._chunk_repository(document)
        else:
            logger.warning(f"No chunking logic for document type: {type(document)}")
            return [{"content": str(document), "metadata": {"source": document.id}}]
    
    @staticmethod
    def _chunk_article(article: ArticleDocument) -> List[dict]:
        """Chunk article document"""
        chunks = []
        
        try:
            if isinstance(article.content, dict):
                # Extract text content from article
                text_content = ""
                
                # Try different common content fields
                for field in ['title', 'content', 'body', 'text', 'description']:
                    if field in article.content and isinstance(article.content[field], str):
                        text_content += article.content[field] + "\n\n"
                
                if not text_content.strip():
                    # Fallback to all string content
                    for value in article.content.values():
                        if isinstance(value, str):
                            text_content += value + "\n\n"
                
                # Simple chunking by paragraphs
                paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
                
                chunk_size = 3  # paragraphs per chunk
                for i in range(0, len(paragraphs), chunk_size):
                    chunk_text = '\n\n'.join(paragraphs[i:i + chunk_size])
                    
                    chunk = {
                        "content": chunk_text,
                        "metadata": {
                            "source": article.id,
                            "type": "article",
                            "platform": article.platform,
                            "author": article.author_full_name,
                            "link": article.link,
                            "chunk_index": i // chunk_size
                        }
                    }
                    chunks.append(chunk)
            
        except Exception as e:
            logger.error(f"Error chunking article {article.id}: {e}")
            # Fallback chunk
            chunks.append({
                "content": str(article.content),
                "metadata": {"source": article.id, "type": "article"}
            })
        
        return chunks
    
    @staticmethod
    def _chunk_repository(repository: RepositoryDocument) -> List[dict]:
        """Chunk repository document"""
        chunks = []
        
        try:
            if isinstance(repository.content, dict):
                # Create chunks for different repository sections
                sections = ['description', 'readme', 'topics', 'languages', 'summary']
                
                for section in sections:
                    if section in repository.content:
                        content = repository.content[section]
                        
                        if isinstance(content, str):
                            chunk = {
                                "content": content,
                                "metadata": {
                                    "source": repository.id,
                                    "type": "repository",
                                    "section": section,
                                    "platform": repository.platform,
                                    "author": repository.author_full_name,
                                    "name": repository.name,
                                    "link": repository.link
                                }
                            }
                            chunks.append(chunk)
                        elif isinstance(content, list):
                            # Handle lists (e.g., topics, languages)
                            list_text = ", ".join(str(item) for item in content)
                            chunk = {
                                "content": f"{section.title()}: {list_text}",
                                "metadata": {
                                    "source": repository.id,
                                    "type": "repository",
                                    "section": section,
                                    "platform": repository.platform,
                                    "author": repository.author_full_name,
                                    "name": repository.name,
                                    "link": repository.link
                                }
                            }
                            chunks.append(chunk)
            
            # If no chunks were created, create a fallback chunk
            if not chunks:
                chunks.append({
                    "content": f"Repository: {repository.name}\n{str(repository.content)}",
                    "metadata": {
                        "source": repository.id,
                        "type": "repository",
                        "platform": repository.platform,
                        "author": repository.author_full_name,
                        "name": repository.name,
                        "link": repository.link
                    }
                })
            
        except Exception as e:
            logger.error(f"Error chunking repository {repository.id}: {e}")
            # Fallback chunk
            chunks.append({
                "content": str(repository.content),
                "metadata": {"source": repository.id, "type": "repository"}
            })
        
        return chunks


class EmbeddingDispatcher:
    """Dispatcher for embedding document chunks"""
    
    @staticmethod
    def dispatch(chunks: List[dict]) -> List[dict]:
        """
        Embed a batch of chunks
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings added
        """
        try:
            # Placeholder for actual embedding logic
            # In a real implementation, you would use a model like OpenAI embeddings,
            # Sentence Transformers, or another embedding service
            
            for chunk in chunks:
                # Mock embedding - replace with actual embedding logic
                chunk["embedding"] = [0.0] * 384  # Placeholder embedding vector
                
                # Add embedding metadata
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["embedding_model"] = "mock-model"
                chunk["metadata"]["embedding_dimension"] = 384
            
            logger.info(f"Embedded {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            # Return chunks with empty embeddings on error
            for chunk in chunks:
                chunk["embedding"] = []
            return chunks
