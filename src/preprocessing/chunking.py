"""
Chunking dispatcher and handlers for processing cleaned documents into chunks
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, cast
from uuid import UUID4, uuid4
import hashlib
import re
from loguru import logger

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain text splitters not available")

from ..domain.vector_documents import (
    CleanedDocument, CleanedPostDocument, CleanedArticleDocument, CleanedRepositoryDocument,
    Chunk, PostChunk, ArticleChunk, RepositoryChunk,
    EmbeddedChunk, EmbeddedPostChunk, EmbeddedArticleChunk, EmbeddedRepositoryChunk
)
from ..domain.enums import DataCategory
from ..config import settings
from ..ingestion.embedder import EmbeddingGenerator

# Type variables for Generic typing
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
ChunkT = TypeVar("ChunkT", bound=Chunk)


def chunk_article(text: str, min_length: int, max_length: int) -> List[str]:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    extracts = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())
            current_chunk = sentence + " "
    if len(current_chunk) >= min_length:
        extracts.append(current_chunk.strip())
    return extracts


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Advanced chunking using LangChain text splitters with token-based chunking.
    
    Args:
        text: Text to chunk
        chunk_size: Character chunk size for initial splitting
        chunk_overlap: Token overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available, falling back to simple chunking")
        return chunk_article(text, min_length=100, max_length=chunk_size)
    
    try:
        # Create a simple embedding model reference
        class EmbeddingModelRef:
            def __init__(self):
                self.model_id = settings.TEXT_EMBEDDING_MODEL_ID
                # Get max input length for sentence-transformers models
                if "all-MiniLM-L6-v2" in self.model_id:
                    self.max_input_length = 512
                elif "all-mpnet-base-v2" in self.model_id:
                    self.max_input_length = 514
                else:
                    self.max_input_length = 512  # Default fallback
        
        embedding_model = EmbeddingModelRef()
        
        # First split by paragraphs using RecursiveCharacterTextSplitter
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"], 
            chunk_size=chunk_size, 
            chunk_overlap=0
        )
        text_split_by_characters = character_splitter.split_text(text)
        
        # Then split by tokens using SentenceTransformersTokenTextSplitter
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=embedding_model.max_input_length,
            model_name=embedding_model.model_id,
        )
        
        chunks_by_tokens = []
        for section in text_split_by_characters:
            chunks_by_tokens.extend(token_splitter.split_text(section))
        
        return chunks_by_tokens
        
    except Exception as e:
        logger.error(f"Error in chunk_text: {e}")
        # Fallback to simple chunking
        return chunk_article(text, min_length=100, max_length=chunk_size)


class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    """Abstract base class for chunking handlers"""
    
    @property
    def metadata(self) -> dict:
        """Metadata for chunking configuration"""
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }
    
    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> List[ChunkT]:
        """
        Chunk the cleaned document into smaller pieces
        
        Args:
            data_model: Cleaned document to chunk
            
        Returns:
            List of document chunks
        """
        pass
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - overlap if end < len(text) else len(text)
        
        return chunks


class PostChunkingHandler(ChunkingDataHandler[CleanedPostDocument, PostChunk]):
    """Handler for chunking post documents"""
    
    def chunk(self, data_model: CleanedPostDocument) -> List[PostChunk]:
        """Chunk post document"""
        try:
            chunks = []
            chunk_size = self.metadata["chunk_size"]
            overlap = self.metadata["chunk_overlap"]
            
            # Split content into chunks
            text_chunks = self._split_text_into_chunks(data_model.content, chunk_size, overlap)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = PostChunk(
                    content=chunk_text,
                    platform=data_model.platform,
                    document_id=data_model.id,
                    author_id=data_model.author_id,
                    author_full_name=data_model.author_full_name,
                    metadata={
                        "chunk_index": i,
                        "source_type": "post",
                        "has_image": bool(data_model.image),
                        "chunk_size": len(chunk_text)
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Post chunked successfully. Created {len(chunks)} chunks from {len(data_model.content)} characters")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking post: {e}")
            return []
    
    @property
    def metadata(self) -> dict:
        """Post-specific chunking metadata"""
        return {
            "chunk_size": 300,  # Smaller chunks for posts
            "chunk_overlap": 30,
            "document_type": "post"
        }


class ArticleChunkingHandler(ChunkingDataHandler[CleanedArticleDocument, ArticleChunk]):
    """Handler for chunking article documents"""
    
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 1000,
            "max_length": 1000,
        }
    
    def chunk(self, data_model: CleanedArticleDocument) -> List[ArticleChunk]:
        data_models_list = []
        cleaned_content = data_model.content
        chunks = chunk_article(
            cleaned_content, min_length=self.metadata["min_length"], max_length=self.metadata["max_length"]
        )
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = ArticleChunk(
                id=uuid4(),  # Use uuid4 instead of UUID constructor
                content=chunk,
                platform=data_model.platform,
                link=data_model.link,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                metadata=self.metadata,
            )
            data_models_list.append(model)
        return data_models_list


class RepositoryChunkingHandler(ChunkingDataHandler[CleanedRepositoryDocument, RepositoryChunk]):
    """Handler for chunking repository documents"""
    
    def chunk(self, data_model: CleanedRepositoryDocument) -> List[RepositoryChunk]:
        """Chunk repository document"""
        try:
            chunks = []
            
            # For repositories, create chunks based on content structure
            # rather than just splitting text
            
            # Main description chunk
            if data_model.content.strip():
                chunk = RepositoryChunk(
                    content=data_model.content,
                    platform=data_model.platform,
                    document_id=data_model.id,
                    author_id=data_model.author_id,
                    author_full_name=data_model.author_full_name,
                    metadata={
                        "chunk_index": 0,
                        "source_type": "repository",
                        "source_name": data_model.name,
                        "source_link": data_model.link,
                        "chunk_type": "description",
                        "chunk_size": len(data_model.content)
                    }
                )
                chunks.append(chunk)
            
            # Create additional chunks if content is very long
            chunk_size = self.metadata["chunk_size"]
            if len(data_model.content) > chunk_size:
                text_chunks = self._split_text_into_chunks(data_model.content, chunk_size, 0)  # No overlap for repos
                
                for i, chunk_text in enumerate(text_chunks[1:], 1):  # Skip first as it's already added
                    chunk = RepositoryChunk(
                        content=chunk_text,
                        platform=data_model.platform,
                        document_id=data_model.id,
                        author_id=data_model.author_id,
                        author_full_name=data_model.author_full_name,
                        metadata={
                            "chunk_index": i,
                            "source_type": "repository",
                            "source_name": data_model.name,
                            "source_link": data_model.link,
                            "chunk_type": "description_part",
                            "chunk_size": len(chunk_text)
                        }
                    )
                    chunks.append(chunk)
            
            logger.info(f"Repository chunked successfully. Created {len(chunks)} chunks from {len(data_model.content)} characters")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking repository: {e}")
            return []
    
    @property
    def metadata(self) -> dict:
        """Repository-specific chunking metadata"""
        return {
            "chunk_size": 800,  # Larger chunks for repositories
            "chunk_overlap": 0,   # No overlap for repository descriptions
            "document_type": "repository"
        }


class ChunkingHandlerFactory:
    """Factory for creating chunking handlers based on data category"""
    
    @staticmethod
    def create_handler(data_category: DataCategory) -> ChunkingDataHandler:
        """
        Create appropriate chunking handler for data category
        
        Args:
            data_category: Category of data to chunk
            
        Returns:
            Appropriate chunking handler instance
            
        Raises:
            ValueError: If data category is not supported
        """
        if data_category == DataCategory.POSTS:
            return PostChunkingHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleChunkingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryChunkingHandler()
        else:
            raise ValueError(f"Unsupported data type: {data_category}")


class ChunkingDispatcher:
    """Dispatcher for chunking documents based on their data category"""
    
    chunking_factory = ChunkingHandlerFactory()
    
    @classmethod
    def dispatch(cls, data_model: CleanedDocument) -> List[Chunk]:
        """
        Dispatch document to appropriate chunking handler
        
        Args:
            data_model: Cleaned document to chunk
            
        Returns:
            List of document chunks
        """
        try:
            # Get data category from the document's Config
            data_category = data_model.Config.get_category()
            
            # Get appropriate handler
            handler = cls.chunking_factory.create_handler(data_category)
            
            # Chunk the document
            chunks = handler.chunk(data_model)
            
            logger.info(
                "Document chunked successfully.",
                data_category=data_category,
                chunk_count=len(chunks),
                chunk_metadata=handler.metadata
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in chunking dispatcher: {e}")
            return [f"Error: {e}"]


