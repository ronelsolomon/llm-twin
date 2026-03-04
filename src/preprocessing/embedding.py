"""
Embedding dispatcher and handlers for processing document chunks into embedded chunks
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, cast, Union
from pathlib import Path
import numpy as np
from loguru import logger

try:
    from sentence_transformers.SentenceTransformer import SentenceTransformer
    from sentence_transformers import AutoTokenizer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")

from ..domain.vector_documents import (
    Chunk, PostChunk, ArticleChunk, RepositoryChunk,
    EmbeddedChunk, EmbeddedPostChunk, EmbeddedArticleChunk, EmbeddedRepositoryChunk
)
from ..domain.enums import DataCategory
from ..config import settings


class SingletonMeta(type):
    """Metaclass for creating singleton classes"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class EmbeddingModelSingleton(metaclass=SingletonMeta):
    """
    Singleton wrapper over SentenceTransformer for embedding generation.
    
    Writing a wrapper over external packages is good practice. When you want to change 
    the third-party tool, you modify only the internal logic of the wrapper instead 
    of the whole code base.
    """
    
    def __init__(
        self,
        model_id: str = settings.TEXT_EMBEDDING_MODEL_ID,
        device: str = settings.RAG_MODEL_DEVICE,
        cache_dir: Path = None,
    ) -> None:
        if not hasattr(self, '_initialized'):
            self._model_id = model_id
            self._device = device
            self._cache_dir = cache_dir
            
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers is required but not available")
            
            self._model = SentenceTransformer(
                self._model_id,
                device=self._device,
                cache_folder=str(cache_dir) if cache_dir else None,
            )
            self._model.eval()
            self._initialized = True
    
    @property
    def model_id(self) -> str:
        """Get the model ID"""
        return self._model_id
    
    @property
    def embedding_size(self) -> int:
        """Get the embedding size by encoding dummy text"""
        try:
            dummy_embedding = self._model.encode("")
            return dummy_embedding.shape[0]
        except Exception as e:
            logger.error(f"Error getting embedding size: {e}")
            # Fallback sizes for common models
            if "all-MiniLM-L6-v2" in self._model_id:
                return 384
            elif "all-mpnet-base-v2" in self._model_id:
                return 768
            else:
                return 384  # Default fallback
    
    @property
    def max_input_length(self) -> int:
        """Get the maximum input length for the model"""
        try:
            return self._model.max_seq_length
        except Exception as e:
            logger.error(f"Error getting max input length: {e}")
            # Fallback lengths for common models
            if "all-MiniLM-L6-v2" in self._model_id:
                return 512
            elif "all-mpnet-base-v2" in self._model_id:
                return 514
            else:
                return 512  # Default fallback
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer for the model"""
        try:
            return self._model.tokenizer
        except Exception as e:
            logger.error(f"Error getting tokenizer: {e}")
            return None
    
    def __call__(
        self, 
        input_text: Union[str, List[str]], 
        to_list: bool = True
    ) -> Union[np.ndarray, List[float], List[List[float]]]:
        """
        Generate embeddings for input text(s)
        
        Args:
            input_text: Single text or list of texts to embed
            to_list: Whether to return embeddings as lists instead of numpy arrays
            
        Returns:
            Embeddings as numpy array or list(s)
        """
        try:
            embeddings = self._model.encode(input_text)
            if to_list:
                embeddings = embeddings.tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for {self._model_id} and {input_text}: {e}")
            return [] if to_list else np.array([])


# Global embedding model instance
embedding_model = EmbeddingModelSingleton()

# Type variables for embedding handlers
ChunkT = TypeVar("ChunkT", bound=Chunk)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    """
    Abstract class for all embedding data handlers.
    
    All data transformations logic for embedding step is done here
    """
    
    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        """Embed a single chunk"""
        return self.embed_batch([data_model])[0]
    
    def embed_batch(self, data_models: List[ChunkT]) -> List[EmbeddedChunkT]:
        """Embed multiple chunks in batch for optimization"""
        try:
            embedding_model_input = [data_model.content for data_model in data_models]
            embeddings = embedding_model(embedding_model_input, to_list=True)
            
            embedded_chunks = [
                self.map_model(data_model, cast(List[float], embedding))
                for data_model, embedding in zip(data_models, embeddings)
            ]
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Error in embed_batch: {e}")
            # Return empty list on error
            return []
    
    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: List[float]) -> EmbeddedChunkT:
        """Map chunk and embedding to embedded chunk entity"""
        pass


class PostEmbeddingHandler(EmbeddingDataHandler[PostChunk, EmbeddedPostChunk]):
    """Handler for embedding post chunks"""
    
    def map_model(self, data_model: PostChunk, embedding: List[float]) -> EmbeddedPostChunk:
        return EmbeddedPostChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
                **data_model.metadata
            }
        )


class ArticleEmbeddingHandler(EmbeddingDataHandler[ArticleChunk, EmbeddedArticleChunk]):
    """Handler for embedding article chunks"""
    
    def map_model(self, data_model: ArticleChunk, embedding: List[float]) -> EmbeddedArticleChunk:
        return EmbeddedArticleChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
                **data_model.metadata
            }
        )


class RepositoryEmbeddingHandler(EmbeddingDataHandler[RepositoryChunk, EmbeddedRepositoryChunk]):
    """Handler for embedding repository chunks"""
    
    def map_model(self, data_model: RepositoryChunk, embedding: List[float]) -> EmbeddedRepositoryChunk:
        return EmbeddedRepositoryChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
                **data_model.metadata
            }
        )


class EmbeddingHandlerFactory:
    """Factory for creating embedding handlers based on data category"""
    
    @staticmethod
    def create_handler(data_category: DataCategory) -> EmbeddingDataHandler:
        """
        Create appropriate embedding handler for data category
        
        Args:
            data_category: Category of data to embed
            
        Returns:
            Appropriate embedding handler instance
            
        Raises:
            ValueError: If data category is not supported
        """
        if data_category == DataCategory.POSTS:
            return PostEmbeddingHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleEmbeddingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryEmbeddingHandler()
        else:
            raise ValueError(f"Unsupported data type: {data_category}")


class EmbeddingDispatcher:
    """Dispatcher for embedding chunks based on their data category"""
    
    embedding_factory = EmbeddingHandlerFactory()
    
    @classmethod
    def dispatch(cls, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """
        Dispatch chunks to appropriate embedding handler
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of embedded chunks
        """
        try:
            if not chunks:
                return []
            
            # Get data category from first chunk's Config
            data_category = chunks[0].Config.get_category()
            
            # Get appropriate handler
            handler = cls.embedding_factory.create_handler(data_category)
            
            # Embed chunks
            embedded_chunks = handler.embed_batch(chunks)
            
            logger.info(
                "Chunks embedded successfully.",
                data_category=data_category,
                chunk_count=len(chunks),
                embedded_count=len(embedded_chunks)
            )
            
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Error in embedding dispatcher: {e}")
            return []
