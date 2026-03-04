"""
Enhanced vector document classes with Config internal classes
"""
from abc import ABC
from typing import Optional, Dict, Any, List, Type, TypeVar, Generic
from uuid import uuid4, UUID
from pydantic import Field, BaseModel
import numpy as np

from .enums import DataCategory

# Type variable for Generic typing
T = TypeVar("T", bound="VectorBaseDocument")

# Mock Qdrant types for now (replace with actual imports when available)
class Record:
    """Mock Record class for Qdrant"""
    def __init__(self, id: str, vector: List[float] = None, payload: Dict[str, Any] = None):
        self.id = id
        self.vector = vector
        self.payload = payload

class PointStruct:
    """Mock PointStruct class for Qdrant"""
    def __init__(self, id: str, vector: List[float] = None, payload: Dict[str, Any] = None):
        self.id = id
        self.vector = vector
        self.payload = payload

# Mock exceptions
class UnexpectedResponse(Exception):
    """Mock exception for Qdrant unexpected responses"""
    pass

class ImproperlyConfigured(Exception):
    """Mock exception for improper configuration"""
    pass

# Mock connection
class MockConnection:
    """Mock Qdrant connection"""
    
    @staticmethod
    def upsert(collection_name: str, points: List[PointStruct]) -> None:
        """Mock upsert operation"""
        print(f"Mock upsert: {len(points)} points to {collection_name}")
    
    @staticmethod
    def scroll(collection_name: str, limit: int = 10, with_payload: bool = True, 
              with_vectors: bool = False, offset: str = None, **kwargs) -> tuple[List[Record], Optional[str]]:
        """Mock scroll operation"""
        print(f"Mock scroll: {collection_name}, limit={limit}, offset={offset}")
        return [], None
    
    @staticmethod
    def search(collection_name: str, query_vector: List[float], limit: int = 10,
              with_payload: bool = True, with_vectors: bool = False, **kwargs) -> List[Record]:
        """Mock search operation"""
        print(f"Mock search: {collection_name}, limit={limit}")
        return []

# Mock connection instance
connection = MockConnection()


class VectorBaseDocument(BaseModel, Generic[T], ABC):
    """Enhanced base class for vector documents with Qdrant integration"""
    
    id: UUID = Field(default_factory=uuid4)
    
    class Config:
        """Configuration for vector document"""
        name: str = ""
        category: DataCategory = DataCategory.ARTICLES
        use_vector_index: bool = False
        
        @classmethod
        def get_collection_name(cls) -> str:
            """Get the collection name from Config"""
            return cls.name
        
        @classmethod
        def get_category(cls) -> DataCategory:
            """Get the data category from Config"""
            return cls.category
        
        @classmethod
        def should_use_vector_index(cls) -> bool:
            """Check if vector index should be used"""
            return cls.use_vector_index
    
    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        """Get collection name with validation"""
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "name"):
            raise ImproperlyConfigured(
                "The class should define a Config class with the 'name' property that reflects collection's name."
            )
        return cls.Config.name
    
    @classmethod
    def bulk_insert(cls: Type[T], documents: List["VectorBaseDocument"]) -> bool:
        """
        Bulk insert documents into vector database with collection creation fallback
        
        Args:
            documents: List of documents to insert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cls._bulk_insert(documents)
        except UnexpectedResponse:
            print(f"Collection '{cls.get_collection_name()}' does not exist. Trying to create collection and reinsert documents.")
            cls.create_collection()
            try:
                cls._bulk_insert(documents)
            except UnexpectedResponse:
                print(f"Failed to insert documents in '{cls.get_collection_name()}'.")
                return False
        return True
    
    @classmethod
    def _bulk_insert(cls: Type[T], documents: List["VectorBaseDocument"]) -> None:
        """
        Private bulk insert method
        
        Args:
            documents: List of documents to insert
        """
        points = [doc.to_point() for doc in documents]
        connection.upsert(collection_name=cls.get_collection_name(), points=points)
    
    @classmethod
    def create_collection(cls: Type[T]) -> None:
        """
        Create collection in vector database
        
        Note: This is a placeholder - implement based on your vector DB requirements
        """
        print(f"Mock creating collection: {cls.get_collection_name()}")
        # In real implementation, this would create the collection with proper configuration
    
    @classmethod
    def bulk_find(cls: Type[T], limit: int = 10, **kwargs) -> tuple[List[T], Optional[UUID]]:
        """
        Find multiple documents using scroll operation
        
        Args:
            limit: Maximum number of documents to return
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (documents, next_offset)
        """
        try:
            documents, next_offset = cls._bulk_find(limit=limit, **kwargs)
        except UnexpectedResponse:
            print(f"Failed to search documents in '{cls.get_collection_name()}'.")
            documents, next_offset = [], None
        return documents, next_offset
    
    @classmethod
    def _bulk_find(cls: Type[T], limit: int = 10, **kwargs) -> tuple[List[T], Optional[UUID]]:
        """
        Private bulk find method using scroll
        
        Args:
            limit: Maximum number of documents to return
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (documents, next_offset)
        """
        collection_name = cls.get_collection_name()
        offset = kwargs.pop("offset", None)
        offset = str(offset) if offset else None
        
        records, next_offset = connection.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", False),
            offset=offset,
            **kwargs,
        )
        
        documents = [cls.from_record(record) for record in records]
        
        if next_offset is not None:
            next_offset = UUID(next_offset, version=4)
            
        return documents, next_offset
    
    @classmethod
    def search(cls: Type[T], query_vector: List[float], limit: int = 10, **kwargs) -> List[T]:
        """
        Search documents using vector similarity
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            **kwargs: Additional parameters
            
        Returns:
            List of matching documents
        """
        try:
            documents = cls._search(query_vector=query_vector, limit=limit, **kwargs)
        except UnexpectedResponse:
            print(f"Failed to search documents in '{cls.get_collection_name()}'.")
            documents = []
        return documents
    
    @classmethod
    def _search(cls: Type[T], query_vector: List[float], limit: int = 10, **kwargs) -> List[T]:
        """
        Private search method
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            **kwargs: Additional parameters
            
        Returns:
            List of matching documents
        """
        collection_name = cls.get_collection_name()
        records = connection.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", False),
            **kwargs,
        )
        documents = [cls.from_record(record) for record in records]
        return documents
    
    @classmethod
    def from_record(cls: Type[T], point: Record) -> T:
        """
        Create document instance from Qdrant record
        
        Args:
            point: Qdrant record point
            
        Returns:
            Document instance
        """
        _id = UUID(point.id, version=4)
        payload = point.payload or {}
        
        attributes = {
            "id": _id,
            **payload,
        }
        
        if cls._has_class_attribute("embedding"):
            payload["embedding"] = point.vector or None
            
        return cls(**attributes)
    
    def to_point(self: T, **kwargs) -> PointStruct:
        """
        Convert document to Qdrant point structure
        
        Args:
            **kwargs: Additional arguments for dict conversion
            
        Returns:
            Qdrant PointStruct
        """
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)
        
        payload = self.dict(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)
        _id = str(payload.pop("id"))
        vector = payload.pop("embedding", {})
        
        if vector and isinstance(vector, np.ndarray):
            vector = vector.tolist()
            
        return PointStruct(id=_id, vector=vector, payload=payload)
    
    @classmethod
    def _has_class_attribute(cls, attr_name: str) -> bool:
        """
        Check if class has a specific attribute
        
        Args:
            attr_name: Name of the attribute to check
            
        Returns:
            True if attribute exists, False otherwise
        """
        return hasattr(cls, attr_name)


class CleanedDocument(VectorBaseDocument, ABC):
    """Base class for cleaned documents"""
    content: str
    platform: str
    author_id: UUID
    author_full_name: str


class CleanedPostDocument(CleanedDocument):
    """Cleaned post document"""
    image: Optional[str] = None
    
    class Config:
        name = "cleaned_posts"
        category = DataCategory.POSTS
        use_vector_index = False


class CleanedArticleDocument(CleanedDocument):
    """Cleaned article document"""
    link: str
    
    class Config:
        name = "cleaned_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


class CleanedRepositoryDocument(CleanedDocument):
    """Cleaned repository document"""
    name: str
    link: str
    
    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False


class Chunk(VectorBaseDocument, ABC):
    """Base class for document chunks"""
    content: str
    platform: str
    document_id: UUID
    author_id: UUID
    author_full_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PostChunk(Chunk):
    """Chunk for post documents"""
    
    class Config:
        name = "post_chunks"
        category = DataCategory.POSTS
        use_vector_index = True


class ArticleChunk(Chunk):
    """Chunk for article documents"""
    
    class Config:
        name = "article_chunks"
        category = DataCategory.ARTICLES
        use_vector_index = True


class RepositoryChunk(Chunk):
    """Chunk for repository documents"""
    
    class Config:
        name = "repository_chunks"
        category = DataCategory.REPOSITORIES
        use_vector_index = True


class EmbeddedChunk(VectorBaseDocument, ABC):
    """Base class for embedded document chunks"""
    content: str
    embedding: Optional[List[float]] = None
    platform: str
    document_id: UUID
    author_id: UUID
    author_full_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddedPostChunk(EmbeddedChunk):
    """Embedded chunk for post documents"""
    
    class Config:
        name = "embedded_post_chunks"
        category = DataCategory.POSTS
        use_vector_index = True


class EmbeddedArticleChunk(EmbeddedChunk):
    """Embedded chunk for article documents"""
    
    class Config:
        name = "embedded_article_chunks"
        category = DataCategory.ARTICLES
        use_vector_index = True


class EmbeddedRepositoryChunk(EmbeddedChunk):
    """Embedded chunk for repository documents"""
    
    class Config:
        name = "embedded_repository_chunks"
        category = DataCategory.REPOSITORIES
        use_vector_index = True
