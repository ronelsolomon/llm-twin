"""
Vector database integration module for the ingestion pipeline.
Handles storage and retrieval of embeddings using Qdrant.
"""

from typing import List, Dict, Any, Optional, Union
import uuid
from dataclasses import dataclass
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not available")

from .chunker import Chunk


@dataclass
class VectorStoreConfig:
    """Configuration for vector store connection."""
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    collection_name: str = "document_chunks"
    embedding_dimension: int = 384
    distance_metric: str = "cosine"


class QdrantVectorStore:
    """Qdrant-based vector store for document chunks."""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the Qdrant vector store.
        
        Args:
            config: Vector store configuration
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")
        
        self.config = config
        self.client = None
        self.collection_name = config.collection_name
        
        self._initialize_client()
        self._ensure_collection()
    
    def _initialize_client(self):
        """Initialize Qdrant client."""
        try:
            if self.config.api_key:
                # Cloud connection
                self.client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    api_key=self.config.api_key,
                    prefer_grpc=False
                )
            else:
                # Local connection
                self.client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port
                )
            
            # Test connection
            self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _ensure_collection(self):
        """Ensure the collection exists."""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections
            )
            
            if not collection_exists:
                self._create_collection()
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def _create_collection(self):
        """Create a new collection."""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        
        distance = distance_map.get(
            self.config.distance_metric.lower(), 
            Distance.COSINE
        )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.config.embedding_dimension,
                distance=distance
            )
        )
    
    def store_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Store embedded chunks in Qdrant.
        
        Args:
            embedded_chunks: List of chunks with embeddings
            
        Returns:
            List of point IDs that were stored
        """
        if not embedded_chunks:
            return []
        
        logger.info(f"Storing {len(embedded_chunks)} chunks in Qdrant")
        
        points = []
        point_ids = []
        
        for chunk_data in embedded_chunks:
            # Generate unique point ID
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Prepare point for Qdrant
            point = PointStruct(
                id=point_id,
                vector=chunk_data['embedding'],
                payload={
                    'text': chunk_data['text'],
                    'chunk_id': chunk_data['chunk_id'],
                    'document_id': chunk_data['document_id'],
                    'document_type': chunk_data['document_type'],
                    'chunk_index': chunk_data['chunk_index'],
                    'metadata': chunk_data['metadata']
                }
            )
            points.append(point)
        
        try:
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
            
            logger.info(f"Successfully stored {len(point_ids)} chunks")
            return point_ids
            
        except Exception as e:
            logger.error(f"Failed to store chunks in Qdrant: {e}")
            raise
    
    def search_similar(self, 
                      query_vector: List[float], 
                      limit: int = 10,
                      score_threshold: float = 0.5,
                      filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions
            
        Returns:
            List of similar chunks with scores
        """
        try:
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Convert results to dictionary format
            similar_chunks = []
            for result in results:
                similar_chunks.append({
                    'id': result.id,
                    'score': result.score,
                    'text': result.payload.get('text', ''),
                    'chunk_id': result.payload.get('chunk_id', ''),
                    'document_id': result.payload.get('document_id', ''),
                    'document_type': result.payload.get('document_type', ''),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'metadata': result.payload.get('metadata', {})
                })
            
            logger.info(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Failed to search in Qdrant: {e}")
            raise
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its chunk ID."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_id",
                            match=models.MatchValue(value=chunk_id)
                        )
                    ]
                ),
                limit=1
            )
            
            if results[0]:  # points
                point = results[0][0]
                return {
                    'id': point.id,
                    'text': point.payload.get('text', ''),
                    'chunk_id': point.payload.get('chunk_id', ''),
                    'document_id': point.payload.get('document_id', ''),
                    'document_type': point.payload.get('document_type', ''),
                    'chunk_index': point.payload.get('chunk_index', 0),
                    'metadata': point.payload.get('metadata', {}),
                    'vector': point.vector
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                'name': self.collection_name,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'status': info.status,
                'optimizer_status': info.optimizer_status,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def clear_collection(self):
        """Clear all points from the collection but keep the collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[models.MatchAll()]
                )
            )
            logger.info(f"Cleared all points from collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1000  # Adjust based on expected chunk count
            )
            
            chunks = []
            for point in results[0]:  # points
                chunks.append({
                    'id': point.id,
                    'text': point.payload.get('text', ''),
                    'chunk_id': point.payload.get('chunk_id', ''),
                    'document_id': point.payload.get('document_id', ''),
                    'document_type': point.payload.get('document_type', ''),
                    'chunk_index': point.payload.get('chunk_index', 0),
                    'metadata': point.payload.get('metadata', {})
                })
            
            # Sort by chunk_index
            chunks.sort(key=lambda x: x['chunk_index'])
            
            logger.info(f"Found {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []
