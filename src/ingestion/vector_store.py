"""
Vector database integration module for the ingestion pipeline.
Handles storage and retrieval of embeddings using Qdrant.
"""

from typing import List, Dict, Any, Optional
import uuid
from dataclasses import dataclass
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
    )
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
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        self.config = config
        self.client: Optional[QdrantClient] = None
        self.collection_name = config.collection_name

        self._initialize_client()
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_client(self) -> None:
        """Initialize Qdrant client and verify connectivity."""
        try:
            kwargs: Dict[str, Any] = {
                "host": self.config.host,
                "port": self.config.port,
                "prefer_grpc": False,
            }
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key

            self.client = QdrantClient(**kwargs)

            # Verify connection is alive
            self.client.get_collections()
            logger.info(
                f"Connected to Qdrant at {self.config.host}:{self.config.port}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            existing = {
                c.name for c in self.client.get_collections().collections
            }
            if self.collection_name not in existing:
                self._create_collection()
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(
                    f"Collection '{self.collection_name}' already exists"
                )
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def _create_collection(self) -> None:
        """Create a new Qdrant collection with the configured parameters."""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        distance = distance_map.get(
            self.config.distance_metric.lower(), Distance.COSINE
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.config.embedding_dimension,
                distance=distance,
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_chunks(
        self, embedded_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store embedded chunks in Qdrant.

        Args:
            embedded_chunks: List of dicts, each containing an 'embedding'
                             key plus chunk metadata.

        Returns:
            List of point IDs (UUIDs) that were stored.
        """
        if not embedded_chunks:
            return []

        logger.info(f"Storing {len(embedded_chunks)} chunks in Qdrant")

        batch_size = 50  # Reduced batch size to stay under payload limits
        all_point_ids: List[str] = []

        try:
            for batch_start in range(0, len(embedded_chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(embedded_chunks))
                batch_chunks = embedded_chunks[batch_start:batch_end]
                
                # Process only current batch
                points: List[PointStruct] = []
                batch_point_ids: List[str] = []

                for chunk_data in batch_chunks:
                    point_id = str(uuid.uuid4())
                    batch_point_ids.append(point_id)

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=chunk_data["embedding"],
                            payload={
                                "text": chunk_data["text"],
                                "chunk_id": chunk_data["chunk_id"],
                                "document_id": chunk_data["document_id"],
                                "document_type": chunk_data["document_type"],
                                "chunk_index": chunk_data["chunk_index"],
                                "metadata": chunk_data["metadata"],
                            },
                        )
                    )

                # Upload current batch
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                
                all_point_ids.extend(batch_point_ids)
                logger.info(f"Stored batch {batch_start//batch_size + 1}: {len(points)} chunks")

            logger.info(f"Successfully stored {len(all_point_ids)} chunks total")
            return all_point_ids

        except Exception as e:
            logger.error(f"Failed to store chunks in Qdrant: {e}")
            raise

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks whose vectors are most similar to *query_vector*.

        Args:
            query_vector:      Query embedding vector.
            limit:             Maximum number of results to return.
            score_threshold:   Minimum similarity score (inclusive).
            filter_conditions: Optional equality filters keyed by payload field.

        Returns:
            List of matching chunks ordered by descending similarity score.
        """
        try:
            search_filter: Optional[models.Filter] = None
            if filter_conditions:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                        for key, value in filter_conditions.items()
                    ]
                )

            # query_points() is the current API (qdrant-client >= 1.7).
            # Older .search() was removed in newer releases.
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold if score_threshold > 0.0 else None,
                with_payload=True,
            )

            # query_points returns a QueryResponse with a .points attribute
            raw_results = response.points if hasattr(response, "points") else response

            similar_chunks = [
                {
                    "id": r.id,
                    "score": r.score,
                    "text": r.payload.get("text", ""),
                    "chunk_id": r.payload.get("chunk_id", ""),
                    "document_id": r.payload.get("document_id", ""),
                    "document_type": r.payload.get("document_type", ""),
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "metadata": r.payload.get("metadata", {}),
                }
                for r in raw_results
            ]

            logger.info(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks

        except Exception as e:
            logger.error(f"Failed to search in Qdrant: {e}")
            raise

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single chunk by its *chunk_id* payload field.

        Returns None if no matching chunk is found.
        """
        try:
            # FIX: pass with_vectors=True so point.vector is populated.
            # Previously the flag was absent, so point.vector was always None.
            points, _next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_id",
                            match=models.MatchValue(value=chunk_id),
                        )
                    ]
                ),
                limit=1,
                with_vectors=True,
                with_payload=True,
            )

            if not points:
                return None

            point = points[0]
            return {
                "id": point.id,
                "text": point.payload.get("text", ""),
                "chunk_id": point.payload.get("chunk_id", ""),
                "document_id": point.payload.get("document_id", ""),
                "document_type": point.payload.get("document_type", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "metadata": point.payload.get("metadata", {}),
                "vector": point.vector,
            }

        except Exception as e:
            logger.error(f"Failed to get chunk '{chunk_id}': {e}")
            return None

    def get_collection_info(self) -> Dict[str, Any]:
        """Return metadata about the active collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            vectors_cfg = info.config.params.vectors
            return {
                "name": self.collection_name,
                "vectors_count": getattr(info, 'vectors_count', info.points_count),
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', info.points_count),
                "points_count": info.points_count,
                "status": info.status,
                "optimizer_status": getattr(info, 'optimizer_status', 'unknown'),
                "vector_size": vectors_cfg.size,
                "distance": vectors_cfg.distance,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def delete_collection(self) -> None:
        """Permanently delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def clear_collection(self) -> None:
        """
        Remove all points from the collection while preserving its schema.

        FIX: the original code called ``models.MatchAll()`` which does not
        exist in qdrant-client and raises ``AttributeError`` at runtime.
        The correct approach is to recreate the collection.
        """
        try:
            # Recreate preserves vector config while wiping all data.
            self.client.delete_collection(self.collection_name)
            self._create_collection()
            logger.info(
                f"Cleared all points from collection: {self.collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks that belong to *document_id*, sorted by index.
        """
        try:
            points, _next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
            )

            chunks = [
                {
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "chunk_id": point.payload.get("chunk_id", ""),
                    "document_id": point.payload.get("document_id", ""),
                    "document_type": point.payload.get("document_type", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "metadata": point.payload.get("metadata", {}),
                }
                for point in points
            ]

            chunks.sort(key=lambda x: x["chunk_index"])
            logger.info(
                f"Found {len(chunks)} chunks for document '{document_id}'"
            )
            return chunks

        except Exception as e:
            logger.error(
                f"Failed to get chunks for document '{document_id}': {e}"
            )
            return []