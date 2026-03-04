"""
Hybrid vector store that can switch between multiple backends.
Uses Qdrant as primary, falls back to FAISS when payload limits are exceeded.
"""

from typing import List, Dict, Any, Optional, Union
import uuid
import json
import os
from dataclasses import dataclass
from pathlib import Path
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

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available")

from .chunker import Chunk


@dataclass
class HybridStoreConfig:
    """Configuration for hybrid vector store."""
    # Qdrant config
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "document_chunks"
    
    # FAISS config
    faiss_index_path: str = "data/faiss_index"
    faiss_metadata_path: str = "data/faiss_metadata.json"
    
    # General config
    embedding_dimension: int = 384
    distance_metric: str = "cosine"
    
    # Switching thresholds
    qdrant_payload_limit_mb: int = 30  # Conservative limit below 32MB
    max_batch_size: int = 25  # Smaller batches to avoid payload limits


class FAISSVectorStore:
    """FAISS-based local vector store for fallback storage."""
    
    def __init__(self, config: HybridStoreConfig):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        self.config = config
        self.index_path = Path(config.faiss_index_path)
        self.metadata_path = Path(config.faiss_metadata_path)
        
        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        
        self._initialize_index()
        self._load_metadata()
    
    def _initialize_index(self):
        """Initialize or load FAISS index."""
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded FAISS index from {self.index_path}")
            else:
                # Create new index based on distance metric
                if self.config.distance_metric.lower() == "cosine":
                    # For cosine similarity, use inner product after normalization
                    self.index = faiss.IndexFlatIP(self.config.embedding_dimension)
                else:
                    # For Euclidean distance
                    self.index = faiss.IndexFlatL2(self.config.embedding_dimension)
                
                logger.info(f"Created new FAISS index with {self.config.embedding_dimension} dimensions")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _load_metadata(self):
        """Load metadata from file."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded {len(self.metadata)} metadata entries")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            self.metadata = []
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def store_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> List[str]:
        """Store chunks in FAISS index."""
        if not embedded_chunks:
            return []
        
        logger.info(f"Storing {len(embedded_chunks)} chunks in FAISS")
        
        point_ids = []
        vectors = []
        
        for chunk_data in embedded_chunks:
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Store metadata
            metadata_entry = {
                "id": point_id,
                "text": chunk_data["text"],
                "chunk_id": chunk_data["chunk_id"],
                "document_id": chunk_data["document_id"],
                "document_type": chunk_data["document_type"],
                "chunk_index": chunk_data["chunk_index"],
                "metadata": chunk_data["metadata"],
            }
            self.metadata.append(metadata_entry)
            
            # Prepare vector for FAISS
            vector = np.array(chunk_data["embedding"], dtype=np.float32)
            if self.config.distance_metric.lower() == "cosine":
                # Normalize for cosine similarity
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        
        # Add vectors to index
        vectors_array = np.vstack(vectors)
        self.index.add(vectors_array)
        
        # Save to disk
        self._save_index()
        self._save_metadata()
        
        logger.info(f"Successfully stored {len(point_ids)} chunks in FAISS")
        return point_ids
    
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in FAISS."""
        try:
            # Prepare query vector
            query = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            if self.config.distance_metric.lower() == "cosine":
                query = query / np.linalg.norm(query)
            
            # Search
            scores, indices = self.index.search(query, min(limit, len(self.metadata)))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if score < score_threshold:
                    continue
                
                metadata = self.metadata[idx]
                
                # Apply filters if specified
                if filter_conditions:
                    match = True
                    for key, value in filter_conditions.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                result = {
                    "id": metadata["id"],
                    "score": float(score),
                    "text": metadata["text"],
                    "chunk_id": metadata["chunk_id"],
                    "document_id": metadata["document_id"],
                    "document_type": metadata["document_type"],
                    "chunk_index": metadata["chunk_index"],
                    "metadata": metadata["metadata"],
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks in FAISS")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search in FAISS: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        return {
            "type": "faiss",
            "vectors_count": self.index.ntotal if self.index else 0,
            "metadata_count": len(self.metadata),
            "index_path": str(self.index_path),
            "distance_metric": self.config.distance_metric,
        }


class HybridVectorStore:
    """Hybrid vector store that switches between Qdrant and FAISS."""
    
    def __init__(self, config: HybridStoreConfig):
        self.config = config
        
        # Initialize stores
        self.qdrant_store = None
        self.faiss_store = None
        
        if QDRANT_AVAILABLE:
            try:
                from .vector_store import QdrantVectorStore, VectorStoreConfig
                qdrant_config = VectorStoreConfig(
                    host=config.qdrant_host,
                    port=config.qdrant_port,
                    api_key=config.qdrant_api_key,
                    collection_name=config.qdrant_collection_name,
                    embedding_dimension=config.embedding_dimension,
                    distance_metric=config.distance_metric,
                )
                self.qdrant_store = QdrantVectorStore(qdrant_config)
                logger.info("Qdrant store initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant: {e}")
        
        if FAISS_AVAILABLE:
            try:
                self.faiss_store = FAISSVectorStore(config)
                logger.info("FAISS store initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS: {e}")
        
        if not self.qdrant_store and not self.faiss_store:
            raise RuntimeError("No vector stores available. Install qdrant-client and/or faiss")
    
    def _estimate_payload_size(self, chunks: List[Dict[str, Any]]) -> float:
        """Estimate payload size in MB."""
        if not chunks:
            return 0.0
        
        # Sample a few chunks to estimate size
        sample_chunks = chunks[:min(5, len(chunks))]
        total_size = 0
        
        for chunk in sample_chunks:
            # Rough estimation of JSON payload size
            payload = {
                "text": chunk["text"],
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_type": chunk["document_type"],
                "chunk_index": chunk["chunk_index"],
                "metadata": chunk["metadata"],
            }
            total_size += len(json.dumps(payload).encode('utf-8'))
        
        avg_size_per_chunk = total_size / len(sample_chunks)
        estimated_total_mb = (avg_size_per_chunk * len(chunks)) / (1024 * 1024)
        
        return estimated_total_mb
    
    def store_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> List[str]:
        """Store chunks using the best available store."""
        if not embedded_chunks:
            return []
        
        logger.info(f"Storing {len(embedded_chunks)} chunks in hybrid store")
        
        # Estimate payload size
        estimated_size_mb = self._estimate_payload_size(embedded_chunks)
        logger.info(f"Estimated payload size: {estimated_size_mb:.2f} MB")
        
        # Try Qdrant first if available and payload is small enough
        if self.qdrant_store and estimated_size_mb < self.config.qdrant_payload_limit_mb:
            try:
                # Use smaller batches for Qdrant
                batch_size = self.config.max_batch_size
                all_point_ids = []
                
                for i in range(0, len(embedded_chunks), batch_size):
                    batch = embedded_chunks[i:i + batch_size]
                    batch_ids = self.qdrant_store.store_chunks(batch)
                    all_point_ids.extend(batch_ids)
                    logger.info(f"Stored batch {i//batch_size + 1} in Qdrant: {len(batch_ids)} chunks")
                
                logger.info(f"Successfully stored all {len(all_point_ids)} chunks in Qdrant")
                return all_point_ids
                
            except Exception as e:
                logger.warning(f"Qdrant storage failed, falling back to FAISS: {e}")
        
        # Fall back to FAISS
        if self.faiss_store:
            try:
                point_ids = self.faiss_store.store_chunks(embedded_chunks)
                logger.info(f"Successfully stored {len(point_ids)} chunks in FAISS")
                return point_ids
            except Exception as e:
                logger.error(f"FAISS storage failed: {e}")
                raise
        
        raise RuntimeError("No available vector store could handle the request")
    
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all available stores."""
        all_results = []
        
        # Search Qdrant first
        if self.qdrant_store:
            try:
                qdrant_results = self.qdrant_store.search_similar(
                    query_vector, limit, score_threshold, filter_conditions
                )
                all_results.extend(qdrant_results)
                logger.info(f"Found {len(qdrant_results)} results in Qdrant")
            except Exception as e:
                logger.warning(f"Qdrant search failed: {e}")
        
        # Search FAISS if needed
        if self.faiss_store and len(all_results) < limit:
            try:
                remaining_limit = limit - len(all_results)
                faiss_results = self.faiss_store.search_similar(
                    query_vector, remaining_limit, score_threshold, filter_conditions
                )
                all_results.extend(faiss_results)
                logger.info(f"Found {len(faiss_results)} results in FAISS")
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:limit]
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get combined statistics from all stores."""
        stats = {
            "type": "hybrid",
            "stores": {},
            "total_vectors": 0,
            "total_points": 0,
        }
        
        if self.qdrant_store:
            try:
                qdrant_stats = self.qdrant_store.get_collection_info()
                stats["stores"]["qdrant"] = qdrant_stats
                stats["total_vectors"] += qdrant_stats.get("vectors_count", 0)
                stats["total_points"] += qdrant_stats.get("points_count", 0)
            except Exception as e:
                logger.warning(f"Failed to get Qdrant stats: {e}")
        
        if self.faiss_store:
            try:
                faiss_stats = self.faiss_store.get_stats()
                stats["stores"]["faiss"] = faiss_stats
                stats["total_vectors"] += faiss_stats.get("vectors_count", 0)
                stats["total_points"] += faiss_stats.get("metadata_count", 0)
            except Exception as e:
                logger.warning(f"Failed to get FAISS stats: {e}")
        
        return stats
