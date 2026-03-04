"""
Vector database operations for document storage and retrieval
"""
from typing import Dict, List, Type, Any
from abc import ABC, abstractmethod
from loguru import logger
from .domain.documents import NoSQLBaseDocument, ArticleDocument, RepositoryDocument
from .domain.vector_documents import (
    VectorBaseDocument, CleanedArticleDocument, CleanedRepositoryDocument,
    ArticleChunk, RepositoryChunk, EmbeddedArticleChunk, EmbeddedRepositoryChunk,
    DataCategory
)


class VectorStoreOperations:
    """Operations for vector database interactions"""
    
    @staticmethod
    def group_by_class(documents: List[Dict[str, Any]]) -> Dict[Type, List[Dict[str, Any]]]:
        """
        Group documents by their class type based on metadata
        
        Args:
            documents: List of document dictionaries with metadata
            
        Returns:
            Dictionary mapping document classes to their respective documents
        """
        grouped = {}
        
        for doc in documents:
            # Extract document type from metadata
            doc_type = "unknown"
            if isinstance(doc, dict) and "metadata" in doc:
                doc_type = doc["metadata"].get("type", "unknown")
            elif hasattr(doc, 'metadata'):
                doc_type = doc.metadata.get("type", "unknown")
            
            # Map document type to appropriate class
            if doc_type == "article":
                if "embedding" in doc:
                    document_class = EmbeddedArticleChunk
                else:
                    document_class = ArticleChunk
            elif doc_type == "repository":
                if "embedding" in doc:
                    document_class = EmbeddedRepositoryChunk
                else:
                    document_class = RepositoryChunk
            else:
                # Default to ArticleChunk for unknown types
                document_class = ArticleChunk
                logger.warning(f"Unknown document type '{doc_type}', defaulting to ArticleChunk")
            
            if document_class not in grouped:
                grouped[document_class] = []
            grouped[document_class].append(doc)
        
        return grouped


class VectorStore:
    """Main vector store interface"""
    
    def __init__(self):
        """Initialize vector store connection"""
        # Placeholder for vector store initialization
        logger.info("Initializing vector store (mock implementation)")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Group documents by type
            grouped_documents = VectorStoreOperations.group_by_class(documents)
            
            # Insert each group
            for document_class, docs in grouped_documents.items():
                logger.info(f"Loading documents into {document_class.Config.get_collection_name()}")
                
                # Process in batches
                batch_size = 4
                for batch in self._batch_documents(docs, batch_size):
                    try:
                        success = self._bulk_insert(document_class, batch)
                        if not success:
                            logger.error(f"Failed to insert batch of {len(batch)} documents")
                            return False
                    except Exception as e:
                        logger.error(f"Error inserting batch: {e}")
                        return False
            
            logger.info("All documents successfully loaded into vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def _bulk_insert(self, document_class: Type, documents: List[Dict[str, Any]]) -> bool:
        """
        Insert multiple documents into the vector database
        
        Args:
            document_class: The document class to use
            documents: List of documents to insert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection_name = document_class.Config.get_collection_name()
            logger.info(f"Bulk inserting {len(documents)} documents into {collection_name}")
            
            # Placeholder for actual vector database insertion
            # In a real implementation, this would connect to Qdrant, Pinecone, etc.
            
            for doc in documents:
                # Extract content and embedding
                content = doc.get("content", "")
                embedding = doc.get("embedding", [])
                metadata = doc.get("metadata", {})
                
                # Validate required fields
                if not content:
                    logger.warning("Document missing content, skipping")
                    continue
                
                if not embedding and document_class.Config.get_category() != DataCategory.POSTS:
                    logger.warning("Document missing embedding, skipping")
                    continue
                
                # Mock insertion - replace with actual vector DB operations
                logger.debug(f"Inserting document: {metadata.get('source', 'unknown')}")
            
            logger.info(f"Successfully inserted {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error during bulk insert: {e}")
            return False
    
    def _batch_documents(self, documents: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """
        Split documents into batches
        
        Args:
            documents: List of documents to batch
            batch_size: Size of each batch
            
        Yields:
            Batches of documents
        """
        for i in range(0, len(documents), batch_size):
            yield documents[i:i + batch_size]
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of similar documents
        """
        # Placeholder for search implementation
        logger.info(f"Searching for: '{query}' (limit: {limit})")
        return []
