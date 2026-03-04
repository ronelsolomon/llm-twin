"""
Main ingestion pipeline orchestrator.
Coordinates cleaning, chunking, embedding, and vector storage.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from .cleaner import DataCleaner
from .chunker import TextChunker, Chunk
from .embedder import EmbeddingGenerator
from .vector_store import QdrantVectorStore, VectorStoreConfig
from .hybrid_vector_store import HybridVectorStore, HybridStoreConfig


@dataclass
class PipelineConfig:
    """Configuration for the ingestion pipeline."""
    
    # Data cleaning
    clean_data: bool = True
    
    # Text chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding generation
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_type: str = "sentence_transformers"
    openai_api_key: Optional[str] = None
    embedding_batch_size: int = 128
    
    # Vector store
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "document_chunks"
    qdrant_distance_metric: str = "cosine"
    
    # Hybrid store settings
    use_hybrid_store: bool = True
    faiss_index_path: str = "data/faiss_index"
    faiss_metadata_path: str = "data/faiss_metadata.json"
    qdrant_payload_limit_mb: int = 30
    max_batch_size: int = 25
    
    # Pipeline settings
    skip_existing: bool = True
    log_level: str = "INFO"


class IngestionPipeline:
    """Main ingestion pipeline that processes documents and stores them in Qdrant."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the ingestion pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.cleaner = None
        self.chunker = None
        self.embedder = None
        self.vector_store = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing ingestion pipeline components")
        
        # Initialize data cleaner
        if self.config.clean_data:
            self.cleaner = DataCleaner()
            logger.info("Data cleaner initialized")
        
        # Initialize text chunker
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        logger.info("Text chunker initialized")
        
        # Initialize embedding generator
        self.embedder = EmbeddingGenerator(
            model_name=self.config.embedding_model,
            model_type=self.config.embedding_model_type,
            openai_api_key=self.config.openai_api_key,
            batch_size=self.config.embedding_batch_size
        )
        logger.info(f"Embedding generator initialized with {self.config.embedding_model}")
        
        # Initialize vector store
        if self.config.use_hybrid_store:
            hybrid_config = HybridStoreConfig(
                qdrant_host=self.config.qdrant_host,
                qdrant_port=self.config.qdrant_port,
                qdrant_api_key=self.config.qdrant_api_key,
                qdrant_collection_name=self.config.qdrant_collection_name,
                faiss_index_path=self.config.faiss_index_path,
                faiss_metadata_path=self.config.faiss_metadata_path,
                embedding_dimension=self.embedder.embedding_dim,
                distance_metric=self.config.qdrant_distance_metric,
                qdrant_payload_limit_mb=self.config.qdrant_payload_limit_mb,
                max_batch_size=self.config.max_batch_size,
            )
            self.vector_store = HybridVectorStore(hybrid_config)
            logger.info("Hybrid vector store initialized (Qdrant + FAISS)")
        else:
            vector_config = VectorStoreConfig(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                api_key=self.config.qdrant_api_key,
                collection_name=self.config.qdrant_collection_name,
                embedding_dimension=self.embedder.embedding_dim,
                distance_metric=self.config.qdrant_distance_metric
            )
            self.vector_store = QdrantVectorStore(vector_config)
            logger.info("Qdrant vector store initialized")
    
    def process_documents(self, documents: List[Dict[str, Any]], 
                         document_type: str) -> Dict[str, Any]:
        """
        Process a list of documents through the entire pipeline.
        
        Args:
            documents: List of documents to process
            document_type: Type of documents ('article', 'repository', etc.)
            
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Starting pipeline for {len(documents)} {document_type} documents")
        
        results = {
            'input_documents': len(documents),
            'processed_documents': 0,
            'total_chunks': 0,
            'stored_chunks': 0,
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Step 1: Clean documents
            if self.cleaner:
                logger.info("Step 1: Cleaning documents")
                cleaned_docs = self.cleaner.clean_documents(documents, document_type)
                results['processed_documents'] = len(cleaned_docs)
            else:
                cleaned_docs = documents
                results['processed_documents'] = len(documents)
            
            # Step 2: Chunk documents
            logger.info("Step 2: Chunking documents")
            chunks = self.chunker.chunk_documents(cleaned_docs, self.cleaner if self.cleaner else None)
            results['total_chunks'] = len(chunks)
            
            # Get chunk statistics
            chunk_stats = self.chunker.get_chunk_stats(chunks)
            results['statistics']['chunking'] = chunk_stats
            
            # Step 3: Generate embeddings
            logger.info("Step 3: Generating embeddings")
            embedded_chunks = self.embedder.embed_chunks(chunks)
            
            # Get embedding statistics
            embedding_stats = self.embedder.get_embedding_statistics(embedded_chunks)
            results['statistics']['embedding'] = embedding_stats
            
            # Step 4: Store in vector database
            logger.info("Step 4: Storing chunks in Qdrant")
            stored_point_ids = self.vector_store.store_chunks(embedded_chunks)
            results['stored_chunks'] = len(stored_point_ids)
            
            # Get vector store statistics
            collection_info = self.vector_store.get_collection_info()
            results['statistics']['vector_store'] = collection_info
            
            logger.info(f"Pipeline completed successfully. Stored {results['stored_chunks']} chunks")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def process_json_file(self, file_path: str, document_type: str) -> Dict[str, Any]:
        """
        Process documents from a JSON file.
        
        Args:
            file_path: Path to JSON file containing documents
            document_type: Type of documents in the file
            
        Returns:
            Dictionary with processing results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return self.process_documents(documents, document_type)
            
        except Exception as e:
            error_msg = f"Failed to process file {file_path}: {str(e)}"
            logger.error(error_msg)
            return {'errors': [error_msg]}
    
    def process_data_directory(self, data_dir: str) -> Dict[str, Any]:
        """
        Process all JSON files in the data directory.
        
        Args:
            data_dir: Path to data directory containing JSON files
            
        Returns:
            Dictionary with combined processing results
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            error_msg = f"Data directory does not exist: {data_dir}"
            logger.error(error_msg)
            return {'errors': [error_msg]}
        
        logger.info(f"Processing JSON files in directory: {data_dir}")
        
        combined_results = {
            'files_processed': 0,
            'total_documents': 0,
            'total_chunks': 0,
            'total_stored': 0,
            'file_results': {},
            'errors': []
        }
        
        # Process each JSON file
        for json_file in data_path.glob("*.json"):
            try:
                # Determine document type from filename
                if 'articles' in json_file.name.lower():
                    document_type = 'article'
                elif 'repositories' in json_file.name.lower():
                    document_type = 'repository'
                elif 'users' in json_file.name.lower():
                    document_type = 'user'
                else:
                    document_type = 'unknown'
                
                logger.info(f"Processing file: {json_file.name} (type: {document_type})")
                
                # Process the file
                file_results = self.process_json_file(str(json_file), document_type)
                
                # Combine results
                combined_results['files_processed'] += 1
                combined_results['total_documents'] += file_results.get('processed_documents', 0)
                combined_results['total_chunks'] += file_results.get('total_chunks', 0)
                combined_results['total_stored'] += file_results.get('stored_chunks', 0)
                combined_results['file_results'][json_file.name] = file_results
                
                # Add any errors
                if file_results.get('errors'):
                    combined_results['errors'].extend(file_results['errors'])
                
            except Exception as e:
                error_msg = f"Failed to process file {json_file.name}: {str(e)}"
                logger.error(error_msg)
                combined_results['errors'].append(error_msg)
        
        logger.info(f"Directory processing completed. Processed {combined_results['files_processed']} files")
        return combined_results
    
    def search_similar_documents(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using a text query.
        
        Args:
            query_text: Query text to search for
            limit: Maximum number of results
            
        Returns:
            List of similar document chunks
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedder.generate_embeddings([query_text])
            if not query_embedding:
                return []
            
            # Search in vector store
            results = self.vector_store.search_similar(
                query_vector=query_embedding[0],
                limit=limit
            )
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics."""
        try:
            collection_info = self.vector_store.get_collection_info()
            
            return {
                'pipeline_config': {
                    'chunk_size': self.config.chunk_size,
                    'embedding_model': self.config.embedding_model,
                    'vector_store': self.config.qdrant_collection_name
                },
                'vector_store_info': collection_info,
                'components_status': {
                    'cleaner': self.cleaner is not None,
                    'chunker': self.chunker is not None,
                    'embedder': self.embedder is not None,
                    'vector_store': self.vector_store is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {'error': str(e)}
    
    def test_pipeline(self, test_text: str = "This is a test document for the ingestion pipeline.") -> Dict[str, Any]:
        """Test the entire pipeline with a sample document."""
        logger.info("Testing pipeline with sample document")
        
        try:
            # Create test document
            test_doc = {
                'id': 'test_doc_001',
                'title': 'Test Document',
                'content': {'main_text': test_text},
                'platform': 'test',
                'author_id': 'test_author',
                'author_full_name': 'Test Author'
            }
            
            # Process through pipeline
            results = self.process_documents([test_doc], 'test')
            
            # Test search
            search_results = self.search_similar_documents(test_text, limit=5)
            results['test_search'] = {
                'query': test_text,
                'results_count': len(search_results),
                'top_result': search_results[0] if search_results else None
            }
            
            logger.info("Pipeline test completed")
            return results
            
        except Exception as e:
            error_msg = f"Pipeline test failed: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}
