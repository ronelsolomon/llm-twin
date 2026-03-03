#!/usr/bin/env python3
"""
Example usage of the ingestion pipeline with Qdrant vector database.

This script demonstrates how to:
1. Set up the ingestion pipeline
2. Process documents from JSON files
3. Search for similar documents
4. Get pipeline statistics
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ingestion import IngestionPipeline, PipelineConfig
from loguru import logger


def main():
    """Main function demonstrating the ingestion pipeline."""
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Pipeline configuration
    config = PipelineConfig(
        # Text chunking settings
        chunk_size=800,
        chunk_overlap=200,
        
        # Embedding settings
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_type="sentence_transformers",
        embedding_batch_size=16,
        
        # Qdrant settings
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection_name="document_chunks",
        qdrant_distance_metric="cosine",
        
        # Pipeline settings
        clean_data=True,
        skip_existing=True
    )
    
    try:
        # Initialize the pipeline
        logger.info("Initializing ingestion pipeline...")
        pipeline = IngestionPipeline(config)
        
        # Test the pipeline with a sample document
        logger.info("Testing pipeline with sample document...")
        test_results = pipeline.test_pipeline(
            "This is a sample document about artificial intelligence and machine learning. "
            "It contains information about neural networks, deep learning, and natural language processing."
        )
        
        if 'error' not in test_results:
            logger.info("✅ Pipeline test successful!")
            logger.info(f"Test results: {test_results}")
        else:
            logger.error(f"❌ Pipeline test failed: {test_results['error']}")
            return
        
        # Process documents from the data directory
        data_dir = Path(__file__).parent.parent / "data"
        if data_dir.exists():
            logger.info(f"Processing documents from {data_dir}...")
            
            # Process all JSON files in the data directory
            results = pipeline.process_data_directory(str(data_dir))
            
            logger.info("📊 Processing Results:")
            logger.info(f"Files processed: {results.get('files_processed', 0)}")
            logger.info(f"Total documents: {results.get('total_documents', 0)}")
            logger.info(f"Total chunks: {results.get('total_chunks', 0)}")
            logger.info(f"Total stored: {results.get('total_stored', 0)}")
            
            if results.get('errors'):
                logger.warning(f"Errors encountered: {len(results['errors'])}")
                for error in results['errors']:
                    logger.warning(f"  - {error}")
        else:
            logger.warning(f"Data directory not found: {data_dir}")
        
        # Demonstrate search functionality
        logger.info("🔍 Testing search functionality...")
        search_queries = [
            "machine learning algorithms",
            "neural networks",
            "data processing",
            "artificial intelligence"
        ]
        
        for query in search_queries:
            results = pipeline.search_similar_documents(query, limit=3)
            logger.info(f"Query: '{query}' - Found {len(results)} results")
            
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. Score: {result['score']:.3f} - {result['text'][:100]}...")
        
        # Get pipeline status
        logger.info("📈 Getting pipeline status...")
        status = pipeline.get_pipeline_status()
        
        logger.info("Pipeline Status:")
        logger.info(f"  Vector store: {status['pipeline_config']['vector_store']}")
        logger.info(f"  Embedding model: {status['pipeline_config']['embedding_model']}")
        logger.info(f"  Chunk size: {status['pipeline_config']['chunk_size']}")
        
        if 'vector_store_info' in status:
            vs_info = status['vector_store_info']
            logger.info(f"  Collection points: {vs_info.get('points_count', 'N/A')}")
            logger.info(f"  Collection vectors: {vs_info.get('vectors_count', 'N/A')}")
        
        logger.info("✅ Ingestion pipeline example completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline example failed: {e}")
        raise


def setup_qdrant_docker():
    """Instructions for setting up Qdrant with Docker."""
    instructions = """
    
🐳 QDRANT SETUP INSTRUCTIONS:

To run Qdrant locally using Docker, execute:

1. Pull the Qdrant image:
   docker pull qdrant/qdrant

2. Run Qdrant container:
   docker run -p 6333:6333 qdrant/qdrant

3. Or use docker-compose (create docker-compose.yml):
   version: '3.8'
   services:
     qdrant:
       image: qdrant/qdrant
       ports:
         - "6333:6333"
       volumes:
         - ./qdrant_data:/qdrant/storage

4. Verify Qdrant is running:
   curl http://localhost:6333/collections

5. For cloud setup, visit: https://cloud.qdrant.tech/

"""
    print(instructions)


if __name__ == "__main__":
    # Check if Qdrant is running (basic check)
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Qdrant is running locally")
        else:
            logger.warning("⚠️  Qdrant may not be properly configured")
            setup_qdrant_docker()
    except:
        logger.warning("⚠️  Cannot connect to Qdrant. Please ensure it's running:")
        setup_qdrant_docker()
        logger.info("Continuing with example anyway...")
    
    main()
