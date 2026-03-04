#!/usr/bin/env python3
"""
Script to process all raw documents from the data/raw folder and ingest them into Qdrant vector database.
This script converts the standalone extracted data into searchable vectors.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion import IngestionPipeline, PipelineConfig


class RawDocumentProcessor:
    """Process raw documents from data/raw folder and ingest into Qdrant."""
    
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)
        self.processed_files = set()
        self.error_files = []
        
    def load_raw_documents(self) -> List[Dict[str, Any]]:
        """Load all raw documents from the raw directory."""
        documents = []
        
        if not self.raw_dir.exists():
            logger.error(f"Raw directory not found: {self.raw_dir}")
            return documents
        
        logger.info(f"Loading raw documents from {self.raw_dir}")
        
        # Process all JSON files in the raw directory
        for file_path in self.raw_dir.glob("*.json"):
            try:
                logger.info(f"Loading file: {file_path.name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different document formats
                if isinstance(data, dict):
                    # Single document format
                    if 'files' in data:
                        # GitHub repository format
                        documents.extend(self._process_github_repo(data, file_path.name))
                    elif 'content' in data and ('link' in data or 'url' in data):
                        # FIX #5: check for both 'link' and 'url' keys to match _process_article
                        doc = self._process_article(data, file_path.name)
                        # FIX #2: guard against None before appending
                        if doc is not None:
                            documents.append(doc)
                    else:
                        # Generic document format
                        doc = self._process_generic_document(data, file_path.name)
                        # FIX #3: guard against None before appending
                        if doc is not None:
                            documents.append(doc)
                        
                elif isinstance(data, list):
                    # Multiple documents format
                    for item in data:
                        if isinstance(item, dict):
                            if 'content' in item and ('link' in item or 'url' in item):
                                # FIX #5: check for both 'link' and 'url' keys
                                doc = self._process_article(item, file_path.name)
                                # FIX #2: guard against None before appending
                                if doc is not None:
                                    documents.append(doc)
                            else:
                                doc = self._process_generic_document(item, file_path.name)
                                # FIX #3: guard against None before appending
                                if doc is not None:
                                    documents.append(doc)
                
                self.processed_files.add(str(file_path))
                logger.info(f"✅ Successfully processed {file_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path.name}: {e}")
                self.error_files.append((str(file_path), str(e)))
                continue
        
        logger.info(f"📊 Loaded {len(documents)} documents from {len(self.processed_files)} files")
        return documents
    
    def _process_github_repo(self, repo_data: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
        """Process GitHub repository data into individual documents."""
        documents = []
        
        repo_name = repo_data.get('name', 'unknown')
        repo_url = repo_data.get('repository_url', '')
        owner = repo_data.get('owner', 'unknown')

        # FIX #4: guard against 'files' being a non-dict type
        files = repo_data.get('files', {})
        if not isinstance(files, dict):
            logger.warning(f"⚠️  'files' field in '{source_file}' is not a dict (got {type(files).__name__}), skipping.")
            return documents

        for file_path, file_info in files.items():
            content = file_info.get('content', '')
            language = file_info.get('language', 'unknown')
            size = file_info.get('size', 0)
            
            if not content or len(content.strip()) < 10:
                continue  # Skip empty or very small files
            
            # Create document for each file
            doc = {
                'text': content,
                'metadata': {
                    'source_type': 'github',
                    'source_file': source_file,
                    'repository_name': repo_name,
                    'repository_url': repo_url,
                    'repository_owner': owner,
                    'file_path': file_path,
                    'language': language,
                    'file_size': size,
                    'document_type': 'code'
                }
            }
            documents.append(doc)
        
        logger.info(f"📁 Processed GitHub repo '{repo_name}': {len(documents)} files")
        return documents
    
    def _process_article(self, article_data: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
        """Process article data into a document."""
        content_data = article_data.get('content', {})
        
        # Extract title and content
        title = content_data.get('Title', '')
        subtitle = content_data.get('Subtitle', '')
        content = content_data.get('Content', '')
        
        # Combine title and content for better searchability
        full_text = f"{title}\n{subtitle}\n{content}" if title or subtitle else content
        
        if not full_text or len(full_text.strip()) < 10:
            return None
        
        # FIX #5: support both 'link' and 'url' as the URL field
        url = article_data.get('link', '') or article_data.get('url', '')

        doc = {
            'text': full_text,
            'metadata': {
                'source_type': 'article',
                'source_file': source_file,
                'url': url,
                'platform': article_data.get('platform', ''),
                'title': title,
                'subtitle': subtitle,
                'author_id': article_data.get('author_id', ''),
                'author_full_name': article_data.get('author_full_name', ''),
                'extracted_at': article_data.get('extracted_at', ''),
                'document_type': 'article'
            }
        }
        
        logger.info(f"📄 Processed article: {title[:50]}...")
        return doc
    
    def _process_generic_document(self, doc_data: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
        """Process generic document format."""
        # Try to extract text content
        text_content = ""
        
        if isinstance(doc_data.get('content'), str):
            text_content = doc_data['content']
        elif isinstance(doc_data.get('content'), dict):
            # Handle nested content structure
            content_dict = doc_data['content']
            if 'Content' in content_dict:
                text_content = content_dict['Content']
            elif 'content' in content_dict:
                text_content = content_dict['content']
            else:
                # Join all string values
                text_content = ' '.join([str(v) for v in content_dict.values() if isinstance(v, str)])
        
        if not text_content or len(text_content.strip()) < 10:
            return None
        
        doc = {
            'text': text_content,
            'metadata': {
                'source_type': 'generic',
                'source_file': source_file,
                'document_type': 'generic',
                'original_data_keys': list(doc_data.keys())
            }
        }
        
        return doc
    
    def setup_pipeline(self) -> IngestionPipeline:
        """Set up the ingestion pipeline with optimal configuration."""
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
            qdrant_collection_name="llm_twin_documents",
            qdrant_distance_metric="cosine",
            
            # Pipeline settings
            clean_data=True,
            skip_existing=True
        )
        
        logger.info("🔧 Setting up ingestion pipeline...")
        pipeline = IngestionPipeline(config)
        
        # Test the pipeline
        test_results = pipeline.test_pipeline(
            "This is a test document to verify the pipeline is working correctly."
        )
        
        if 'error' in test_results:
            raise Exception(f"Pipeline test failed: {test_results['error']}")
        
        logger.info("✅ Pipeline setup complete and tested")
        return pipeline
    
    def ingest_documents(self, documents: List[Dict[str, Any]], pipeline: IngestionPipeline) -> Dict[str, Any]:
        """Ingest documents into the vector database."""
        logger.info(f"🚀 Starting ingestion of {len(documents)} documents...")
        
        # Group documents by type for better processing
        github_docs = []
        article_docs = []
        other_docs = []
        
        for doc in documents:
            doc_type = doc.get('metadata', {}).get('document_type', 'other')
            if doc_type == 'code':
                github_docs.append(doc)
            elif doc_type == 'article':
                article_docs.append(doc)
            else:
                other_docs.append(doc)
        
        results = {
            'total_documents': len(documents),
            'successful_ingestions': 0,
            'failed_ingestions': 0,
            'total_chunks': 0,
            'errors': []
        }
        
        # Process each document type
        if github_docs:
            logger.info(f"Processing {len(github_docs)} GitHub documents...")
            try:
                github_results = pipeline.process_documents(github_docs, 'repository')
                results['successful_ingestions'] += github_results.get('processed_documents', 0)
                results['total_chunks'] += github_results.get('total_chunks', 0)
                results['errors'].extend(github_results.get('errors', []))
                logger.info(f"✅ GitHub docs: {github_results.get('processed_documents', 0)} processed, {github_results.get('total_chunks', 0)} chunks")
            except Exception as e:
                logger.error(f"❌ Error processing GitHub documents: {e}")
                results['failed_ingestions'] += len(github_docs)
                results['errors'].append(f"GitHub processing: {str(e)}")
        
        if article_docs:
            logger.info(f"Processing {len(article_docs)} article documents...")
            try:
                article_results = pipeline.process_documents(article_docs, 'article')
                results['successful_ingestions'] += article_results.get('processed_documents', 0)
                results['total_chunks'] += article_results.get('total_chunks', 0)
                results['errors'].extend(article_results.get('errors', []))
                logger.info(f"✅ Article docs: {article_results.get('processed_documents', 0)} processed, {article_results.get('total_chunks', 0)} chunks")
            except Exception as e:
                logger.error(f"❌ Error processing article documents: {e}")
                results['failed_ingestions'] += len(article_docs)
                results['errors'].append(f"Article processing: {str(e)}")
        
        if other_docs:
            logger.info(f"Processing {len(other_docs)} other documents...")
            try:
                other_results = pipeline.process_documents(other_docs, 'other')
                results['successful_ingestions'] += other_results.get('processed_documents', 0)
                results['total_chunks'] += other_results.get('total_chunks', 0)
                results['errors'].extend(other_results.get('errors', []))
                logger.info(f"✅ Other docs: {other_results.get('processed_documents', 0)} processed, {other_results.get('total_chunks', 0)} chunks")
            except Exception as e:
                logger.error(f"❌ Error processing other documents: {e}")
                results['failed_ingestions'] += len(other_docs)
                results['errors'].append(f"Other processing: {str(e)}")
        
        return results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline from raw documents to vector database."""
        logger.info("🎯 Starting full document processing pipeline...")
        
        # Step 1: Load raw documents
        documents = self.load_raw_documents()
        
        if not documents:
            logger.error("❌ No documents to process")
            return {'success': False, 'error': 'No documents found'}
        
        # Step 2: Setup pipeline
        try:
            pipeline = self.setup_pipeline()
        except Exception as e:
            logger.error(f"❌ Failed to setup pipeline: {e}")
            return {'success': False, 'error': f'Pipeline setup failed: {e}'}
        
        # Step 3: Ingest documents
        ingestion_results = self.ingest_documents(documents, pipeline)
        
        # Step 4: Get final status
        try:
            status = pipeline.get_pipeline_status()
            vector_store_info = status.get('vector_store_info', {})
        except Exception as e:
            logger.warning(f"Could not get pipeline status: {e}")
            vector_store_info = {}
        
        # Compile final results
        final_results = {
            'success': True,
            'raw_files_processed': len(self.processed_files),
            'raw_files_failed': len(self.error_files),
            'documents_loaded': ingestion_results['total_documents'],
            'documents_successful': ingestion_results['successful_ingestions'],
            'documents_failed': ingestion_results['failed_ingestions'],
            'total_chunks_created': ingestion_results['total_chunks'],
            'vector_store_points': vector_store_info.get('points_count', 'N/A'),
            'vector_store_vectors': vector_store_info.get('vectors_count', 'N/A'),
            'errors': ingestion_results['errors'] + self.error_files
        }
        
        return final_results


def main():
    """Main function to run the document processing."""
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    logger.info("🚀 Starting Raw Document Processing Pipeline")
    
    # Check Qdrant connection
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Qdrant is running and accessible")
        else:
            # FIX #1: added missing 'f' prefix to make this an f-string
            logger.warning(f"⚠️  Qdrant responded with status code: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Cannot connect to Qdrant: {e}")
        logger.error("Please ensure Qdrant is running on localhost:6333")
        logger.error("You can start it with: docker run -p 6333:6333 qdrant/qdrant")
        return
    
    # Initialize processor
    processor = RawDocumentProcessor()
    
    # Run the full pipeline
    try:
        results = processor.run_full_pipeline()
        
        # Display results
        if results.get('success', False):
            logger.info("🎉 Pipeline completed!")
            logger.info("📊 Final Results:")
            logger.info(f"  Raw files processed: {results.get('raw_files_processed', 0)}")
            logger.info(f"  Raw files failed: {results.get('raw_files_failed', 0)}")
            logger.info(f"  Documents loaded: {results.get('documents_loaded', 0)}")
            logger.info(f"  Documents successful: {results.get('documents_successful', 0)}")
            logger.info(f"  Documents failed: {results.get('documents_failed', 0)}")
            logger.info(f"  Total chunks created: {results.get('total_chunks_created', 0)}")
            logger.info(f"  Vector store points: {results.get('vector_store_points', 'N/A')}")
            logger.info(f"  Vector store vectors: {results.get('vector_store_vectors', 'N/A')}")
            
            if results.get('errors'):
                logger.warning(f"⚠️  {len(results['errors'])} errors encountered:")
                for error in results['errors'][:5]:  # Show first 5 errors
                    logger.warning(f"  - {error}")
                if len(results['errors']) > 5:
                    logger.warning(f"  ... and {len(results['errors']) - 5} more errors")
        else:
            logger.error("❌ Pipeline failed!")
            logger.error(f"Error: {results.get('error', 'Unknown error')}")
        
        logger.info("✅ Raw document processing pipeline completed!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
