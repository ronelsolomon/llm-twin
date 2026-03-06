#!/usr/bin/env python3
"""
Complete ETL Pipeline Runner - From Data Extraction to Vector Database
This script orchestrates the entire pipeline:
1. Web crawling and data extraction
2. Data processing and cleaning
3. Vector embedding and storage in Qdrant
"""
import sys
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.domain.documents import UserDocument, ArticleDocument, RepositoryDocument
from scripts.process_raw_documents import RawDocumentProcessor


class CompleteETLPipeline:
    """Complete ETL pipeline from web crawling to vector database."""
    
    def __init__(self):
        self.project_root = project_root
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Complete ETL Pipeline initialized")
    
    def check_dependencies(self) -> bool:
        """Check if all dependencies are available."""
        logger.info("🔍 Checking dependencies...")
        
        # Check Qdrant
        try:
            import requests
            response = requests.get("http://localhost:6333/collections", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Qdrant is running and accessible")
                qdrant_ok = True
            else:
                logger.warning(f"⚠️  Qdrant responded with status code: {response.status_code}")
                qdrant_ok = False
        except Exception as e:
            logger.error(f"❌ Cannot connect to Qdrant: {e}")
            logger.error("Please ensure Qdrant is running on localhost:6333")
            logger.error("You can start it with: docker run -p 6333:6333 qdrant/qdrant")
            qdrant_ok = False
        
        # Check required modules
        required_modules = [
            'src.crawlers.medium',
            'src.crawlers.github', 
            'src.crawlers.linkedin',
            'src.ingestion.pipeline'
        ]
        
        modules_ok = True
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✅ Module {module} available")
            except ImportError as e:
                logger.error(f"❌ Module {module} not available: {e}")
                modules_ok = False
        
        return qdrant_ok and modules_ok
    
    def run_web_crawling(self, user_full_name: str, links: List[str]) -> Dict[str, Any]:
        """Run web crawling to extract raw data."""
        logger.info(f"🕷️  Starting web crawling for {user_full_name}")
        
        try:
            from scripts.simple_web_crawling import simple_digital_data_etl_pipeline
            
            result = simple_digital_data_etl_pipeline(
                user_full_name=user_full_name,
                links=links
            )
            
            logger.info(f"✅ Web crawling completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Web crawling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_data_ingestion(self) -> Dict[str, Any]:
        """Run data ingestion from JSON files to document system."""
        logger.info("📥 Starting data ingestion from JSON files")
        
        try:
            from scripts.simple_data_ingestion import ingest_articles, ingest_repositories, ingest_users
            
            # Ingest all data types
            articles_count = ingest_articles(str(self.data_dir))
            repos_count = ingest_repositories(str(self.data_dir))
            users_count = ingest_users(str(self.data_dir))
            
            result = {
                'success': True,
                'articles_ingested': articles_count,
                'repositories_ingested': repos_count,
                'users_ingested': users_count,
                'total_documents': articles_count + repos_count + users_count
            }
            
            logger.info(f"✅ Data ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Data ingestion failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_vector_processing(self) -> Dict[str, Any]:
        """Run vector processing and storage."""
        logger.info("🔄 Starting vector processing and storage")
        
        try:
            processor = RawDocumentProcessor(str(self.raw_dir))
            result = processor.run_full_pipeline()
            
            logger.info(f"✅ Vector processing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Vector processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_complete_pipeline(self, user_full_name: str, links: List[str]) -> Dict[str, Any]:
        """Run the complete ETL pipeline."""
        logger.info("🎯 Starting complete ETL pipeline")
        
        # Check dependencies first
        if not self.check_dependencies():
            return {'success': False, 'error': 'Dependencies not satisfied'}
        
        results = {
            'pipeline_start_time': time.time(),
            'stages': {},
            'final_results': {}
        }
        
        try:
            # Stage 1: Web Crawling
            logger.info("📋 Stage 1: Web Crawling")
            crawling_result = self.run_web_crawling(user_full_name, links)
            results['stages']['web_crawling'] = crawling_result
            
            if not crawling_result.get('success', False):
                logger.error("❌ Web crawling failed, stopping pipeline")
                return {'success': False, 'error': 'Web crawling failed', 'results': results}
            
            # Stage 2: Data Ingestion
            logger.info("📋 Stage 2: Data Ingestion")
            ingestion_result = self.run_data_ingestion()
            results['stages']['data_ingestion'] = ingestion_result
            
            # Stage 3: Vector Processing
            logger.info("📋 Stage 3: Vector Processing")
            vector_result = self.run_vector_processing()
            results['stages']['vector_processing'] = vector_result
            
            # Compile final results
            results['pipeline_end_time'] = time.time()
            results['pipeline_duration'] = results['pipeline_end_time'] - results['pipeline_start_time']
            results['success'] = True
            
            # Summary
            results['final_results'] = {
                'total_duration_minutes': results['pipeline_duration'] / 60,
                'web_crawling_success': crawling_result.get('success', False),
                'data_ingestion_success': ingestion_result.get('success', False),
                'vector_processing_success': vector_result.get('success', False),
                'total_documents_processed': ingestion_result.get('total_documents', 0),
                'total_chunks_created': vector_result.get('total_chunks_created', 0),
            }
            
            # Get vector store statistics from hybrid store
            if hasattr(self, 'vector_store') and hasattr(self.vector_store, 'get_collection_info'):
                try:
                    collection_info = self.vector_store.get_collection_info()
                    results['final_results']['vector_store_points'] = collection_info.get('total_points', 'N/A')
                    results['final_results']['vector_store_vectors'] = collection_info.get('total_vectors', 'N/A')
                    results['final_results']['vector_store_details'] = collection_info.get('stores', {})
                except Exception as e:
                    logger.warning(f"Failed to get vector store stats: {e}")
                    results['final_results']['vector_store_points'] = 'N/A'
                    results['final_results']['vector_store_vectors'] = 'N/A'
            else:
                results['final_results']['vector_store_points'] = 'N/A'
                results['final_results']['vector_store_vectors'] = 'N/A'
            
            logger.info("🎉 Complete ETL pipeline finished successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Complete pipeline failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def get_ronel_solomon_links(self) -> List[str]:
        """Get Ronel Solomon's links from configuration."""
        config_file = self.project_root / "configs" / "digital_data_etl_ronel_solomon.yaml"
        
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                return config.get('parameters', {}).get('links', [])
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        # Fallback links
        return [
            # GitHub repositories
            "https://github.com/ronelsolomon/2D-Farming-Game.git",
            "https://github.com/ronelsolomon/2d-game.git",
            "https://github.com/ronelsolomon/2d-game-dev.git",
            "https://github.com/ronelsolomon/AI-Engineer-questions.git",
            "https://github.com/ronelsolomon/ai-videos.git",
            "https://github.com/ronelsolomon/aiadds.git",
            "https://github.com/ronelsolomon/air-quality-MLOPs.git",
            "https://github.com/ronelsolomon/aivideo.git",
            "https://github.com/ronelsolomon/aleoex.git",
            "https://github.com/ronelsolomon/assementTemplate.git",
            "https://github.com/ronelsolomon/AWS-backend.git",
            "https://github.com/ronelsolomon/cebras.git",
            "https://github.com/ronelsolomon/changeAudio.git",
            "https://github.com/ronelsolomon/crawlerx.git",
            "https://github.com/ronelsolomon/csvDashboardRetetion.git",
            "https://github.com/ronelsolomon/curify-gallery.git",
            "https://github.com/ronelsolomon/dev.git",
            "https://github.com/ronelsolomon/dma.git",
            "https://github.com/ronelsolomon/ETL-pipeline.git",
            "https://github.com/ronelsolomon/expresive.git",
            "https://github.com/ronelsolomon/file-audio.git",
            "https://github.com/ronelsolomon/filesummarize.git",
            "https://github.com/ronelsolomon/FindYoutube.git",
            "https://github.com/ronelsolomon/genAI.git",
            "https://github.com/ronelsolomon/Graph-k-means.git",
            "https://github.com/ronelsolomon/healthcare.git",
            "https://github.com/ronelsolomon/hearpython.git",
            "https://github.com/ronelsolomon/interview_questions.git",
            "https://github.com/ronelsolomon/java-project.git",
            "https://github.com/ronelsolomon/keboola-mcp.git",
        ]
    
    def display_pipeline_summary(self, results: Dict[str, Any]):
        """Display a summary of the pipeline results."""
        logger.info("📊 Pipeline Summary:")
        logger.info("=" * 50)
        
        if results.get('success'):
            final_results = results.get('final_results', {})
            logger.info(f"✅ Pipeline Status: SUCCESS")
            logger.info(f"⏱️  Total Duration: {final_results.get('total_duration_minutes', 0):.2f} minutes")
            logger.info(f"📄 Documents Processed: {final_results.get('total_documents_processed', 0)}")
            logger.info(f"🧩 Chunks Created: {final_results.get('total_chunks_created', 0)}")
            logger.info(f"🗄️  Vector Store Points: {final_results.get('vector_store_points', 'N/A')}")
            
            # Stage details
            stages = results.get('stages', {})
            for stage_name, stage_result in stages.items():
                status = "✅" if stage_result.get('success', False) else "❌"
                logger.info(f"{status} {stage_name.replace('_', ' ').title()}: {stage_result.get('success', False)}")
        else:
            logger.error("❌ Pipeline Status: FAILED")
            logger.error(f"Error: {results.get('error', 'Unknown error')}")
        
        logger.info("=" * 50)


def main():
    """Main function to run the complete ETL pipeline."""
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    logger.info("🚀 Complete ETL Pipeline Runner")
    logger.info("This pipeline will: 1) Crawl web data, 2) Ingest to document system, 3) Process and store in vector database")
    
    pipeline = CompleteETLPipeline()
    
    # Menu options
    logger.info("\nSelect pipeline option:")
    logger.info("1. Run complete pipeline for Ronel Solomon")
    logger.info("2. Run complete pipeline for custom user")
    logger.info("3. Run only web crawling")
    logger.info("4. Run only data ingestion")
    logger.info("5. Run only vector processing")
    logger.info("6. Check dependencies")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            # Run for Ronel Solomon
            user_name = "Ronel Solomon"
            links = pipeline.get_ronel_solomon_links()
            logger.info(f"Running complete pipeline for {user_name} with {len(links)} links")
            results = pipeline.run_complete_pipeline(user_name, links)
            
        elif choice == "2":
            # Custom user
            user_name = input("Enter user full name: ").strip()
            links_input = input("Enter links (comma-separated): ").strip()
            links = [link.strip() for link in links_input.split(",") if link.strip()]
            
            if not links:
                logger.error("No links provided. Exiting.")
                return
            
            logger.info(f"Running complete pipeline for {user_name} with {len(links)} links")
            results = pipeline.run_complete_pipeline(user_name, links)
            
        elif choice == "3":
            # Only web crawling
            user_name = input("Enter user full name: ").strip()
            links_input = input("Enter links (comma-separated): ").strip()
            links = [link.strip() for link in links_input.split(",") if link.strip()]
            
            results = pipeline.run_web_crawling(user_name, links)
            
        elif choice == "4":
            # Only data ingestion
            results = pipeline.run_data_ingestion()
            
        elif choice == "5":
            # Only vector processing
            results = pipeline.run_vector_processing()
            
        elif choice == "6":
            # Check dependencies
            deps_ok = pipeline.check_dependencies()
            results = {'success': deps_ok}
            
        else:
            logger.error("Invalid choice. Exiting.")
            return
        
        # Display results
        pipeline.display_pipeline_summary(results)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
