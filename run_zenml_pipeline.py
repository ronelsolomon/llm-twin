#!/usr/bin/env python3
"""
Run ZenML pipeline and show dashboard
"""
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def start_zenml_dashboard():
    """Start ZenML dashboard in background"""
    print("Starting ZenML dashboard...")
    
    # Set environment variable for macOS
    env = {"OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES"}
    
    # Start zenml server in background
    try:
        process = subprocess.Popen(
            ["zenml", "login", "--local"],
            env={**subprocess.os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Open browser
        webbrowser.open("http://localhost:8237")
        print("ZenML dashboard should be available at: http://localhost:8237")
        print("If not accessible, try: OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES zenml login --local")
        
        return process
        
    except Exception as e:
        print(f"Failed to start ZenML dashboard: {e}")
        return None

def run_data_pipeline():
    """Run the data ingestion pipeline"""
    print("Running data ingestion pipeline...")
    
    try:
        from scripts.simple_data_ingestion import main as run_ingestion
        result = run_ingestion()
        print(f"Simple ingestion completed: {result}")
        return result
    except Exception as e:
        print(f"Simple ingestion failed: {e}")
        return None

def run_web_crawling_pipeline():
    """Run the enhanced web crawling pipeline"""
    print("Running web crawling pipeline...")
    
    try:
        from scripts.enhanced_digital_data_etl import digital_data_etl_pipeline
        from zenml.client import Client
        
        # Initialize ZenML project with proper name
        try:
            client = Client()
            project_name = "digital_data_etl_maxime"
            client.create_project(project_name, delete_if_exists=True)
        except Exception as e:
            print(f"Warning: Could not create ZenML project: {e}")
            print("Continuing without explicit project creation...")
        
        # Example: Run with Maxime Labonne configuration
        maxime_links = [
            # Personal Blog
            "https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html",
            "https://mlabonne.github.io/blog/posts/2024-07-15_The_Rise_of_Agentic_Data_Generation.html",
            # Substack
            "https://maximelabonne.substack.com/p/uncensor-any-llm-with-abliteration-d30148b7d43e",
            "https://maximelabonne.substack.com/p/create-mixtures-of-experts-with-mergekit-11b318c99562",
            "https://maximelabonne.substack.com/p/merge-large-language-models-with-mergekit-2118fb392b54",
        ]
        
        result = digital_data_etl_pipeline(
            user_full_name="Maxime Labonne",
            links=maxime_links
        )
        
        print(f"Web crawling pipeline completed: {result}")
        return result
        
    except Exception as e:
        print(f"Web crawling pipeline failed: {e}")
        return None

def run_ronel_solomon_pipeline():
    """Run pipeline for Ronel Solomon's data"""
    print("Running Ronel Solomon's data collection...")
    
    try:
        from scripts.simple_web_crawling import simple_digital_data_etl_pipeline
        
        ronel_links = [
            # GitHub repositories from existing data
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
        
        result = simple_digital_data_etl_pipeline(
            user_full_name="Ronel Solomon",
            links=ronel_links
        )
        
        print(f"Ronel Solomon pipeline completed: {result}")
        return result
        
    except Exception as e:
        print(f"Ronel Solomon pipeline failed: {e}")
        return None

def run_paul_iusztin_pipeline():
    """Run pipeline for Paul Iusztin's data"""
    print("Running Paul Iusztin's data collection...")
    
    try:
        from scripts.enhanced_digital_data_etl import digital_data_etl_pipeline
        from zenml.client import Client
        
        # Initialize ZenML project with proper name
        try:
            client = Client()
            project_name = "digital_data_etl_paul"
            client.create_project(project_name, delete_if_exists=True)
        except Exception as e:
            print(f"Warning: Could not create ZenML project: {e}")
            print("Continuing without explicit project creation...")
        
        paul_links = [
            # Medium
            "https://medium.com/decodingml/an-end-to-end-framework-for-production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f",
            "https://medium.com/decodingml/a-real-time-retrieval-system-for-rag-on-social-media-data-9cc01d50a2a0",
            "https://medium.com/decodingml/sota-python-streaming-pipelines-for-fine-tuning-llms-and-rag-in-real-time-82eb07795b87",
            # Substack
            "https://decodingml.substack.com/p/real-time-feature-pipelines-with?r=1ttoeh",
            "https://decodingml.substack.com/p/building-ml-systems-the-right-way?r=1ttoeh",
            "https://decodingml.substack.com/p/reduce-your-pytorchs-code-latency?r=1ttoeh",
        ]
        
        result = digital_data_etl_pipeline(
            user_full_name="Paul Iusztin",
            links=paul_links
        )
        
        print(f"Paul Iusztin pipeline completed: {result}")
        return result
        
    except Exception as e:
        print(f"Paul Iusztin pipeline failed: {e}")
        return None

def main():
    """Main function"""
    print("=== Enhanced ZenML Data Warehouse Pipeline ===")
    print("\nSelect pipeline to run:")
    print("1. Simple data ingestion (JSON files only)")
    print("2. Web crawling pipeline - Ronel Solomon")
    print("3. Web crawling pipeline - Maxime Labonne")
    print("4. Web crawling pipeline - Paul Iusztin")
    print("5. Run all pipelines")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n1. Running simple data ingestion pipeline...")
            pipeline_result = run_data_pipeline()
            
        elif choice == "2":
            print("\n2. Running web crawling pipeline for Ronel Solomon...")
            pipeline_result = run_ronel_solomon_pipeline()
            
        elif choice == "3":
            print("\n3. Running web crawling pipeline for Maxime Labonne...")
            pipeline_result = run_web_crawling_pipeline()
            
        elif choice == "4":
            print("\n4. Running web crawling pipeline for Paul Iusztin...")
            pipeline_result = run_paul_iusztin_pipeline()
            
        elif choice == "5":
            print("\n5. Running all pipelines...")
            
            print("\n--- Simple Data Ingestion ---")
            simple_result = run_data_pipeline()
            
            print("\n--- Web Crawling - Ronel Solomon ---")
            ronel_result = run_ronel_solomon_pipeline()
            
            print("\n--- Web Crawling - Maxime Labonne ---")
            maxime_result = run_web_crawling_pipeline()
            
            print("\n--- Web Crawling - Paul Iusztin ---")
            paul_result = run_paul_iusztin_pipeline()
            
            pipeline_result = {
                "simple_ingestion": simple_result,
                "ronel_crawling": ronel_result,
                "maxime_crawling": maxime_result,
                "paul_crawling": paul_result
            }
            
        else:
            print("Invalid choice. Exiting.")
            return
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        return
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return
    
    # Show results
    if pipeline_result:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"Results: {pipeline_result}")
        
        # Show data files
        data_dir = Path("data")
        if data_dir.exists():
            print(f"\n📁 Data files in {data_dir}:")
            for file in data_dir.glob("*.json"):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
        
        # Query example for Ronel Solomon
        print(f"\n🔍 Example query - Ronel Solomon's articles:")
        try:
            from src.domain.documents import ArticleDocument, UserDocument
            
            # Find user by name
            users = UserDocument.find_all(first_name="Ronel", last_name="Solomon")
            if users:
                user = users[0]
                articles = ArticleDocument.find_all(author_id=str(user.id))
                print(f"User ID: {user.id}")
                print(f"User name: {user.first_name} {user.last_name}")
                print(f"Number of articles: {len(articles)}")
                if articles:
                    print(f"First article link: {articles[0].link}")
            else:
                print("User not found")
        except Exception as e:
            print(f"Query failed: {e}")
    
    else:
        print("❌ Pipeline failed or was cancelled.")
    
    print(f"\n🌐 To view ZenML dashboard manually:")
    print("  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES")
    print("  zenml login --local")
    print("  # Then open http://localhost:8237 in browser")
    
    print(f"\n🔧 To run individual pipelines:")
    print("  python scripts/enhanced_digital_data_etl.py")
    print("  python scripts/data_warehouse_pipeline.py")
    print("  python scripts/simple_data_ingestion.py")
    
    print(f"\n📋 To run with poetry commands:")
    print("  poetry poe run-digital-data-etl-ronel")

if __name__ == "__main__":
    main()
