#!/usr/bin/env python3
"""
Run the RAG feature engineering pipeline
"""
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_feature_engineering_pipeline():
    """Run the feature engineering pipeline for specified authors"""
    print("=== RAG Feature Engineering Pipeline ===")
    
    try:
        from src.pipelines.feature_engineering import feature_engineering
        
        # Example authors - you can modify this list
        author_full_names = [
            "Ronel Solomon",
            "Maxime Labonne", 
            "Paul Iusztin"
        ]
        
        print(f"Running feature engineering pipeline for authors: {author_full_names}")
        
        # Run the pipeline
        result = feature_engineering(author_full_names=author_full_names)
        
        print(f"✅ Feature engineering pipeline completed successfully!")
        print(f"Pipeline result: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Feature engineering pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("Select option:")
    print("1. Run feature engineering pipeline for all authors")
    print("2. Run feature engineering pipeline for specific authors")
    
    try:
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == "1":
            run_feature_engineering_pipeline()
        elif choice == "2":
            authors_input = input("Enter author names (comma-separated): ").strip()
            authors = [author.strip() for author in authors_input.split(",")]
            
            # Override the authors in the pipeline
            from src.pipelines.feature_engineering import feature_engineering
            result = feature_engineering(author_full_names=authors)
            
            print(f"✅ Pipeline completed for authors: {authors}")
            print(f"Result: {result}")
        else:
            print("Invalid choice. Exiting.")
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
