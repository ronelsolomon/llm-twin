#!/usr/bin/env python3
"""
Script to run the data warehouse ingestion pipeline
"""
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_warehouse_pipeline import data_warehouse_ingestion_pipeline
from zenml.client import Client


def main():
    """Run the data ingestion pipeline"""
    
    # Initialize ZenML if needed
    try:
        client = Client()
        if "data_warehouse" not in [p.name for p in client.list_projects()]:
            client.create_project("data_warehouse")
            print("Created 'data_warehouse' project")
    except Exception as e:
        print(f"ZenML initialization error: {e}")
        print("Make sure ZenML is installed and initialized")
        return
    
    # Define data path
    data_path = "/Users/ronel/Downloads/llm twin/data"
    
    print(f"Starting data ingestion from: {data_path}")
    
    try:
        # Run the pipeline
        result = data_warehouse_ingestion_pipeline(data_path)
        print(f"Pipeline completed successfully!")
        print(f"Results: {result}")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
