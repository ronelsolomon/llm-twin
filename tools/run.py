#!/usr/bin/env python3
"""
Tools module for running ETL pipelines with configuration support
"""
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.enhanced_digital_data_etl import digital_data_etl_pipeline
from zenml.client import Client


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_etl_with_config(config_file: str, no_cache: bool = False) -> Dict[str, Any]:
    """Run ETL pipeline with configuration file"""
    
    # Load configuration
    config = load_config(config_file)
    
    # Extract parameters
    parameters = config.get('parameters', {})
    user_full_name = parameters.get('user_full_name')
    links = parameters.get('links', [])
    
    if not user_full_name:
        raise ValueError("user_full_name is required in configuration")
    
    if not links:
        raise ValueError("links are required in configuration")
    
    print(f"Running ETL pipeline for: {user_full_name}")
    print(f"Number of links to process: {len(links)}")
    
    # Initialize ZenML project
    Client().create_project("digital_data_etl", delete_if_exists=True)
    
    # Run pipeline
    result = digital_data_etl_pipeline(
        user_full_name=user_full_name,
        links=links
    )
    
    return result


def run_feature_engineering_with_config(config_file: str, no_cache: bool = False) -> Dict[str, Any]:
    """Run feature engineering pipeline with configuration file"""
    
    # Load configuration
    config = load_config(config_file)
    
    # Extract parameters
    parameters = config.get('parameters', {})
    author_full_names = parameters.get('author_full_names', [])
    
    if not author_full_names:
        raise ValueError("author_full_names are required in configuration")
    
    print(f"Running feature engineering pipeline for authors: {author_full_names}")
    
    # Import and run feature engineering pipeline
    from src.pipelines.feature_engineering import run_feature_engineering_with_config as run_pipeline
    
    result = run_pipeline(config_path=config_file)
    
    return result


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Run digital data ETL pipeline')
    parser.add_argument('--run-etl', action='store_true', help='Run the ETL pipeline')
    parser.add_argument('--run-feature-engineering', action='store_true', help='Run the feature engineering pipeline')
    parser.add_argument('--etl-config-filename', type=str, help='Configuration file name')
    parser.add_argument('--feature-engineering-config-filename', type=str, help='Feature engineering configuration file name')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    args = parser.parse_args()
    
    if args.run_etl and args.etl_config_filename:
        # Run ETL pipeline with configuration
        config_file = f"configs/{args.etl_config_filename}"
        
        try:
            result = run_etl_with_config(config_file, args.no_cache)
            print(f"✅ ETL Pipeline completed successfully!")
            print(f"Results: {result}")
            
        except Exception as e:
            print(f"❌ ETL Pipeline failed: {e}")
            sys.exit(1)
    
    elif args.run_feature_engineering:
        # Run feature engineering pipeline with configuration
        if args.feature_engineering_config_filename:
            config_file = f"configs/{args.feature_engineering_config_filename}"
        else:
            config_file = "configs/feature_engineering.yaml"  # Default config
        
        try:
            result = run_feature_engineering_with_config(config_file, args.no_cache)
            print(f"✅ Feature Engineering Pipeline completed successfully!")
            print(f"Results: {result}")
            
        except Exception as e:
            print(f"❌ Feature Engineering Pipeline failed: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
