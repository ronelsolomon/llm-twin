from zenml import pipeline
from typing import List, Optional
import yaml
from pathlib import Path


@pipeline
def feature_engineering(author_full_names: List[str]) -> None:
    """
    RAG feature engineering pipeline that processes documents through five core phases:
    1. Extracting raw documents from data warehouse
    2. Cleaning documents
    3. Chunking and embedding documents
    4. Loading cleaned documents to vector DB
    5. Loading embedded documents to vector DB
    
    Args:
        author_full_names: List of author full names to process
    """
    from llm_engineering.interfaces.orchestrator.steps import feature_engineering as fe_steps
    
    # Step 1: Extract raw documents from data warehouse
    raw_documents = fe_steps.query_data_warehouse(author_full_names)
    
    # Step 2: Clean the documents
    cleaned_documents = fe_steps.clean_documents(raw_documents)
    
    # Step 3: Load cleaned documents to vector database
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)
    
    # Step 4: Chunk and embed the cleaned documents
    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    
    # Step 5: Load embedded documents to vector database
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)
    
    return [last_step_1.invocation_id, last_step_2.invocation_id]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_feature_engineering_with_config(config_path: Optional[str] = None) -> None:
    """
    Run feature engineering pipeline with configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Default config path if not provided
    if config_path is None:
        config_path = "configs/feature_engineering.yaml"
    
    # Load configuration
    config = load_config(config_path)
    parameters = config.get('parameters', {})
    author_full_names = parameters.get('author_full_names', [])
    
    if not author_full_names:
        raise ValueError("No author_full_names found in configuration")
    
    # Run pipeline with configuration
    pipeline_with_config = feature_engineering.with_options(config_path=config_path)
    result = pipeline_with_config(author_full_names=author_full_names)
    
    return result
