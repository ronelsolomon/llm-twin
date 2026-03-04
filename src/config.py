from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    # RAG Settings
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"
    
    # QdrantDB Vector DB Settings
    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = ""
    QDRANT_APIKEY: str | None = None
    
    # Add other settings as needed
    # ... # More settings...


# Global settings instance
settings = Settings()
