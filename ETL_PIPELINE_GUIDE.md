# Complete ETL Pipeline Documentation

## Overview

This project implements a comprehensive ETL (Extract, Transform, Load) pipeline that collects digital data from various sources, processes it, and stores it in a vector database for RAG (Retrieval-Augmented Generation) applications.

## Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Web Sources   │───▶│   Raw Data       │───▶│  Processed      │───▶│  Vector DB       │
│                 │    │   Extraction     │    │  Documents     │    │  (Qdrant)        │
│ • Medium        │    │                  │    │                 │    │                  │
│ • GitHub        │    │ • JSON Files     │    │ • Cleaning      │    │ • Embeddings     │
│ • LinkedIn      │    │ • Raw Content    │    │ • Chunking      │    │ • Similarity     │
│ • Other         │    │ • Metadata       │    │ • Embedding     │    │ • Search         │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

## Complete Pipeline Stages

### Stage 1: Data Extraction (Web Crawling)
**Location**: `src/crawlers/` and `scripts/simple_web_crawling.py`

**Purpose**: Extract raw data from web sources
- **MediumCrawler**: Scrapes articles and profiles
- **GithubCrawler**: Clones and analyzes repositories
- **LinkedInCrawler**: Extracts profile and post content
- **BaseSeleniumCrawler**: Anti-detection web scraping

**Output**: Raw JSON files in `data/raw/`

### Stage 2: Data Ingestion
**Location**: `scripts/simple_data_ingestion.py`

**Purpose**: Load raw data into document system
- Ingests JSON files into NoSQL document system
- Creates UserDocument, ArticleDocument, RepositoryDocument
- Stores in `data/` directory (articles.json, repositories.json, users.json)

**Output**: Structured documents in document system

### Stage 3: Vector Processing
**Location**: `scripts/process_raw_documents.py` and `src/ingestion/`

**Purpose**: Convert documents to searchable vectors
- **DataCleaner**: Clean and preprocess text
- **TextChunker**: Split documents into chunks
- **EmbeddingGenerator**: Generate embeddings using sentence transformers
- **QdrantVectorStore**: Store vectors in Qdrant database

**Output**: Vector embeddings stored in Qdrant

## Key Components

### Crawlers (`src/crawlers/`)
- **base.py**: Abstract base crawler interface
- **selenium_base.py**: Selenium-based crawler with anti-detection
- **medium.py**: Medium article scraper
- **github.py**: GitHub repository scraper
- **linkedin.py**: LinkedIn content scraper

### Ingestion Pipeline (`src/ingestion/`)
- **pipeline.py**: Main orchestrator
- **cleaner.py**: Data cleaning and preprocessing
- **chunker.py**: Text chunking for better retrieval
- **embedder.py**: Embedding generation
- **vector_store.py**: Qdrant integration

### Document Models (`src/domain/`)
- **documents.py**: NoSQL document classes
- **enums.py**: Document type enumerations

## Usage

### Quick Start - Run Everything
```bash
# Run the complete pipeline
python main.py

# Choose option 1 for Ronel Solomon's data
# Choose option 2 for custom user data
```

### Individual Components

#### 1. Web Crawling Only
```bash
python scripts/simple_web_crawling.py
```

#### 2. Data Ingestion Only
```bash
python scripts/simple_data_ingestion.py
```

#### 3. Vector Processing Only
```bash
python scripts/process_raw_documents.py
```

#### 4. ZenML Pipeline (Advanced)
```bash
python run_zenml_pipeline.py
```

## Configuration

### User Configuration
Edit `configs/digital_data_etl_ronel_solomon.yaml`:
```yaml
parameters:
  user_full_name: Ronel Solomon
  links:
    - https://github.com/ronelsolomon/repository.git
    - https://medium.com/@user/article
    - https://linkedin.com/in/user
```

### Pipeline Configuration
The pipeline uses these default settings:
- **Chunk size**: 800 characters
- **Chunk overlap**: 200 characters
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector store**: Qdrant (localhost:6333)
- **Collection**: llm_twin_documents

## Data Flow

### Input Sources
1. **GitHub Repositories**: Source code, README files, documentation
2. **Medium Articles**: Blog posts, technical articles
3. **LinkedIn Content**: Profile information, posts, articles
4. **Custom Sources**: Any web content via crawlers

### Processing Steps
1. **Extraction**: Web crawlers fetch raw content
2. **Cleaning**: Remove HTML, normalize text, extract metadata
3. **Chunking**: Split long documents into searchable chunks
4. **Embedding**: Convert text to vector representations
5. **Storage**: Store vectors with metadata in Qdrant

### Output Formats
- **Raw JSON**: Unprocessed web data (`data/raw/`)
- **Structured Documents**: Cleaned documents (`data/*.json`)
- **Vector Chunks**: Embeddings in Qdrant
- **Metadata**: Source information, timestamps, document types

## Dependencies

### Required Services
- **Qdrant**: Vector database (docker run -p 6333:6333 qdrant/qdrant)
- **Chrome/ChromeDriver**: For Selenium-based crawling

### Python Packages
See `requirements.txt` for complete list:
- `selenium`: Web automation
- `sentence-transformers`: Text embeddings
- `qdrant-client`: Vector database client
- `loguru`: Logging
- `pydantic`: Data validation
- `zenml`: Pipeline orchestration (optional)

## Example Usage

### Complete Pipeline for Ronel Solomon
```python
from main import CompleteETLPipeline

pipeline = CompleteETLPipeline()
results = pipeline.run_complete_pipeline(
    user_full_name="Ronel Solomon",
    links=[
        "https://github.com/ronelsolomon/2D-Farming-Game.git",
        "https://github.com/ronelsolomon/ai-videos.git",
        # ... more links
    ]
)
```

### Search in Vector Database
```python
from src.ingestion.pipeline import IngestionPipeline, PipelineConfig

config = PipelineConfig()
pipeline = IngestionPipeline(config)

# Search for similar content
results = pipeline.search_similar_documents(
    query_text="machine learning pipelines",
    limit=10
)
```

## Monitoring and Debugging

### Logging
All components use `loguru` for structured logging:
```python
from loguru import logger
logger.info("Processing document: {document_id}")
```

### Pipeline Status
```python
status = pipeline.get_pipeline_status()
print(f"Vector store points: {status['vector_store_info']['points_count']}")
```

### Debug Files
HTML dumps and debug info stored in `debug/` directory.

## Performance Considerations

### Optimization Tips
1. **Batch Processing**: Process documents in batches for better performance
2. **Chunk Size**: Adjust chunk size based on content type (800 for code, 1000 for articles)
3. **Embedding Batch Size**: Use batch size 16-32 for optimal GPU utilization
4. **Rate Limiting**: Respect website rate limits when crawling

### Scaling
- **Horizontal Scaling**: Multiple crawler instances
- **Vector Store Scaling**: Qdrant cluster for large datasets
- **Caching**: Redis cache for frequently accessed documents

## Troubleshooting

### Common Issues
1. **Qdrant Connection**: Ensure Qdrant is running on localhost:6333
2. **Selenium Issues**: Check ChromeDriver compatibility
3. **Memory Issues**: Reduce batch size for large datasets
4. **Rate Limiting**: Add delays between requests

### Error Recovery
- Pipeline checkpoints allow resuming from failed stages
- Error logs stored in pipeline results
- Failed documents can be reprocessed individually

## Advanced Features

### Custom Crawlers
```python
from src.crawlers.base import BaseCrawler

class CustomCrawler(BaseCrawler):
    def extract(self, url: str, user: UserDocument):
        # Custom extraction logic
        pass
```

### Custom Embeddings
```python
from src.ingestion.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator(
    model_name="your-custom-model",
    model_type="openai",  # or "huggingface"
    openai_api_key="your-key"
)
```

### Pipeline Extensions
- Add custom document types
- Implement custom cleaning rules
- Add metadata enrichment
- Integrate with external APIs

## Production Deployment

### Docker Setup
```dockerfile
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

### Environment Variables
```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export OPENAI_API_KEY=your-key
export LOG_LEVEL=INFO
```

### Monitoring
- Use ZenML dashboard for pipeline monitoring
- Monitor Qdrant collection size and performance
- Set up alerts for pipeline failures

## Contributing

### Adding New Sources
1. Create crawler in `src/crawlers/`
2. Add document model in `src/domain/`
3. Update pipeline configuration
4. Add tests

### Code Style
- Follow PEP 8
- Use type hints
- Add comprehensive tests
- Document all public methods

## License

[Add your license information here]
