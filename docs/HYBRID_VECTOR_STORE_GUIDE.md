# Hybrid Vector Store Guide

## Overview

The hybrid vector store combines Qdrant (cloud/remote) and FAISS (local) storage to provide unlimited capacity with automatic fallback when payload limits are exceeded.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ingestion     │    │   Hybrid Store   │    │   Search        │
│   Pipeline      │───▶│   Manager       │◀───│   Pipeline      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │                    │
                    ▼                    ▼
            ┌──────────────┐    ┌──────────────┐
            │   Qdrant     │    │    FAISS     │
            │ (Primary)    │    │ (Fallback)   │
            └──────────────┘    └──────────────┘
```

## How It Works

1. **Payload Estimation**: Estimates JSON payload size before upload
2. **Smart Routing**: Uses Qdrant for small payloads (< 30MB)
3. **Automatic Fallback**: Switches to FAISS when payload exceeds limits
4. **Unified Search**: Searches both stores seamlessly
5. **Persistent Storage**: FAISS provides unlimited local storage

## Configuration

### Basic Setup

```python
from src.ingestion import PipelineConfig

config = PipelineConfig(
    # Enable hybrid store
    use_hybrid_store=True,
    
    # Qdrant settings
    qdrant_host="localhost",
    qdrant_port=6333,
    qdrant_collection_name="document_chunks",
    
    # FAISS settings  
    faiss_index_path="data/faiss_index",
    faiss_metadata_path="data/faiss_metadata.json",
    
    # Thresholds
    qdrant_payload_limit_mb=30,
    max_batch_size=25,
)
```

### Using Configuration File

```yaml
# configs/hybrid_vector_store.yaml
hybrid:
  enabled: true
  
qdrant:
  host: "localhost"
  port: 6333
  collection_name: "document_chunks"
  
faiss:
  index_path: "data/faiss_index"
  metadata_path: "data/faiss_metadata.json"
  
thresholds:
  qdrant_payload_limit_mb: 30
  max_batch_size: 25
```

## Usage Examples

### 1. Document Ingestion

```bash
# Uses hybrid store automatically
python scripts/process_raw_documents.py
```

### 2. Search/Query

```bash
# Interactive search mode
python scripts/inference_pipeline.py

# Single query
python scripts/inference_pipeline.py "What repositories do I have related to AI?"

# Batch queries
python scripts/inference_pipeline.py --batch questions.txt
```

### 3. Programmatic Usage

```python
from src.ingestion.hybrid_vector_store import HybridVectorStore, HybridStoreConfig

# Initialize hybrid store
config = HybridStoreConfig(
    qdrant_payload_limit_mb=30,
    max_batch_size=25
)
store = HybridVectorStore(config)

# Store chunks
embedded_chunks = [...]  # Your embedded chunks
point_ids = store.store_chunks(embedded_chunks)

# Search
results = store.search_similar(
    query_vector=embedding,
    limit=5,
    score_threshold=0.5
)
```

## Store Switching Logic

### Decision Tree

```
Is hybrid store enabled?
├─ Yes → Estimate payload size
│   ├─ Payload < 30MB → Try Qdrant
│   │   ├─ Success → Store in Qdrant
│   │   └─ Failed → Fall back to FAISS
│   └─ Payload ≥ 30MB → Use FAISS directly
└─ No → Use Qdrant only (may fail on large payloads)
```

### Batch Processing

- **Qdrant**: Small batches (25 chunks) to avoid payload limits
- **FAISS**: Large batches (unlimited) for efficiency
- **Automatic**: System chooses optimal batch size per store

## Performance Characteristics

| Store | Capacity | Latency | Persistence | Use Case |
|--------|----------|----------|-------------|----------|
| Qdrant | Limited (32MB) | Low | Remote | Small datasets, production |
| FAISS | Unlimited | Medium | Local | Large datasets, fallback |

## Monitoring & Statistics

### Get Store Statistics

```python
# Get combined statistics
stats = store.get_collection_info()
print(f"Total vectors: {stats['total_vectors']}")
print(f"Store details: {stats['stores']}")

# Example output:
{
  "type": "hybrid",
  "total_vectors": 14693,
  "total_points": 14693,
  "stores": {
    "qdrant": {
      "vectors_count": 0,
      "points_count": 0
    },
    "faiss": {
      "vectors_count": 14693,
      "metadata_count": 14693,
      "index_path": "data/faiss_index"
    }
  }
}
```

### Log Messages

```
INFO | Storing 14693 chunks in hybrid store
INFO | Estimated payload size: 204.75 MB
INFO | Storing 14693 chunks in FAISS
INFO | Successfully stored 14693 chunks in FAISS

INFO | Found 5 relevant documents
INFO | Found 5 results in FAISS
```

## Troubleshooting

### Common Issues

1. **Payload Too Large**
   ```
   ERROR | Payload error: JSON payload larger than allowed
   ```
   **Solution**: Hybrid store automatically falls back to FAISS

2. **FAISS Not Available**
   ```
   WARNING | faiss not available
   ```
   **Solution**: Install with `pip install faiss-cpu`

3. **Memory Issues**
   ```
   ERROR | Out of memory during FAISS indexing
   ```
   **Solution**: Use smaller batch sizes or process in chunks

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check store availability
store = HybridVectorStore(config)
print(f"Qdrant available: {store.qdrant_store is not None}")
print(f"FAISS available: {store.faiss_store is not None}")
```

## Best Practices

### 1. Configuration
- Set conservative payload limits (25-30MB)
- Use appropriate batch sizes per store
- Monitor storage usage

### 2. Performance
- Let hybrid store choose optimal routing
- Use FAISS for large datasets
- Keep Qdrant for frequently accessed data

### 3. Reliability
- Monitor fallback events
- Track store statistics
- Test both stores independently

## Migration Guide

### From Qdrant-Only

1. Install FAISS: `pip install faiss-cpu`
2. Enable hybrid mode: `use_hybrid_store=True`
3. Configure paths for FAISS storage
4. Test with existing data

### From FAISS-Only

1. Install and configure Qdrant
2. Enable hybrid mode
3. Set appropriate payload limits
4. Migrate data gradually

## Advanced Configuration

### Custom Thresholds

```python
config = HybridStoreConfig(
    qdrant_payload_limit_mb=50,  # Higher limit for better Qdrant
    max_batch_size=50,          # Larger batches
)
```

### Multiple Collections

```python
# Separate stores per document type
article_store = HybridVectorStore(
    config=HybridStoreConfig(
        qdrant_collection_name="articles",
        faiss_index_path="data/articles_index"
    )
)

repo_store = HybridVectorStore(
    config=HybridStoreConfig(
        qdrant_collection_name="repositories", 
        faiss_index_path="data/repos_index"
    )
)
```

## Future Enhancements

- **Load Balancing**: Distribute queries across stores
- **Cache Layer**: Cache frequently accessed vectors
- **Compression**: Reduce payload sizes
- **Multi-Region**: Geo-distributed storage
- **Real-time Sync**: Keep stores synchronized

---

This hybrid approach gives you the best of both worlds: Qdrant's performance for normal workloads and FAISS's unlimited capacity for large datasets.
