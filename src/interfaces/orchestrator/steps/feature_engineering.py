"""
Feature engineering steps for RAG pipeline
"""
from typing import List, Dict, Any
from zenml import step
from loguru import logger


@step
def query_data_warehouse(author_full_names: List[str]) -> List[Dict[str, Any]]:
    """
    Query data warehouse to extract raw documents for specified authors
    
    Args:
        author_full_names: List of author full names to query
        
    Returns:
        List of raw document dictionaries
    """
    logger.info(f"Querying data warehouse for authors: {author_full_names}")
    
    # Import here to avoid circular imports
    from src.domain.documents import ArticleDocument, UserDocument
    
    raw_documents = []
    
    for author_name in author_full_names:
        # Find user by full name
        users = UserDocument.find_all(full_name=author_name)
        
        if not users:
            logger.warning(f"No user found for author: {author_name}")
            continue
            
        user = users[0]
        logger.info(f"Found user {author_name} with ID: {user.id}")
        
        # Get all articles for this user
        articles = ArticleDocument.find_all(author_id=str(user.id))
        logger.info(f"Found {len(articles)} articles for {author_name}")
        
        for article in articles:
            raw_documents.append({
                'id': str(article.id),
                'content': article.content,
                'link': article.link,
                'platform': article.platform,
                'author_id': article.author_id,
                'author_full_name': article.author_full_name,
                'created_at': article.created_at,
                'updated_at': article.updated_at
            })
    
    logger.info(f"Total raw documents extracted: {len(raw_documents)}")
    return raw_documents


@step
def clean_documents(raw_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and preprocess raw documents
    
    Args:
        raw_documents: List of raw document dictionaries
        
    Returns:
        List of cleaned document dictionaries
    """
    logger.info(f"Cleaning {len(raw_documents)} documents")
    
    import re
    from typing import Dict, Any
    
    cleaned_documents = []
    
    for doc in raw_documents:
        try:
            # Extract text content from the document
            content = doc.get('content', {})
            
            # Handle different content structures
            if isinstance(content, dict):
                # If content is a dict, try to extract text
                text_content = content.get('text', '') or content.get('content', '') or str(content)
            else:
                text_content = str(content)
            
            # Clean the text
            cleaned_text = re.sub(r'\s+', ' ', text_content)  # Normalize whitespace
            cleaned_text = cleaned_text.strip()
            
            # Create cleaned document
            cleaned_doc = {
                'id': doc['id'],
                'content': cleaned_text,
                'link': doc['link'],
                'platform': doc['platform'],
                'author_id': doc['author_id'],
                'author_full_name': doc['author_full_name'],
                'created_at': doc['created_at'],
                'updated_at': doc['updated_at']
            }
            
            # Only include documents with meaningful content
            if len(cleaned_text) > 50:  # Minimum content length
                cleaned_documents.append(cleaned_doc)
            else:
                logger.warning(f"Document {doc['id']} has too short content after cleaning")
                
        except Exception as e:
            logger.error(f"Error cleaning document {doc.get('id', 'unknown')}: {e}")
            continue
    
    logger.info(f"Successfully cleaned {len(cleaned_documents)} documents")
    return cleaned_documents


@step
def chunk_and_embed(cleaned_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk documents and generate embeddings
    
    Args:
        cleaned_documents: List of cleaned document dictionaries
        
    Returns:
        List of chunked and embedded document dictionaries
    """
    logger.info(f"Chunking and embedding {len(cleaned_documents)} documents")
    
    # Import settings
    from src.config import settings
    
    # Import sentence transformers for embeddings
    from sentence_transformers import SentenceTransformer
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Initialize embedding model
    logger.info(f"Loading embedding model: {settings.TEXT_EMBEDDING_MODEL_ID}")
    embedding_model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL_ID, device=settings.RAG_MODEL_DEVICE)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    embedded_documents = []
    
    for doc in cleaned_documents:
        try:
            # Split document into chunks
            chunks = text_splitter.split_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                # Generate embedding for the chunk
                embedding = embedding_model.encode(chunk, convert_to_tensor=False)
                
                # Create chunk document
                chunk_doc = {
                    'id': f"{doc['id']}_chunk_{i}",
                    'content': chunk,
                    'link': doc['link'],
                    'platform': doc['platform'],
                    'author_id': doc['author_id'],
                    'author_full_name': doc['author_full_name'],
                    'chunk_index': i,
                    'document_id': doc['id'],
                    'embedding': embedding.tolist(),  # Convert to list for serialization
                    'created_at': doc['created_at'],
                    'updated_at': doc['updated_at']
                }
                
                embedded_documents.append(chunk_doc)
                
        except Exception as e:
            logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
            continue
    
    logger.info(f"Generated {len(embedded_documents)} chunks with embeddings")
    return embedded_documents


@step
def load_to_vector_db(documents: List[Dict[str, Any]]) -> Any:
    """
    Load documents to vector database (Qdrant)
    
    Args:
        documents: List of document dictionaries with or without embeddings
        
    Returns:
        Qdrant operation result
    """
    logger.info(f"Loading {len(documents)} documents to vector database")
    
    # Import settings
    from src.config import settings
    
    # Import Qdrant client
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    
    # Initialize Qdrant client
    if settings.USE_QDRANT_CLOUD:
        client = QdrantClient(
            url=settings.QDRANT_CLOUD_URL,
            api_key=settings.QDRANT_APIKEY
        )
    else:
        client = QdrantClient(
            host=settings.QDRANT_DATABASE_HOST,
            port=settings.QDRANT_DATABASE_PORT
        )
    
    # Determine if documents have embeddings
    has_embeddings = documents and 'embedding' in documents[0]
    
    # Create collection if it doesn't exist
    collection_name = "documents"
    
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        if not collection_exists:
            # Create collection
            vector_size = len(documents[0]['embedding']) if has_embeddings else 384  # Default size for MiniLM
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
        
        # Prepare points for upsert
        points = []
        for doc in documents:
            # Create point struct
            point = PointStruct(
                id=doc['id'],
                vector=doc['embedding'] if has_embeddings else None,
                payload={
                    'content': doc['content'],
                    'link': doc['link'],
                    'platform': doc['platform'],
                    'author_id': doc['author_id'],
                    'author_full_name': doc['author_full_name'],
                    'created_at': doc['created_at'],
                    'updated_at': doc['updated_at'],
                    **({'chunk_index': doc['chunk_index'], 'document_id': doc['document_id']} if 'chunk_index' in doc else {})
                }
            )
            points.append(point)
        
        # Upsert points to Qdrant
        if has_embeddings:
            result = client.upsert(
                collection_name=collection_name,
                points=points
            )
        else:
            # For documents without embeddings, we'll need to generate them first
            # For now, skip loading documents without embeddings
            logger.warning("Skipping documents without embeddings")
            result = None
        
        logger.info(f"Successfully loaded {len(points)} documents to vector database")
        return result
        
    except Exception as e:
        logger.error(f"Error loading documents to vector database: {e}")
        raise
