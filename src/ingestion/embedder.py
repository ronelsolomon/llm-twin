"""
Embedding generation module for the ingestion pipeline.
Generates vector embeddings for text chunks using various models.
"""

from typing import List, Optional, Union
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not available")

from .chunker import Chunk


class EmbeddingGenerator:
    """Generates embeddings for text chunks using various models."""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model_type: str = "sentence_transformers",
                 openai_api_key: Optional[str] = None,
                 batch_size: int = 128):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the model to use
            model_type: Type of model ('sentence_transformers', 'openai')
            openai_api_key: OpenAI API key (required for OpenAI models)
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.model = None
        self.embedding_dim = None
        
        self._initialize_model(openai_api_key)
    
    def _initialize_model(self, openai_api_key: Optional[str]):
        """Initialize the embedding model."""
        try:
            if self.model_type == "sentence_transformers":
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError("sentence-transformers is required for this model type")
                
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
                
            elif self.model_type == "openai":
                if not OPENAI_AVAILABLE:
                    raise ImportError("openai is required for this model type")
                
                if not openai_api_key:
                    raise ValueError("OpenAI API key is required for OpenAI models")

                # FIX #1: openai >= 1.0 uses a client instance, not a module-level
                # api_key + openai.Embedding.create().  The old pattern was removed
                # in the v1 SDK and raises AttributeError at runtime.
                self.model = openai.OpenAI(api_key=openai_api_key)
                self.embedding_dim = self._get_openai_embedding_dimension()
                logger.info(f"OpenAI model initialized. Embedding dimension: {self.embedding_dim}")
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {e}")
            raise
    
    def _get_openai_embedding_dimension(self) -> int:
        """Get the embedding dimension for OpenAI models."""
        if "text-embedding-ada-002" in self.model_name:
            return 1536
        elif "text-embedding-3-small" in self.model_name:
            return 1536
        elif "text-embedding-3-large" in self.model_name:
            return 3072
        else:
            return 1536
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # FIX #2: silently drop None / non-string entries that would crash both
        # the sentence-transformer encoder and the OpenAI API call.
        cleaned_texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
        
        try:
            if self.model_type == "sentence_transformers":
                return self._generate_sentence_transformer_embeddings(cleaned_texts)
            elif self.model_type == "openai":
                return self._generate_openai_embeddings(cleaned_texts)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch_texts = texts[i:i + self.batch_size]
            
            # Log progress every 10 batches or for first/last batch
            if batch_num == 1 or batch_num == total_batches or batch_num % 10 == 0:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings.tolist())
                
            except Exception as e:
                logger.error(f"Failed to embed batch {batch_num}: {e}")
                zero_emb = [0.0] * self.embedding_dim
                embeddings.extend([zero_emb] * len(batch_texts))
        
        return embeddings
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # FIX #1 (continued): use the v1 SDK client API.
                # Old (broken): self.model.Embedding.create(input=..., model=...)
                # New (correct): self.model.embeddings.create(input=..., model=...)
                # Response is now an object, not a dict, so use attribute access.
                response = self.model.embeddings.create(
                    input=batch_texts,
                    model=self.model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Failed to embed batch {i//self.batch_size} with OpenAI: {e}")
                zero_emb = [0.0] * self.embedding_dim
                embeddings.extend([zero_emb] * len(batch_texts))
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[dict]:
        """
        Generate embeddings for chunks and return enriched chunk data.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of dictionaries with chunk data and embeddings
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.model_name}")
        logger.info(f"Batch size: {self.batch_size}, Estimated batches: {(len(chunks) + self.batch_size - 1) // self.batch_size}")
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.generate_embeddings(texts)

        # FIX #3: zip() silently truncates to the shorter list, so a length
        # mismatch between chunks and embeddings would drop data with no
        # warning.  Validate lengths before zipping.
        if len(embeddings) != len(chunks):
            logger.error(
                f"Embedding count mismatch: {len(embeddings)} embeddings for "
                f"{len(chunks)} chunks. Results may be incomplete."
            )
        
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_data = {
                'text': chunk.text,
                'chunk_id': chunk.chunk_id,
                'document_id': chunk.document_id,
                'document_type': chunk.document_type,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata,
                'embedding': embedding
            }
            embedded_chunks.append(chunk_data)
        
        logger.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def get_embedding_statistics(self, embedded_chunks: List[dict]) -> dict:
        """Get statistics about the generated embeddings."""
        if not embedded_chunks:
            return {}
        
        embeddings = [chunk['embedding'] for chunk in embedded_chunks]
        embedding_array = np.array(embeddings)
        
        return {
            'total_embeddings': len(embeddings),
            'embedding_dimension': len(embeddings[0]) if embeddings else 0,
            'mean_norm': np.mean(np.linalg.norm(embedding_array, axis=1)),
            'std_norm': np.std(np.linalg.norm(embedding_array, axis=1)),
            'model_used': self.model_name,
            'model_type': self.model_type
        }
    
    def test_embedding(self, test_text: str = "This is a test text for embedding generation.") -> List[float]:
        """Test the embedding generation with a sample text."""
        logger.info(f"Testing embedding generation with: '{test_text}'")
        
        try:
            embedding = self.generate_embeddings([test_text])
            if embedding:
                logger.info(f"Test successful. Embedding dimension: {len(embedding[0])}")
                return embedding[0]
            else:
                logger.error("Test failed: No embedding generated")
                return []
                
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return []
