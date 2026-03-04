"""
Text chunking module for the ingestion pipeline.
Splits documents into manageable chunks for embedding and storage.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document as LangchainDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available, using fallback chunker")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using character-based chunking")


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    document_id: str
    document_type: str
    chunk_index: int
    metadata: Dict[str, Any]


class TextChunker:
    """Handles text chunking for various document types."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in tokens/characters
            chunk_overlap: Overlap between consecutive chunks
            separators: List of separators to split on (for LangChain)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        if LANGCHAIN_AVAILABLE:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=self._get_length_function
            )
        else:
            self.splitter = None
    
    def _get_length_function(self, text: str) -> int:
        """Get the length function for chunking."""
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                return len(encoding.encode(text))
            except:
                pass
        return len(text)
    
    def _extract_text(self, document: Dict[str, Any]) -> str:
        """
        Extract text from a document dict.

        Documents produced by process_raw_documents.py have the structure:
            {'text': '...', 'metadata': {...}}
        but legacy documents may also carry a top-level 'content' key or
        nest content under metadata.  We check all known locations in order
        of preference so the chunker never silently returns an empty string.
        """
        # Primary location used by our pipeline
        if isinstance(document.get('text'), str) and document['text'].strip():
            return document['text']

        # Legacy / alternative key names
        for key in ('content', 'page_content', 'body', 'raw_text'):
            if isinstance(document.get(key), str) and document[key].strip():
                return document[key]

        # Content nested inside metadata
        metadata = document.get('metadata', {})
        for key in ('text', 'content', 'Content', 'body'):
            if isinstance(metadata.get(key), str) and metadata[key].strip():
                return metadata[key]

        return ''

    def _extract_document_type(self, document: Dict[str, Any]) -> str:
        """
        Extract the document type from a document dict.

        Our pipeline stores document_type inside the 'metadata' sub-dict,
        but older code stored it as a top-level 'platform' key.
        """
        # Primary location
        metadata = document.get('metadata', {})
        if metadata.get('document_type'):
            return metadata['document_type']

        # Fallback top-level keys (legacy)
        for key in ('document_type', 'platform', 'type', 'source_type'):
            if document.get(key):
                return document[key]

        return 'unknown'

    def chunk_document(self, document: Dict[str, Any], text: str) -> List[Chunk]:
        """
        Chunk a single document's text.
        
        Args:
            document: Original document with metadata
            text: Extracted text to chunk
            
        Returns:
            List of chunks with metadata
        """
        if not text or not text.strip():
            logger.warning(f"No text to chunk for document {document.get('id', 'unknown')}")
            return []
        
        document_id = document.get('id', 'unknown')

        # FIX: read document_type from metadata, not from the top-level 'platform' key
        document_type = self._extract_document_type(document)
        
        if LANGCHAIN_AVAILABLE and self.splitter:
            return self._chunk_with_langchain(document, text, document_id, document_type)
        else:
            return self._chunk_fallback(document, text, document_id, document_type)
    
    def _chunk_with_langchain(self, document: Dict[str, Any], text: str, 
                            document_id: str, document_type: str) -> List[Chunk]:
        """Chunk using LangChain's text splitter."""
        try:
            # Create LangChain document
            lc_doc = LangchainDocument(
                page_content=text,
                metadata={
                    'document_id': document_id,
                    'document_type': document_type,
                    **{k: v for k, v in document.items() if k != 'content'}
                }
            )
            
            # Split the document
            split_docs = self.splitter.split_documents([lc_doc])
            
            # Convert to our Chunk format
            chunks = []
            for i, split_doc in enumerate(split_docs):
                chunk = Chunk(
                    text=split_doc.page_content,
                    chunk_id=f"{document_id}_chunk_{i}",
                    document_id=document_id,
                    document_type=document_type,
                    chunk_index=i,
                    metadata=split_doc.metadata
                )
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} chunks for document {document_id} using LangChain")
            return chunks
            
        except Exception as e:
            logger.error(f"LangChain chunking failed for document {document_id}: {e}")
            return self._chunk_fallback(document, text, document_id, document_type)
    
    def _chunk_fallback(self, document: Dict[str, Any], text: str,
                       document_id: str, document_type: str) -> List[Chunk]:
        """Fallback chunking method using simple character splitting."""
        chunks = []
        text_length = len(text)
        
        if text_length <= self.chunk_size:
            # Text is small enough for a single chunk
            chunk = Chunk(
                text=text,
                chunk_id=f"{document_id}_chunk_0",
                document_id=document_id,
                document_type=document_type,
                chunk_index=0,
                metadata={**document, 'chunk_index': 0}
            )
            chunks.append(chunk)
        else:
            # Split into overlapping chunks
            start = 0
            chunk_index = 0
            
            while start < text_length:
                end = start + self.chunk_size
                
                # Try to break at a sentence or paragraph
                if end < text_length:
                    # Look for sentence endings
                    sentence_breaks = ['. ', '! ', '? ', '\n\n']
                    for break_char in sentence_breaks:
                        break_pos = text.rfind(break_char, start, end)
                        if break_pos != -1:
                            end = break_pos + len(break_char)
                            break
                
                chunk_text = text[start:end].strip()
                
                if chunk_text:
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        document_id=document_id,
                        document_type=document_type,
                        chunk_index=chunk_index,
                        metadata={**document, 'chunk_index': chunk_index}
                    )
                    chunks.append(chunk)
                
                # Move to next chunk with overlap
                start = max(start + 1, end - self.chunk_overlap)
                chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks for document {document_id} using fallback method")
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]],
                        cleaner=None) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk.  Each document is expected
                to have the structure {'text': '...', 'metadata': {...}}.
            cleaner: Optional DataCleaner instance.  If provided its
                extract_main_text() is tried first; the built-in extraction
                logic is used as a reliable fallback so that documents are
                never silently dropped when the cleaner returns empty string.
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            try:
                # FIX: use our own robust extraction instead of relying solely
                # on the cleaner, which returns '' for our {'text':...} format.
                text = ''
                if cleaner is not None:
                    try:
                        text = cleaner.extract_main_text(doc) or ''
                    except Exception:
                        pass

                # Fall back to direct extraction if cleaner gave nothing
                if not text or not text.strip():
                    text = self._extract_text(doc)

                # Chunk the document
                chunks = self.chunk_document(doc, text)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Created total of {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'documents_processed': len(set(chunk.document_id for chunk in chunks))
        }
