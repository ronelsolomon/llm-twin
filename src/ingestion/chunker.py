"""
Text chunking module for the ingestion pipeline.
Splits documents into manageable chunks for embedding and storage.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document as LangchainDocument
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
        document_type = document.get('platform', 'unknown')
        
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
                       cleaner) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk
            cleaner: DataCleaner instance for text extraction
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            try:
                # Extract main text from document
                text = cleaner.extract_main_text(doc)
                
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
