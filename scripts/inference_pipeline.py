#!/usr/bin/env python3
"""
Inference pipeline that queries vector database for relevant context and generates answers using LLM.
This script provides a complete RAG (Retrieval-Augmented Generation) system.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from loguru import logger

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion import IngestionPipeline, PipelineConfig


class InferencePipeline:
    """RAG pipeline for querying vector database and generating answers."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize inference pipeline.
        
        Args:
            config: Optional pipeline configuration for vector store connection
        """
        if config is None:
            # Default configuration matching ingestion pipeline
            config = PipelineConfig(
                qdrant_host="localhost",
                qdrant_port=6333,
                qdrant_collection_name="document_chunks",
                qdrant_distance_metric="cosine",
                use_hybrid_store=True,  # Enable hybrid store
                faiss_index_path="data/faiss_index",
                faiss_metadata_path="data/faiss_metadata.json",
            )
        
        self.config = config
        self.pipeline = None
        self.llm_client = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize vector store and LLM components."""
        logger.info("Initializing inference pipeline components...")
        
        # Initialize vector store (reusing ingestion pipeline)
        try:
            self.pipeline = IngestionPipeline(self.config)
            logger.info("✅ Vector store initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise
        
        # Initialize LLM client (using OpenAI by default)
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client for answer generation."""
        # Try OpenAI first
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                import openai
                self.llm_client = openai.OpenAI(api_key=openai_api_key)
                self.llm_type = "openai"
                self.llm_model = "gpt-3.5-turbo"
                logger.info("✅ OpenAI LLM initialized")
                return
            except ImportError:
                logger.warning("OpenAI library not available")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Fallback to a simple response if no LLM is available
        logger.warning("⚠️  No LLM client available. Will return context only.")
        self.llm_client = None
        self.llm_type = "none"
    
    def search_relevant_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents in vector database.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        logger.info(f"🔍 Searching for relevant documents: '{query}'")
        
        try:
            # Generate embedding for query
            from src.ingestion.chunker import Chunk
            query_chunk = Chunk(
                text=query,
                chunk_id="query",
                document_id="query", 
                document_type="query",
                chunk_index=0,
                metadata={}
            )
            embedded_query = self.pipeline.embedder.embed_chunks([query_chunk])[0]
            query_embedding = embedded_query['embedding']
            
            # Use hybrid vector store search
            results = self.pipeline.vector_store.search_similar(
                query_vector=query_embedding,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} relevant documents")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. Score: {result['score']:.3f} - {result['text'][:100]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching documents: {e}")
            return []
    
    def generate_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate formatted context from search results.
        
        Args:
            search_results: List of search results with metadata
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            # Extract relevant information
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            
            # Format context with source information
            source_info = []
            if metadata.get('repository_name'):
                source_info.append(f"Repo: {metadata['repository_name']}")
            if metadata.get('file_path'):
                source_info.append(f"File: {metadata['file_path']}")
            if metadata.get('title'):
                source_info.append(f"Title: {metadata['title']}")
            if metadata.get('url'):
                source_info.append(f"URL: {metadata['url']}")
            
            source_str = " | ".join(source_info) if source_info else "Unknown source"
            
            context_part = f"""
Document {i} (Score: {result.get('score', 0):.3f})
Source: {source_str}
Content: {text}
---"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using LLM based on query and context.
        
        Args:
            query: Original user query
            context: Retrieved context from vector database
            
        Returns:
            Generated answer
        """
        if not self.llm_client:
            return f"Based on the retrieved context:\n\n{context}"
        
        logger.info("🤖 Generating answer using LLM...")
        
        # Create prompt for LLM
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the context doesn't contain 
the answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            if self.llm_type == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                answer = response.choices[0].message.content.strip()
                logger.info("✅ Answer generated successfully")
                return answer
            else:
                return f"LLM type {self.llm_type} not implemented yet."
                
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def query(self, question: str, max_context_docs: int = 5) -> Dict[str, Any]:
        """
        Complete RAG query: search for context and generate answer.
        
        Args:
            question: User's question
            max_context_docs: Maximum number of context documents to retrieve
            
        Returns:
            Dictionary with query results and answer
        """
        logger.info(f"🎯 Processing query: '{question}'")
        
        # Step 1: Search for relevant documents
        search_results = self.search_relevant_documents(question, limit=max_context_docs)
        
        # Step 2: Generate context
        context = self.generate_context(search_results)
        
        # Step 3: Generate answer
        answer = self.generate_answer(question, context)
        
        # Compile results
        results = {
            'question': question,
            'context_documents': len(search_results),
            'context': context,
            'answer': answer,
            'sources': [
                {
                    'text': result.get('text', '')[:200] + '...',
                    'score': result.get('score', 0),
                    'metadata': result.get('metadata', {})
                }
                for result in search_results
            ]
        }
        
        return results
    
    def interactive_mode(self):
        """Run the inference pipeline in interactive mode."""
        print("🤖 LLM Twin Inference Pipeline")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'help' for available commands")
        print()
        
        while True:
            try:
                question = input("❓ Ask a question: ").strip()
                
                if question.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if not question:
                    continue
                
                # Process the query
                results = self.query(question)
                self._display_results(results)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error in interactive mode: {e}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
Available commands:
- Type your question directly to query the vector database
- 'help' - Show this help message
- 'quit' or 'exit' - Exit the program

Example questions:
- "What repositories do I have related to AI?"
- "Show me Python code for data processing"
- "What articles have I written about machine learning?"
- "Find code related to web scraping"
"""
        print(help_text)
    
    def _display_results(self, results: Dict[str, Any]):
        """Display query results in a formatted way."""
        print(f"\n🎯 Question: {results['question']}")
        print(f"📚 Context Sources: {results['context_documents']} documents")
        print("\n💡 Answer:")
        print(results['answer'])
        
        if results.get('sources'):
            print(f"\n📖 Sources:")
            for i, source in enumerate(results['sources'], 1):
                metadata = source.get('metadata', {})
                source_name = (
                    metadata.get('repository_name') or 
                    metadata.get('title') or 
                    metadata.get('file_path') or 
                    'Unknown'
                )
                print(f"  {i}. {source_name} (Score: {source['score']:.3f})")
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of results for each question
        """
        logger.info(f"🔄 Processing batch of {len(questions)} questions")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            result = self.query(question)
            results.append(result)
        
        return results


def main():
    """Main function to run the inference pipeline."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("🚀 Starting LLM Twin Inference Pipeline")
    
    # Check Qdrant connection
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            collections = response.json().get('result', {}).get('collections', [])
            collection_names = [c['name'] for c in collections]
            logger.info(f"✅ Qdrant connected. Available collections: {collection_names}")
        else:
            logger.error("❌ Cannot connect to Qdrant")
            return
    except Exception as e:
        logger.error(f"❌ Cannot connect to Qdrant: {e}")
        logger.error("Please ensure Qdrant is running on localhost:6333")
        return
    
    # Initialize inference pipeline
    try:
        inference = InferencePipeline()
    except Exception as e:
        logger.error(f"❌ Failed to initialize inference pipeline: {e}")
        return
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # Batch mode - read questions from file
            questions_file = sys.argv[2] if len(sys.argv) > 2 else "questions.txt"
            
            if Path(questions_file).exists():
                with open(questions_file, 'r', encoding='utf-8') as f:
                    questions = [line.strip() for line in f if line.strip()]
                
                logger.info(f"Processing {len(questions)} questions from {questions_file}")
                results = inference.batch_query(questions)
                
                # Save results
                output_file = "inference_results.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Batch processing complete. Results saved to {output_file}")
            else:
                logger.error(f"Questions file not found: {questions_file}")
        else:
            # Single query mode
            question = " ".join(sys.argv[1:])
            results = inference.query(question)
            inference._display_results(results)
    else:
        # Interactive mode
        inference.interactive_mode()


if __name__ == "__main__":
    main()
