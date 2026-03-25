"""
Enhanced Retrieval Pipeline with Query Expansion and Reranking
Implements the 6-step process: query expansion, self-querying, filtered search, collection, and reranking
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from .chunker import Chunk
from .hybrid_vector_store import HybridVectorStore, HybridStoreConfig
from .embedder import EmbeddingGenerator

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available for reranking")


@dataclass
class QueryExpansion:
    """Expanded query with metadata"""
    original_query: str
    expanded_queries: List[str]
    query_metadata: Dict[str, Any]
    expansion_strategy: str


@dataclass
class RetrievalConfig:
    """Configuration for enhanced retrieval pipeline"""
    # Query expansion
    expansion_strategies: List[str] = None
    max_expanded_queries: int = 5
    
    # Search parameters
    top_k_per_query: int = 10
    final_top_k: int = 5
    
    # Reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_threshold: float = 0.2
    
    # Filtering
    enable_metadata_filtering: bool = True
    filter_fields: List[str] = None
    
    def __post_init__(self):
        if self.expansion_strategies is None:
            self.expansion_strategies = ["synonym", "paraphrase", "domain_specific"]
        if self.filter_fields is None:
            self.filter_fields = ["document_type", "author", "topic"]


class QueryExpander:
    """Expands queries using multiple strategies"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        
    def expand_query(self, query: str, domain_context: Optional[Dict] = None) -> QueryExpansion:
        """Expand query using multiple strategies"""
        expanded_queries = [query]  # Include original
        query_metadata = self._extract_metadata(query, domain_context)
        
        for strategy in self.config.expansion_strategies:
            strategy_queries = self._apply_expansion_strategy(query, strategy, domain_context)
            expanded_queries.extend(strategy_queries)
        
        # Limit to max expanded queries
        expanded_queries = expanded_queries[:self.config.max_expanded_queries]
        
        return QueryExpansion(
            original_query=query,
            expanded_queries=expanded_queries,
            query_metadata=query_metadata,
            expansion_strategy=",".join(self.config.expansion_strategies)
        )
    
    def _extract_metadata(self, query: str, domain_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract metadata for filtering (self-querying step)"""
        metadata = {}
        
        # Extract potential author names
        if "by " in query.lower():
            potential_author = query.lower().split("by ")[-1].split()[0]
            metadata["potential_author"] = potential_author
        
        # Extract document types
        doc_types = ["article", "blog", "repository", "code", "documentation"]
        for doc_type in doc_types:
            if doc_type in query.lower():
                metadata["document_type"] = doc_type
        
        # Extract time-based filters
        time_words = ["recent", "latest", "new", "old", "previous"]
        for time_word in time_words:
            if time_word in query.lower():
                metadata["time_preference"] = time_word
        
        # Add domain context if provided
        if domain_context:
            metadata.update(domain_context)
        
        return metadata
    
    def _apply_expansion_strategy(self, query: str, strategy: str, domain_context: Optional[Dict] = None) -> List[str]:
        """Apply specific expansion strategy"""
        if strategy == "synonym":
            return self._expand_with_synonyms(query)
        elif strategy == "paraphrase":
            return self._expand_with_paraphrases(query)
        elif strategy == "domain_specific":
            return self._expand_domain_specific(query, domain_context)
        elif strategy == "decomposition":
            return self._expand_by_decomposition(query)
        else:
            return []
    
    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms"""
        synonym_map = {
            "machine learning": ["ML", "artificial intelligence", "AI", "deep learning"],
            "code": ["programming", "software", "development", "implementation"],
            "article": ["blog post", "documentation", "paper", "writeup"],
            "analysis": ["review", "examination", "study", "investigation"],
            "performance": ["speed", "efficiency", "optimization", "benchmark"]
        }
        
        expanded = []
        for term, synonyms in synonym_map.items():
            if term.lower() in query.lower():
                for synonym in synonyms:
                    expanded_query = query.lower().replace(term.lower(), synonym)
                    expanded.append(expanded_query)
        
        return expanded[:2]  # Limit synonyms per term
    
    def _expand_with_paraphrases(self, query: str) -> List[str]:
        """Generate paraphrased versions of the query"""
        paraphrases = []
        
        # Simple paraphrase patterns
        if "how to" in query.lower():
            paraphrases.append(query.lower().replace("how to", "steps for"))
            paraphrases.append(query.lower().replace("how to", "guide to"))
        elif "what is" in query.lower():
            paraphrases.append(query.lower().replace("what is", "explain"))
            paraphrases.append(query.lower().replace("what is", "definition of"))
        elif "benefits of" in query.lower():
            paraphrases.append(query.lower().replace("benefits of", "advantages of"))
            paraphrases.append(query.lower().replace("benefits of", "pros of"))
        
        return paraphrases[:2]
    
    def _expand_domain_specific(self, query: str, domain_context: Optional[Dict] = None) -> List[str]:
        """Expand with domain-specific terminology"""
        if not domain_context:
            return []
        
        domain_terms = domain_context.get("domain_terms", [])
        expanded = []
        
        # Add domain-specific context to query
        for term in domain_terms:
            if term not in query.lower():
                expanded_query = f"{query} {term}"
                expanded.append(expanded_query)
        
        return expanded[:2]
    
    def _expand_by_decomposition(self, query: str) -> List[str]:
        """Decompose complex queries into simpler sub-queries"""
        # Split by conjunctions
        conjunctions = ["and", "or", "but", "while", "with"]
        sub_queries = []
        
        for conj in conjunctions:
            if f" {conj} " in query.lower():
                parts = query.lower().split(f" {conj} ")
                sub_queries.extend([part.strip() for part in parts if part.strip()])
        
        return sub_queries[:2]


class Reranker:
    """Reranks search results using cross-encoder models"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.model = None
        self._initialize_reranker()
    
    def _initialize_reranker(self):
        """Initialize the cross-encoder reranker"""
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("Cross-encoder not available, reranking disabled")
            return
        
        try:
            self.model = CrossEncoder(self.config.reranker_model)
            logger.info(f"Loaded reranker: {self.config.reranker_model}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            self.model = None
    
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results based on relevance to query"""
        if not self.model or not results:
            return results
        
        # Prepare query-document pairs
        query_doc_pairs = [(query, result["text"]) for result in results]
        
        try:
            # Compute relevance scores
            scores = self.model.predict(query_doc_pairs)
            
            # Add scores to results
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
            
            # Filter by threshold and sort
            filtered_results = [
                result for result in results 
                if result["rerank_score"] >= self.config.rerank_threshold
            ]
            
            # Sort by rerank score
            filtered_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(f"Reranked {len(results)} results to {len(filtered_results)} after filtering")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results


class EnhancedRetrievalPipeline:
    """Enhanced retrieval pipeline with query expansion and reranking"""
    
    def __init__(self, 
                 vector_store: HybridVectorStore,
                 embedder: EmbeddingGenerator,
                 config: RetrievalConfig = None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        
        self.query_expander = QueryExpander(self.config)
        self.reranker = Reranker(self.config)
        
        logger.info("Enhanced retrieval pipeline initialized")
    
    def retrieve(self, 
                query: str, 
                domain_context: Optional[Dict] = None,
                top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the 6-step retrieval process:
        1. Query expansion
        2. Self-querying (metadata extraction)
        3. Filtered vector search
        4. Collect results
        5. Reranking
        6. Return top K results
        """
        if top_k is None:
            top_k = self.config.final_top_k
        
        logger.info(f"Starting enhanced retrieval for query: '{query}'")
        
        # Step 1: Query expansion
        query_expansion = self.query_expander.expand_query(query, domain_context)
        expanded_queries = query_expansion.expanded_queries
        query_metadata = query_expansion.query_metadata
        
        logger.info(f"Expanded to {len(expanded_queries)} queries: {expanded_queries}")
        
        # Step 2 & 3: Self-querying and filtered vector search
        all_results = []
        search_metadata = self._build_search_filters(query_metadata)
        
        for expanded_query in expanded_queries:
            # Generate embedding for expanded query
            query_embedding = self.embedder.generate_embeddings([expanded_query])[0]
            
            # Search with filters
            results = self.vector_store.search_similar(
                query_vector=query_embedding,
                limit=self.config.top_k_per_query,
                filter_conditions=search_metadata
            )
            
            # Add query info to results
            for result in results:
                result["matched_query"] = expanded_query
                result["expansion_strategy"] = query_expansion.expansion_strategy
            
            all_results.extend(results)
        
        logger.info(f"Collected {len(all_results)} raw results from all queries")
        
        # Step 4: Deduplicate and collect results
        deduplicated_results = self._deduplicate_results(all_results)
        logger.info(f"After deduplication: {len(deduplicated_results)} results")
        
        # Step 5: Reranking
        reranked_results = self.reranker.rerank(query, deduplicated_results)
        
        # Step 6: Return top K results
        final_results = reranked_results[:top_k]
        
        # Prepare comprehensive response
        retrieval_result = {
            "query": query,
            "expanded_queries": expanded_queries,
            "query_metadata": query_metadata,
            "search_filters": search_metadata,
            "total_raw_results": len(all_results),
            "deduplicated_count": len(deduplicated_results),
            "reranked_count": len(reranked_results),
            "final_results": final_results,
            "retrieval_metadata": {
                "expansion_strategies": self.config.expansion_strategies,
                "reranker_model": self.config.reranker_model if self.reranker.model else None,
                "top_k_per_query": self.config.top_k_per_query,
                "final_top_k": top_k
            }
        }
        
        logger.info(f"Retrieval complete: {len(final_results)} final results")
        return retrieval_result
    
    def _build_search_filters(self, query_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build search filters from extracted metadata"""
        if not self.config.enable_metadata_filtering:
            return None
        
        filters = {}
        
        # Map metadata to filter fields
        if "potential_author" in query_metadata:
            filters["author"] = query_metadata["potential_author"]
        
        if "document_type" in query_metadata:
            filters["document_type"] = query_metadata["document_type"]
        
        # Add other metadata filters
        for field in self.config.filter_fields:
            if field in query_metadata:
                filters[field] = query_metadata[field]
        
        return filters if filters else None
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity"""
        seen_texts = set()
        deduplicated = []
        
        for result in results:
            text = result.get("text", "").strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                deduplicated.append(result)
        
        return deduplicated
    
    def evaluate_retrieval_quality(self, retrieval_result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the quality of retrieval results"""
        final_results = retrieval_result.get("final_results", [])
        
        if not final_results:
            return {"overall_score": 0.0}
        
        # Calculate quality metrics
        metrics = {
            "coverage_score": min(len(final_results) / self.config.final_top_k, 1.0),
            "diversity_score": self._calculate_diversity(final_results),
            "relevance_score": self._calculate_average_relevance(final_results),
            "expansion_utilization": len(retrieval_result.get("expanded_queries", [])) / self.config.max_expanded_queries
        }
        
        # Overall score
        metrics["overall_score"] = sum(metrics.values()) / len(metrics)
        
        return metrics
    
    def _calculate_diversity(self, results: List[Dict[str, Any]]) -> float:
        """Calculate diversity of document types in results"""
        if not results:
            return 0.0
        
        doc_types = set()
        for result in results:
            doc_type = result.get("document_type", "unknown")
            doc_types.add(doc_type)
        
        return min(len(doc_types) / 3, 1.0)  # Normalize to max 3 types
    
    def _calculate_average_relevance(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average relevance score from reranking"""
        if not results:
            return 0.0
        
        rerank_scores = [r.get("rerank_score", r.get("score", 0)) for r in results]
        return np.mean(rerank_scores) if rerank_scores else 0.0
