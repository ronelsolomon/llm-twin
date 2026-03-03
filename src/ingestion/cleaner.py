"""
Data cleaning module for the ingestion pipeline.
Handles text preprocessing and normalization.
"""

import re
from typing import List, Dict, Any
from loguru import logger


class DataCleaner:
    """Cleans and preprocesses text data from various sources."""
    
    def __init__(self):
        self.patterns_to_remove = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
            r'<[^>]+>',  # HTML tags
            r'\[edit\]',  # Wikipedia edit markers
            r'\{\{[^}]*\}\}',  # Wikipedia template markers
            r'&[a-zA-Z]+;',  # HTML entities
            r'\s+',  # Multiple whitespace
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove patterns
        cleaned = text
        for pattern in self.patterns_to_remove:
            cleaned = re.sub(pattern, ' ', cleaned)
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"]', '', cleaned)
        
        return cleaned.strip()
    
    def clean_article_content(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Clean article content while preserving metadata."""
        cleaned_article = article.copy()
        
        # Clean main content
        if 'content' in article and isinstance(article['content'], dict):
            cleaned_content = {}
            for key, value in article['content'].items():
                if isinstance(value, str):
                    cleaned_content[key] = self.clean_text(value)
                else:
                    cleaned_content[key] = value
            cleaned_article['content'] = cleaned_content
        
        # Clean title if exists
        if 'title' in cleaned_article and isinstance(cleaned_article['title'], str):
            cleaned_article['title'] = self.clean_text(cleaned_article['title'])
        
        return cleaned_article
    
    def clean_repository_content(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """Clean repository content while preserving metadata."""
        cleaned_repo = repo.copy()
        
        # Clean README and description
        if 'content' in repo and isinstance(repo['content'], dict):
            cleaned_content = {}
            for key, value in repo['content'].items():
                if isinstance(value, str):
                    cleaned_content[key] = self.clean_text(value)
                else:
                    cleaned_content[key] = value
            cleaned_repo['content'] = cleaned_content
        
        return cleaned_repo
    
    def clean_documents(self, documents: List[Dict[str, Any]], document_type: str) -> List[Dict[str, Any]]:
        """Clean a list of documents based on their type."""
        cleaned_docs = []
        
        for doc in documents:
            try:
                if document_type.lower() == 'article':
                    cleaned_doc = self.clean_article_content(doc)
                elif document_type.lower() == 'repository':
                    cleaned_doc = self.clean_repository_content(doc)
                else:
                    # Generic cleaning
                    cleaned_doc = doc.copy()
                    if 'content' in cleaned_doc and isinstance(cleaned_doc['content'], str):
                        cleaned_doc['content'] = self.clean_text(cleaned_doc['content'])
                
                cleaned_docs.append(cleaned_doc)
                
            except Exception as e:
                logger.warning(f"Failed to clean document {doc.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Cleaned {len(cleaned_docs)} {document_type} documents")
        return cleaned_docs
    
    def extract_main_text(self, document: Dict[str, Any]) -> str:
        """Extract the main text content from a document for chunking."""
        text_parts = []
        
        # Handle article content
        if 'content' in document and isinstance(document['content'], dict):
            for key, value in document['content'].items():
                if isinstance(value, str) and len(value.strip()) > 10:
                    text_parts.append(value)
        
        # Handle repository content
        elif 'content' in document and isinstance(document['content'], dict):
            for key in ['readme', 'description', 'summary']:
                if key in document['content'] and isinstance(document['content'][key], str):
                    text_parts.append(document['content'][key])
        
        # Handle simple content
        elif 'content' in document and isinstance(document['content'], str):
            text_parts.append(document['content'])
        
        # Add title if available
        if 'title' in document and isinstance(document['title'], str):
            text_parts.insert(0, document['title'])
        
        return '\n\n'.join(text_parts)
