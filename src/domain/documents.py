import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path


class NoSQLBaseDocument:
    """Simple file-based NoSQL document base class"""
    
    def __init__(self, **kwargs):
        # Set absolute path to data directory relative to this file
        self._db_path = Path(__file__).parent.parent / "data"
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.id = kwargs.get('id', self._generate_id())
        self.created_at = kwargs.get('created_at', datetime.now().isoformat())
        self.updated_at = kwargs.get('updated_at', datetime.now().isoformat())
    
    def _generate_id(self) -> str:
        """Generate unique ID based on timestamp and random"""
        import uuid
        return str(uuid.uuid4())
    
    @classmethod
    def _get_collection_path(cls) -> Path:
        """Get the collection file path"""
        # Use absolute path relative to this file
        db_path = Path(__file__).parent.parent / "data"
        db_path.mkdir(exist_ok=True)
        collection_name = cls.__name__.lower()
        # Handle special pluralization cases
        if 'repository' in collection_name:
            collection_name = collection_name.replace('repositorydocument', 'repositories')
        elif 'article' in collection_name:
            collection_name = collection_name.replace('articledocument', 'articles')
        elif 'user' in collection_name:
            collection_name = collection_name.replace('userdocument', 'users')
        else:
            # Fallback to original logic for other document types
            collection_name = collection_name.replace('document', 's')
        return db_path / f"{collection_name}.json"
    
    @classmethod
    def _load_collection(cls) -> List[Dict]:
        """Load all documents from collection file"""
        collection_path = cls._get_collection_path()
        if not collection_path.exists():
            return []
        
        try:
            with open(collection_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    @classmethod
    def _save_collection(cls, data: List[Dict]) -> None:
        """Save all documents to collection file"""
        collection_path = cls._get_collection_path()
        with open(collection_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save(self) -> None:
        """Save document to database"""
        collection = self._load_collection()
        
        # Convert document to dict
        doc_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                doc_dict[key] = value
            else:
                doc_dict[key] = str(value)
        
        # Update timestamp
        doc_dict['updated_at'] = datetime.now().isoformat()
        
        # Find existing document by ID or append new
        existing_index = None
        for i, doc in enumerate(collection):
            if doc.get('id') == self.id:
                existing_index = i
                break
        
        if existing_index is not None:
            collection[existing_index] = doc_dict
        else:
            collection.append(doc_dict)
        
        self._save_collection(collection)
    
    @classmethod
    def find(cls, **kwargs) -> Optional['NoSQLBaseDocument']:
        """Find document by criteria"""
        collection = cls._load_collection()
        
        for doc_dict in collection:
            match = True
            for key, value in kwargs.items():
                if doc_dict.get(key) != value:
                    match = False
                    break
            if match:
                return cls(**doc_dict)
        
        return None
    
    @classmethod
    def find_all(cls, **kwargs) -> List['NoSQLBaseDocument']:
        """Find all documents matching criteria"""
        collection = cls._load_collection()
        results = []
        
        for doc_dict in collection:
            match = True
            for key, value in kwargs.items():
                if doc_dict.get(key) != value:
                    match = False
                    break
            if match:
                results.append(cls(**doc_dict))
        
        return results
    
    @classmethod
    def get_or_create(cls, **kwargs) -> 'NoSQLBaseDocument':
        """Get existing document or create new one"""
        existing = cls.find(**kwargs)
        if existing:
            return existing
        return cls(**kwargs)
    
    @classmethod
    def get_all(cls) -> List['NoSQLBaseDocument']:
        """Get all documents in collection"""
        collection = cls._load_collection()
        return [cls(**doc_dict) for doc_dict in collection]


class UserDocument(NoSQLBaseDocument):
    """User document model"""
    def __init__(self, id: str = None, full_name: str = None, first_name: str = None, last_name: str = None, **kwargs):
        # Handle both full_name and separate first/last name
        if full_name and not first_name and not last_name:
            name_parts = full_name.split()
            first_name = name_parts[0] if name_parts else ""
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
        elif not full_name and first_name and last_name:
            full_name = f"{first_name} {last_name}"
        
        super().__init__(id=id, full_name=full_name, first_name=first_name, last_name=last_name, **kwargs)


class ArticleDocument(NoSQLBaseDocument):
    """Article document model"""
    def __init__(self, content: Dict[str, Any], link: str, platform: str, 
                 author_id: str, author_full_name: str, **kwargs):
        super().__init__(
            content=content,
            link=link,
            platform=platform,
            author_id=author_id,
            author_full_name=author_full_name,
            **kwargs
        )


class RepositoryDocument(NoSQLBaseDocument):
    """Repository document model"""
    def __init__(self, content: Dict[str, Any], name: str, link: str, platform: str,
                 author_id: str, author_full_name: str, **kwargs):
        super().__init__(
            content=content,
            name=name,
            link=link,
            platform=platform,
            author_id=author_id,
            author_full_name=author_full_name,
            **kwargs
        )
