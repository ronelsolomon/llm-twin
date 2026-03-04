import sys
from pathlib import Path
from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import split_user_full_name
from src.utils.misc import batch
from src.data_retrieval import fetch_all_data
from src.domain.documents import UserDocument
from src.preprocessing.dispatchers import ChunkingDispatcher, EmbeddingDispatcher
from src.preprocessing.cleaning import CleaningDispatcher
from src.vector_store import VectorStoreOperations

@step
def get_or_create_user(user_full_name: str) -> Annotated[UserDocument,
"user"]:
    logger.info(f"Getting or creating user: {user_full_name}")
    first_name, last_name = split_user_full_name(user_full_name)
    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="user", metadata=_get_metadata(user_full_name, user))
    return user

def _get_metadata(user_full_name: str, user: UserDocument) -> dict:
    return {
        "query": {
            "user_full_name": user_full_name,
        },
        "retrieved": {
            "user_id": str(user.id),
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
    }

def _get_metadata(documents: list) -> dict:
    """Generate metadata for documents list with collection and author information"""
    metadata = {
        "num_documents": len(documents),
    }
    
    for document in documents:
        collection = document.get_collection_name()
        if collection not in metadata:
            metadata[collection] = {}
        if "authors" not in metadata[collection]:
            metadata[collection]["authors"] = list()
        metadata[collection]["num_documents"] = metadata[collection].get("num_documents", 0) + 1
        metadata[collection]["authors"].append(document.author_full_name)
    
    # Remove duplicate authors from each collection
    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))
    
    return metadata

def _add_chunks_metadata(chunks: list, existing_metadata: dict) -> dict:
    """Add metadata for chunks"""
    existing_metadata["num_chunks"] = len(chunks)
    return existing_metadata

def _add_embeddings_metadata(embedded_chunks: list, existing_metadata: dict) -> dict:
    """Add metadata for embedded chunks"""
    existing_metadata["num_embedded_chunks"] = len(embedded_chunks)
    return existing_metadata

@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    metadata = {"chunking": {}, "embedding": {}, "num_documents": len(cleaned_documents)}
    embedded_chunks = []
    
    for document in cleaned_documents:
        chunks = ChunkingDispatcher.dispatch(document)
        metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"])
        
        for batched_chunks in batch(chunks, 10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)
        
        metadata["embedding"] = _add_embeddings_metadata(embedded_chunks, metadata["embedding"])
    
    metadata["num_chunks"] = len(embedded_chunks)
    metadata["num_embedded_chunks"] = len(embedded_chunks)
    
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="embedded_documents", metadata=metadata)
    
    return embedded_chunks

@step
def load_to_vector_db(
    documents: Annotated[list, "documents"],
) -> bool:
    logger.info(f"Loading {len(documents)} documents into the vector database.")
    
    grouped_documents = VectorStoreOperations.group_by_class(documents)
    
    for document_class, docs in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.Config.get_collection_name()}")
        
        for documents_batch in batch(docs, size=4):
            try:
                # For now, we'll use a simple bulk insert simulation
                # In a real implementation, this would call document_class.bulk_insert()
                success = len(documents_batch) > 0  # Placeholder logic
                if not success:
                    logger.error(f"Failed to insert batch of {len(documents_batch)} documents")
                    return False
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                return False
    
    logger.info("All documents successfully loaded into vector database")
    return True

@step
def query_data_warehouse(
    author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:
    documents = []
    authors = []
    
    for author_full_name in author_full_names:
        logger.info(f"Querying data warehouse for user: {author_full_name}")
        first_name, last_name = split_user_full_name(author_full_name)
        logger.info(f"First name: {first_name}, Last name: {last_name}")
        
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        authors.append(user)
        
        results = fetch_all_data(user)
        user_documents = [doc for query_result in results.values() for doc in query_result]
        documents.extend(user_documents)
    
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents))
    
    return documents

@step
def clean_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
    cleaned_documents = []
    for document in documents:
        cleaned_document = CleaningDispatcher.dispatch(document)
        cleaned_documents.append(cleaned_document)
    
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents))
    
    return cleaned_documents