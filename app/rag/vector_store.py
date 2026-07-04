"""
Vector store interface for chunk persistence.

Provides abstraction over different vector databases (ChromaDB, Milvus, Qdrant, etc.)
Makes it easy to swap implementations without changing the indexer.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path


class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    All vector stores must implement these methods for:
    - Adding documents with embeddings
    - Checking if documents exist
    - Getting collection statistics
    """
    
    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        Add documents with embeddings to the store.
        
        Args:
            ids: List of unique document IDs
            embeddings: List of embedding vectors
            documents: List of document texts
            metadatas: List of metadata dictionaries
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists in the store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            bool: True if exists
        """
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Get count of documents in the store.
        
        Returns:
            int: Number of documents
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all documents from the store.
        
        Returns:
            bool: True if successful
        """
        pass


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of vector store.
    
    Provides persistence for embedded code chunks.
    """
    
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        distance_metric: str = "cosine"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
            distance_metric: Distance metric ("cosine", "l2", "ip")
        """
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.persist_directory = persist_directory
            self.collection_name = collection_name
            self.distance_metric = distance_metric
            
            # Ensure directory exists
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_metric}
            )
            
            from app.core import logger
            logger.info(
                f"ChromaVectorStore initialized: "
                f"collection={collection_name}, "
                f"distance={distance_metric}, "
                f"count={self.count()}"
            )
        
        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        except Exception as e:
            from app.core import logger
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        Add documents to ChromaDB.
        
        Args:
            ids: Document IDs
            embeddings: Embedding vectors
            documents: Document texts
            metadatas: Metadata dictionaries
            
        Returns:
            bool: True if successful
        """
        if not ids:
            return True
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            from app.core import logger
            logger.error(f"Failed to add {len(ids)} documents to ChromaDB: {e}")
            return False
    
    def document_exists(self, doc_id: str) -> bool:
        """
        Check if document exists.
        
        Args:
            doc_id: Document ID
            
        Returns:
            bool: True if exists
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception:
            return False
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by ID.
        
        Args:
            ids: Document IDs
            
        Returns:
            bool: True if successful
        """
        if not ids:
            return True
        
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception as e:
            from app.core import logger
            logger.error(f"Failed to delete {len(ids)} documents: {e}")
            return False
    
    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """
        Delete documents by metadata filter.
        
        Args:
            where: Metadata filter (e.g., {"repository_id": "abc123"})
            
        Returns:
            int: Number of documents deleted
        """
        try:
            # First, query to get IDs matching the filter
            result = self.collection.get(where=where)
            ids_to_delete = result.get("ids", [])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                from app.core import logger
                logger.info(f"Deleted {len(ids_to_delete)} documents by metadata filter")
                return len(ids_to_delete)
            
            return 0
        except Exception as e:
            from app.core import logger
            logger.error(f"Failed to delete by metadata: {e}")
            return 0
    
    def count(self) -> int:
        """
        Get document count.
        
        Returns:
            int: Number of documents
        """
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    def clear(self) -> bool:
        """
        Clear all documents.
        
        Returns:
            bool: True if successful
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            return True
        except Exception as e:
            from app.core import logger
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information.
        
        Returns:
            dict: Collection metadata
        """
        return {
            "name": self.collection_name,
            "count": self.count(),
            "distance_metric": self.distance_metric,
            "persist_directory": str(self.persist_directory),
        }
