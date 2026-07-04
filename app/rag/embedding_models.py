"""
Embedding model interfaces and implementations.

Provides a common interface for different embedding models while
maintaining the progressive enrichment pattern.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from app.core import logger


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    
    All embedding models must implement encode() to generate embeddings.
    """
    
    @abstractmethod
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embedding vectors.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings to unit vectors
            
        Returns:
            np.ndarray: Array of embeddings, shape (len(texts), embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass


class SentenceTransformerModel(EmbeddingModel):
    """
    Embedding model using sentence-transformers library.
    
    Supports hundreds of pre-trained models from HuggingFace.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        max_seq_length: Optional[int] = None
    ):
        """
        Initialize SentenceTransformer model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cpu" or "cuda")
            max_seq_length: Maximum sequence length (tokens)
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model_name = model_name
            self._device = device
            
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
            
            if max_seq_length:
                self.model.max_seq_length = max_seq_length
            
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Model loaded: {model_name}, "
                f"dim={self._embedding_dim}, "
                f"device={device}"
            )
        
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embedding vectors.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to encode {len(texts)} texts: {e}")
            raise
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


class HuggingFaceModel(EmbeddingModel):
    """
    Embedding model using transformers library directly.
    
    Useful for models not in sentence-transformers.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512
    ):
        """
        Initialize HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            max_length: Maximum sequence length
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self._model_name = model_name
            self._device = device
            self._max_length = max_length
            
            logger.info(f"Loading HuggingFace model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
            
            self._embedding_dim = self.model.config.hidden_size
            
            logger.info(
                f"Model loaded: {model_name}, "
                f"dim={self._embedding_dim}, "
                f"device={device}"
            )
        
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embedding vectors.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            np.ndarray: Array of embeddings
        """
        import torch
        
        if not texts:
            return np.array([])
        
        embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt"
                ).to(self._device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    # Use [CLS] token embedding (first token)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            embeddings = np.vstack(embeddings)
            
            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to encode {len(texts)} texts: {e}")
            raise
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


def create_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_length: Optional[int] = None
) -> EmbeddingModel:
    """
    Factory function to create embedding model.
    
    Args:
        model_name: Model identifier (defaults to settings.embedding_model)
        device: Device to run on (defaults to settings.embedding_device)
        max_length: Max sequence length (defaults to settings.embedding_max_length)
        
    Returns:
        EmbeddingModel: Initialized embedding model
    """
    from app.core import settings
    
    model_name = model_name or settings.embedding_model
    device = device or settings.embedding_device
    max_length = max_length or settings.embedding_max_length
    
    # Try SentenceTransformer first (easier, more models)
    try:
        return SentenceTransformerModel(
            model_name=model_name,
            device=device,
            max_seq_length=max_length
        )
    except Exception as e:
        logger.warning(f"Failed to load as SentenceTransformer: {e}")
        
        # Fallback to HuggingFace
        try:
            return HuggingFaceModel(
                model_name=model_name,
                device=device,
                max_length=max_length
            )
        except Exception as e:
            logger.error(f"Failed to load as HuggingFace model: {e}")
            raise
