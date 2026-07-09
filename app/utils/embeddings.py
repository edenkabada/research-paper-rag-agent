"""
Embedding generation utilities
"""
import numpy as np
from typing import List, Optional
import structlog
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = structlog.get_logger()

class EmbeddingGenerator:
    """Generate embeddings for text using SentenceTransformers"""
    
    def __init__(self):
        """Initialize the embedding model"""
        try:
            # Use a default model since settings doesn't have this configured
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded", model='all-MiniLM-L6-v2')
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            # Fallback to a simpler model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.warning("Using fallback embedding model", model='all-MiniLM-L6-v2')
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error("Error generating embeddings", error=str(e))
            # Return zero embeddings as fallback
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return [[0.0] * embedding_dim for _ in texts]
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error("Error generating single embedding", error=str(e))
            # Return zero embedding as fallback
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return [0.0] * embedding_dim
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error("Error calculating similarity", error=str(e))
            return 0.0
