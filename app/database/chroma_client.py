"""
ChromaDB client for vector database operations
"""
import uuid
from typing import List, Dict, Any, Optional
import structlog
from datetime import datetime
import chromadb
from chromadb.config import Settings
import requests
import json

from app.config import settings, get_israel_time

logger = structlog.get_logger()

class OllamaEmbeddings:
    """Custom Ollama embeddings client that uses the correct API endpoint"""
    
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip('/')
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self._get_embedding(text)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama API using the correct endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            logger.error("Failed to get embedding from Ollama", error=str(e), text=text[:100])
            raise

class ChromaClient:
    """Client for interacting with ChromaDB vector database"""
    
    def __init__(self):
        """Initialize ChromaDB client"""
        self.client = chromadb.PersistentClient(
            path="./chroma_data",  # Use local path
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize Ollama embeddings with custom implementation
        self.ollama_embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://ollama:11434"
        )
        
        # Get or create collection for academic papers with custom embedding function
        self.collection = self.client.get_or_create_collection(
            name="academic_papers",
            metadata={"description": "Academic research papers for RAG with Ollama embeddings"}
        )
        
        logger.info("ChromaDB client initialized with Ollama embeddings", 
                   collection_name="academic_papers",
                   db_path="./chroma_data",
                   embedding_model="nomic-embed-text")
    
    def add_documents(self, chunks: List[str], document_name: str) -> str:
        """
        Add document chunks to the vector database
        
        Args:
            chunks: List of text chunks from the document
            document_name: Name of the source document
            
        Returns:
            str: Document ID
        """
        document_id = str(uuid.uuid4())
        
        # Extract text content from chunk objects
        texts = []
        for chunk in chunks:
            if isinstance(chunk, dict) and "text" in chunk:
                texts.append(chunk["text"])
            elif isinstance(chunk, str):
                texts.append(chunk)
            else:
                texts.append(str(chunk))
        
        # Generate embeddings using Ollama
        try:
            embeddings = self.ollama_embeddings.embed_documents(texts)
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            raise Exception(f"Failed to generate embeddings: {str(e)}")
        
        # Prepare data for ChromaDB
        ids = [f"{document_id}_{i}" for i in range(len(texts))]
        metadatas = [
            {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": i,
                "upload_time": get_israel_time().isoformat(),
                "chunk_length": len(text)
            }
            for i, text in enumerate(texts)
        ]
        
        # Add to collection with custom embeddings
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        logger.info("Added document to ChromaDB with Ollama embeddings",
                   document_id=document_id,
                   document_name=document_name,
                   chunks_count=len(chunks))
        
        return document_id
    
    def query_documents(self, 
                       query: str, 
                       top_k: int = 5,
                       document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Query documents for relevant chunks
        
        Args:
            query: Search query
            top_k: Number of top results to return
            document_ids: Optional list of specific document IDs to search
            
        Returns:
            List of relevant document chunks with metadata
        """
        # Generate query embedding using Ollama
        try:
            query_embedding = self.ollama_embeddings.embed_query(query)
        except Exception as e:
            logger.error("Failed to generate query embedding", error=str(e))
            return []
        
        # Prepare query filters
        where_filter = None
        if document_ids:
            where_filter = {"document_id": {"$in": document_ids}}
        
        # Query the collection with custom embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity_score": 1 - distance,  # Convert distance to similarity
                    "chunk_id": results['ids'][0][i]
                })
        
        logger.info("Queried documents with Ollama embeddings",
                   query=query,
                   results_count=len(formatted_results),
                   top_k=top_k)
        
        return formatted_results
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the database
        
        Returns:
            List of document information with consistent field names
        """
        try:
            # Get all items from collection
            all_items = self.collection.get()
            
            # Group by document_id
            documents = {}
            for metadata in all_items['metadatas']:
                doc_id = metadata['document_id']
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,  # Changed from "id" to "document_id"
                        "document_name": metadata['document_name'],  # Changed from "name"
                        "chunks_count": 0,
                        "upload_time": metadata['upload_time']
                    }
                documents[doc_id]["chunks_count"] += 1
            
            logger.info("Listed documents", document_count=len(documents))
            return list(documents.values())
            
        except Exception as e:
            logger.error("Error listing documents", error=str(e))
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            bool: True if successfully deleted
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results['ids']:
                logger.warning("Document not found for deletion", document_id=document_id)
                return False
            
            # Delete all chunks
            self.collection.delete(ids=results['ids'])
            
            logger.info("Deleted document from ChromaDB",
                       document_id=document_id,
                       chunks_deleted=len(results['ids']))
            
            return True
            
        except Exception as e:
            logger.error("Error deleting document", 
                        document_id=document_id, 
                        error=str(e))
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": "academic_papers"
            }
        except Exception as e:
            logger.error("Error getting collection stats", error=str(e))
            return {"total_chunks": 0, "collection_name": "academic_papers"}
    
    def reset_collection(self) -> bool:
        """
        Reset/clear the entire collection
        
        Returns:
            bool: True if successfully reset
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection("academic_papers")
            
            self.collection = self.client.get_or_create_collection(
                name="academic_papers",
                metadata={"description": "Academic research papers for RAG with Ollama embeddings"}
            )
            
            logger.info("Reset ChromaDB collection with Ollama embeddings")
            return True
            
        except Exception as e:
            logger.error("Error resetting collection", error=str(e))
            return False
