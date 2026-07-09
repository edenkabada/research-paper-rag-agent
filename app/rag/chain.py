"""
RAG Chain implementation for academic paper Q&A
"""
import time
from typing import List, Dict, Any, Optional
import structlog
import ollama

from app.database.chroma_client import ChromaClient
from app.rag.prompt_templates import (
    create_system_message, 
    create_user_message, 
    format_context,
    CONFIDENCE_PROMPT
)
from app.rag.output_parser import RAGOutputParser
from app.models import QueryResponse
from app.config import settings

logger = structlog.get_logger()

class RAGChain:
    """RAG implementation for querying academic papers"""
    
    def __init__(self, chroma_client: ChromaClient):
        """
        Initialize RAG chain
        
        Args:
            chroma_client: ChromaDB client for document retrieval
        """
        self.chroma_client = chroma_client
        self.output_parser = RAGOutputParser()
        self.ollama_client = ollama.Client(host=settings.ollama_host)
        
        # Store last query data for comprehensive logging
        self._last_retrieved_chunks = []
        self._last_processing_metadata = {}
        
        logger.info("RAG chain initialized",
                   ollama_host=settings.ollama_host,
                   model=settings.ollama_model)
    
    def get_last_retrieved_chunks(self) -> List[Dict[str, Any]]:
        """Get the retrieved chunks from the last query for logging purposes"""
        return self._last_retrieved_chunks
    
    def get_last_processing_metadata(self) -> Dict[str, Any]:
        """Get the processing metadata from the last query for logging purposes"""
        return self._last_processing_metadata
    
    def retrieve_documents(self, 
                          question: str, 
                          top_k: int = 5,
                          document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the question
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            document_ids: Optional list of specific document IDs to search
            
        Returns:
            List of relevant document chunks
        """
        try:
            retrieved_docs = self.chroma_client.query_documents(
                query=question,
                top_k=top_k,
                document_ids=document_ids
            )
            
            logger.info("Retrieved documents",
                       question=question,
                       retrieved_count=len(retrieved_docs),
                       top_k=top_k,
                       document_ids=document_ids)
            
            return retrieved_docs
            
        except Exception as e:
            logger.error("Error retrieving documents", 
                        question=question, 
                        error=str(e))
            return []
    
    def generate_answer(self, 
                       question: str, 
                       context: str) -> str:
        """
        Generate answer using Ollama LLM
        
        Args:
            question: User's question
            context: Formatted context from retrieved documents
            
        Returns:
            Generated answer
        """
        try:
            # Prepare messages
            messages = [
                create_system_message(),
                create_user_message(question, context)
            ]
            
            # Generate response using Ollama
            response = self.ollama_client.chat(
                model=settings.ollama_model,
                messages=messages,
                options={
                    "temperature": 0.1,  # Low temperature for factual responses
                    "top_p": 0.9,
                    "max_tokens": 1000,
                    "stop": ["Human:", "Assistant:"]
                }
            )
            
            answer = response['message']['content'].strip()
            
            logger.info("Generated answer using Ollama",
                       question=question,
                       answer_length=len(answer),
                       model=settings.ollama_model)
            
            return answer
            
        except Exception as e:
            logger.error("Error generating answer", 
                        question=question, 
                        error=str(e))
            return "I apologize, but I encountered an error while generating the answer. Please try again."
    
    def estimate_confidence(self, 
                           question: str, 
                           answer: str,
                           retrieved_docs: List[Dict[str, Any]]) -> float:
        """
        Estimate confidence score for the answer
        
        Args:
            question: Original question
            answer: Generated answer
            retrieved_docs: Retrieved documents used
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple, reliable confidence estimation
        confidence_factors = []
        
        # Factor 1: Number of sources (weight: 0.3)
        source_count = len(retrieved_docs) if retrieved_docs else 0
        if source_count >= 3:
            source_factor = 0.9
        elif source_count >= 2:
            source_factor = 0.7
        elif source_count >= 1:
            source_factor = 0.5
        else:
            source_factor = 0.2
        confidence_factors.append(source_factor * 0.3)
        
        # Factor 2: Answer length (weight: 0.3)
        answer_length = len(answer.split()) if answer else 0
        if 20 <= answer_length <= 200:
            length_factor = 0.8
        elif answer_length >= 10:
            length_factor = 0.6
        elif answer_length >= 5:
            length_factor = 0.4
        else:
            length_factor = 0.2
        confidence_factors.append(length_factor * 0.3)
        
        # Factor 3: Base confidence (weight: 0.4)
        base_factor = 0.7  # Default confidence for functioning system
        confidence_factors.append(base_factor * 0.4)
        
        # Calculate final score
        confidence_score = sum(confidence_factors)
        
        # Ensure bounds
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return confidence_score
    
    def query(self, 
              question: str, 
              top_k: int = 5,
              include_confidence: bool = True,
              document_ids: Optional[List[str]] = None) -> QueryResponse:
        """
        Process a complete RAG query
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            include_confidence: Whether to include confidence estimation
            document_ids: Optional list of specific document IDs to search
            
        Returns:
            QueryResponse with answer and metadata
        """
        start_time = time.time()
        retrieval_start = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(question, top_k, document_ids)
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_docs:
                logger.warning("No documents retrieved for question", question=question)
                return QueryResponse(
                    success=True,
                    answer="I couldn't find any relevant information in the uploaded papers to answer your question.",
                    sources=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Format context
            context = format_context(retrieved_docs)
            
            # Step 3: Generate answer
            llm_start = time.time()
            answer = self.generate_answer(question, context)
            llm_time = time.time() - llm_start
            
            # Step 4: Parse output
            parsed_response = self.output_parser.parse_full_response(
                answer=answer,
                retrieved_docs=retrieved_docs
            )
            
            # Step 5: Estimate confidence if requested
            final_confidence = 0.5  # Default fallback
            if include_confidence:
                try:
                    calculated_confidence = self.estimate_confidence(question, answer, retrieved_docs)
                    # Ensure we have a valid confidence score
                    if calculated_confidence is not None and 0.0 <= calculated_confidence <= 1.0:
                        final_confidence = calculated_confidence
                    else:
                        final_confidence = 0.6  # Better default for working system
                except Exception as e:
                    logger.error("Error calculating confidence", error=str(e))
                    final_confidence = 0.5  # Default fallback
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Store retrieved chunks for logging (not in API response)
            self._last_retrieved_chunks = retrieved_docs
            self._last_processing_metadata = {
                "retrieval_time": retrieval_time,
                "llm_generation_time": llm_time,
                "llm_model": settings.ollama_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "max_retrieval_docs": top_k,
                "temperature": 0.1,
                "error_occurred": False
            }
            
            # Create response
            response = QueryResponse(
                success=True,
                answer=parsed_response["answer"],
                sources=parsed_response["sources"],
                confidence_score=final_confidence,  # Use our calculated confidence
                processing_time=processing_time
            )
            
            logger.info("Completed RAG query",
                       question=question,
                       processing_time=processing_time,
                       retrieved_chunks=len(retrieved_docs),
                       confidence_score=parsed_response.get("confidence_score"))
            
            return response
            
        except Exception as e:
            logger.error("Error in RAG query", question=question, error=str(e))
            
            return QueryResponse(
                success=False,
                answer="I apologize, but I encountered an error while processing your question. Please try again.",
                sources=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of RAG components
        
        Returns:
            Health status dictionary
        """
        health_status = {
            "chroma_db": False,
            "ollama": False,
            "overall": False
        }
        
        try:
            # Check ChromaDB
            stats = self.chroma_client.get_collection_stats()
            health_status["chroma_db"] = True
            health_status["chroma_stats"] = stats
            
        except Exception as e:
            logger.error("ChromaDB health check failed", error=str(e))
        
        try:
            # Check Ollama
            models = self.ollama_client.list()
            health_status["ollama"] = True
            health_status["ollama_models"] = [model['name'] for model in models['models']]
            
        except Exception as e:
            logger.error("Ollama health check failed", error=str(e))
        
        # Overall health
        health_status["overall"] = health_status["chroma_db"] and health_status["ollama"]
        
        return health_status
