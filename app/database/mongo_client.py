"""
MongoDB client for query logging and metadata storage
"""
from pymongo import MongoClient as PyMongoClient
from typing import Dict, Any, List, Optional
import structlog
from datetime import datetime
import time

from app.config import settings, get_israel_time

logger = structlog.get_logger()

class MongoClient:
    """Client for interacting with MongoDB"""
    
    def __init__(self):
        """Initialize MongoDB client"""
        self.client = PyMongoClient(settings.mongo_uri)
        self.db = self.client[settings.mongo_database]
        self.queries_collection = self.db.queries
        self.documents_collection = self.db.documents
        self.logs_collection = self.db.application_logs
        
        # Create indexes for better performance
        self._create_indexes()
        
        logger.info("MongoDB client initialized",
                   database=settings.mongo_database,
                   uri=settings.mongo_uri.split('@')[-1])  # Hide credentials in logs
    
    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Index on timestamp for queries
            self.queries_collection.create_index([("timestamp", -1)])
            
            # Index on document_id for documents
            self.documents_collection.create_index([("document_id", 1)])
            
            # Index on timestamp for application logs
            self.logs_collection.create_index([("timestamp", -1)])
            self.logs_collection.create_index([("endpoint", 1)])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.warning("Error creating MongoDB indexes", error=str(e))
    
    def log_query(self,
                  question: str,
                  answer: str,
                  sources: List[Dict[str, Any]],
                  retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
                  confidence_score: Optional[float] = None,
                  processing_time: Optional[float] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a comprehensive query interaction with all processing details
        
        Args:
            question: User's question
            answer: Generated answer
            sources: List of source documents with citations
            retrieved_chunks: Raw document chunks retrieved from vector search
            confidence_score: Optional confidence score
            processing_time: Time taken to process the query
            metadata: Additional metadata including performance metrics
            
        Returns:
            str: Query log ID
        """
        # Performance metrics
        current_time = get_israel_time()
        performance_metrics = {
            "processing_time_seconds": processing_time,
            "retrieval_time": metadata.get("retrieval_time") if metadata else None,
            "llm_generation_time": metadata.get("llm_generation_time") if metadata else None,
            "total_chunks_processed": len(retrieved_chunks) if retrieved_chunks else 0,
            "sources_cited": len(sources),
            "answer_length_chars": len(answer),
            "question_length_chars": len(question)
        }
        
        # Source citations with enhanced metadata
        enhanced_sources = []
        for source in sources:
            enhanced_source = {
                "document_name": source.get("document_name"),
                "chunk_id": source.get("chunk_id"),
                "relevance_score": source.get("relevance_score"),
                "citation_used": True,
                "chunk_position": source.get("chunk_position", 0)
            }
            enhanced_sources.append(enhanced_source)
        
        # Retrieved document chunks (raw retrieval data)
        chunk_details = []
        if retrieved_chunks:
            for i, chunk in enumerate(retrieved_chunks):
                chunk_detail = {
                    "chunk_index": i,
                    "chunk_id": chunk.get("id", f"chunk_{i}"),
                    "document_name": chunk.get("document", "unknown"),
                    "content_preview": chunk.get("content", "")[:200] + "..." if len(chunk.get("content", "")) > 200 else chunk.get("content", ""),
                    "content_length": len(chunk.get("content", "")),
                    "similarity_score": chunk.get("distance", 0.0),
                    "metadata": chunk.get("metadata", {})
                }
                chunk_details.append(chunk_detail)
        
        # Comprehensive query log
        query_log = {
            # Core identification
            "timestamp": current_time,
            "session_id": metadata.get("session_id") if metadata else None,
            
            # User query details
            "user_query": {
                "question": question,
                "question_length": len(question),
                "question_type": self._classify_question_type(question),
                "language": "en"  # Could be enhanced with language detection
            },
            
            # Retrieved document chunks (raw retrieval)
            "retrieval_data": {
                "chunks_retrieved": chunk_details,
                "total_chunks": len(chunk_details),
                "retrieval_method": "vector_similarity",
                "embedding_model": "all-MiniLM-L6-v2"
            },
            
            # Generated answer
            "generated_answer": {
                "answer": answer,
                "answer_length": len(answer),
                "confidence_score": confidence_score,
                "generation_model": metadata.get("llm_model", "llama2:7b-chat-q4_0") if metadata else "llama2:7b-chat-q4_0",
                "generation_successful": len(answer) > 0
            },
            
            # Source citations
            "source_citations": enhanced_sources,
            
            # Processing metadata
            "processing_metadata": {
                "rag_pipeline_version": "1.0.0",
                "chunk_size": metadata.get("chunk_size", 1000) if metadata else 1000,
                "chunk_overlap": metadata.get("chunk_overlap", 200) if metadata else 200,
                "max_retrieval_docs": metadata.get("max_retrieval_docs", 5) if metadata else 5,
                "temperature": metadata.get("temperature", 0.1) if metadata else 0.1
            },
            
            # Performance metrics
            "performance_metrics": performance_metrics,
            
            # System context
            "system_context": {
                "database_collections_used": ["research_papers"],
                "services_involved": ["chromadb", "ollama", "mongodb"],
                "error_occurred": metadata.get("error_occurred", False) if metadata else False,
                "error_message": metadata.get("error_message") if metadata else None
            }
        }
        
        try:
            result = self.queries_collection.insert_one(query_log)
            log_id = str(result.inserted_id)
            
            logger.info("Comprehensive query logged to MongoDB",
                       log_id=log_id,
                       question_length=len(question),
                       sources_count=len(sources),
                       chunks_retrieved=len(chunk_details),
                       processing_time=processing_time)
            
            return log_id
            
        except Exception as e:
            logger.error("Error logging comprehensive query to MongoDB", error=str(e))
            return ""
    
    def _classify_question_type(self, question: str) -> str:
        """
        Classify the type of question for analytics
        
        Args:
            question: User's question
            
        Returns:
            Question type classification
        """
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["what", "define", "definition"]):
            return "definition"
        elif any(word in question_lower for word in ["how", "process", "method"]):
            return "process"
        elif any(word in question_lower for word in ["why", "reason", "cause"]):
            return "explanation"
        elif any(word in question_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        elif any(word in question_lower for word in ["list", "enumerate", "examples"]):
            return "enumeration"
        elif "?" in question:
            return "question"
        else:
            return "general"
    
    def log_document_upload(self,
                           document_id: str,
                           document_name: str,
                           chunks_count: int,
                           file_size: Optional[int] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log document upload information
        
        Args:
            document_id: Document ID
            document_name: Name of the document
            chunks_count: Number of chunks created
            file_size: Size of the original file
            metadata: Additional metadata
            
        Returns:
            str: Document log ID
        """
        document_log = {
            "timestamp": get_israel_time(),
            "document_id": document_id,
            "document_name": document_name,
            "chunks_count": chunks_count,
            "file_size": file_size,
            "metadata": metadata or {}
        }
        
        try:
            result = self.documents_collection.insert_one(document_log)
            log_id = str(result.inserted_id)
            
            logger.info("Logged document upload to MongoDB",
                       log_id=log_id,
                       document_id=document_id,
                       document_name=document_name)
            
            return log_id
            
        except Exception as e:
            logger.error("Error logging document upload to MongoDB", error=str(e))
            return ""
    
    def get_query_history(self,
                         limit: int = 100,
                         skip: int = 0) -> List[Dict[str, Any]]:
        """
        Get query history
        
        Args:
            limit: Maximum number of queries to return
            skip: Number of queries to skip
            
        Returns:
            List of query logs
        """
        try:
            cursor = self.queries_collection.find().sort("timestamp", -1).skip(skip).limit(limit)
            queries = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for query in queries:
                query["_id"] = str(query["_id"])
            
            return queries
            
        except Exception as e:
            logger.error("Error getting query history", error=str(e))
            return []
    
    def get_document_logs(self,
                         limit: int = 100,
                         skip: int = 0) -> List[Dict[str, Any]]:
        """
        Get document upload logs
        
        Args:
            limit: Maximum number of logs to return
            skip: Number of logs to skip
            
        Returns:
            List of document logs
        """
        try:
            cursor = self.documents_collection.find().sort("timestamp", -1).skip(skip).limit(limit)
            documents = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                doc["_id"] = str(doc["_id"])
            
            return documents
            
        except Exception as e:
            logger.error("Error getting document logs", error=str(e))
            return []
    
    def get_query_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about queries
        
        Returns:
            Analytics data
        """
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_queries": {"$sum": 1},
                        "avg_processing_time": {"$avg": "$processing_time"},
                        "avg_question_length": {"$avg": "$question_length"},
                        "avg_answer_length": {"$avg": "$answer_length"},
                        "avg_sources_count": {"$avg": "$sources_count"}
                    }
                }
            ]
            
            result = list(self.queries_collection.aggregate(pipeline))
            
            if result:
                analytics = result[0]
                analytics.pop("_id", None)
                return analytics
            else:
                return {
                    "total_queries": 0,
                    "avg_processing_time": 0,
                    "avg_question_length": 0,
                    "avg_answer_length": 0,
                    "avg_sources_count": 0
                }
                
        except Exception as e:
            logger.error("Error getting query analytics", error=str(e))
            return {}
    
    def delete_query_logs(self, older_than_days: int = 30) -> int:
        """
        Delete old query logs
        
        Args:
            older_than_days: Delete logs older than this many days
            
        Returns:
            Number of deleted logs
        """
        try:
            cutoff_date = datetime.utcnow() - datetime.timedelta(days=older_than_days)
            
            result = self.queries_collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            logger.info("Deleted old query logs",
                       deleted_count=result.deleted_count,
                       older_than_days=older_than_days)
            
            return result.deleted_count
            
        except Exception as e:
            logger.error("Error deleting old query logs", error=str(e))
            return 0
    
    def log_application_activity(self,
                                endpoint: str,
                                method: str,
                                status_code: int,
                                message: str,
                                additional_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Log general application activity
        
        Args:
            endpoint: The endpoint that was accessed
            method: HTTP method (GET, POST, etc.) or SYSTEM for internal events
            status_code: HTTP status code or custom code for system events
            message: Description of the activity
            additional_data: Additional data to log
            
        Returns:
            str: Log record ID if successful, None if failed
        """
        try:
            log_record = {
                "timestamp": get_israel_time(),
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "message": message,
                "additional_data": additional_data or {}
            }
            
            result = self.logs_collection.insert_one(log_record)
            
            logger.debug("Application activity logged",
                        endpoint=endpoint,
                        method=method,
                        status_code=status_code)
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Error logging application activity",
                        endpoint=endpoint,
                        error=str(e))
            return None
    
    def get_application_logs(self,
                           limit: int = 100,
                           skip: int = 0,
                           endpoint_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve application logs
        
        Args:
            limit: Maximum number of logs to return
            skip: Number of logs to skip (for pagination)
            endpoint_filter: Optional filter by endpoint
            
        Returns:
            Dictionary containing logs and metadata
        """
        try:
            # Build query filter
            query_filter = {}
            if endpoint_filter:
                query_filter["endpoint"] = {"$regex": endpoint_filter, "$options": "i"}
            
            # Get logs sorted by timestamp (most recent first)
            cursor = self.logs_collection.find(query_filter).sort("timestamp", -1).skip(skip).limit(limit)
            logs = list(cursor)
            
            # Convert ObjectIds to strings
            for log in logs:
                log["_id"] = str(log["_id"])
                if isinstance(log.get("timestamp"), datetime):
                    log["timestamp"] = log["timestamp"].isoformat()
            
            # Get total count for pagination
            total_count = self.logs_collection.count_documents(query_filter)
            
            logger.info("Retrieved application logs",
                       count=len(logs),
                       total_count=total_count,
                       limit=limit,
                       skip=skip)
            
            return {
                "logs": logs,
                "count": len(logs),
                "total_count": total_count,
                "limit": limit,
                "skip": skip
            }
            
        except Exception as e:
            logger.error("Error retrieving application logs", error=str(e))
            return {
                "logs": [],
                "count": 0,
                "total_count": 0,
                "limit": limit,
                "skip": skip
            }
    
    def health_check(self) -> bool:
        """
        Check if MongoDB connection is healthy
        
        Returns:
            bool: True if connection is healthy
        """
        try:
            # Ping the database
            self.client.admin.command('ping')
            return True
            
        except Exception as e:
            logger.error("MongoDB health check failed", error=str(e))
            return False
