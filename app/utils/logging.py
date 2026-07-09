"""
MongoDB logging functions for the RAG Academic Papers AI Agent
"""
import structlog
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.config import get_israel_time
from app.database.mongo_client import MongoClient

logger = structlog.get_logger()

class RAGLogger:
    """Enhanced logging utilities for RAG operations"""
    
    def __init__(self, mongo_client: MongoClient):
        """Initialize RAG logger with MongoDB client"""
        self.mongo_client = mongo_client
        
    def log_pdf_upload(self, 
                      filename: str,
                      document_id: str,
                      chunks_count: int,
                      file_size: int,
                      processing_time: float,
                      success: bool = True,
                      error_message: Optional[str] = None) -> str:
        """
        Log PDF upload operation
        
        Args:
            filename: Name of the uploaded PDF file
            document_id: Generated document ID
            chunks_count: Number of text chunks created
            file_size: Size of the uploaded file in bytes
            processing_time: Time taken to process the file
            success: Whether the upload was successful
            error_message: Error message if upload failed
            
        Returns:
            Log entry ID
        """
        try:
            upload_log = {
                "timestamp": get_israel_time(),
                "operation": "pdf_upload",
                "filename": filename,
                "document_id": document_id if success else None,
                "chunks_count": chunks_count,
                "file_size_bytes": file_size,
                "processing_time_seconds": processing_time,
                "success": success,
                "error_message": error_message,
                "metadata": {
                    "file_extension": filename.split('.')[-1].lower() if '.' in filename else None,
                    "chunks_per_mb": chunks_count / (file_size / 1024 / 1024) if file_size > 0 else 0
                }
            }
            
            # Insert into MongoDB
            result = self.mongo_client.db.uploads.insert_one(upload_log)
            
            logger.info("PDF upload logged",
                       filename=filename,
                       success=success,
                       chunks_count=chunks_count,
                       log_id=str(result.inserted_id))
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to log PDF upload", error=str(e), filename=filename)
            return ""
    
    def log_system_error(self,
                        operation: str,
                        error_message: str,
                        error_type: str,
                        context: Optional[Dict[str, Any]] = None) -> str:
        """
        Log system errors for monitoring and debugging
        
        Args:
            operation: The operation that failed
            error_message: Error message
            error_type: Type of error (e.g., 'ValidationError', 'DatabaseError')
            context: Additional context information
            
        Returns:
            Log entry ID
        """
        try:
            error_log = {
                "timestamp": get_israel_time(),
                "log_type": "system_error",
                "operation": operation,
                "error_message": error_message,
                "error_type": error_type,
                "context": context or {},
                "severity": self._get_error_severity(error_type)
            }
            
            # Insert into MongoDB
            result = self.mongo_client.db.system_logs.insert_one(error_log)
            
            logger.error("System error logged",
                        operation=operation,
                        error_type=error_type,
                        log_id=str(result.inserted_id))
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to log system error", error=str(e))
            return ""
    
    def log_performance_metrics(self,
                               operation: str,
                               metrics: Dict[str, Any]) -> str:
        """
        Log performance metrics for system monitoring
        
        Args:
            operation: The operation being measured
            metrics: Performance metrics dictionary
            
        Returns:
            Log entry ID
        """
        try:
            performance_log = {
                "timestamp": get_israel_time(),
                "log_type": "performance_metrics",
                "operation": operation,
                "metrics": metrics,
                "benchmarks": self._get_performance_benchmarks(operation, metrics)
            }
            
            # Insert into MongoDB
            result = self.mongo_client.db.performance_logs.insert_one(performance_log)
            
            logger.info("Performance metrics logged",
                       operation=operation,
                       log_id=str(result.inserted_id))
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to log performance metrics", error=str(e))
            return ""
    
    def log_user_activity(self,
                         activity_type: str,
                         details: Dict[str, Any],
                         user_session: Optional[str] = None) -> str:
        """
        Log user activity for analytics
        
        Args:
            activity_type: Type of user activity
            details: Activity details
            user_session: Optional user session identifier
            
        Returns:
            Log entry ID
        """
        try:
            activity_log = {
                "timestamp": get_israel_time(),
                "log_type": "user_activity",
                "activity_type": activity_type,
                "details": details,
                "user_session": user_session,
                "israel_timezone": True  # Flag to indicate Israel time usage
            }
            
            # Insert into MongoDB
            result = self.mongo_client.db.user_activities.insert_one(activity_log)
            
            logger.info("User activity logged",
                       activity_type=activity_type,
                       log_id=str(result.inserted_id))
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to log user activity", error=str(e))
            return ""
    
    def _get_error_severity(self, error_type: str) -> str:
        """Determine error severity based on error type"""
        high_severity = ['DatabaseError', 'ConnectionError', 'SystemError']
        medium_severity = ['ValidationError', 'ProcessingError']
        
        if error_type in high_severity:
            return 'high'
        elif error_type in medium_severity:
            return 'medium'
        else:
            return 'low'
    
    def _get_performance_benchmarks(self, operation: str, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Get performance benchmarks for comparison"""
        benchmarks = {}
        
        # Processing time benchmarks
        if 'processing_time' in metrics:
            time = metrics['processing_time']
            if operation == 'pdf_upload':
                if time < 5.0:
                    benchmarks['processing_speed'] = 'excellent'
                elif time < 15.0:
                    benchmarks['processing_speed'] = 'good'
                else:
                    benchmarks['processing_speed'] = 'needs_improvement'
            elif operation == 'query_processing':
                if time < 2.0:
                    benchmarks['response_speed'] = 'excellent'
                elif time < 10.0:
                    benchmarks['response_speed'] = 'good'
                else:
                    benchmarks['response_speed'] = 'needs_improvement'
        
        return benchmarks
    
    def get_recent_logs(self, 
                       log_type: str, 
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent logs of a specific type
        
        Args:
            log_type: Type of logs to retrieve
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        try:
            collection_map = {
                'queries': self.mongo_client.db.queries,
                'uploads': self.mongo_client.db.uploads,
                'system_errors': self.mongo_client.db.system_logs,
                'performance': self.mongo_client.db.performance_logs,
                'user_activity': self.mongo_client.db.user_activities
            }
            
            collection = collection_map.get(log_type)
            if not collection:
                logger.warning("Unknown log type requested", log_type=log_type)
                return []
            
            cursor = collection.find().sort("timestamp", -1).limit(limit)
            logs = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for log in logs:
                if '_id' in log:
                    log['_id'] = str(log['_id'])
            
            return logs
            
        except Exception as e:
            logger.error("Failed to retrieve recent logs", error=str(e), log_type=log_type)
            return []


# Global logger instance (will be initialized with MongoDB client)
rag_logger: Optional[RAGLogger] = None

def initialize_rag_logger(mongo_client: MongoClient):
    """Initialize the global RAG logger instance"""
    global rag_logger
    rag_logger = RAGLogger(mongo_client)
    logger.info("RAG logger initialized with MongoDB client")

def get_rag_logger() -> Optional[RAGLogger]:
    """Get the global RAG logger instance"""
    return rag_logger
