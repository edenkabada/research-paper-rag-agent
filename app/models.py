from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.config import get_israel_time


class PDFUploadRequest(BaseModel):
    """Request model for PDF upload"""
    files: List[str] = Field(..., description="List of PDF files to upload")
    
    class Config:
        schema_extra = {
            "example": {
                "files": ["research_paper_1.pdf", "research_paper_2.pdf"]
            }
        }


class PDFUploadResponse(BaseModel):
    """Response model for PDF upload"""
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Response message")
    document_ids: List[str] = Field(..., description="List of processed document IDs")
    total_chunks: int = Field(..., description="Total number of text chunks created")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully processed 2 documents",
                "document_ids": ["doc_1", "doc_2"],
                "total_chunks": 45
            }
        }


class QueryRequest(BaseModel):
    """Request model for research query"""
    question: str = Field(
        ..., 
        min_length=10, 
        max_length=500,
        description="Research question about the uploaded papers"
    )
    max_results: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of relevant chunks to retrieve"
    )
    document_names: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific document names to search."
    )
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What are the main findings regarding climate change impacts on biodiversity?",
                "max_results": 5,
                "document_names": ["climate_study.pdf", "biodiversity_report.pdf"]
            }
        }


class SourceCitation(BaseModel):
    """Model for source citations with enhanced metadata"""
    document_name: str = Field(..., description="Name of the source document")
    chunk_id: str = Field(..., description="ID of the relevant text chunk")
    relevance_score: float = Field(..., description="Relevance score for this source")
    citation_confidence: Optional[float] = Field(
        default=None, 
        description="Confidence in citation accuracy (0.0-1.0)"
    )
    contribution_type: Optional[str] = Field(
        default="supporting",
        description="Type of contribution: direct, supporting, contradictory, indirect"
    )
    content_snippet: Optional[str] = Field(
        default=None,
        description="Relevant content snippet from the source"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "document_name": "climate_biodiversity_study.pdf",
                "chunk_id": "chunk_12",
                "relevance_score": 0.85,
                "citation_confidence": 0.92,
                "contribution_type": "direct",
                "content_snippet": "The study found that climate change significantly impacts..."
            }
        }


class QueryResponse(BaseModel):
    """Response model for research query"""
    success: bool = Field(..., description="Query processing success status")
    answer: str = Field(..., description="AI-generated answer to the question")
    sources: List[SourceCitation] = Field(..., description="List of source citations")
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the answer"
    )
    processing_time: float = Field(..., description="Query processing time in seconds")
    multi_document_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Analysis of multi-document context and relationships"
    )
    context_synthesis: Optional[str] = Field(
        default=None,
        description="How information was synthesized across documents"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "answer": "Based on the research papers, climate change significantly impacts biodiversity through habitat loss, temperature changes, and altered precipitation patterns...",
                "sources": [
                    {
                        "document_name": "climate_biodiversity_study.pdf",
                        "chunk_id": "chunk_12",
                        "relevance_score": 0.85
                    }
                ],
                "confidence_score": 0.82,
                "processing_time": 2.34
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Success status")
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid file format",
                "details": {"supported_formats": ["pdf"]}
            }
        }


class QueryLog(BaseModel):
    """Model for logging queries to MongoDB"""
    timestamp: datetime = Field(default_factory=get_israel_time)
    query: str = Field(..., description="User's question")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceCitation] = Field(..., description="Source citations")
    processing_time: float = Field(..., description="Processing time in seconds")
    retrieved_chunks: List[Dict[str, Any]] = Field(..., description="Retrieved document chunks")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2025-07-30T15:30:00+03:00",
                "query": "What are the main findings?",
                "answer": "The main findings include...",
                "sources": [],
                "processing_time": 2.5,
                "retrieved_chunks": [],
                "metadata": {"model": "research-assistant"}
            }
        }
