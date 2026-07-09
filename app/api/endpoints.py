"""
API route definitions for the RAG Academic Papers AI Agent
"""
import os
from flask import Blueprint, request, jsonify, Response
from pydantic import ValidationError
import structlog

from app.models import QueryRequest, QueryResponse, PDFUploadResponse
from app.utils.pdf_processor import PDFProcessor
from app.database.chroma_client import ChromaClient
from app.database.mongo_client import MongoClient
from app.rag.chain import RAGChain
from app.config import get_israel_time

logger = structlog.get_logger()

# Create blueprint for API routes
api_bp = Blueprint('api', __name__)

# Initialize components (will be passed from main app)
pdf_processor = None
chroma_client = None
mongo_client = None
rag_chain = None
rag_logger = None

def init_components(pdf_proc, chroma_cl, mongo_cl, rag_ch, rag_log):
    """Initialize components for the API routes"""
    global pdf_processor, chroma_client, mongo_client, rag_chain, rag_logger
    pdf_processor = pdf_proc
    chroma_client = chroma_cl
    mongo_client = mongo_cl
    rag_chain = rag_ch
    rag_logger = rag_log

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if mongo_client:
        mongo_client.log_application_activity(
            endpoint="/health",
            method="GET",
            status_code=200,
            message="Health check performed"
        )
    
    return jsonify({
        "status": "healthy",
        "service": "rag-academic-agent",
        "version": "1.0.0"
    })

@api_bp.route('/', methods=['GET'])
@api_bp.route('/interface', methods=['GET'])
def query_interface():
    """Serve the query interface HTML page"""
    try:
        # Read the HTML file
        html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'query_interface.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return Response(html_content, mimetype='text/html')
    except Exception as e:
        logger.error("Error serving query interface", error=str(e))
        return jsonify({"error": "Interface not available"}), 500

@api_bp.route('/papers', methods=['POST'])
def upload_papers():
    """
    Upload and process PDF research papers
    ---
    tags:
      - Papers
    consumes:
      - multipart/form-data
    parameters:
      - name: files
        in: formData
        type: file
        required: true
        description: PDF files to upload
    responses:
      200:
        description: Successfully uploaded and processed papers
        schema:
          $ref: '#/definitions/PDFUploadResponse'
      400:
        description: Bad request - invalid files
      500:
        description: Internal server error
    """
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400
        
        document_ids = []
        processed_count = 0
        response_data_list = []
        
        # Create temporary uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        for file in files:
            if file and file.filename.lower().endswith('.pdf'):
                logger.info("Starting PDF processing", filename=file.filename)
                
                # Save file temporarily
                temp_path = os.path.join("uploads", file.filename)
                file.save(temp_path)
                logger.info("File saved temporarily", filename=file.filename)
                
                # Get file size
                file_size = os.path.getsize(temp_path)
                
                # Process PDF
                logger.info("Processing PDF content", filename=file.filename)
                result = pdf_processor.process_pdf(temp_path)
                
                if result["success"]:
                    logger.info("PDF processed, storing in vector database", 
                               filename=file.filename, 
                               chunks_count=result["total_chunks"])
                    
                    # Store in vector database
                    doc_id = chroma_client.add_documents(
                        chunks=result["chunks"],
                        document_name=file.filename
                    )
                    document_ids.append(doc_id)
                    processed_count += 1
                    
                    # Extract text content from chunks for preview
                    text_content = ""
                    for chunk in result["chunks"]:
                        if isinstance(chunk, dict) and "text" in chunk:
                            text_content += chunk["text"] + " "
                        elif isinstance(chunk, str):
                            text_content += chunk + " "
                        else:
                            text_content += str(chunk) + " "
                    
                    # Create response data for this file
                    response_data = {
                        "record_id": doc_id,
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": get_israel_time().isoformat(),
                        "content_length": len(text_content),
                        "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
                    }
                    response_data_list.append(response_data)
                    
                    # Log the upload using RAG logger
                    if rag_logger:
                        rag_logger.log_pdf_upload(
                            filename=file.filename,
                            document_id=doc_id,
                            chunks_count=result["total_chunks"],
                            file_size=file_size,
                            processing_time=result.get("processing_time", 0.0)
                        )
                    
                    logger.info("Successfully processed PDF", 
                              filename=file.filename, 
                              chunks_count=result["total_chunks"],
                              document_id=doc_id)
                    
                    # Clean up temp file
                    os.remove(temp_path)
            else:
                # File is not a PDF - log the rejected file
                logger.warning("Rejected non-PDF file", filename=file.filename if file else "unnamed")

        # Check if any files were actually processed
        if processed_count == 0:
            return jsonify({
                "error": "You can only upload PDF files"
            }), 400

        # Log successful upload
        if mongo_client:
            mongo_client.log_application_activity(
                endpoint="/papers",
                method="POST",
                status_code=200,
                message=f"Successfully uploaded {len(response_data_list)} PDF(s)",
                additional_data={
                    "files_count": len(response_data_list),
                    "filenames": [data["filename"] for data in response_data_list]
                }
            )

        return jsonify({
            "success": True,
            "message": "PDF file uploaded and processed successfully.",
            "data": response_data_list[0] if len(response_data_list) == 1 else response_data_list
        })

    except Exception as e:
        logger.error("Error uploading papers", error=str(e), exc_info=True)
        if rag_logger:
            rag_logger.log_system_error("pdf_upload", str(e), "error")
        return jsonify({"error": "Failed to process papers"}), 500



@api_bp.route('/query', methods=['POST'])
def query_papers():
    """
    Query the uploaded papers with natural language questions
    ---
    tags:
      - Query
    consumes:
      - application/json
      - application/x-www-form-urlencoded
    parameters:
      - name: question
        in: formData
        type: string
        required: true
        description: "Your research question about the uploaded papers. MUST include PDF file specification using format: 'Your question, search from filename.pdf'"
        example: "When did the worldwide depression begin, search from economics_paper.pdf"
      - name: document_names
        in: formData
        type: array
        items:
          type: string
        required: false
        description: "Alternative: Specific document names to search (if not specified in question text)"
      - name: body
        in: body
        required: false
        schema:
          $ref: '#/definitions/QueryRequest'
    responses:
      200:
        description: Successfully generated answer
        schema:
          $ref: '#/definitions/QueryResponse'
      400:
        description: Bad request - invalid query or missing PDF file specification
      500:
        description: Internal server error
    """
    # Handle POST request - process the query
    try:
        # Check if it's form data (simple text input) or JSON
        if request.content_type and 'application/json' in request.content_type:
            # Handle JSON request (existing functionality)
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            query_request = QueryRequest(**data)

            # REQUIRE PDF file specification for JSON requests too
            if not query_request.document_names or len(query_request.document_names) == 0:
                return jsonify({
                    "error": "Must enter which PDF file to search. Please specify document_names in your JSON request or include PDF file in your question text."
                }), 400
        else:
            # Handle form data (simple text input)
            question = request.form.get('question') or request.args.get('question')
            if not question:
                return jsonify({"error": "Question parameter is required"}), 400

            # Parse document names from the question text
            document_names = None
            original_question = question

            # Look for patterns like "search from filename.pdf" or "search in filename.pdf"
            import re
            search_patterns = [
                r',\s*search\s+from\s+([^,\n]+\.pdf)(?:\s|$)',
                r',\s*search\s+in\s+([^,\n]+\.pdf)(?:\s|$)',
                r'\s+search\s+from\s+([^,\n]+\.pdf)(?:\s|$)',
                r'\s+search\s+in\s+([^,\n]+\.pdf)(?:\s|$)'
            ]

            for pattern in search_patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    document_names = [match.group(1).strip()]
                    # Remove the search instruction from the question
                    question = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
                    break

            # Also check for fallback parameter (for compatibility)
            if not document_names:
                document_names = request.form.getlist('document_names') if request.form.getlist('document_names') else None

            # REQUIRE PDF file specification - return error if not found
            if not document_names:
                return jsonify({
                    "error": "Must enter which PDF file to search. Please specify a PDF file in your query using format: 'Your question, search from filename.pdf'"
                }), 400
            
            # Create QueryRequest object
            query_request = QueryRequest(
                question=question,
                max_results=5,  # Fixed value
                document_names=document_names
            )
        
        # Resolve document names to IDs if needed
        resolved_document_ids = None
        if query_request.document_names:
            # Get all documents and filter by names
            all_documents = chroma_client.list_documents()
            resolved_document_ids = [
                doc['document_id'] for doc in all_documents 
                if doc['document_name'] in query_request.document_names
            ]
            if not resolved_document_ids:
                return jsonify({
                    "error": f"No documents found with names: {', '.join(query_request.document_names)}"
                }), 400
        
        logger.info("Processing query", 
                   question=query_request.question,
                   document_names=query_request.document_names,
                   resolved_to_ids=resolved_document_ids)
        
        # Log user activity
        if rag_logger:
            rag_logger.log_user_activity("query", {
                "question": query_request.question,
                "document_names": query_request.document_names
            })
        
        # Generate response using RAG chain
        response = rag_chain.query(
            question=query_request.question,
            top_k=query_request.max_results,
            include_confidence=True,  # Default to True for now
            document_ids=resolved_document_ids
        )
        
        # Get comprehensive data for logging
        retrieved_chunks = rag_chain.get_last_retrieved_chunks()
        processing_metadata = rag_chain.get_last_processing_metadata()
        
        # Log performance metrics using RAG logger
        if rag_logger:
            rag_logger.log_performance_metrics("query", {
                "processing_time": response.processing_time,
                "chunks_retrieved": len(retrieved_chunks) if retrieved_chunks else 0,
                "sources_count": len(response.sources),
                "confidence_score": response.confidence_score
            })
        
        # Log the comprehensive interaction
        mongo_client.log_query(
            question=query_request.question,
            answer=response.answer,
            sources=[source.model_dump() for source in response.sources],
            retrieved_chunks=retrieved_chunks,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            metadata=processing_metadata
        )
        
        logger.info("Successfully processed query", 
                   question=query_request.question,
                   sources_count=len(response.sources))
        
        # Log application activity
        if mongo_client:
            mongo_client.log_application_activity(
                endpoint="/query",
                method="POST",
                status_code=200,
                message="Query processed successfully",
                additional_data={
                    "question": query_request.question[:100] + "..." if len(query_request.question) > 100 else query_request.question,
                    "sources_count": len(response.sources),
                    "processing_time": response.processing_time,
                    "confidence_score": response.confidence_score,
                    "document_names": query_request.document_names
                }
            )
        
        # Get available documents to include in response
        try:
            available_documents = chroma_client.list_documents()
        except Exception as e:
            logger.warning("Failed to get available documents for response", error=str(e))
            available_documents = []
        
        # Create enhanced response with available documents
        query_response = response.model_dump()
        query_response.update({
            "available_documents": available_documents,
            "total_documents": len(available_documents),
            "filtered_documents": resolved_document_ids if resolved_document_ids else "all",
            "query_scope": f"Searched {len(resolved_document_ids) if resolved_document_ids else len(available_documents)} document(s)"
        })
        
        return jsonify(query_response)
        
    except ValidationError as e:
        logger.warning("Validation error", error=str(e))
        if rag_logger:
            rag_logger.log_system_error("query_validation", str(e), "warning")
        return jsonify({"error": "Invalid request format", "details": e.errors()}), 400
    except Exception as e:
        logger.error("Error processing query", error=str(e), exc_info=True)
        if rag_logger:
            rag_logger.log_system_error("query_processing", str(e), "error")
        return jsonify({"error": "Failed to process query"}), 500

@api_bp.route('/analytics/logs', methods=['GET'])
def get_analytics_logs():
    """
    Get recent system logs and analytics
    ---
    tags:
      - Analytics
    parameters:
      - name: limit
        in: query
        type: integer
        default: 50
        description: Number of recent logs to retrieve
    responses:
      200:
        description: Recent logs retrieved successfully
      500:
        description: Internal server error
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        if rag_logger:
            logs = rag_logger.get_recent_logs(limit)
            return jsonify({"logs": logs, "count": len(logs)})
        else:
            return jsonify({"error": "Logging not available"}), 503
    except Exception as e:
        logger.error("Error retrieving analytics", error=str(e))
        return jsonify({"error": "Failed to retrieve analytics"}), 500

@api_bp.route('/logs/application', methods=['GET'])
def get_application_logs():
    """
    Retrieve application logs
    ---
    tags:
      - Logs
    parameters:
      - name: fields
        in: query
        type: string
        required: false
        description: An optional fields mask
    responses:
      200:
        description: Application logs retrieved successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
            logs:
              type: array
              items:
                type: object
                properties:
                  _id:
                    type: string
                  timestamp:
                    type: string
                  endpoint:
                    type: string
                  method:
                    type: string
                  status_code:
                    type: integer
                  message:
                    type: string
                  additional_data:
                    type: object
            count:
              type: integer
            total_count:
              type: integer
      500:
        description: Internal server error
    """
    try:
        # Get query parameters
        fields = request.args.get('fields')
        limit = 100  # Fixed limit
        skip = 0     # Fixed skip
        endpoint_filter = None  # No filtering
        
        # Get logs from MongoDB
        if mongo_client:
            result = mongo_client.get_application_logs(
                limit=limit,
                skip=skip,
                endpoint_filter=endpoint_filter
            )
            
            return jsonify({
                "success": True,
                "message": f"Retrieved {result['count']} log records",
                "logs": result["logs"],
                "count": result["count"],
                "total_count": result["total_count"],
                "pagination": {
                    "limit": limit,
                    "skip": skip,
                    "has_more": skip + result["count"] < result["total_count"]
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": "Database not available"
            }), 503
            
    except Exception as e:
        logger.error("Error retrieving application logs", error=str(e))
        return jsonify({
            "success": False,
            "error": "Failed to retrieve application logs"
        }), 500

@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500
