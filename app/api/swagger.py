"""
Swagger/OpenAPI documentation setup
"""
from flask import Flask
from flask_swagger_ui import get_swaggerui_blueprint

# Swagger UI configuration
SWAGGER_URL = '/api/docs'
API_URL = '/api/swagger.json'

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "RAG Academic Papers AI Agent",
        'dom_id': '#swagger-ui',
        'layout': 'StandaloneLayout'
    }
)

def setup_swagger(app: Flask):
    """
    Setup Swagger documentation for the Flask app
    
    Args:
        app: Flask application instance
    """
    
    # Register Swagger UI blueprint
    app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
    
    @app.route('/api/swagger.json')
    def swagger_json():
        """Return the Swagger JSON specification"""
        swagger_spec = {
            "swagger": "2.0",
            "info": {
                "title": "RAG Academic Papers AI Agent",
                "description": "API for uploading PDF research papers and querying them using RAG (Retrieval-Augmented Generation)",
                "version": "1.0.0",
                "contact": {
                    "name": "RAG API Support",
                    "email": "support@example.com"
                }
            },
            "host": "localhost:5001",
            "basePath": "/",
            "schemes": ["http", "https"],
            "consumes": ["application/json", "multipart/form-data"],
            "produces": ["application/json"],
            "paths": {
                "/health": {
                    "get": {
                        "tags": ["Health"],
                        "summary": "Health check endpoint",
                        "description": "Check if the service is running",
                        "responses": {
                            "200": {
                                "description": "Service is healthy",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "service": {"type": "string"},
                                        "version": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "/papers": {
                    "post": {
                        "tags": ["Papers"],
                        "summary": "Upload PDF research papers",
                        "description": "Upload one or more PDF files to be processed and indexed",
                        "consumes": ["multipart/form-data"],
                        "parameters": [
                            {
                                "name": "files",
                                "in": "formData",
                                "type": "file",
                                "required": True,
                                "description": "PDF files to upload (multiple files supported)"
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Successfully uploaded and processed papers",
                                "schema": {"$ref": "#/definitions/UploadResponse"}
                            },
                            "400": {
                                "description": "Bad request - invalid files",
                                "schema": {"$ref": "#/definitions/ErrorResponse"}
                            },
                            "500": {
                                "description": "Internal server error",
                                "schema": {"$ref": "#/definitions/ErrorResponse"}
                            }
                        }
                    }
                },
                "/query": {
                    "post": {
                        "tags": ["Query"],
                        "summary": "Query uploaded papers",
                        "description": "Ask questions about the uploaded papers using natural language. Optionally specify which document to search by adding 'search from filename.pdf' to your question.",
                        "consumes": ["application/x-www-form-urlencoded"],
                        "parameters": [
                            {
                                "name": "question",
                                "in": "formData",
                                "type": "string",
                                "required": True,
                                "description": "Your research question. To search specific documents, add 'search from filename.pdf' to your question.",
                                "example": "When did World War 1 start, search from worldwar1.pdf"
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Successfully generated answer",
                                "schema": {"$ref": "#/definitions/QueryResponse"}
                            },
                            "400": {
                                "description": "Bad request - invalid query",
                                "schema": {"$ref": "#/definitions/ErrorResponse"}
                            },
                            "500": {
                                "description": "Internal server error",
                                "schema": {"$ref": "#/definitions/ErrorResponse"}
                            }
                        }
                    }
                },
                "/logs/application": {
                    "get": {
                        "tags": ["Logs"],
                        "summary": "Get application logs",
                        "description": "Retrieve application activity logs including uploads, queries, and system events",
                        "parameters": [
                            {
                                "name": "fields",
                                "in": "query",
                                "type": "string",
                                "required": False,
                                "description": "An optional fields mask"
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Successfully retrieved application logs",
                                "schema": {"$ref": "#/definitions/LogsResponse"}
                            },
                            "500": {
                                "description": "Internal server error",
                                "schema": {"$ref": "#/definitions/ErrorResponse"}
                            }
                        }
                    }
                }
            },
            "definitions": {
                "QueryRequest": {
                    "type": "object",
                    "required": ["question"],
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask about the uploaded papers. To search specific documents, add 'search from filename.pdf' to your question.",
                            "example": "When did World War 1 start, search from worldwar1.pdf"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Number of relevant document chunks to retrieve",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        }
                    }
                },
                "QueryResponse": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The generated answer to the question"
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of source document names used"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "Confidence score between 0.0 and 1.0",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "processing_time": {
                            "type": "number",
                            "description": "Time taken to process the query in seconds"
                        },
                        "retrieved_chunks": {
                            "type": "integer",
                            "description": "Number of document chunks retrieved"
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key findings extracted from the answer"
                        }
                    }
                },
                "UploadResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "description": "Whether the upload was successful"
                        },
                        "message": {
                            "type": "string",
                            "description": "Success or error message"
                        },
                        "document_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of generated document IDs"
                        },
                        "total_processed": {
                            "type": "integer",
                            "description": "Number of documents successfully processed"
                        }
                    }
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error message"
                        },
                        "details": {
                            "type": "object",
                            "description": "Additional error details"
                        }
                    }
                },
                "LogsResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "description": "Whether the request was successful"
                        },
                        "message": {
                            "type": "string",
                            "description": "Success message"
                        },
                        "logs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "_id": {"type": "string"},
                                    "timestamp": {"type": "string"},
                                    "endpoint": {"type": "string"},
                                    "method": {"type": "string"},
                                    "status_code": {"type": "integer"},
                                    "message": {"type": "string"},
                                    "additional_data": {"type": "object"}
                                }
                            },
                            "description": "Array of log records"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of logs returned"
                        },
                        "total_count": {
                            "type": "integer",
                            "description": "Total number of logs available"
                        },
                        "pagination": {
                            "type": "object",
                            "properties": {
                                "limit": {"type": "integer"},
                                "skip": {"type": "integer"},
                                "has_more": {"type": "boolean"}
                            }
                        }
                    }
                }
            },
            "tags": [
                {
                    "name": "Health",
                    "description": "Health check endpoints"
                },
                {
                    "name": "Papers", 
                    "description": "PDF upload and management"
                },
                {
                    "name": "Query",
                    "description": "Question answering with RAG"
                },
                {
                    "name": "Logs",
                    "description": "Application activity logs"
                }
            ]
        }
        
        return swagger_spec
