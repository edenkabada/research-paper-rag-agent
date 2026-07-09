"""
Main Flask application for the RAG Academic Papers AI Agent
"""
import logging
from flask import Flask
from flask_cors import CORS
import structlog

from app.config import settings
from app.utils.pdf_processor import PDFProcessor
from app.utils.logging import RAGLogger
from app.database.chroma_client import ChromaClient
from app.database.mongo_client import MongoClient
from app.rag.chain import RAGChain
from app.api.swagger import setup_swagger
from app.api.endpoints import api_bp, init_components

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Initialize components
    pdf_processor = PDFProcessor()
    chroma_client = ChromaClient()
    mongo_client = MongoClient()
    rag_chain = RAGChain(chroma_client)
    rag_logger = RAGLogger(mongo_client)
    
    # Initialize API components
    init_components(pdf_processor, chroma_client, mongo_client, rag_chain, rag_logger)
    
    # Register API blueprint
    app.register_blueprint(api_bp)
    
    # Setup Swagger documentation
    setup_swagger(app)
    
    # Log application startup
    mongo_client.log_application_activity(
        endpoint="/startup",
        method="SYSTEM",
        status_code=200,
        message="Application started",
        additional_data={"version": "1.0.0"}
    )
    
    logger.info("Flask application created successfully")
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
