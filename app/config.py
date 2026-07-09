import os
from pydantic_settings import BaseSettings
from typing import Optional
from datetime import datetime, timezone, timedelta


# Israel timezone utility function
def get_israel_time():
    """Get current time in Israel timezone (IST/IDT)"""
    # Israel is UTC+2 (IST) in winter, UTC+3 (IDT) in summer
    # This is a simplified version - for production, use pytz
    israel_offset = timedelta(hours=3)  # Using IDT (summer time)
    israel_tz = timezone(israel_offset)
    return datetime.now(israel_tz)


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Flask settings
    flask_host: str = "0.0.0.0"
    flask_port: int = 5000
    flask_debug: bool = True
    
        # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "tinyllama"
    
    # ChromaDB settings
    chroma_host: str = "http://localhost:8000"
    chroma_collection: str = "research_papers"
    
    # MongoDB settings
    mongo_uri: str = "mongodb://admin:password123@localhost:27017/rag_system?authSource=admin"
    mongo_database: str = "rag_system"
    mongo_collection: str = "query_logs"
    
    # File upload settings
    upload_folder: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: set = {"pdf"}
    
    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Timezone settings
    timezone_name: str = "Israel"
    timezone_offset_hours: int = 3  # IDT (Israel Daylight Time) UTC+3
    max_retrieval_docs: int = 5
    
    # API settings
    api_title: str = "Academic Research RAG API"
    api_version: str = "1.0.0"
    api_description: str = "AI-powered question answering system for academic research papers"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.upload_folder, exist_ok=True)
