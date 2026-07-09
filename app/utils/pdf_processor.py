import os
import hashlib
import re
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import pytz
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..config import settings


class PDFProcessor:
    """Handles PDF text extraction and chunking with multilingual support"""
    
    def __init__(self):
        # Enhanced text splitter with Hebrew-aware separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=[
                "\n\n", "\n", 
                ". ", "! ", "? ",  # English sentence endings
                "׃", ".", "!", "?",  # Hebrew and mixed punctuation
                "; ", ": ", " - ", " – ", " — ",  # Other separators
                " ", ""
            ]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing of Hebrew and English"""
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving Hebrew text structure
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        # Remove common PDF artifacts while preserving Hebrew characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Control characters
        
        # Normalize Hebrew text direction markers (if any)
        text = re.sub(r'[\u200E\u200F\u202A-\u202E]', '', text)  # LTR/RTL marks
        
        # Clean up spacing around Hebrew punctuation
        text = re.sub(r'\s+([׃.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([׃.,;:!?])\s+', r'\1 ', text)  # Normalize space after punctuation
        
        return text.strip()
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect if text contains Hebrew, English, or mixed content"""
        if not text:
            return {"hebrew": False, "english": False, "mixed": False}
        
        # Hebrew character range: \u0590-\u05FF
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
        # English character range: \u0041-\u007A
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = hebrew_chars + english_chars
        if total_chars == 0:
            return {"hebrew": False, "english": False, "mixed": False}
        
        hebrew_ratio = hebrew_chars / total_chars
        english_ratio = english_chars / total_chars
        
        return {
            "hebrew": hebrew_ratio > 0.1,
            "english": english_ratio > 0.1,
            "mixed": hebrew_ratio > 0.1 and english_ratio > 0.1,
            "hebrew_ratio": hebrew_ratio,
            "english_ratio": english_ratio
        }
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 with Hebrew support"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num}: {e}")
                        continue
                
                # Clean and normalize the extracted text
                return self.clean_text(text)
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber with Hebrew support"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text with better handling for mixed RTL/LTR content
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num}: {e}")
                        continue
            
            # Clean and normalize the extracted text
            return self.clean_text(text)
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using best available method"""
        # Try pdfplumber first (better text extraction)
        text = self.extract_text_pdfplumber(pdf_path)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text.strip():
            text = self.extract_text_pypdf2(pdf_path)
        
        return text.strip()
    
    def create_chunks(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks with Hebrew language support"""
        try:
            # Check if text contains Hebrew
            is_hebrew = self.detect_language(text) == "Hebrew"
            
            # Configure text splitter based on language
            if is_hebrew:
                # For Hebrew text, use smaller chunks and different separators
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=self.text_splitter["Hebrew"]["separators"],
                    length_function=len,
                    is_separator_regex=False,
                )
            else:
                # For English or mixed content
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=self.text_splitter["English"]["separators"],
                    length_function=len,
                )
            
            # Split text into chunks
            chunks = splitter.split_text(text)
            
            # Clean each chunk
            cleaned_chunks = []
            for chunk in chunks:
                cleaned_chunk = self.clean_text(chunk)
                if cleaned_chunk.strip():  # Only add non-empty chunks
                    cleaned_chunks.append(cleaned_chunk)
            
            return cleaned_chunks
            
        except Exception as e:
            print(f"Error creating chunks: {e}")
            # Fallback: simple splitting
            words = text.split()
            chunks = []
            for i in range(0, len(words), chunk_size // 10):  # Rough word-based chunking
                chunk = " ".join(words[i:i + chunk_size // 10])
                if chunk.strip():
                    chunks.append(self.clean_text(chunk))
            return chunks
    
    def generate_chunk_id(self, document_name: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        base_name = Path(document_name).stem
        return f"{base_name}_chunk_{chunk_index}"
    
    def generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID based on file content"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return f"doc_{file_hash[:8]}"
        except Exception:
            # Fallback to filename-based ID
            return f"doc_{Path(file_path).stem}"
    
    def validate_pdf(self, file_path: str) -> bool:
        """Validate if file is a proper PDF"""
        try:
            with open(file_path, 'rb') as file:
                # Check PDF header
                header = file.read(4)
                if header != b'%PDF':
                    return False
                
                # Try to open with PyPDF2
                file.seek(0)
                PyPDF2.PdfReader(file)
                return True
        except Exception:
            return False
    
    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process a single PDF file completely with Hebrew support"""
        try:
            # Validate PDF
            if not self.validate_pdf(file_path):
                raise ValueError("Invalid PDF file")
            
            # Extract filename
            document_name = Path(file_path).name
            
            # Extract text with enhanced error handling
            text = self.extract_text(file_path)
            if not text.strip():
                raise ValueError("No extractable text found in PDF")
            
            # Detect language for better processing
            language = self.detect_language(text)
            
            # Create chunks with language-aware processing
            chunks = self.create_chunks(text, 1000, 200)  # Using default chunk sizes
            if not chunks:
                raise ValueError("No text chunks created from PDF")
            
            # Generate document ID
            document_id = self.generate_document_id(file_path)
            
            return {
                "success": True,
                "document_id": document_id,
                "document_name": document_name,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "text_length": len(text),
                "detected_language": language,
                "has_hebrew": "Hebrew" in language or language == "Hebrew",
                "processing_timestamp": datetime.now(pytz.timezone('Asia/Jerusalem')).isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "document_name": Path(file_path).name if file_path else "unknown"
            }
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic information about a PDF file"""
        try:
            stat = os.stat(file_path)
            return {
                "filename": Path(file_path).name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime
            }
        except Exception as e:
            return {"error": str(e)}
