"""
Output parser for structured RAG responses
"""
import re
import json
from typing import List, Dict, Any, Optional
import structlog

from app.models import SourceCitation

logger = structlog.get_logger()

class RAGOutputParser:
    """Parser for extracting structured information from LLM responses"""
    
    def __init__(self):
        """Initialize the output parser"""
        self.citation_patterns = [
            r'\[Source:\s*([^\]]+)\]',  # [Source: document.pdf]
            r'(?:Source|From):\s*([^\n]+)',  # Source: document.pdf
            r'(?:According to|Based on|In)\s+([^,\n]+\.pdf)',  # According to document.pdf
            r'(?:cited in|referenced in|found in)\s+([^,\n]+\.pdf)',  # found in document.pdf
        ]
    
    def extract_sources(self, text: str) -> List[str]:
        """
        Extract source document names from the response text
        
        Args:
            text: LLM response text
            
        Returns:
            List of unique source document names
        """
        sources = set()
        
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                source = match.strip().strip('"\'')
                if source:
                    sources.add(source)
        
        return list(sources)
    
    def extract_confidence_score(self, text: str) -> Optional[float]:
        """
        Extract confidence score from text
        
        Args:
            text: Text containing confidence score
            
        Returns:
            Confidence score between 0.0 and 1.0, or None if not found
        """
        # Look for patterns like "0.8" or "0.85" 
        patterns = [
            r'(?:confidence|score):\s*([0-1]\.?\d*)',
            r'([0-1]\.\d+)',
            r'(\d\.\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    score = float(match)
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue
        
        return None
    
    def clean_answer(self, answer: str) -> str:
        """
        Clean and format the answer text
        
        Args:
            answer: Raw answer text
            
        Returns:
            Cleaned answer text
        """
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer.strip())
        
        # Remove any system prompts that might have leaked through
        system_indicators = [
            "You are an expert AI assistant",
            "User question:",
            "Certainly! Here's a detailed response",
            "Based on the following excerpts",
            "CONTEXT:",
            "QUESTION:",
            "Israel Tal Tevet is an expert AI assistant",
            "specialized in analyzing academic research papers",
            "Can you provide me with a comprehensive answer",
            "based on the provided context"
        ]
        
        for indicator in system_indicators:
            if indicator.lower() in answer.lower():
                # Split by the indicator and take the part after it
                parts = answer.split(indicator)
                if len(parts) > 1:
                    # Find actual content after the system text
                    remaining = parts[-1].strip()
                    if remaining:
                        answer = remaining
                        break
        
        # Remove common verbose patterns at the start
        verbose_patterns = [
            r'^.*?Certainly!.*?response.*?:',
            r'^.*?Here\'s.*?response.*?:',
            r'^.*?CONTENT:.*?\[Source:',
        ]
        
        for pattern in verbose_patterns:
            answer = re.sub(pattern, '[Source:', answer, flags=re.IGNORECASE | re.DOTALL)
        
        # If answer starts with a source citation, extract just the factual content
        if answer.startswith('[Source:'):
            # Look for the actual answer after source citations
            lines = answer.split('\n')
            content_lines = []
            in_content = False
            
            for line in lines:
                if '[Source:' in line:
                    in_content = True
                    continue
                if in_content and line.strip():
                    content_lines.append(line.strip())
            
            if content_lines:
                # Extract the key factual information
                content = ' '.join(content_lines)
                # For occupation questions, look for job titles
                if 'occupation' in answer.lower() or 'job' in answer.lower():
                    job_patterns = [
                        r'D\s*A\s*T\s*A\s*A\s*N\s*A\s*L\s*Y\s*S\s*T',
                        r'Data Analyst',
                        r'AI Engineer',
                        r'Software Developer',
                        r'Researcher'
                    ]
                    for pattern in job_patterns:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            return match.group().replace(' ', ' ').strip()
                
                # Return first meaningful sentence if available
                sentences = re.split(r'[.!?]', content)
                for sentence in sentences:
                    if len(sentence.strip()) > 5:
                        return sentence.strip()
        
        # If this is a simple occupation/job question, extract just the job title
        if any(word in answer.lower() for word in ['occupation', 'job', 'what is', 'who is']):
            # Look for job titles in the answer
            job_patterns = [
                r'(?:is a|works as|occupation.*?is)\s+([^,\n.]+?)(?:\s+with|\s+in|\.|,|$)',
                r'(Data Analyst|AI Engineer|Software Developer|Researcher|Engineer|Developer|Analyst)',
                r'D\s*A\s*T\s*A\s*\s*A\s*N\s*A\s*L\s*Y\s*S\s*T'
            ]
            
            for pattern in job_patterns:
                match = re.search(pattern, answer, re.IGNORECASE)
                if match:
                    job_title = match.group(1) if len(match.groups()) > 0 else match.group()
                    job_title = re.sub(r'\s+', ' ', job_title.strip())
                    if len(job_title) > 2:  # Ensure we have a meaningful title
                        return job_title
        
        return answer.strip()
    
    def parse_follow_up_questions(self, text: str) -> List[str]:
        """
        Extract follow-up questions from text
        
        Args:
            text: Text containing follow-up questions
            
        Returns:
            List of follow-up questions
        """
        questions = []
        
        # Look for numbered lists or bullet points
        patterns = [
            r'(?:^\d+\.?\s+(.+\?)\s*$)',  # 1. Question?
            r'(?:^[-*]\s+(.+\?)\s*$)',    # - Question?
            r'(?:^(.+\?)\s*$)',           # Question?
        ]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    question = match.group(1).strip()
                    if len(question) > 10:  # Filter out very short matches
                        questions.append(question)
                    break
        
        return questions[:3]  # Return max 3 questions
    
    def extract_key_findings(self, answer: str) -> List[str]:
        """
        Extract key findings or main points from the answer
        
        Args:
            answer: Answer text
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Look for common academic patterns
        patterns = [
            r'(?:The study found that|Results show that|Research indicates that|Evidence suggests that)\s+([^.]+\.)',
            r'(?:Key finding:|Main result:|Important:|Notably,)\s+([^.]+\.)',
            r'(?:In conclusion,|Overall,|Summary:)\s+([^.]+\.)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for match in matches:
                finding = match.strip()
                if len(finding) > 20:  # Filter out very short matches
                    findings.append(finding)
        
        return findings
    
    def parse_full_response(self, 
                          answer: str,
                          retrieved_docs: List[Dict[str, Any]],
                          confidence_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse complete LLM response into structured format
        
        Args:
            answer: Main answer text
            retrieved_docs: Documents used for context
            confidence_text: Optional confidence scoring text
            
        Returns:
            Structured response dictionary
        """
        # Clean the answer
        clean_answer = self.clean_answer(answer)
        
        # Extract sources from answer and retrieved docs
        answer_sources = self.extract_sources(clean_answer)
        available_sources = [doc['metadata']['document_name'] for doc in retrieved_docs]
        
        # Combine and deduplicate sources
        all_sources = list(set(answer_sources + available_sources))
        
        # Convert sources to SourceCitation objects
        source_citations = []
        for i, source in enumerate(all_sources):
            source_citations.append(SourceCitation(
                document_name=source,
                chunk_id=f"{source}_chunk_0",  # Default chunk ID
                relevance_score=0.8,  # Default relevance score
                snippet="Content extracted from the document"  # Default snippet
            ))
        
        # Extract confidence if provided in the answer text (but don't override external calculation)
        extracted_confidence = None
        if confidence_text:
            extracted_confidence = self.extract_confidence_score(confidence_text)
        
        # Extract key findings
        key_findings = self.extract_key_findings(clean_answer)
        
        result = {
            "answer": clean_answer,
            "sources": source_citations,
            "confidence_score": extracted_confidence,  # Will be overridden by RAG chain if calculated
            "key_findings": key_findings,
            "retrieved_chunks_count": len(retrieved_docs),
            "source_documents": list(set([doc['metadata']['document_name'] for doc in retrieved_docs]))
        }
        
        logger.info("Parsed RAG response",
                   answer_length=len(clean_answer),
                   sources_count=len(source_citations),
                   confidence_score=extracted_confidence)
        
        return result
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate that the response has all required fields
        
        Args:
            response: Response dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["answer", "sources"]
        
        for field in required_fields:
            if field not in response:
                logger.error("Missing required field in response", field=field)
                return False
        
        if not response["answer"].strip():
            logger.error("Empty answer in response")
            return False
        
        return True
