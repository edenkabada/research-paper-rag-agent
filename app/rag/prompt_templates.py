"""
Prompt templates for RAG system
"""

# System prompt for academic paper Q&A
ACADEMIC_QA_SYSTEM_PROMPT = """Answer questions based on the provided document context. For simple factual questions (what, who, when, where), give only the direct answer. For complex questions, provide comprehensive responses with sources."""

# Template for formatting context from retrieved documents
CONTEXT_TEMPLATE = """Based on the following excerpts from academic papers, please answer the user's question:

CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive answer based on the context above, including specific citations to the source documents."""

# Template for formatting individual document chunks in context
DOCUMENT_CHUNK_TEMPLATE = """
[Source: {document_name}]
{content}
---
"""

# Follow-up questions prompt template
FOLLOW_UP_QUESTIONS_PROMPT = """Based on the question "{question}" and the answer provided, suggest 3 relevant follow-up questions that would help the user explore this topic further within the context of the available academic papers.

Follow-up questions:"""

# Confidence scoring prompt
CONFIDENCE_PROMPT = """Rate your confidence in the answer you just provided on a scale of 0.0 to 1.0, where:
- 1.0 = Very confident, answer is well-supported by multiple clear sources
- 0.8 = Confident, answer is supported by good sources with minor gaps
- 0.6 = Moderately confident, answer is supported but sources are limited
- 0.4 = Low confidence, answer is partially supported or sources are unclear
- 0.2 = Very low confidence, answer is speculative based on limited information
- 0.0 = No confidence, insufficient information to answer

Confidence score (number only):"""

# Citation extraction prompt
CITATION_EXTRACTION_PROMPT = """From your answer, extract all the specific document sources that were referenced. List them as a JSON array of document names.

For example: ["paper1.pdf", "research_study_2023.pdf"]

Document sources:"""

def format_context(retrieved_docs: list) -> str:
    """
    Format retrieved documents into context for the LLM
    
    Args:
        retrieved_docs: List of retrieved document chunks with metadata
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for doc in retrieved_docs:
        formatted_chunk = DOCUMENT_CHUNK_TEMPLATE.format(
            document_name=doc['metadata']['document_name'],
            content=doc['content']
        )
        context_parts.append(formatted_chunk)
    
    return "\n".join(context_parts)

def create_qa_prompt(question: str, context: str) -> str:
    """
    Create the main Q&A prompt for English responses
    
    Args:
        question: User's question
        context: Formatted context from retrieved documents
        
    Returns:
        Complete prompt for the LLM
    """
    # Detect if this is a simple factual question
    simple_question_words = ['what is', 'who is', 'when', 'where', 'how many', 'which']
    is_simple_question = any(question.lower().strip().startswith(word) for word in simple_question_words)
    
    if is_simple_question:
        # Use very simple template for direct answers
        template = f"""{context}

Question: {question}
Answer:"""
    else:
        # Use comprehensive template for complex questions
        template = f"""Context from documents:
{context}

Question: {question}
Answer:"""
    
    return template

def create_system_message() -> dict:
    """
    Create system message for the LLM
    
    Returns:
        System message dictionary
    """
    return {
        "role": "system",
        "content": ACADEMIC_QA_SYSTEM_PROMPT
    }

def create_user_message(question: str, context: str) -> dict:
    """
    Create user message with question and context, with language detection
    
    Args:
        question: User's question
        context: Formatted context
        
    Returns:
        User message dictionary
    """
    return {
        "role": "user", 
        "content": create_qa_prompt(question, context)
    }
