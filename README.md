# RAG Academic Papers AI Agent

An intelligent document analysis system that enables users to upload PDF research papers and query them using natural language questions. Built with RAG (Retrieval-Augmented Generation) architecture for accurate, source-cited responses.

## Features

- **PDF Upload & Processing**: Upload multiple academic papers (PDFs) for analysis
- **Intelligent Q&A**: Ask natural language questions about your papers
- **Source Citations**: All answers include proper source citations
- **Vector Search**: Advanced semantic search using ChromaDB
- **Custom LLM**: Specialized Ollama model for academic research
- **Comprehensive Logging**: MongoDB-based query and interaction logging
- **REST API**: Well-documented API with Swagger/OpenAPI
- **Docker Deployment**: Fully containerized with Docker Compose

## Architecture

The system consists of several microservices:

- **Flask API**: Main application server with REST endpoints
- **Ollama**: Custom LLM for generating responses
- **ChromaDB**: Vector database for document embeddings
- **MongoDB**: Query logging and metadata storage
- **Mongo Express**: Database administration GUI

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- At least 8GB RAM (for Ollama model)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd rag-academic-papers
```

### 2. Start Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Wait for services to initialize (especially Ollama model download)
docker-compose logs -f ollama
```

### 3. Verify Installation

```bash
# Check all services are running
docker-compose ps

# Test the API
curl http://localhost:5001/health
```

### 4. Access the Application

- **API Documentation**: http://localhost:5001/api/docs
- **Database GUI**: http://localhost:8081 (admin/admin)
- **API Base URL**: http://localhost:5001

## Usage

### Upload Papers

```bash
curl -X POST -F "files=@paper1.pdf" -F "files=@paper2.pdf" \
  http://localhost:5001/papers
```

### Query Papers

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings about machine learning?"}' \
  http://localhost:5001/query
```

### List Documents

```bash
curl http://localhost:5001/documents
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/papers` | Upload PDF papers |
| POST | `/query` | Ask questions |
| GET | `/documents` | List uploaded documents |
| DELETE | `/documents/{id}` | Delete a document |

### Request/Response Examples

#### Upload Papers
```json
// Request: multipart/form-data with files
// Response:
{
  "success": true,
  "message": "Successfully processed 2 documents",
  "document_ids": ["uuid1", "uuid2"],
  "total_processed": 2
}
```

#### Query Papers
```json
// Request:
{
  "question": "What are the main research methodologies used?",
  "top_k": 5,
  "include_confidence": true
}

// Response:
{
  "answer": "The papers describe several methodologies...",
  "sources": ["paper1.pdf", "paper2.pdf"],
  "confidence_score": 0.85,
  "processing_time": 2.3,
  "retrieved_chunks": 5,
  "key_findings": ["Finding 1", "Finding 2"]
}
```

## Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start services (without Flask app)
docker-compose up -d ollama chromadb mongodb mongo-express

# Run Flask app locally
export FLASK_ENV=development
export OLLAMA_HOST=http://localhost:11434
export MONGO_URI=mongodb://admin:password123@localhost:27017/rag_system?authSource=admin
python app/main.py
```

### Project Structure

```
├── app/                    # Main Flask application
│   ├── main.py            # Flask app entry point
│   ├── config.py          # Configuration settings
│   ├── models.py          # Pydantic data models
│   ├── __init__.py       
│   ├── utils/             # Utility modules
│   │   ├── pdf_processor.py
│   │   ├── embeddings.py
│   │   └── logging.py
│   │   └── __init__.py
│   ├── database/          # Database clients
│   │   ├── chroma_client.py
│   │   └── mongo_client.py
│   │   └── __init__.py
│   ├── rag/               # RAG implementation
│   │   ├── chain.py
│   │   ├── __init__.py
│   │   ├── prompt_templates.py
│   │   └── output_parser.py
│   └── api/               # API documentation
│       └── swagger.py
│       └── __init__.py
│       └── endpoints.py
├── modelfile/             # Ollama model configuration
│   └── Modelfile
├── docker-compose.yml     # Docker services orchestration
├── Dockerfile            # Flask app container
└── requirements.txt      # Python dependencies
└── README.md      
```

### Configuration

Environment variables can be set in `app/config.py`:

- `OLLAMA_HOST`: Ollama server URL
- `OLLAMA_MODEL`: Model name to use
- `CHROMA_DB_PATH`: ChromaDB storage path
- `MONGODB_URI`: MongoDB connection string
- `EMBEDDING_MODEL`: Sentence transformer model

## Monitoring & Logging

### View Logs

```bash
# Application logs
docker-compose logs -f flask-app

# Ollama logs
docker-compose logs -f ollama

# Database logs
docker-compose logs -f mongodb
```

### Database Access

Access MongoDB through Mongo Express at http://localhost:8081:
- Username: `admin`
- Password: `admin`

Navigate to the `rag_system` database to view:
- `queries`: Query logs and responses
- `documents`: Document upload logs

## Testing

### Manual Testing

1. Upload a test PDF:
```bash
curl -X POST -F "files=@test_paper.pdf" http://localhost:5001/papers
```

2. Query the paper:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the main hypothesis?"}' \
  http://localhost:5001/query
```

3. Check the response includes proper citations and sources.

### Health Checks

```bash
# API health
curl http://localhost:5001/health

# ChromaDB health
curl http://localhost:8000/api/v1/heartbeat

# MongoDB health
docker-compose exec mongodb mongosh --eval "db.runCommand('ping')"
```

## Troubleshooting

### Common Issues

1. **Ollama model not loading**:
   ```bash
   docker-compose logs ollama
   # Wait for model download to complete
   ```

2. **Memory issues**:
   - Ensure Docker has at least 8GB RAM allocated
   - Consider using a smaller model in `modelfile/Modelfile`

3. **ChromaDB connection errors**:
   ```bash
   docker-compose restart chromadb
   ```

4. **Flask app crashes**:
   ```bash
   docker-compose logs flask-app
   # Check for dependency issues
   ```

### Reset Everything

```bash
# Stop and remove all containers, volumes, and networks
docker-compose down -v
docker-compose up -d
```

## Performance Tuning

### For Production

1. **Increase Ollama memory**:
   ```yaml
   # In docker-compose.yml under ollama service
   deploy:
     resources:
       limits:
         memory: 16G
   ```

2. **Use production WSGI server**:
   ```dockerfile
   # In Dockerfile
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
   ```

3. **Optimize ChromaDB**:
   - Use persistent volumes
   - Configure batch sizes for large document sets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Check the troubleshooting section
- Review Docker logs for error details
- Ensure all prerequisites are met
- Verify service health endpoints

## Future Enhancements

- [ ] Multi-user support with authentication
- [ ] Advanced document preprocessing (OCR, table extraction)
- [ ] Support for additional file formats (Word, PowerPoint)
- [ ] Real-time query suggestions
- [ ] Export functionality for Q&A sessions
- [ ] Integration with academic databases
- [ ] Advanced analytics dashboard
