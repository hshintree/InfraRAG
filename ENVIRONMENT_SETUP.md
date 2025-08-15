# InfraRAG Environment Setup Guide

This guide provides step-by-step instructions to set up the complete development environment for InfraRAG, a legal document RAG system for drafting purchase agreements.

## Prerequisites

- Conda installed on your system
- Git
- Docker (for supporting services)

## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd InfraRAG
```

### 2. Create Conda Environment

Create the conda environment using the provided environment file:

```bash
conda env create -f environment-cpu.yml
```

**Note**: If you have CUDA-capable GPU, you can use `environment-cuda.yml` instead for GPU acceleration.

### 3. Activate the Environment

```bash
conda activate infra-rag
```

### 4. Install Additional Dependencies

Some packages need to be installed separately via pip due to conda compatibility issues:

```bash
# Core retrieval and NLP packages
pip install qdrant-client opensearch-py sentence-transformers spacy pymupdf

# Document processing packages
pip install pdfplumber unstructured python-docx rank-bm25 rapidfuzz
```

### 5. Verify Installation

Test that all key packages are working:

```bash
python -c "
import qdrant_client
import opensearchpy
import sentence_transformers
import spacy
import fitz  # PyMuPDF
import pdfplumber
import unstructured
from docx import Document
import rank_bm25
import rapidfuzz
import langchain
import llama_index
print('✅ All key packages imported successfully!')
"
```

### 6. Set Up Supporting Services (Optional)

Start the vector database and search services using Docker:

```bash
# Start Qdrant (vector database) and OpenSearch (BM25 search)
docker-compose -f docker-compose.rag.yml up -d

# Verify services are running
docker ps
```

This will start:
- **Qdrant** on port 6333 (vector database)
- **OpenSearch** on port 9200 (BM25 search engine)
- **OpenSearch Dashboards** on port 5601 (web UI)

### 7. Environment Activation for Future Sessions

For future development sessions, simply activate the environment:

```bash
conda activate infra-rag
```

To update dependencies when new packages are added:

```bash
conda env update -f environment-cpu.yml
```

## Package Overview

The environment includes packages for:

### Core Web Framework
- **FastAPI**: Modern web framework for APIs
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Retrieval Stack
- **Qdrant Client**: Vector database client
- **OpenSearch**: BM25 search engine
- **Rank-BM25**: Local BM25 implementation

### NLP & Embeddings
- **OpenAI**: API client for embeddings and LLMs
- **Sentence Transformers**: Local embedding models
- **Transformers**: Hugging Face transformers
- **Spacy**: NLP processing

### Document Processing
- **PyMuPDF**: PDF processing
- **PDFPlumber**: PDF text extraction
- **Unstructured**: Document parsing
- **python-docx**: Word document generation
- **lxml**: XML processing

### Orchestration
- **LlamaIndex**: RAG framework
- **LangChain**: LLM application framework

### Development Tools
- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **MyPy**: Type checking
- **Pytest**: Testing framework

## Troubleshooting

### Common Issues

1. **Conda environment creation fails**: 
   - Try using `environment-cpu.yml` instead of `environment-cuda.yml`
   - Update conda: `conda update conda`

2. **Import errors after installation**:
   - Ensure you've activated the environment: `conda activate infra-rag`
   - Install missing packages individually with pip

3. **Docker services won't start**:
   - Check if ports 6333, 9200, 5601 are available
   - Ensure Docker daemon is running

### Package Compatibility Notes

- Some packages (like `opentelemetry-instrumentation-fastapi`) had version constraints with Python 3.11
- We use compatible versions in the environment files to avoid conflicts
- Additional packages are installed via pip after conda environment creation

## Architecture Components

This environment supports the full InfraRAG pipeline:

1. **Data Ingestion**: PDF/XML parsing with PyMuPDF, pdfplumber, lxml
2. **Indexing**: Hybrid search with Qdrant (vectors) + OpenSearch (BM25)
3. **Retrieval**: Multi-stage retrieval with reranking
4. **Planning**: Clause selection and document structure planning
5. **Drafting**: RAG-based content generation
6. **Validation**: Rule-based checks and schema validation
7. **Assembly**: DOCX generation with citations
8. **Review**: Web UI for human-in-the-loop editing

## Next Steps

After environment setup:

1. Configure API keys in `.env` file (OpenAI, etc.)
2. Start the supporting services with Docker
3. Run the ingestion pipeline to process your legal documents
4. Test the retrieval and generation components
5. Launch the web interface for document drafting

For detailed usage instructions, see the main README.md file.