#!/bin/bash

# InfraRAG Environment Setup Script
# This script automates the environment setup process

set -e  # Exit on any error

echo "🚀 Setting up InfraRAG development environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install conda first."
    exit 1
fi

echo "✅ Conda found"

# Create conda environment
echo "📦 Creating conda environment from environment-cpu.yml..."
conda env create -f environment-cpu.yml

echo "🔧 Activating environment..."
source ~/.bashrc
conda activate infra-rag

echo "📥 Installing additional dependencies via pip..."
pip install qdrant-client opensearch-py sentence-transformers spacy pymupdf
pip install pdfplumber unstructured python-docx rank-bm25 rapidfuzz

echo "🧪 Testing package imports..."
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

echo "🐳 Checking Docker availability..."
if command -v docker &> /dev/null; then
    echo "✅ Docker found. You can start services with:"
    echo "   docker-compose -f docker-compose.rag.yml up -d"
else
    echo "⚠️  Docker not found. Install Docker to run Qdrant and OpenSearch services."
fi

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment in future sessions:"
echo "   conda activate infra-rag"
echo ""
echo "To start supporting services:"
echo "   docker-compose -f docker-compose.rag.yml up -d"
echo ""
echo "See ENVIRONMENT_SETUP.md for detailed documentation."