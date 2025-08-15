"""
Document ingestion pipeline for legal documents.
Handles parsing, chunking, normalization, and indexing.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import hashlib
from datetime import datetime

from .schema import LegalDocument, ProcessedChunk, ChunkMetadata, LEGAL_TERM_SYNONYMS, INFRA_FINANCE_TERMS
from .parsers.xml_parser import XMLLegalParser
from .parsers.pdf_parser import PDFLegalParser


class DocumentIngestionPipeline:
    """Pipeline for ingesting and processing legal documents"""
    
    def __init__(self, output_dir: str = "./processed_documents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.xml_parser = XMLLegalParser()
        self.pdf_parser = PDFLegalParser()
        
        self.max_chunk_size = 1000  # characters
        self.chunk_overlap = 100    # characters
    
    def ingest_document(self, file_path: str) -> LegalDocument:
        """Ingest a single document (PDF or XML)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        if file_path.suffix.lower() == '.xml':
            document = self.xml_parser.parse_document(str(file_path))
        elif file_path.suffix.lower() == '.pdf':
            document = self.pdf_parser.parse_document(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        document = self._normalize_document(document)
        
        self._save_document(document)
        
        return document
    
    def _normalize_document(self, document: LegalDocument) -> LegalDocument:
        """Normalize legal terminology and add semantic tags"""
        for section in document.sections:
            section.tags = self._normalize_tags(section.tags, section.text)
        
        for section in document.sections:
            section.text = self._normalize_terminology(section.text)
        
        return document
    
    def _normalize_tags(self, existing_tags: List[str], content: str) -> List[str]:
        """Normalize and enhance semantic tags"""
        tags = set(existing_tags)
        content_lower = content.lower()
        
        for canonical_term, synonyms in LEGAL_TERM_SYNONYMS.items():
            for synonym in synonyms:
                if synonym.lower() in content_lower:
                    tags.add(canonical_term)
                    break
        
        for term, description in INFRA_FINANCE_TERMS.items():
            if term.lower() in content_lower or description.lower() in content_lower:
                tags.add(term.lower().replace(' ', '_'))
        
        return list(tags)
    
    def _normalize_terminology(self, text: str) -> str:
        """Normalize legal terminology in text"""
        normalized_text = text
        
        for canonical_term, synonyms in LEGAL_TERM_SYNONYMS.items():
            for synonym in synonyms:
                import re
                pattern = re.compile(re.escape(synonym), re.IGNORECASE)
                normalized_text = pattern.sub(canonical_term.replace('_', ' ').title(), normalized_text)
        
        return normalized_text
    
    def chunk_document(self, document: LegalDocument) -> List[ProcessedChunk]:
        """Create chunks from document for retrieval system"""
        chunks = []
        
        for section in document.sections:
            section_chunks = self._chunk_section(document, section)
            chunks.extend(section_chunks)
        
        for definition in document.definitions:
            def_chunk = self._chunk_definition(document, definition)
            chunks.append(def_chunk)
        
        return chunks
    
    def _chunk_section(self, document: LegalDocument, section) -> List[ProcessedChunk]:
        """Chunk a section into smaller pieces"""
        chunks = []
        text = section.text
        
        if len(text) <= self.max_chunk_size:
            chunk = ProcessedChunk(
                metadata=ChunkMetadata(
                    chunk_id=f"{document.metadata.document_id}_{section.id}_0",
                    document_id=document.metadata.document_id,
                    section_id=section.id,
                    chunk_index=0,
                    chunk_type="clause",
                    tags=section.tags,
                    source_citation=f"{document.metadata.title}, Section {section.id}"
                ),
                content=text
            )
            chunks.append(chunk)
        else:
            chunk_texts = self._split_text_with_overlap(text)
            for i, chunk_text in enumerate(chunk_texts):
                chunk = ProcessedChunk(
                    metadata=ChunkMetadata(
                        chunk_id=f"{document.metadata.document_id}_{section.id}_{i}",
                        document_id=document.metadata.document_id,
                        section_id=section.id,
                        chunk_index=i,
                        chunk_type="clause",
                        tags=section.tags,
                        source_citation=f"{document.metadata.title}, Section {section.id}"
                    ),
                    content=chunk_text
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_definition(self, document: LegalDocument, definition) -> ProcessedChunk:
        """Create chunk for a definition"""
        content = f'"{definition.term}" means {definition.definition}'
        
        chunk = ProcessedChunk(
            metadata=ChunkMetadata(
                chunk_id=f"{document.metadata.document_id}_def_{self._hash_string(definition.term)}",
                document_id=document.metadata.document_id,
                section_id=definition.section_id or "definitions",
                chunk_index=0,
                chunk_type="definition",
                tags=["definition", definition.term.lower().replace(' ', '_')],
                source_citation=f"{document.metadata.title}, Definition: {definition.term}"
            ),
            content=content
        )
        
        return chunk
    
    def _split_text_with_overlap(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            if end < len(text):
                sentence_end = text.rfind('.', start + self.max_chunk_size - 100, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _hash_string(self, text: str) -> str:
        """Create hash for string"""
        return hashlib.md5(text.encode()).hexdigest()[:8]
    
    def _save_document(self, document: LegalDocument):
        """Save processed document to disk"""
        output_file = self.output_dir / f"{document.metadata.document_id}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(document.dict(), f, indent=2, ensure_ascii=False, default=str)
    
    def batch_ingest(self, file_paths: List[str]) -> List[LegalDocument]:
        """Ingest multiple documents"""
        documents = []
        
        for file_path in file_paths:
            try:
                document = self.ingest_document(file_path)
                documents.append(document)
                print(f"Successfully ingested: {file_path}")
            except Exception as e:
                print(f"Error ingesting {file_path}: {e}")
        
        return documents
    
    def get_corpus_stats(self, documents: List[LegalDocument]) -> Dict[str, Any]:
        """Get statistics about the ingested corpus"""
        stats = {
            "total_documents": len(documents),
            "total_sections": sum(len(doc.sections) for doc in documents),
            "total_definitions": sum(len(doc.definitions) for doc in documents),
            "industries": {},
            "jurisdictions": {},
            "clause_types": {},
            "document_types": {}
        }
        
        for doc in documents:
            industry = doc.metadata.industry
            stats["industries"][industry] = stats["industries"].get(industry, 0) + 1
            
            jurisdiction = doc.metadata.jurisdiction
            stats["jurisdictions"][jurisdiction] = stats["jurisdictions"].get(jurisdiction, 0) + 1
            
            doc_type = doc.metadata.document_type
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
            
            for section in doc.sections:
                clause_type = section.clause_type.value
                stats["clause_types"][clause_type] = stats["clause_types"].get(clause_type, 0) + 1
        
        return stats


def main():
    """CLI interface for document ingestion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest legal documents")
    parser.add_argument("files", nargs="+", help="Document files to ingest")
    parser.add_argument("--output-dir", default="./processed_documents", help="Output directory")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")
    
    args = parser.parse_args()
    
    pipeline = DocumentIngestionPipeline(output_dir=args.output_dir)
    
    documents = pipeline.batch_ingest(args.files)
    
    if args.stats:
        stats = pipeline.get_corpus_stats(documents)
        print("\nCorpus Statistics:")
        print(json.dumps(stats, indent=2))
    
    print(f"\nIngested {len(documents)} documents successfully.")


if __name__ == "__main__":
    main()
