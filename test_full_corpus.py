"""
Test script for full corpus ingestion and analysis
"""

import sys
import os
sys.path.append('./src')

from src.ingestion import DocumentIngestionPipeline
from src.parsers.xml_parser import XMLLegalParser
from src.parsers.pdf_parser import PDFLegalParser
import json

def test_full_corpus():
    """Test ingestion pipeline on all documents in the corpus"""
    print("=== Testing Full Corpus Ingestion ===\n")
    
    documents = [
        ("129435.pdf", "129435_sanitized.xml"),
        ("153885.pdf", "153885_sanitized.xml"), 
        ("39638.pdf", "39638_sanitized.xml")
    ]
    
    pipeline = DocumentIngestionPipeline(output_dir="./corpus_output")
    processed_documents = []
    all_chunks = []
    
    for pdf_file, xml_file in documents:
        print(f"Processing document pair: {pdf_file} / {xml_file}")
        
        if os.path.exists(xml_file):
            try:
                xml_doc = pipeline.ingest_document(xml_file)
                xml_chunks = pipeline.chunk_document(xml_doc)
                
                print(f"  XML ({xml_file}):")
                print(f"    - Document ID: {xml_doc.metadata.document_id}")
                print(f"    - Title: {xml_doc.metadata.title}")
                print(f"    - Industry: {xml_doc.metadata.industry}")
                print(f"    - Jurisdiction: {xml_doc.metadata.jurisdiction}")
                print(f"    - Parties: {len(xml_doc.metadata.parties)}")
                print(f"    - Sections: {len(xml_doc.sections)}")
                print(f"    - Definitions: {len(xml_doc.definitions)}")
                print(f"    - Chunks generated: {len(xml_chunks)}")
                
                processed_documents.append(xml_doc)
                all_chunks.extend(xml_chunks)
                
            except Exception as e:
                print(f"  ✗ XML parsing failed for {xml_file}: {e}")
        
        if os.path.exists(pdf_file):
            try:
                pdf_doc = pipeline.ingest_document(pdf_file)
                pdf_chunks = pipeline.chunk_document(pdf_doc)
                
                print(f"  PDF ({pdf_file}):")
                print(f"    - Document ID: {pdf_doc.metadata.document_id}")
                print(f"    - Title: {pdf_doc.metadata.title}")
                print(f"    - Industry: {pdf_doc.metadata.industry}")
                print(f"    - Jurisdiction: {pdf_doc.metadata.jurisdiction}")
                print(f"    - Parties: {len(pdf_doc.metadata.parties)}")
                print(f"    - Sections: {len(pdf_doc.sections)}")
                print(f"    - Definitions: {len(pdf_doc.definitions)}")
                print(f"    - Chunks generated: {len(pdf_chunks)}")
                
                processed_documents.append(pdf_doc)
                all_chunks.extend(pdf_chunks)
                
            except Exception as e:
                print(f"  ✗ PDF parsing failed for {pdf_file}: {e}")
        
        print()
    
    if processed_documents:
        stats = pipeline.get_corpus_stats(processed_documents)
        
        print("=== Corpus Statistics ===")
        print(f"Total documents processed: {stats['total_documents']}")
        print(f"Total sections: {stats['total_sections']}")
        print(f"Total definitions: {stats['total_definitions']}")
        print(f"Total chunks: {len(all_chunks)}")
        
        print(f"\nIndustries: {dict(stats['industries'])}")
        print(f"Jurisdictions: {dict(stats['jurisdictions'])}")
        print(f"Document types: {dict(stats['document_types'])}")
        
        print(f"\nTop clause types:")
        clause_types = sorted(stats['clause_types'].items(), key=lambda x: x[1], reverse=True)
        for clause_type, count in clause_types[:10]:
            print(f"  {clause_type}: {count}")
        
        chunk_types = {}
        all_tags = {}
        
        for chunk in all_chunks:
            chunk_type = chunk.metadata.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            for tag in chunk.metadata.tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1
        
        print(f"\nChunk types: {dict(chunk_types)}")
        print(f"\nTop semantic tags:")
        top_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
        for tag, count in top_tags[:15]:
            print(f"  {tag}: {count}")
        
        results = {
            "corpus_stats": stats,
            "chunk_analysis": {
                "total_chunks": len(all_chunks),
                "chunk_types": chunk_types,
                "top_tags": dict(top_tags[:20])
            },
            "documents": [
                {
                    "id": doc.metadata.document_id,
                    "title": doc.metadata.title,
                    "industry": doc.metadata.industry,
                    "jurisdiction": doc.metadata.jurisdiction,
                    "parties": len(doc.metadata.parties),
                    "sections": len(doc.sections),
                    "definitions": len(doc.definitions)
                }
                for doc in processed_documents
            ]
        }
        
        with open("corpus_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Detailed analysis saved to corpus_analysis.json")
        print(f"✓ Processed documents saved to ./corpus_output/")
        
        return True
    else:
        print("❌ No documents were successfully processed")
        return False

if __name__ == "__main__":
    success = test_full_corpus()
    if success:
        print("\n🎉 Full corpus ingestion test completed successfully!")
    else:
        print("\n❌ Full corpus ingestion test failed")
