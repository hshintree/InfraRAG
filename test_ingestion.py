"""
Test script for document ingestion pipeline
"""

import sys
import os
sys.path.append('./src')

from src.ingestion import DocumentIngestionPipeline
from src.parsers.xml_parser import XMLLegalParser

def test_xml_parsing():
    """Test XML parsing on the provided sample document"""
    print("Testing XML parsing...")
    
    xml_file = "/home/ubuntu/attachments/2e36673c-6c1f-4844-b6cb-6d8445bf86ea/129435_sanitized.xml"
    
    if not os.path.exists(xml_file):
        print(f"XML file not found: {xml_file}")
        return False
    
    try:
        parser = XMLLegalParser()
        document = parser.parse_document(xml_file)
        
        print(f"✓ Successfully parsed XML document")
        print(f"  Document ID: {document.metadata.document_id}")
        print(f"  Title: {document.metadata.title}")
        print(f"  Industry: {document.metadata.industry}")
        print(f"  Jurisdiction: {document.metadata.jurisdiction}")
        print(f"  Parties: {len(document.metadata.parties)}")
        print(f"  Sections: {len(document.sections)}")
        print(f"  Definitions: {len(document.definitions)}")
        
        print("\nFirst 3 sections:")
        for i, section in enumerate(document.sections[:3]):
            print(f"  {section.id}: {section.title} ({section.clause_type})")
        
        return True
        
    except Exception as e:
        print(f"✗ XML parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ingestion_pipeline():
    """Test the full ingestion pipeline"""
    print("\nTesting ingestion pipeline...")
    
    try:
        pipeline = DocumentIngestionPipeline(output_dir="./test_output")
        
        xml_file = "/home/ubuntu/attachments/2e36673c-6c1f-4844-b6cb-6d8445bf86ea/129435_sanitized.xml"
        
        if os.path.exists(xml_file):
            document = pipeline.ingest_document(xml_file)
            chunks = pipeline.chunk_document(document)
            
            print(f"✓ Successfully processed document")
            print(f"  Generated {len(chunks)} chunks")
            print(f"  Chunk types: {set(chunk.metadata.chunk_type for chunk in chunks)}")
            
            print("\nSample chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"  Chunk {i+1}: {chunk.metadata.chunk_type} - {len(chunk.content)} chars")
                print(f"    Tags: {chunk.metadata.tags}")
                print(f"    Content preview: {chunk.content[:100]}...")
            
            return True
        else:
            print(f"XML file not found: {xml_file}")
            return False
            
    except Exception as e:
        print(f"✗ Ingestion pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing InfraRAG Ingestion System ===\n")
    
    xml_success = test_xml_parsing()
    pipeline_success = test_ingestion_pipeline()
    
    print(f"\n=== Test Results ===")
    print(f"XML Parsing: {'✓ PASS' if xml_success else '✗ FAIL'}")
    print(f"Ingestion Pipeline: {'✓ PASS' if pipeline_success else '✗ FAIL'}")
    
    if xml_success and pipeline_success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")
