"""
Hybrid indexing system for legal documents.
Combines dense vector search with BM25 sparse search.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from qdrant_client.http import models
except ImportError:
    print("Warning: qdrant-client not installed. Vector search will not be available.")
    QdrantClient = None

try:
    from opensearchpy import OpenSearch
except ImportError:
    print("Warning: opensearch-py not installed. BM25 search will not be available.")
    OpenSearch = None

from .schema import ProcessedChunk, LegalDocument
from .ingestion import DocumentIngestionPipeline


class HybridIndexer:
    """Hybrid indexing system combining vector and BM25 search"""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        collection_name: str = "legal_documents"
    ):
        self.collection_name = collection_name
        
        if QdrantClient:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        else:
            self.qdrant_client = None
        
        if OpenSearch:
            self.opensearch_client = OpenSearch([{
                'host': opensearch_host,
                'port': opensearch_port
            }])
        else:
            self.opensearch_client = None
        
        self.embedding_dim = 384  # Dimension for sentence-transformers/all-MiniLM-L6-v2
    
    def setup_indices(self):
        """Set up vector and BM25 indices"""
        self._setup_qdrant_collection()
        self._setup_opensearch_index()
    
    def _setup_qdrant_collection(self):
        """Set up Qdrant collection for vector search"""
        if not self.qdrant_client:
            print("Qdrant client not available, skipping vector index setup")
            return
        
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Qdrant collection already exists: {self.collection_name}")
        
        except Exception as e:
            print(f"Error setting up Qdrant collection: {e}")
    
    def _setup_opensearch_index(self):
        """Set up OpenSearch index for BM25 search"""
        if not self.opensearch_client:
            print("OpenSearch client not available, skipping BM25 index setup")
            return
        
        try:
            index_name = f"{self.collection_name}_bm25"
            
            if not self.opensearch_client.indices.exists(index=index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "section_id": {"type": "keyword"},
                            "chunk_type": {"type": "keyword"},
                            "content": {
                                "type": "text",
                                "analyzer": "english"
                            },
                            "tags": {"type": "keyword"},
                            "source_citation": {"type": "text"},
                            "metadata": {"type": "object"}
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }
                
                self.opensearch_client.indices.create(
                    index=index_name,
                    body=mapping
                )
                print(f"Created OpenSearch index: {index_name}")
            else:
                print(f"OpenSearch index already exists: {index_name}")
        
        except Exception as e:
            print(f"Error setting up OpenSearch index: {e}")
    
    def index_chunks(self, chunks: List[ProcessedChunk], embeddings: Optional[List[List[float]]] = None):
        """Index chunks in both vector and BM25 stores"""
        if not chunks:
            return
        
        if self.qdrant_client and embeddings:
            self._index_chunks_qdrant(chunks, embeddings)
        
        if self.opensearch_client:
            self._index_chunks_opensearch(chunks)
    
    def _index_chunks_qdrant(self, chunks: List[ProcessedChunk], embeddings: List[List[float]]):
        """Index chunks in Qdrant for vector search"""
        try:
            points = []
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    point = PointStruct(
                        id=chunk.metadata.chunk_id,
                        vector=embeddings[i],
                        payload={
                            "chunk_id": chunk.metadata.chunk_id,
                            "document_id": chunk.metadata.document_id,
                            "section_id": chunk.metadata.section_id,
                            "chunk_index": chunk.metadata.chunk_index,
                            "chunk_type": chunk.metadata.chunk_type,
                            "content": chunk.content,
                            "tags": chunk.metadata.tags,
                            "source_citation": chunk.metadata.source_citation
                        }
                    )
                    points.append(point)
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Indexed {len(points)} chunks in Qdrant")
        
        except Exception as e:
            print(f"Error indexing chunks in Qdrant: {e}")
    
    def _index_chunks_opensearch(self, chunks: List[ProcessedChunk]):
        """Index chunks in OpenSearch for BM25 search"""
        try:
            index_name = f"{self.collection_name}_bm25"
            
            bulk_body = []
            for chunk in chunks:
                bulk_body.append({
                    "index": {
                        "_index": index_name,
                        "_id": chunk.metadata.chunk_id
                    }
                })
                
                bulk_body.append({
                    "chunk_id": chunk.metadata.chunk_id,
                    "document_id": chunk.metadata.document_id,
                    "section_id": chunk.metadata.section_id,
                    "chunk_type": chunk.metadata.chunk_type,
                    "content": chunk.content,
                    "tags": chunk.metadata.tags,
                    "source_citation": chunk.metadata.source_citation,
                    "metadata": {
                        "chunk_index": chunk.metadata.chunk_index
                    }
                })
            
            response = self.opensearch_client.bulk(body=bulk_body)
            
            if response.get("errors"):
                print("Some documents failed to index in OpenSearch")
            else:
                print(f"Indexed {len(chunks)} chunks in OpenSearch")
        
        except Exception as e:
            print(f"Error indexing chunks in OpenSearch: {e}")
    
    def search_vector(self, query_embedding: List[float], limit: int = 20, filters: Optional[Dict] = None) -> List[Dict]:
        """Search using vector similarity"""
        if not self.qdrant_client:
            return []
        
        try:
            must_conditions = []
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
            
            filter_condition = models.Filter(must=must_conditions) if must_conditions else None
            
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_condition,
                limit=limit
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []
    
    def search_bm25(self, query: str, limit: int = 20, filters: Optional[Dict] = None) -> List[Dict]:
        """Search using BM25"""
        if not self.opensearch_client:
            return []
        
        try:
            index_name = f"{self.collection_name}_bm25"
            
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content": {
                                        "query": query,
                                        "operator": "or"
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": limit
            }
            
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        filter_conditions.append({"terms": {key: value}})
                    else:
                        filter_conditions.append({"term": {key: value}})
                
                search_body["query"]["bool"]["filter"] = filter_conditions
            
            response = self.opensearch_client.search(
                index=index_name,
                body=search_body
            )
            
            formatted_results = []
            for hit in response["hits"]["hits"]:
                formatted_results.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"]
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Error in BM25 search: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 20,
        alpha: float = 0.5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Hybrid search combining vector and BM25 results"""
        vector_results = self.search_vector(query_embedding, limit * 2, filters)
        bm25_results = self.search_bm25(query, limit * 2, filters)
        
        combined_results = {}
        
        for result in vector_results:
            chunk_id = result["id"]
            combined_results[chunk_id] = {
                "chunk_id": chunk_id,
                "content": result["payload"]["content"],
                "source_citation": result["payload"]["source_citation"],
                "tags": result["payload"]["tags"],
                "vector_score": result["score"],
                "bm25_score": 0.0,
                "metadata": result["payload"]
            }
        
        for result in bm25_results:
            chunk_id = result["id"]
            if chunk_id in combined_results:
                combined_results[chunk_id]["bm25_score"] = result["score"]
            else:
                combined_results[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": result["source"]["content"],
                    "source_citation": result["source"]["source_citation"],
                    "tags": result["source"]["tags"],
                    "vector_score": 0.0,
                    "bm25_score": result["score"],
                    "metadata": result["source"]
                }
        
        for result in combined_results.values():
            vector_score = result["vector_score"]
            bm25_score = result["bm25_score"]
            
            hybrid_score = alpha * vector_score + (1 - alpha) * (bm25_score / 10.0)  # Scale BM25 score
            result["hybrid_score"] = hybrid_score
        
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the indices"""
        stats = {
            "qdrant": {},
            "opensearch": {}
        }
        
        if self.qdrant_client:
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                stats["qdrant"] = {
                    "status": collection_info.status,
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count
                }
            except Exception as e:
                stats["qdrant"]["error"] = str(e)
        
        if self.opensearch_client:
            try:
                index_name = f"{self.collection_name}_bm25"
                index_stats = self.opensearch_client.indices.stats(index=index_name)
                stats["opensearch"] = {
                    "docs_count": index_stats["indices"][index_name]["total"]["docs"]["count"],
                    "store_size": index_stats["indices"][index_name]["total"]["store"]["size_in_bytes"]
                }
            except Exception as e:
                stats["opensearch"]["error"] = str(e)
        
        return stats


def main():
    """CLI interface for indexing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index legal documents")
    parser.add_argument("--setup", action="store_true", help="Set up indices")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--documents-dir", default="./processed_documents", help="Directory with processed documents")
    
    args = parser.parse_args()
    
    indexer = HybridIndexer()
    
    if args.setup:
        print("Setting up indices...")
        indexer.setup_indices()
    
    if args.stats:
        stats = indexer.get_index_stats()
        print("Index Statistics:")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
