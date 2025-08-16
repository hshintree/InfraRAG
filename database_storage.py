"""
Database storage layer for InfraRAG system.
Provides PostgreSQL connection and session management with pgvector support.
"""

import os
from typing import Generator, List, Dict, Any, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import numpy as np

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    print("Warning: pgvector not installed. Vector operations will not be available.")
    Vector = None

Base = declarative_base()


class TextChunk(Base):
    """SQLAlchemy model for text chunks"""
    __tablename__ = 'text_chunks'
    
    id = Column(Integer, primary_key=True)
    source = Column(String(255))
    symbol = Column(String(255))
    document_id = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_id = Column(String(255), unique=True, nullable=False)
    section_id = Column(String(255))
    chunk_type = Column(String(100))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384)) if Vector else Column(Text)
    tags = Column(ARRAY(String))
    source_citation = Column(Text)
    chunk_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DataStorage:
    """Database storage manager for PostgreSQL with pgvector"""
    
    def __init__(self):
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5434")
        self.db_name = os.getenv("DB_NAME", "infrarag_db")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "infrarag_secure_pw_2024")
        
        print(f"Connecting to database: {self.db_host}:{self.db_port}/{self.db_name}")
        
        self.connection_string = (
            f"postgresql://{self.db_user}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}"
        )
        
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema and extensions"""
        with self.get_session() as db:
            try:
                db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                
                Base.metadata.create_all(bind=self.engine)
                
                db.execute(text("CREATE INDEX IF NOT EXISTS idx_text_chunks_document_id ON text_chunks(document_id)"))
                db.execute(text("CREATE INDEX IF NOT EXISTS idx_text_chunks_chunk_id ON text_chunks(chunk_id)"))
                db.execute(text("CREATE INDEX IF NOT EXISTS idx_text_chunks_source ON text_chunks(source)"))
                
                db.commit()
                
            except Exception as e:
                print(f"Error initializing database: {e}")
                db.rollback()
                raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def save_chunks_to_database(self, chunks: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None):
        """Save chunks to database with idempotency checks"""
        with self.get_session() as db:
            try:
                for i, chunk_data in enumerate(chunks):
                    existing_chunk = db.query(TextChunk).filter(
                        TextChunk.chunk_id == chunk_data['chunk_id']
                    ).first()
                    
                    if existing_chunk:
                        print(f"Chunk {chunk_data['chunk_id']} already exists, skipping...")
                        continue
                    
                    embedding = None
                    if embeddings and i < len(embeddings):
                        embedding = embeddings[i]
                    
                    new_chunk = TextChunk(
                        source=chunk_data.get('source'),
                        symbol=chunk_data.get('symbol'),
                        document_id=chunk_data['document_id'],
                        chunk_index=chunk_data['chunk_index'],
                        chunk_id=chunk_data['chunk_id'],
                        section_id=chunk_data.get('section_id'),
                        chunk_type=chunk_data.get('chunk_type'),
                        content=chunk_data['content'],
                        embedding=embedding,
                        tags=chunk_data.get('tags', []),
                        source_citation=chunk_data.get('source_citation'),
                        chunk_metadata=chunk_data.get('metadata', {})
                    )
                    
                    db.add(new_chunk)
                
                db.commit()
                print(f"Successfully saved {len(chunks)} chunks to database")
                
            except Exception as e:
                print(f"Error saving chunks to database: {e}")
                db.rollback()
                raise
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in database"""
        with self.get_session() as db:
            return db.query(TextChunk).count()
    
    def get_chunks_with_embeddings_count(self) -> int:
        """Get number of chunks with embeddings"""
        with self.get_session() as db:
            return db.query(TextChunk).filter(TextChunk.embedding.isnot(None)).count()
    
    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search chunks using full-text search"""
        with self.get_session() as db:
            result = db.execute(text("""
                SELECT chunk_id, source, section_id, chunk_type, content,
                       tags, source_citation, chunk_metadata,
                       ts_rank(to_tsvector('english', content), plainto_tsquery('english', :query)) as rank
                FROM text_chunks 
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                ORDER BY rank DESC
                LIMIT :limit
            """), {'query': query, 'limit': limit})
            
            chunks = []
            for row in result:
                chunks.append({
                    'chunk_id': row.chunk_id,
                    'source': row.source,
                    'section_id': row.section_id,
                    'chunk_type': row.chunk_type,
                    'content': row.content,
                    'tags': row.tags,
                    'source_citation': row.source_citation,
                    'metadata': row.chunk_metadata,
                    'rank': float(row.rank)
                })
            
            return chunks
    
    def vector_search(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search chunks using vector similarity"""
        if not Vector:
            print("pgvector not available, falling back to text search")
            return []
        
        with self.get_session() as db:
            result = db.execute(text("""
                SELECT chunk_id, source, section_id, chunk_type, content,
                       tags, source_citation, chunk_metadata,
                       1 - (embedding <=> :query_vector) as similarity
                FROM text_chunks 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :query_vector
                LIMIT :limit
            """), {
                'query_vector': str(query_embedding),
                'limit': limit
            })
            
            chunks = []
            for row in result:
                chunks.append({
                    'chunk_id': row.chunk_id,
                    'source': row.source,
                    'section_id': row.section_id,
                    'chunk_type': row.chunk_type,
                    'content': row.content,
                    'tags': row.tags,
                    'source_citation': row.source_citation,
                    'metadata': row.chunk_metadata,
                    'similarity': float(row.similarity)
                })
            
            return chunks
