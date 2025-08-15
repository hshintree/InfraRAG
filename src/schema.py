"""
Canonical JSON schema for legal document ingestion and normalization.
Defines the structure for infrastructure project finance purchase agreements.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class PartyRole(str, Enum):
    """Standard party roles in infrastructure finance agreements"""
    SELLER = "Seller"
    BUYER = "Buyer"
    PURCHASER = "Purchaser"
    VENDOR = "Vendor"
    CONTRACTOR = "Contractor"
    LENDER = "Lender"
    GUARANTOR = "Guarantor"
    TRUSTEE = "Trustee"
    AGENT = "Agent"
    OTHER = "Other"


class ClauseType(str, Enum):
    """Standard clause types in infrastructure agreements"""
    PARTIES = "Parties"
    DEFINITIONS = "Definitions"
    PURCHASE_AND_SALE = "Purchase and Sale"
    PRICE = "Price"
    ADJUSTMENTS = "Adjustments"
    CLOSING = "Closing"
    CONDITIONS_PRECEDENT = "Conditions Precedent"
    REPRESENTATIONS_WARRANTIES = "Representations and Warranties"
    COVENANTS = "Covenants"
    INDEMNITIES = "Indemnities"
    LIMITATIONS = "Limitations"
    GOVERNING_LAW = "Governing Law"
    DISPUTE_RESOLUTION = "Dispute Resolution"
    NOTICES = "Notices"
    TERMINATION = "Termination"
    FORCE_MAJEURE = "Force Majeure"
    MISCELLANEOUS = "Miscellaneous"
    OTHER = "Other"


class Party(BaseModel):
    """Legal party in the agreement"""
    name: str = Field(..., description="Full legal name of the party")
    role: PartyRole = Field(..., description="Role of the party in the agreement")
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction of incorporation/organization")
    address: Optional[str] = Field(None, description="Principal place of business")
    entity_type: Optional[str] = Field(None, description="Type of legal entity (corporation, LLC, etc.)")


class Definition(BaseModel):
    """Defined term in the agreement"""
    term: str = Field(..., description="The defined term")
    definition: str = Field(..., description="The definition text")
    section_id: Optional[str] = Field(None, description="Section where defined")


class Section(BaseModel):
    """Document section/clause"""
    id: str = Field(..., description="Section identifier (e.g., '3.1', '10.7')")
    title: str = Field(..., description="Section title/heading")
    text: str = Field(..., description="Full section text content")
    clause_type: ClauseType = Field(..., description="Categorized clause type")
    tags: List[str] = Field(default_factory=list, description="Semantic tags for the clause")
    definitions: List[str] = Field(default_factory=list, description="Defined terms referenced in this section")
    parent_section: Optional[str] = Field(None, description="Parent section ID for subsections")
    page_number: Optional[int] = Field(None, description="Page number in source document")


class DocumentMetadata(BaseModel):
    """Document-level metadata"""
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    document_type: str = Field(..., description="Type of agreement (Purchase Agreement, etc.)")
    jurisdiction: str = Field(..., description="Primary jurisdiction (e.g., 'US-NY', 'UK-England')")
    governing_law: Optional[str] = Field(None, description="Governing law jurisdiction")
    industry: str = Field(..., description="Industry sector (e.g., 'Power', 'LNG', 'Infrastructure')")
    effective_date: Optional[datetime] = Field(None, description="Agreement effective date")
    execution_date: Optional[datetime] = Field(None, description="Agreement execution date")
    termination_date: Optional[datetime] = Field(None, description="Agreement termination date")
    parties: List[Party] = Field(..., description="All parties to the agreement")
    source_file: str = Field(..., description="Original source file path")
    source_format: str = Field(..., description="Source format (PDF, XML, DOCX)")
    processing_date: datetime = Field(default_factory=datetime.utcnow, description="Date processed")


class LegalDocument(BaseModel):
    """Complete legal document structure"""
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    sections: List[Section] = Field(..., description="All document sections/clauses")
    definitions: List[Definition] = Field(..., description="All defined terms")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChunkMetadata(BaseModel):
    """Metadata for individual chunks used in retrieval"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    section_id: str = Field(..., description="Source section ID")
    chunk_index: int = Field(..., description="Index within the section")
    chunk_type: str = Field(..., description="Type of chunk (clause, definition, schedule)")
    tags: List[str] = Field(default_factory=list, description="Semantic tags")
    source_citation: str = Field(..., description="Citation reference for the chunk")


class ProcessedChunk(BaseModel):
    """Individual chunk for retrieval system"""
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    content: str = Field(..., description="Chunk text content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    
    
LEGAL_TERM_SYNONYMS = {
    "purchase_price_adjustment": ["true-up", "price adjustment", "purchase price true-up"],
    "material_adverse_effect": ["material adverse change", "MAC", "MAE"],
    "conditions_precedent": ["closing conditions", "conditions to closing"],
    "representations_warranties": ["reps and warranties", "representations", "warranties"],
    "indemnification": ["indemnity", "indemnities", "hold harmless"],
    "force_majeure": ["act of god", "unforeseeable circumstances"],
    "governing_law": ["applicable law", "choice of law"],
    "dispute_resolution": ["arbitration", "litigation", "mediation"],
    "termination": ["expiry", "expiration", "end"],
    "assignment": ["transfer", "novation"],
}

INFRA_FINANCE_TERMS = {
    "CROD": "Contract Rate of Delivery",
    "transmission_service": "electrical transmission services",
    "point_of_delivery": "delivery point",
    "billing_demand": "maximum demand charge",
    "energy_charge": "electricity usage charge",
    "curtailment": "reduction in delivery",
    "liquidated_damages": "predetermined damages",
    "patronage": "cooperative member benefits",
    "operating_procedures": "operational guidelines",
}
