from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SlotSpec:
    slot: str
    doc_type: Optional[str] = None
    industry: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryBundle:
    base_queries: List[str]
    must_tokens: List[str]
    should_tokens: List[str]
    must_not_tokens: List[str]
    heading_like: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalPackage:
    slot: str
    items: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    definitions: List[Dict[str, Any]]
    debug: Dict[str, Any] = field(default_factory=dict) 