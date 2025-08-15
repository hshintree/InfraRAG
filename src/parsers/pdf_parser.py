"""
PDF parser for legal documents.
Extracts text and structure from PDF files using PyMuPDF.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
from ..schema import LegalDocument, DocumentMetadata, Section, Definition, Party, PartyRole, ClauseType


class PDFLegalParser:
    """Parser for PDF legal documents"""
    
    def __init__(self):
        self.section_patterns = [
            r'^(\d+)\.\s+([A-Z][^.]+)\.',  # "1. SECTION TITLE."
            r'^(\d+\.\d+)\s+([A-Z][^.]+)',  # "1.1 Subsection Title"
            r'^([A-Z]+)\.\s+([A-Z][^.]+)',  # "A. Section Title"
        ]
    
    def parse_document(self, pdf_path: str) -> LegalDocument:
        """Parse PDF document and return structured legal document"""
        doc = fitz.open(pdf_path)
        
        full_text = ""
        page_texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            page_texts.append(page_text)
            full_text += page_text + "\n"
        
        doc.close()
        
        metadata = self._extract_metadata(full_text, pdf_path)
        
        sections = self._extract_sections(full_text, page_texts)
        
        definitions = self._extract_definitions(full_text, sections)
        
        return LegalDocument(
            metadata=metadata,
            sections=sections,
            definitions=definitions
        )
    
    def _extract_metadata(self, text: str, source_file: str) -> DocumentMetadata:
        """Extract document-level metadata from PDF text"""
        lines = text.split('\n')[:20]
        title = "Unknown Document"
        for line in lines:
            line = line.strip()
            if len(line) > 10 and line.isupper() and 'agreement' in line.lower():
                title = line.title()
                break
        
        parties = self._extract_parties_from_text(text)
        
        effective_date, execution_date = self._extract_dates_from_text(text)
        
        jurisdiction, governing_law = self._extract_jurisdiction_from_text(text)
        
        industry = self._determine_industry_from_text(text)
        
        document_id = source_file.split('/')[-1].replace('.pdf', '')
        
        return DocumentMetadata(
            document_id=document_id,
            title=title,
            document_type="Purchase Agreement",
            jurisdiction=jurisdiction,
            governing_law=governing_law,
            industry=industry,
            effective_date=effective_date,
            execution_date=execution_date,
            parties=parties,
            source_file=source_file,
            source_format="PDF"
        )
    
    def _extract_parties_from_text(self, text: str) -> List[Party]:
        """Extract parties from PDF text"""
        parties = []
        
        party_patterns = [
            r'between\s+([^,]+),?\s*\(["\']?([^"\']+)["\']?\)',  # "between Company Name ("Seller")"
            r'([A-Z][A-Z\s&,\.]+(?:INC|LLC|CORP|CORPORATION|COMPANY))[,\s]*\(["\']?([^"\']+)["\']?\)',
        ]
        
        for pattern in party_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for name, role_text in matches:
                name = name.strip()
                role_text = role_text.strip().lower()
                
                role = PartyRole.OTHER
                if 'seller' in role_text or 'vendor' in role_text:
                    role = PartyRole.SELLER
                elif 'buyer' in role_text or 'purchaser' in role_text:
                    role = PartyRole.BUYER
                
                parties.append(Party(
                    name=name,
                    role=role,
                    entity_type=self._extract_entity_type(name)
                ))
        
        return parties
    
    def _extract_entity_type(self, name: str) -> Optional[str]:
        """Extract entity type from company name"""
        name_lower = name.lower()
        if 'inc' in name_lower or 'incorporated' in name_lower:
            return "Corporation"
        elif 'llc' in name_lower:
            return "Limited Liability Company"
        elif 'corp' in name_lower or 'corporation' in name_lower:
            return "Corporation"
        elif 'company' in name_lower:
            return "Company"
        return None
    
    def _extract_dates_from_text(self, text: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract dates from PDF text"""
        effective_date = None
        execution_date = None
        
        date_patterns = [
            r'effective\s+(?:date\s+)?(?:of\s+)?([A-Za-z]+\s+\d{1,2},\s+\d{4})',
            r'dated\s+(?:as\s+of\s+)?([A-Za-z]+\s+\d{1,2},\s+\d{4})',
            r'(\d{1,2})\s+day\s+of\s+([A-Za-z]+),\s+(\d{4})',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 3:
                    day, month, year = match
                    parsed_date = self._parse_date_components(day, month, year)
                else:
                    parsed_date = self._parse_date_string(match)
                
                if parsed_date:
                    if not effective_date:
                        effective_date = parsed_date
                    elif not execution_date:
                        execution_date = parsed_date
        
        return effective_date, execution_date
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse date string in various formats"""
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        match = re.match(r'([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})', date_str.strip())
        if match:
            month_name, day, year = match.groups()
            month = months.get(month_name.lower())
            if month:
                try:
                    return datetime(int(year), month, int(day))
                except ValueError:
                    pass
        
        return None
    
    def _parse_date_components(self, day: str, month: str, year: str) -> Optional[datetime]:
        """Parse date from separate components"""
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        month_num = months.get(month.lower())
        if month_num:
            try:
                return datetime(int(year), month_num, int(day))
            except ValueError:
                pass
        
        return None
    
    def _extract_jurisdiction_from_text(self, text: str) -> Tuple[str, Optional[str]]:
        """Extract jurisdiction and governing law from text"""
        jurisdiction = "US"
        governing_law = None
        
        gov_law_patterns = [
            r'governed\s+by\s+(?:the\s+)?laws?\s+of\s+(?:the\s+)?(?:state\s+of\s+)?([A-Za-z\s]+)',
            r'construed\s+(?:in\s+)?accordance\s+with\s+(?:the\s+)?laws?\s+of\s+([A-Za-z\s]+)',
        ]
        
        for pattern in gov_law_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                state = match.strip().lower()
                if 'new york' in state:
                    governing_law = "US-NY"
                    jurisdiction = "US-NY"
                elif 'delaware' in state:
                    governing_law = "US-DE"
                    jurisdiction = "US-DE"
                elif 'california' in state:
                    governing_law = "US-CA"
                    jurisdiction = "US-CA"
                break
        
        return jurisdiction, governing_law
    
    def _determine_industry_from_text(self, text: str) -> str:
        """Determine industry from document content"""
        text_lower = text.lower()
        
        if 'infrastructure' in text_lower:
            return "Infrastructure"
        elif 'power' in text_lower and 'electric' in text_lower:
            return "Power"
        elif 'lng' in text_lower or 'natural gas' in text_lower:
            return "LNG"
        elif 'oil' in text_lower or 'petroleum' in text_lower:
            return "Oil & Gas"
        else:
            return "General"
    
    def _extract_sections(self, text: str, page_texts: List[str]) -> List[Section]:
        """Extract document sections from PDF text"""
        sections = []
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        section_counter = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            section_match = None
            for pattern in self.section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_match = match
                    break
            
            if section_match:
                if current_section:
                    sections.append(current_section)
                
                section_id = section_match.group(1)
                title = section_match.group(2).strip()
                
                clause_type = self._classify_clause_type(title)
                
                current_section = Section(
                    id=section_id,
                    title=title,
                    text="",  # Will be filled as we collect content
                    clause_type=clause_type,
                    tags=self._extract_tags_from_title(title),
                    definitions=[],
                    page_number=self._find_page_number(line, page_texts)
                )
                current_content = [line]
                section_counter += 1
            
            elif current_section:
                current_content.append(line)
        
        if current_section:
            current_section.text = '\n'.join(current_content)
            current_section.definitions = self._extract_referenced_definitions(current_section.text)
            sections.append(current_section)
        
        return sections
    
    def _find_page_number(self, line: str, page_texts: List[str]) -> Optional[int]:
        """Find which page a line appears on"""
        for page_num, page_text in enumerate(page_texts):
            if line in page_text:
                return page_num + 1
        return None
    
    def _classify_clause_type(self, title: str) -> ClauseType:
        """Classify clause type based on title"""
        title_lower = title.lower()
        
        if 'purchase' in title_lower or 'sale' in title_lower:
            return ClauseType.PURCHASE_AND_SALE
        elif 'price' in title_lower or 'payment' in title_lower:
            return ClauseType.PRICE
        elif 'closing' in title_lower:
            return ClauseType.CLOSING
        elif 'condition' in title_lower:
            return ClauseType.CONDITIONS_PRECEDENT
        elif 'representation' in title_lower or 'warrant' in title_lower:
            return ClauseType.REPRESENTATIONS_WARRANTIES
        elif 'covenant' in title_lower:
            return ClauseType.COVENANTS
        elif 'indemnif' in title_lower:
            return ClauseType.INDEMNITIES
        elif 'governing law' in title_lower:
            return ClauseType.GOVERNING_LAW
        elif 'dispute' in title_lower or 'arbitration' in title_lower:
            return ClauseType.DISPUTE_RESOLUTION
        elif 'notice' in title_lower:
            return ClauseType.NOTICES
        elif 'termination' in title_lower:
            return ClauseType.TERMINATION
        else:
            return ClauseType.OTHER
    
    def _extract_tags_from_title(self, title: str) -> List[str]:
        """Extract semantic tags from section title"""
        tags = []
        title_lower = title.lower()
        
        if 'price' in title_lower:
            tags.append('pricing')
        if 'closing' in title_lower:
            tags.append('closing')
        if 'condition' in title_lower:
            tags.append('conditions')
        if 'indemnif' in title_lower:
            tags.append('indemnification')
        if 'termination' in title_lower:
            tags.append('termination')
        
        return tags
    
    def _extract_referenced_definitions(self, text: str) -> List[str]:
        """Extract defined terms referenced in the text"""
        definitions = []
        
        quoted_terms = re.findall(r'"([A-Z][^"]*)"', text)
        definitions.extend(quoted_terms)
        
        caps_terms = re.findall(r'\b([A-Z]{2,})\b', text)
        definitions.extend(caps_terms)
        
        return list(set(definitions))  # Remove duplicates
    
    def _extract_definitions(self, text: str, sections: List[Section]) -> List[Definition]:
        """Extract defined terms from the document"""
        definitions = []
        
        for section in sections:
            if 'definition' in section.title.lower():
                def_patterns = [
                    r'"([^"]+)"\s+means\s+([^.]+\.)',
                    r'"([^"]+)"\s+shall\s+mean\s+([^.]+\.)',
                    r'([A-Z][A-Za-z\s]+)\s+means\s+([^.]+\.)',
                ]
                
                for pattern in def_patterns:
                    matches = re.findall(pattern, section.text)
                    for term, definition in matches:
                        definitions.append(Definition(
                            term=term.strip(),
                            definition=definition.strip(),
                            section_id=section.id
                        ))
        
        inline_patterns = [
            r'"([^"]+)"\s*\([^)]*\)\s*means\s+([^.]+\.)',
            r'([A-Z][A-Za-z\s]+)\s*\([^)]*\)\s*means\s+([^.]+\.)',
        ]
        
        for pattern in inline_patterns:
            matches = re.findall(pattern, text)
            for term, definition in matches:
                definitions.append(Definition(
                    term=term.strip(),
                    definition=definition.strip()
                ))
        
        return definitions
