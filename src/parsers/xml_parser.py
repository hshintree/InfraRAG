"""
XML parser for legal documents with TEI markup.
Extracts structured content from XML files with legal document markup.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
from ..schema import LegalDocument, DocumentMetadata, Section, Definition, Party, PartyRole, ClauseType


class XMLLegalParser:
    """Parser for XML legal documents with TEI markup"""
    
    def __init__(self):
        self.namespaces = {
            'tei': 'http://www.tei-c.org/ns/1.0',
            'xml': 'http://www.w3.org/XML/1998/namespace'
        }
    
    def parse_document(self, xml_path: str) -> LegalDocument:
        """Parse XML document and return structured legal document"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        metadata = self._extract_metadata(root, xml_path)
        
        sections = self._extract_sections(root)
        
        definitions = self._extract_definitions(root, sections)
        
        return LegalDocument(
            metadata=metadata,
            sections=sections,
            definitions=definitions
        )
    
    def _extract_metadata(self, root: ET.Element, source_file: str) -> DocumentMetadata:
        """Extract document-level metadata"""
        title_elem = root.find('.//{http://www.tei-c.org/ns/1.0}p[@ana="title-page intro-page"]')
        title = title_elem.text.strip() if title_elem is not None else "Unknown Document"
        
        parties = self._extract_parties(root)
        
        effective_date, execution_date = self._extract_dates(root)
        
        jurisdiction, governing_law = self._extract_jurisdiction(root)
        
        industry = self._determine_industry(root)
        
        document_id = source_file.split('/')[-1].replace('.xml', '')
        
        return DocumentMetadata(
            document_id=document_id,
            title=title,
            document_type="Wholesale Power Contract",
            jurisdiction=jurisdiction,
            governing_law=governing_law,
            industry=industry,
            effective_date=effective_date,
            execution_date=execution_date,
            parties=parties,
            source_file=source_file,
            source_format="XML"
        )
    
    def _extract_parties(self, root: ET.Element) -> List[Party]:
        """Extract parties from the document"""
        parties = []
        
        for elem in root.findall('.//{http://www.tei-c.org/ns/1.0}orgName'):
            if elem.text:
                name = elem.text.strip()
                
                role = PartyRole.OTHER
                parent_text = ""
                for p_elem in root.findall('.//{http://www.tei-c.org/ns/1.0}p'):
                    if elem in list(p_elem):
                        parent_text = ET.tostring(p_elem, encoding='unicode', method='text')
                        break
                
                if 'seller' in parent_text.lower():
                    role = PartyRole.SELLER
                elif 'buyer' in parent_text.lower() or 'purchaser' in parent_text.lower():
                    role = PartyRole.BUYER
                
                jurisdiction = None
                if 'north dakota' in parent_text.lower():
                    jurisdiction = "US-ND"
                elif 'colorado' in parent_text.lower():
                    jurisdiction = "US-CO"
                
                parties.append(Party(
                    name=name,
                    role=role,
                    jurisdiction=jurisdiction,
                    entity_type="Cooperative Corporation"
                ))
        
        return parties
    
    def _extract_dates(self, root: ET.Element) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract effective and execution dates"""
        effective_date = None
        execution_date = None
        
        for date_elem in root.findall('.//{http://www.tei-c.org/ns/1.0}date'):
            if date_elem.text:
                date_text = date_elem.text.strip()
                
                parsed_date = self._parse_date(date_text)
                if parsed_date:
                    parent_text = ""
                    for p_elem in root.findall('.//{http://www.tei-c.org/ns/1.0}p'):
                        if date_elem in list(p_elem):
                            parent_text = ET.tostring(p_elem, encoding='unicode', method='text').lower()
                            break
                    
                    if 'effective' in parent_text:
                        effective_date = parsed_date
                    elif 'execution' in parent_text or 'made as of' in parent_text:
                        execution_date = parsed_date
        
        return effective_date, execution_date
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse various date formats"""
        date_patterns = [
            r'(\w+)\s+(\d{1,2}),\s+(\d{4})',  # "September 27, 2017"
            r'(\d{1,2})\s+(\w+)\s+(\d{4})',   # "27 September 2017"
            r'(\d{4})-(\d{2})-(\d{2})',       # "2017-09-27"
        ]
        
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, date_text.lower())
            if match:
                try:
                    if pattern == date_patterns[0]:  # Month Day, Year
                        month_name, day, year = match.groups()
                        month = months.get(month_name.lower())
                        if month:
                            return datetime(int(year), month, int(day))
                    elif pattern == date_patterns[2]:  # YYYY-MM-DD
                        year, month, day = match.groups()
                        return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        return None
    
    def _extract_jurisdiction(self, root: ET.Element) -> Tuple[str, Optional[str]]:
        """Extract jurisdiction and governing law"""
        jurisdiction = "US"
        governing_law = None
        
        for elem in root.findall('.//{http://www.tei-c.org/ns/1.0}p'):
            if elem.get('base') == 'governing law' or 'governing law' in (elem.text or '').lower():
                text = ET.tostring(elem, encoding='unicode', method='text')
                if 'colorado' in text.lower():
                    governing_law = "US-CO"
                    jurisdiction = "US-CO"
                elif 'new york' in text.lower():
                    governing_law = "US-NY"
        
        return jurisdiction, governing_law
    
    def _determine_industry(self, root: ET.Element) -> str:
        """Determine industry from document content"""
        content = ET.tostring(root, encoding='unicode', method='text').lower()
        
        if 'power' in content and 'electric' in content:
            return "Power"
        elif 'lng' in content or 'natural gas' in content:
            return "LNG"
        elif 'infrastructure' in content:
            return "Infrastructure"
        else:
            return "Energy"
    
    def _extract_sections(self, root: ET.Element) -> List[Section]:
        """Extract document sections and clauses"""
        sections = []
        
        for elem in root.findall('.//{http://www.tei-c.org/ns/1.0}p[@xml:id]', self.namespaces):
            section_id = elem.get('{http://www.w3.org/XML/1998/namespace}id', '').replace('H_', '')
            base_attr = elem.get('base', '')
            toc_number = elem.get('toc_number', '')
            
            if base_attr or toc_number:
                title = base_attr.replace('_', ' ').title() if base_attr else f"Section {toc_number}"
                
                content = self._get_section_content(elem, root)
                
                clause_type = self._classify_clause_type(title, content)
                
                tags = self._extract_tags(content, title)
                
                definitions = self._extract_referenced_definitions(content)
                
                sections.append(Section(
                    id=toc_number or section_id,
                    title=title,
                    text=content,
                    clause_type=clause_type,
                    tags=tags,
                    definitions=definitions
                ))
        
        return sections
    
    def _get_section_content(self, header_elem: ET.Element, root: ET.Element) -> str:
        """Get content for a section by collecting following paragraphs"""
        content_parts = []
        
        header_text = ET.tostring(header_elem, encoding='unicode', method='text').strip()
        content_parts.append(header_text)
        
        all_paras = root.findall('.//{http://www.tei-c.org/ns/1.0}p')
        header_index = -1
        for i, p in enumerate(all_paras):
            if p == header_elem:
                header_index = i
                break
        
        if header_index >= 0:
            for i in range(header_index + 1, len(all_paras)):
                current = all_paras[i]
                if current.get('{http://www.w3.org/XML/1998/namespace}id') and current.get('base') and current.get('toc_number'):
                    break
                
                para_text = ET.tostring(current, encoding='unicode', method='text').strip()
                if para_text:
                    content_parts.append(para_text)
        
        return ' '.join(content_parts)
    
    def _classify_clause_type(self, title: str, content: str) -> ClauseType:
        """Classify clause type based on title and content"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        if 'rate' in title_lower or 'price' in title_lower:
            return ClauseType.PRICE
        elif 'delivery' in title_lower or 'sale' in title_lower:
            return ClauseType.PURCHASE_AND_SALE
        elif 'governing law' in title_lower:
            return ClauseType.GOVERNING_LAW
        elif 'notice' in title_lower:
            return ClauseType.NOTICES
        elif 'term' in title_lower or 'termination' in title_lower:
            return ClauseType.TERMINATION
        elif 'assignment' in title_lower:
            return ClauseType.MISCELLANEOUS
        elif 'audit' in title_lower:
            return ClauseType.COVENANTS
        elif 'continuity' in title_lower or 'service' in title_lower:
            return ClauseType.COVENANTS
        else:
            return ClauseType.OTHER
    
    def _extract_tags(self, content: str, title: str) -> List[str]:
        """Extract semantic tags from content"""
        tags = []
        content_lower = content.lower()
        title_lower = title.lower()
        
        if 'rate' in title_lower:
            tags.append('pricing')
        if 'delivery' in title_lower:
            tags.append('delivery')
        if 'law' in title_lower:
            tags.append('governing_law')
        
        if 'liquidated damages' in content_lower:
            tags.append('liquidated_damages')
        if 'curtailment' in content_lower:
            tags.append('curtailment')
        if 'transmission' in content_lower:
            tags.append('transmission')
        if 'arbitration' in content_lower:
            tags.append('arbitration')
        
        return tags
    
    def _extract_referenced_definitions(self, content: str) -> List[str]:
        """Extract defined terms referenced in the content"""
        definitions = []
        
        defined_term_patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # "Contract Rate"
            r'\b([A-Z]{2,})\b',  # "CROD"
        ]
        
        for pattern in defined_term_patterns:
            matches = re.findall(pattern, content)
            definitions.extend(matches)
        
        return list(set(definitions))  # Remove duplicates
    
    def _extract_definitions(self, root: ET.Element, sections: List[Section]) -> List[Definition]:
        """Extract defined terms from the document"""
        definitions = []
        
        for section in sections:
            if 'definition' in section.title.lower():
                def_matches = re.findall(r'"([^"]+)"\s+means\s+([^.]+)', section.text)
                for term, definition in def_matches:
                    definitions.append(Definition(
                        term=term,
                        definition=definition.strip(),
                        section_id=section.id
                    ))
        
        content = ET.tostring(root, encoding='unicode', method='text')
        inline_defs = re.findall(r'"([^"]+)"\s*\([^)]*\)\s*means\s+([^.]+)', content)
        for term, definition in inline_defs:
            definitions.append(Definition(
                term=term,
                definition=definition.strip()
            ))
        
        return definitions
