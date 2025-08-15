"""
Debug script to understand XML structure
"""

import xml.etree.ElementTree as ET
import sys

def analyze_xml_structure(xml_file):
    """Analyze the structure of the XML document"""
    print(f"Analyzing XML structure: {xml_file}")
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    print(f"Root element: {root.tag}")
    print(f"Root attributes: {root.attrib}")
    print(f"Namespaces: {dict(root.nsmap) if hasattr(root, 'nsmap') else 'Not available'}")
    
    all_elements = set()
    for elem in root.iter():
        all_elements.add(elem.tag)
    
    print(f"\nAll element types found: {sorted(all_elements)}")
    
    print(f"\nLooking for paragraphs with xml:id...")
    namespaces = {'xml': 'http://www.w3.org/XML/1998/namespace'}
    
    xml_id_elements = root.findall('.//p[@xml:id]', namespaces)
    print(f"Found {len(xml_id_elements)} elements with @xml:id using namespaces")
    
    try:
        xml_id_elements_no_ns = root.findall('.//p[@xml:id]')
        print(f"Found {len(xml_id_elements_no_ns)} elements with @xml:id without namespaces")
    except Exception as e:
        print(f"Error without namespaces: {e}")
    
    id_elements = root.findall('.//p[@id]')
    print(f"Found {len(id_elements)} elements with @id")
    
    print(f"\nFirst 10 paragraph elements:")
    paragraphs = root.findall('.//p')
    for i, p in enumerate(paragraphs[:10]):
        attrs = dict(p.attrib)
        text = (p.text or '').strip()[:100]
        print(f"  {i+1}: {attrs} - '{text}'")
    
    org_names = root.findall('.//orgName')
    print(f"\nFound {len(org_names)} orgName elements:")
    for org in org_names:
        print(f"  - {org.text}")
    
    dates = root.findall('.//date')
    print(f"\nFound {len(dates)} date elements:")
    for date in dates:
        print(f"  - {date.text}")

if __name__ == "__main__":
    xml_file = "/home/ubuntu/attachments/2e36673c-6c1f-4844-b6cb-6d8445bf86ea/129435_sanitized.xml"
    analyze_xml_structure(xml_file)
