SELECT section_id, title 
FROM clauses 
WHERE document_id='39638_sanitized'
  AND content ILIKE '%arbitral award%'
  AND content ILIKE '%LIBOR%'
  AND content ILIKE '%360%';

SELECT section_id, title 
FROM clauses 
WHERE document_id='39638_sanitized'
  AND section_id BETWEEN '216' AND '218'
  AND content ILIKE '%arbitral award%';

  