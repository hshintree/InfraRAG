-- One row per section_id
SELECT document_id, section_id, MIN(title) AS title,
       COUNT(*) AS chunk_count,
       COUNT(DISTINCT content_hash) AS distinct_bodies
FROM clauses
WHERE document_id='39638_sanitized'
GROUP BY 1,2
ORDER BY section_id::int;

