-- One row per section_id
-- SELECT document_id, section_id, MIN(title) AS title,
--        COUNT(*) AS chunk_count,
--        COUNT(DISTINCT content_hash) AS distinct_bodies
-- FROM clauses
-- WHERE document_id='39638_sanitized'
-- GROUP BY 1,2
-- ORDER BY section_id::int;

SELECT relation, COUNT(*) AS edges
FROM clause_edges GROUP BY 1;

-- confirm uniqueness
SELECT COUNT(*) AS total, COUNT(DISTINCT (src_clause_id, dst_clause_id, relation)) AS distinct
FROM clause_edges;

SELECT document_id, COUNT(*) AS total,
       COUNT(heading_number) AS have_heading,
       COUNT(clause_number)  AS have_clause  -- if you added it
FROM clauses GROUP BY 1 ORDER BY 1;