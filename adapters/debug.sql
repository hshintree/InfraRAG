-- One row per section_id
-- SELECT document_id, section_id, MIN(title) AS title,
--        COUNT(*) AS chunk_count,
--        COUNT(DISTINCT content_hash) AS distinct_bodies
-- FROM clauses
-- WHERE document_id='39638_sanitized'
-- GROUP BY 1,2
-- ORDER BY section_id::int;

-- SELECT relation, COUNT(*) AS edges
-- FROM clause_edges GROUP BY 1;

-- -- confirm uniqueness
-- SELECT COUNT(*) AS total, COUNT(DISTINCT (src_clause_id, dst_clause_id, relation)) AS distinct
-- FROM clause_edges;

-- SELECT document_id, COUNT(*) AS total,
--        COUNT(heading_number) AS have_heading,
--        COUNT(clause_number)  AS have_clause  -- if you added it
-- FROM clauses GROUP BY 1 ORDER BY 1;
-- SELECT document_id, heading_number, title, COUNT(*) AS n
-- FROM clauses
-- WHERE heading_number IS NOT NULL
-- GROUP BY 1,2,3
-- HAVING COUNT(*) > 1
-- ORDER BY n DESC, document_id, heading_number
-- LIMIT 30;

-- Ensure extension
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- -- Same doc, longish clauses, very similar content
-- SELECT c1.document_id, c1.section_id a, c2.section_id b,
--        c1.title, similarity(c1.content, c2.content) sim
-- FROM clauses c1
-- JOIN clauses c2
--   ON c1.document_id = c2.document_id
--  AND c1.id < c2.id
--  AND length(c1.content) > 200 AND length(c2.content) > 200
--  AND similarity(c1.content, c2.content) > 0.92
-- ORDER BY sim DESC
-- LIMIT 25;
-- Are headings hierarchical like "10.2.1"?
SELECT (heading_number ~ '^[0-9]+(\.[0-9]+)*$') AS hierarchical, COUNT(*)
FROM clauses
WHERE heading_number IS NOT NULL
GROUP BY 1;