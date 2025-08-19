import os
import json
import argparse
import psycopg
from typing import Any, Dict, List

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "infra_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme_local_pw")


def _connect():
	return psycopg.connect(
		host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
	)


def _get_scalar(cur, sql: str, params: tuple = ()) -> Any:
	cur.execute(sql, params)
	row = cur.fetchone()
	return row[0] if row else None


def _columns(cur, table: str) -> List[str]:
	cur.execute(
		"""
		SELECT column_name FROM information_schema.columns
		WHERE table_name=%s ORDER BY ordinal_position
		""",
		(table,),
	)
	return [r[0] for r in cur.fetchall()]


def _indexes(cur, table: str) -> List[str]:
	cur.execute(
		"""
		SELECT indexname FROM pg_indexes WHERE schemaname='public' AND tablename=%s
		""",
		(table,),
	)
	return [r[0] for r in cur.fetchall()]


def status(document_id: str = "", heading_like: str = "", sample: int = 5) -> Dict[str, Any]:
	with _connect() as conn, conn.cursor() as cur:
		out: Dict[str, Any] = {"db": DB_NAME}
		# Counts
		out["counts"] = {
			"documents": _get_scalar(cur, "SELECT COUNT(*) FROM documents"),
			"clauses": _get_scalar(cur, "SELECT COUNT(*) FROM clauses"),
			"clauses_with_embedding": _get_scalar(cur, "SELECT COUNT(*) FROM clauses WHERE embedding IS NOT NULL"),
			"clauses_without_embedding": _get_scalar(cur, "SELECT COUNT(*) FROM clauses WHERE embedding IS NULL"),
			"documents_in_clauses": _get_scalar(cur, "SELECT COUNT(DISTINCT document_id) FROM clauses"),
		}
		# Columns / indexes
		cols = _columns(cur, "clauses")
		idxs = _indexes(cur, "clauses")
		out["clauses_columns"] = cols
		out["clauses_indexes"] = idxs
		# Heading coverage
		cur.execute(
			"""
			SELECT heading_number, COUNT(*) AS n
			FROM clauses
			WHERE heading_number IS NOT NULL
			GROUP BY heading_number
			ORDER BY n DESC, heading_number ASC
			LIMIT 10
			"""
		)
		out["top_heading_numbers"] = [{"heading_number": r[0], "count": int(r[1])} for r in cur.fetchall()]
		# Clause types
		cur.execute(
			"""
			SELECT clause_type, COUNT(*) AS n
			FROM clauses
			GROUP BY clause_type
			ORDER BY n DESC NULLS LAST
			LIMIT 15
			"""
		)
		out["clause_types"] = [{"clause_type": r[0], "count": int(r[1])} for r in cur.fetchall()]
		# Optional filters
		filters = []
		params: List[Any] = []
		if document_id:
			filters.append("document_id = %s")
			params.append(document_id)
		if heading_like:
			filters.append("heading_number LIKE %s")
			params.append(heading_like)
		where = ("WHERE " + " AND ".join(filters)) if filters else ""
		cur.execute(
			f"""
			SELECT document_id, section_id, title, heading_number
			FROM clauses
			{where}
			ORDER BY document_id, heading_number NULLS LAST, id
			LIMIT %s
			""",
			tuple(params + [sample]),
		)
		out["sample_rows"] = [
			{"document_id": r[0], "section_id": r[1], "title": r[2], "heading_number": r[3]} for r in cur.fetchall()
		]
		return out


def main():
	parser = argparse.ArgumentParser(description="Show database status for InfraRAG")
	parser.add_argument("--document-id", default="", help="Filter sample rows to a document_id")
	parser.add_argument("--heading-like", default="", help="Filter sample rows by heading_number pattern, e.g., '20.%'")
	parser.add_argument("--sample", type=int, default=8, help="How many sample rows to show")
	parser.add_argument("--json", action="store_true", help="Output JSON only")
	args = parser.parse_args()

	info = status(document_id=args.document_id, heading_like=args.heading_like, sample=args.sample)
	if args.json:
		print(json.dumps(info))
		return
	print(json.dumps(info, indent=2))


if __name__ == "__main__":
	main() 