"""Verify store-config ingestion — one chunk per store, each stamped with
metadata_json.Metadata.store_id.

Run this AFTER re-uploading the store-config JSON (e.g. stores_corrected.json)
so it re-ingests through StoreConfigSchema
(backend/services/contextual_ingestion_service.py). It confirms:

  * the file produced ONE chunk per store (not one big blob), and
  * each chunk carries metadata_json->'Metadata'->>'store_id', which is the
    exact path the store-scoped KB retrieval filters on
    (backend/retrieval/scoped_retrieval.py::retrieve_within_store, migration 065).

Usage (from repo root, same env the API uses so DATABASE_URL is set):

    .venv/Scripts/python -m backend.scripts.check_store_config_chunks
    .venv/Scripts/python -m backend.scripts.check_store_config_chunks --name stores_corrected --store 3001

Exit code 0 = at least one store-tagged chunk found; 1 = none (re-ingest
didn't take, or the file still needs re-upload).
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import text

from backend.db.connection import engine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--name",
        default="stores_corrected",
        help="substring of the document name to check (default: stores_corrected)",
    )
    ap.add_argument(
        "--store",
        default=None,
        help="optional store id to assert is present (e.g. 3001)",
    )
    args = ap.parse_args()

    like = f"%{args.name}%"
    with engine.connect() as conn:
        docs = conn.execute(
            text(
                "SELECT id::text AS id, name, file_type "
                "FROM documents WHERE name ILIKE :like AND status = 'active'"
            ),
            {"like": like},
        ).mappings().all()

        if not docs:
            print(f"No active document matching name ILIKE {like!r}. "
                  f"Has it been uploaded?")
            return 1

        total_tagged = 0
        for d in docs:
            rows = conn.execute(
                text(
                    """
                    SELECT
                        (metadata_json->'Metadata'->>'store_id') AS store_id,
                        chunk_type,
                        section_heading
                    FROM chunks
                    WHERE document_id = :id
                    ORDER BY (metadata_json->'Metadata'->>'store_id')
                    """
                ),
                {"id": d["id"]},
            ).mappings().all()

            tagged = [r for r in rows if r["store_id"]]
            total_tagged += len(tagged)

            print(f"\nDocument: {d['name']}  (file_type={d['file_type']})")
            print(f"  chunks total          : {len(rows)}")
            print(f"  chunks with store_id  : {len(tagged)}")
            store_ids = [r["store_id"] for r in tagged]
            print(f"  store_ids             : {store_ids}")

            if len(rows) == 1 and not tagged:
                print("  [!] Single untagged chunk -- this file has NOT been "
                      "re-ingested through StoreConfigSchema yet. Delete + "
                      "re-upload it from admin.")

            if args.store:
                hit = [r for r in tagged if str(r["store_id"]) == str(args.store)]
                mark = "FOUND" if hit else "MISSING"
                print(f"  store {args.store:<8}       : {mark}")
                if hit:
                    print(f"     heading: {hit[0]['section_heading']!r} "
                          f"chunk_type={hit[0]['chunk_type']}")

    print(f"\nTotal store-tagged chunks across matches: {total_tagged}")
    return 0 if total_tagged > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
