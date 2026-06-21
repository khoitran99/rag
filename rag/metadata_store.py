"""Persistent metadata store mapping chunk ids back to their source document and page.

Backed by SQLite so chunk -> (document, page, text) lookups are cheap at query time and
survive process restarts. A companion ``document_hashes`` table records each document's content
hash, which incremental ingestion uses to skip unchanged files. Citations and incremental
ingestion both rely on this store.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rag.models import Chunk


class MetadataStore:
    def __init__(self, path: str | Path):
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                document TEXT NOT NULL,
                page     INTEGER NOT NULL,
                text     TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_hashes (
                document     TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def hash_of(self, document: str) -> str | None:
        """The last recorded content hash for a document, or None if never ingested."""
        row = self._conn.execute(
            "SELECT content_hash FROM document_hashes WHERE document = ?", (document,)
        ).fetchone()
        return row[0] if row else None

    def set_hash(self, document: str, content_hash: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO document_hashes (document, content_hash) VALUES (?, ?)",
            (document, content_hash),
        )
        self._conn.commit()

    def add(self, chunks: list[Chunk]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks (chunk_id, document, page, text) VALUES (?, ?, ?, ?)",
            [(c.chunk_id, c.document, c.page, c.text) for c in chunks],
        )
        self._conn.commit()

    def documents(self) -> list[str]:
        """Every distinct document currently held, used to detect files removed from the folder."""
        rows = self._conn.execute("SELECT DISTINCT document FROM chunks").fetchall()
        return [r[0] for r in rows]

    def chunk_ids_for(self, document: str) -> list[int]:
        """The chunk ids belonging to a document, so its vectors can be removed from the index."""
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE document = ?", (document,)
        ).fetchall()
        return [r[0] for r in rows]

    def delete_by_document(self, document: str) -> None:
        """Remove a document entirely — its chunks and its recorded hash."""
        self._conn.execute("DELETE FROM chunks WHERE document = ?", (document,))
        self._conn.execute("DELETE FROM document_hashes WHERE document = ?", (document,))
        self._conn.commit()

    def get(self, chunk_id: int) -> Chunk:
        row = self._conn.execute(
            "SELECT chunk_id, document, page, text FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise KeyError(chunk_id)
        return Chunk(chunk_id=row[0], document=row[1], page=row[2], text=row[3])

    def close(self) -> None:
        self._conn.close()
