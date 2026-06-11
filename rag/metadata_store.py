"""Persistent metadata store mapping chunk ids back to their source document and page.

Backed by SQLite so chunk -> (document, page, text) lookups are cheap at query time and
survive process restarts. Citations and (later) incremental ingestion both rely on this.
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
        self._conn.commit()

    def add(self, chunks: list[Chunk]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks (chunk_id, document, page, text) VALUES (?, ?, ?, ?)",
            [(c.chunk_id, c.document, c.page, c.text) for c in chunks],
        )
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
