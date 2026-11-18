"""LanceDB chunk store. Every read takes the caller's allowed access tags and applies them as a
pre-filter, so chunks the user may not see never enter the candidate set."""

from __future__ import annotations

import threading
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from .auth import ACCESS_TAGS

TABLE = "chunks"


def _tag_filter(tags: list[str]) -> str:
    safe = [t for t in tags if t in ACCESS_TAGS]  # whitelist: never interpolate user input
    if not safe:
        return "access_tag = '__none__'"
    return "access_tag IN (" + ", ".join(f"'{t}'" for t in safe) + ")"


def _id_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class ChunkStore:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(path))
        self._lock = threading.RLock()

    def _table(self):
        try:
            return self.db.open_table(TABLE)
        except Exception:
            return None

    def add(self, rows: list[dict]) -> None:
        """rows: {id, doc_id, chunk_index, text, access_tag, vector}"""
        if not rows:
            return
        dim = len(rows[0]["vector"])
        schema = pa.schema([
            ("id", pa.string()), ("doc_id", pa.string()), ("chunk_index", pa.int32()),
            ("text", pa.string()), ("access_tag", pa.string()),
            ("vector", pa.list_(pa.float32(), dim)),
        ])
        data = pa.Table.from_pylist(
            [{**r, "vector": np.asarray(r["vector"], dtype=np.float32).tolist()} for r in rows], schema=schema
        )
        with self._lock:
            tbl = self._table()
            if tbl is None:
                self.db.create_table(TABLE, data=data)
            else:
                tbl.add(data)

    def delete_doc(self, doc_id: str) -> None:
        with self._lock:
            tbl = self._table()
            if tbl is not None:
                tbl.delete(f"doc_id = {_id_literal(doc_id)}")

    def set_doc_tag(self, doc_id: str, tag: str) -> None:
        assert tag in ACCESS_TAGS
        with self._lock:
            tbl = self._table()
            if tbl is not None:
                tbl.update(where=f"doc_id = {_id_literal(doc_id)}", values={"access_tag": tag})

    def count(self, doc_id: str | None = None) -> int:
        tbl = self._table()
        if tbl is None:
            return 0
        return tbl.count_rows(f"doc_id = {_id_literal(doc_id)}" if doc_id else None)

    def vector_search(self, vector: np.ndarray, tags: list[str], k: int = 20) -> list[dict]:
        tbl = self._table()
        if tbl is None:
            return []
        res = (
            tbl.search(np.asarray(vector, dtype=np.float32))
            .metric("cosine")
            .where(_tag_filter(tags), prefilter=True)
            .limit(k)
            .select(["id", "doc_id", "chunk_index", "text", "access_tag"])
            .to_list()
        )
        return res

    def all_permitted(self, tags: list[str], limit: int = 50_000) -> list[dict]:
        """All chunk texts the caller may read (used for the BM25 side of hybrid search)."""
        tbl = self._table()
        if tbl is None:
            return []
        return (
            tbl.search()
            .where(_tag_filter(tags))
            .select(["id", "doc_id", "chunk_index", "text", "access_tag"])
            .limit(limit)
            .to_list()
        )

    def chunks_for_doc(self, doc_id: str) -> list[dict]:
        tbl = self._table()
        if tbl is None:
            return []
        rows = (
            tbl.search().where(f"doc_id = {_id_literal(doc_id)}")
            .select(["id", "doc_id", "chunk_index", "text", "access_tag"]).limit(100_000).to_list()
        )
        return sorted(rows, key=lambda r: r["chunk_index"])
