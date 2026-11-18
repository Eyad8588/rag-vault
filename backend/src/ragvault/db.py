"""SQLite metadata store: users, documents, audit log and the knowledge graph tables."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('reader','editor','admin')),
    clearances    TEXT NOT NULL,           -- JSON list of access tags the user may read
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    title         TEXT NOT NULL,
    topic         TEXT,
    topic_conf    REAL,
    sensitivity   TEXT,
    sens_conf     REAL,
    access_tag    TEXT NOT NULL,
    tag_source    TEXT NOT NULL,           -- 'classifier' or 'manual'
    uploaded_by   TEXT NOT NULL,
    created_at    REAL NOT NULL,
    n_chunks      INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL,           -- 'indexing' | 'graphing' | 'ready' | 'error'
    error         TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    username  TEXT,
    role      TEXT,
    action    TEXT NOT NULL,
    target    TEXT,
    details   TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    key     TEXT UNIQUE NOT NULL,          -- normalised name used for de-duplication
    name    TEXT NOT NULL,
    type    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mentions (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    doc_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id  TEXT NOT NULL,
    PRIMARY KEY (entity_id, doc_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS relations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id    INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    dst_id    INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation  TEXT NOT NULL,
    doc_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mentions_doc ON mentions(doc_id);
CREATE INDEX IF NOT EXISTS idx_relations_doc ON relations(doc_id);
"""


class DB:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        with self.tx() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple | list = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---- users -------------------------------------------------------------
    def upsert_user(self, username: str, password_hash: str, role: str, clearances: list[str]) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO users(username,password_hash,role,clearances,created_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(username) DO UPDATE SET password_hash=excluded.password_hash, role=excluded.role, "
                "clearances=excluded.clearances",
                (username, password_hash, role, json.dumps(clearances), time.time()),
            )

    def get_user(self, username: str) -> dict[str, Any] | None:
        u = self.one("SELECT * FROM users WHERE username=?", (username,))
        if u:
            u["clearances"] = json.loads(u["clearances"])
        return u

    def list_users(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT username, role, clearances, created_at FROM users ORDER BY username")
        for r in rows:
            r["clearances"] = json.loads(r["clearances"])
        return rows

    def update_user(self, username: str, role: str | None, clearances: list[str] | None) -> None:
        with self.tx() as c:
            if role is not None:
                c.execute("UPDATE users SET role=? WHERE username=?", (role, username))
            if clearances is not None:
                c.execute("UPDATE users SET clearances=? WHERE username=?", (json.dumps(clearances), username))

    def delete_user(self, username: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM users WHERE username=?", (username,))

    # ---- audit -------------------------------------------------------------
    def audit(self, username: str | None, role: str | None, action: str, target: str | None = None,
              details: dict | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO audit(ts,username,role,action,target,details) VALUES (?,?,?,?,?,?)",
                (time.time(), username, role, action, target, json.dumps(details or {})),
            )

    def list_audit(self, limit: int = 200, action: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audit"
        params: list[Any] = []
        if action:
            sql += " WHERE action=?"
            params.append(action)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.query(sql, params)
        for r in rows:
            r["details"] = json.loads(r["details"] or "{}")
        return rows
