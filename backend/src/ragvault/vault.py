"""The Vault service: ingest, remove and query documents. The HTTP layer (api.py) is a thin wrapper."""

from __future__ import annotations

import logging
import time
import uuid

from . import graph as kg
from .auth import ACCESS_TAGS, DEFAULT_CLEARANCES, SENSITIVITY_TO_TAG, hash_password, readable_tags
from .classify import DocClassifier
from .config import Settings
from .db import DB
from .llm import LLM, Embedder
from .parse import chunk_text, extract_text, guess_title
from .search import answer, hybrid_search
from .store import ChunkStore

log = logging.getLogger(__name__)


class PermissionDenied(Exception):
    pass


class Vault:
    def __init__(self, settings: Settings, embedder: Embedder, llm: LLM, classifier: DocClassifier | None = None):
        settings.ensure_dirs()
        self.s = settings
        self.db = DB(settings.sqlite_path)
        self.store = ChunkStore(settings.lancedb_path)
        self.embedder = embedder
        self.llm = llm
        self.classifier = classifier or DocClassifier(settings.models_dir)

    # ------------------------------------------------------------------ users
    def seed_users(self) -> None:
        """Create the demo users on first run (password = username). Change them in the admin panel."""
        if self.db.list_users():
            return
        for name in ("admin", "editor", "reader"):
            self.db.upsert_user(name, hash_password(name), name, DEFAULT_CLEARANCES[name])

    # ------------------------------------------------------------------ ingest
    def ingest(self, user: dict, filename: str, data: bytes, access_tag: str | None = None,
               build_graph: bool = True) -> dict:
        """Parse → chunk → embed → classify → tag → store. Graph extraction is a separate step
        (`extract_graph`) so the API can run it in the background."""
        if user["role"] not in ("editor", "admin"):
            raise PermissionDenied("Only editors and admins can add documents")
        if access_tag is not None and access_tag not in ACCESS_TAGS:
            raise ValueError(f"access_tag must be one of {ACCESS_TAGS}")
        text = extract_text(filename, data)
        if not text.strip():
            raise ValueError("No text could be extracted from the file")
        chunks = chunk_text(text, self.s.chunk_size, self.s.chunk_overlap)
        doc_id = uuid.uuid4().hex[:12]
        title = guess_title(filename, text)
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO documents(id,filename,title,access_tag,tag_source,uploaded_by,created_at,status) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, filename, title, access_tag or "restricted", "manual" if access_tag else "classifier",
                 user["username"], time.time(), "indexing"),
            )
        try:
            vectors = self.embedder.embed(chunks)
            labels = self.classifier.classify(chunks, vectors)
            tag = access_tag or SENSITIVITY_TO_TAG[labels["sensitivity"]]
            self.store.add([
                {"id": f"{doc_id}:{i}", "doc_id": doc_id, "chunk_index": i, "text": t, "access_tag": tag,
                 "vector": vectors[i]}
                for i, t in enumerate(chunks)
            ])
            (self.s.uploads_dir / f"{doc_id}_{filename}").write_bytes(data)
            with self.db.tx() as c:
                c.execute(
                    "UPDATE documents SET topic=?, topic_conf=?, sensitivity=?, sens_conf=?, access_tag=?, "
                    "n_chunks=?, status=? WHERE id=?",
                    (labels["topic"], labels["topic_conf"], labels["sensitivity"], labels["sens_conf"], tag,
                     len(chunks), "graphing" if build_graph else "ready", doc_id),
                )
        except Exception as e:
            with self.db.tx() as c:
                c.execute("UPDATE documents SET status='error', error=? WHERE id=?", (str(e), doc_id))
            raise
        self.db.audit(user["username"], user["role"], "add_doc", doc_id,
                      {"filename": filename, "title": title, "chunks": len(chunks), "access_tag": tag,
                       "topic": labels["topic"], "sensitivity": labels["sensitivity"],
                       "reasons": labels["reasons"], "tag_source": "manual" if access_tag else "classifier"})
        return self.get_doc(doc_id)

    def extract_graph(self, doc_id: str) -> None:
        try:
            chunks = self.store.chunks_for_doc(doc_id)[: self.s.graph_max_chunks]
            for ch in chunks:
                try:
                    data = kg.extract(self.llm, ch["text"])
                except Exception as e:  # one bad LLM reply should not kill the document
                    log.warning("entity extraction failed for %s: %s", ch["id"], e)
                    continue
                if self.get_doc(doc_id) is None:  # removed while we were extracting
                    return
                kg.save_extraction(self.db, doc_id, ch["id"], data)
            with self.db.tx() as c:
                c.execute("UPDATE documents SET status='ready' WHERE id=?", (doc_id,))
        except Exception as e:
            log.exception("graph extraction failed")
            with self.db.tx() as c:
                c.execute("UPDATE documents SET status='ready', error=? WHERE id=?", (f"graph: {e}", doc_id))

    # ------------------------------------------------------------------ remove / retag
    def remove(self, user: dict, doc_id: str) -> None:
        if user["role"] != "admin":
            raise PermissionDenied("Only admins can remove documents")
        doc = self.get_doc(doc_id)
        if doc is None:
            raise KeyError(doc_id)
        self.store.delete_doc(doc_id)
        kg.delete_doc(self.db, doc_id)
        with self.db.tx() as c:
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        for f in self.s.uploads_dir.glob(f"{doc_id}_*"):
            f.unlink(missing_ok=True)
        self.db.audit(user["username"], user["role"], "remove_doc", doc_id,
                      {"filename": doc["filename"], "title": doc["title"], "chunks": doc["n_chunks"]})

    def set_access_tag(self, user: dict, doc_id: str, tag: str) -> dict:
        if user["role"] != "admin":
            raise PermissionDenied("Only admins can change a document's access tag")
        if tag not in ACCESS_TAGS:
            raise ValueError(f"access_tag must be one of {ACCESS_TAGS}")
        doc = self.get_doc(doc_id)
        if doc is None:
            raise KeyError(doc_id)
        self.store.set_doc_tag(doc_id, tag)
        with self.db.tx() as c:
            c.execute("UPDATE documents SET access_tag=?, tag_source='manual' WHERE id=?", (tag, doc_id))
        self.db.audit(user["username"], user["role"], "retag_doc", doc_id, {"from": doc["access_tag"], "to": tag})
        return self.get_doc(doc_id)

    # ------------------------------------------------------------------ read
    def get_doc(self, doc_id: str) -> dict | None:
        return self.db.one("SELECT * FROM documents WHERE id=?", (doc_id,))

    def visible_docs(self, user: dict) -> list[dict]:
        tags = readable_tags(user)
        if not tags:
            return []
        ph = ",".join("?" * len(tags))
        return self.db.query(f"SELECT * FROM documents WHERE access_tag IN ({ph}) ORDER BY created_at DESC", tags)

    def query(self, user: dict, question: str, k: int = 6) -> dict:
        tags = readable_tags(user)  # permission filter computed BEFORE retrieval
        t0 = time.perf_counter()
        hits = hybrid_search(self.store, self.embedder, question, tags, k=k)
        docs = {d["id"]: d for d in self.visible_docs(user)}
        hits = [h for h in hits if h["doc_id"] in docs]
        titles = {i: d["title"] for i, d in docs.items()}
        text = answer(self.llm, question, hits, titles)
        citations = [
            {"n": i, "doc_id": h["doc_id"], "title": titles[h["doc_id"]], "filename": docs[h["doc_id"]]["filename"],
             "chunk_index": h["chunk_index"], "access_tag": h["access_tag"], "text": h["text"],
             "score": round(h["score"], 5), "via": h["via"]}
            for i, h in enumerate(hits, 1)
        ]
        self.db.audit(user["username"], user["role"], "query", None,
                      {"question": question[:500], "allowed_tags": tags,
                       "cited_docs": sorted({c["doc_id"] for c in citations}),
                       "ms": round(1000 * (time.perf_counter() - t0))})
        return {"answer": text, "citations": citations}

    def graph(self, user: dict) -> dict:
        docs = {d["id"]: d for d in self.visible_docs(user)}
        g = kg.build_graph(self.db, list(docs))
        return kg.graph_json(g, docs)
