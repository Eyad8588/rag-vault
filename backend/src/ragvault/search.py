"""Hybrid retrieval (vector + BM25, fused with reciprocal rank fusion) and cited answers.

Permission filtering happens first: both retrievers only ever see chunks whose access tag is in
the caller's readable tags."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .llm import LLM, Embedder
from .store import ChunkStore

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the and or of to in on for is are was were be by with as at from that this it its "
           "what who how why when which do does did".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


def hybrid_search(store: ChunkStore, embedder: Embedder, query: str, tags: list[str], k: int = 6,
                  candidates: int = 25, rrf_k: int = 60) -> list[dict]:
    if not tags:
        return []
    qvec = embedder.embed([query])[0]
    vec_hits = store.vector_search(qvec, tags, k=candidates)

    permitted = store.all_permitted(tags)
    bm25_hits: list[dict] = []
    if permitted:
        bm25 = BM25Okapi([tokenize(r["text"]) or ["_"] for r in permitted])
        scores = bm25.get_scores(tokenize(query) or ["_"])
        order = sorted(range(len(permitted)), key=lambda i: scores[i], reverse=True)[:candidates]
        bm25_hits = [permitted[i] | {"bm25": float(scores[i])} for i in order if scores[i] > 0]

    fused: dict[str, dict] = {}
    for source, hits in (("vector", vec_hits), ("bm25", bm25_hits)):
        for rank, h in enumerate(hits):
            item = fused.setdefault(h["id"], {k_: h[k_] for k_ in ("id", "doc_id", "chunk_index", "text",
                                                                   "access_tag")} | {"score": 0.0, "via": []})
            item["score"] += 1.0 / (rrf_k + rank + 1)
            item["via"].append(source)
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:k]
    # Defence in depth: never return a chunk outside the caller's tags even if a store bug let it through.
    return [r for r in ranked if r["access_tag"] in tags]


ANSWER_SYSTEM = (
    "You answer questions using ONLY the numbered sources provided. Cite every claim with the source "
    "number in square brackets, e.g. [1] or [2][3]. If the sources do not contain the answer, say you "
    "don't know based on the available documents. Be concise."
)


def answer(llm: LLM, question: str, hits: list[dict], doc_titles: dict[str, str]) -> str:
    if not hits:
        return "I couldn't find anything in the documents you have access to that answers this."
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] (from \"{doc_titles.get(h['doc_id'], h['doc_id'])}\")\n{h['text']}")
    prompt = "SOURCES:\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}\n\nAnswer with citations:"
    return llm.chat([{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}])
