"""Knowledge graph: entities and relations extracted by the local LLM, stored in SQLite and
served as a NetworkX graph filtered to the documents the caller may read."""

from __future__ import annotations

import json
import logging
import re

import networkx as nx

from .db import DB
from .llm import LLM

log = logging.getLogger(__name__)

ENTITY_TYPES = ("person", "organization", "location", "product", "technology", "event", "concept")

EXTRACT_PROMPT = """Extract the most important named entities and the relations between them from the text.
Return JSON only, in this exact shape:
{{"entities": [{{"name": "...", "type": "{types}"}}],
  "relations": [{{"source": "<entity name>", "target": "<entity name>", "relation": "<short verb phrase>"}}]}}
Rules: at most 10 entities, at most 10 relations, use names exactly as written in the text,
only relations between entities you listed.

TEXT:
{text}"""


def normalise(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\-.&]", "", name)).strip().lower()


def extract(llm: LLM, text: str) -> dict:
    prompt = EXTRACT_PROMPT.format(types="|".join(ENTITY_TYPES), text=text[:3500])
    raw = llm.chat([{"role": "user", "content": prompt}], json_mode=True, temperature=0.0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(m.group(0)) if m else {}
    ents, rels = [], []
    for e in data.get("entities", []) or []:
        if isinstance(e, dict) and isinstance(e.get("name"), str) and 1 < len(e["name"]) <= 80:
            t = str(e.get("type", "concept")).lower()
            ents.append({"name": e["name"].strip(), "type": t if t in ENTITY_TYPES else "concept"})
    names = {normalise(e["name"]) for e in ents}
    for r in data.get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        s, t, rel = r.get("source"), r.get("target"), r.get("relation")
        if all(isinstance(x, str) for x in (s, t, rel)) and normalise(s) in names and normalise(t) in names \
                and normalise(s) != normalise(t):
            rels.append({"source": s, "target": t, "relation": rel.strip()[:60]})
    return {"entities": ents, "relations": rels}


def save_extraction(db: DB, doc_id: str, chunk_id: str, data: dict) -> None:
    with db.tx() as c:
        ids: dict[str, int] = {}
        for e in data["entities"]:
            key = normalise(e["name"])
            if not key:
                continue
            c.execute("INSERT OR IGNORE INTO entities(key,name,type) VALUES (?,?,?)", (key, e["name"], e["type"]))
            eid = c.execute("SELECT id FROM entities WHERE key=?", (key,)).fetchone()[0]
            ids[key] = eid
            c.execute("INSERT OR IGNORE INTO mentions(entity_id,doc_id,chunk_id) VALUES (?,?,?)",
                      (eid, doc_id, chunk_id))
        for r in data["relations"]:
            s, t = ids.get(normalise(r["source"])), ids.get(normalise(r["target"]))
            if s and t:
                c.execute("INSERT INTO relations(src_id,dst_id,relation,doc_id,chunk_id) VALUES (?,?,?,?,?)",
                          (s, t, r["relation"], doc_id, chunk_id))


def delete_doc(db: DB, doc_id: str) -> None:
    """Remove a document's mentions/relations, then any entity no longer mentioned anywhere."""
    with db.tx() as c:
        c.execute("DELETE FROM relations WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM mentions WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM entities WHERE id NOT IN (SELECT DISTINCT entity_id FROM mentions)")


def build_graph(db: DB, doc_ids: list[str]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    if not doc_ids:
        return g
    ph = ",".join("?" * len(doc_ids))
    for row in db.query(
        f"SELECT e.id, e.name, e.type, m.doc_id FROM mentions m JOIN entities e ON e.id=m.entity_id "
        f"WHERE m.doc_id IN ({ph})", doc_ids
    ):
        if row["id"] not in g:
            g.add_node(row["id"], name=row["name"], type=row["type"], docs=set())
        g.nodes[row["id"]]["docs"].add(row["doc_id"])
    for row in db.query(f"SELECT src_id, dst_id, relation, doc_id FROM relations WHERE doc_id IN ({ph})", doc_ids):
        if row["src_id"] in g and row["dst_id"] in g:
            g.add_edge(row["src_id"], row["dst_id"], relation=row["relation"], doc_id=row["doc_id"])
    return g


def graph_json(g: nx.MultiDiGraph, docs: dict[str, dict]) -> dict:
    degree = dict(g.degree())
    nodes = [
        {
            "id": n, "name": d["name"], "type": d["type"], "degree": degree.get(n, 0),
            "docs": [{"id": x, "title": docs[x]["title"], "access_tag": docs[x]["access_tag"]}
                     for x in sorted(d["docs"]) if x in docs],
        }
        for n, d in g.nodes(data=True)
    ]
    links = [{"source": u, "target": v, "relation": d["relation"], "doc_id": d["doc_id"]}
             for u, v, d in g.edges(data=True)]
    return {"nodes": nodes, "links": links}
