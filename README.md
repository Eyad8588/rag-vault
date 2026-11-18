# RAG Vault

A knowledge base where **every answer cites its sources**, **users only retrieve what their role and
clearance allow**, and the knowledge can be explored as an **interactive graph**. Everything runs on
local models through [Ollama](https://ollama.com): no cloud APIs.

```
upload ─► parse (PDF/MD/TXT) ─► chunk ─► embed (qwen3-embedding:0.6b)
                                            │
                                            ├─► PyTorch classifier ─► topic + sensitivity ─► access tag
                                            ├─► LanceDB (chunks + vectors + access tag)
                                            └─► local LLM extracts entities/relations ─► graph (SQLite + NetworkX)

question ─► user's readable tags ─► LanceDB vector search  ┐  (both pre-filtered by tag)
                                   └► BM25 over permitted chunks ┘─► RRF fusion ─► qwen3.5 answer with [n] citations
```

## Features

| | |
|---|---|
| **Roles** | `reader` asks questions · `editor` also adds documents · `admin` also removes documents, changes access tags, manages users/roles/clearances and reads the audit log |
| **Access tags** | Every document is `public`, `internal` or `restricted`. Each user has a set of *clearances* (tags they may read). Admins read everything. |
| **Permission filtering before retrieval** | The user's readable tags become a LanceDB `prefilter` on the vector search and limit the BM25 corpus, so restricted chunks never enter the candidate set (and a final check drops anything outside the tags). Document lists, document detail (404, not 403, so existence doesn't leak) and the graph are filtered the same way. |
| **Classifier → default tag** | A PyTorch MLP on the chunk embeddings labels topic (AG News classes) and sensitivity (`public / internal / confidential`). Sensitivity sets the default tag (`confidential → restricted`). The most sensitive chunk wins, and a regex for secrets/PII (SSNs, card numbers, API keys, private keys, `password:`) forces `confidential` (fail closed). Editors can override the tag at upload; admins can re-tag later. |
| **Hybrid search** | Cosine vector search + BM25 (rank-bm25), fused with reciprocal rank fusion. |
| **Answers** | `qwen3.5:4b` by default (switch to `qwen3.5:9b` with an env var), told to cite every claim as `[n]`. Citations show the passage, document, chunk and which retriever found it. |
| **Knowledge graph** | Entities + relations extracted per chunk by the local LLM (JSON mode), stored in SQLite, served through NetworkX. Removing a document removes its mentions, relations and any entity no longer mentioned anywhere. |
| **Audit log** | Logins (and failed logins), every query (question, allowed tags, cited docs, latency), add/remove/re-tag, user changes. |
| **UI** | Next.js: role login, chat with clickable citations, documents (upload / remove / re-tag), admin (users, roles, clearances), audit log, force-directed graph view (click a node → its documents and relations). |

## Requirements

- macOS / Linux with [uv](https://docs.astral.sh/uv/) and Node.js 20+
- Ollama running locally with these models:
  ```bash
  ollama pull qwen3.5:4b
  ollama pull qwen3-embedding:0.6b
  ```

## Run it

**1. Backend** (FastAPI on port 8710)

```bash
cd backend
uv sync
```

```bash
uv run python -m ragvault.ml.compare
```

That trains the classifiers, writes `reports/classifier.md` and saves `models/topic_mlp.pt` + `models/sensitivity_mlp.pt` (≈10–15 min the first time: it downloads AG News and an 11M-parameter BERT, and embeds 10k texts with Ollama). Without these files the API still runs, but topic is `unknown` and sensitivity comes from the regex rules only.

```bash
uv run uvicorn ragvault.api:app --port 8710
```

On first start three demo users are created, **password = username**: `reader`, `editor`, `admin`. Change them in the admin panel before using real data.

**2. Frontend** (Next.js on port 3710)

```bash
cd frontend
npm install
```

```bash
npm run dev -- --port 3710
```

Open http://localhost:3710, click **editor**, and upload the files in `samples/` (a public launch post, an internal roadmap, an internal security handbook and confidential board minutes). Then sign in as **reader** and ask *"What salary was approved for the new CTO?"*: the reader gets no answer, the editor does.

**Load the samples from the command line** (optional, backend must be running):

```bash
cd backend && uv run python -m ragvault.load_samples
```

## Tests

```bash
cd backend && uv run pytest -q
```

The tests run offline (a hashing embedder and a fake LLM replace Ollama). They cover:

- `test_reader_cannot_retrieve_restricted_doc`: a restricted document never shows up in a reader's citations, answer, document list, document detail or graph, while an editor does retrieve it
- `test_filter_is_applied_before_retrieval`: the raw vector and BM25 retrievers only return chunks with allowed tags
- `test_removing_doc_removes_chunks_and_graph_nodes`: chunks, mentions, relations and orphaned entities are deleted; entities shared with other documents stay
- role enforcement (403/401), clearance changes, manual tag override, audit entries for add/query/remove, citations, chunking, secret-pattern escalation, LLM-output validation

## Configuration

All settings are environment variables (see `backend/src/ragvault/config.py`):

| Variable | Default |
|---|---|
| `RAGVAULT_CHAT_MODEL` | `qwen3.5:4b` (use `qwen3.5:9b` for better answers) |
| `RAGVAULT_EMBED_MODEL` | `qwen3-embedding:0.6b` (changing it requires re-training the classifier and re-ingesting) |
| `RAGVAULT_OLLAMA_URL` | `http://localhost:11434` |
| `RAGVAULT_DATA_DIR` | `./data` (SQLite, LanceDB, uploaded files) |
| `RAGVAULT_JWT_SECRET` | random, generated once into `data/.jwt_secret` |
| `RAGVAULT_GRAPH_MAX_CHUNKS` | `8` chunks per document sent to the LLM for entity extraction |
| `RAGVAULT_FAKE_MODELS` | set to `1` to run the API without Ollama (hashing embedder + fake LLM) |
| `NEXT_PUBLIC_API_URL` (frontend) | `http://localhost:8710` |

## Layout

```
backend/src/ragvault/
  api.py        FastAPI routes, auth dependencies
  vault.py      ingest / remove / re-tag / query service
  auth.py       roles, permissions, clearances, password hashing, JWT
  store.py      LanceDB chunk store with tag pre-filtering
  search.py     hybrid retrieval (vector + BM25 + RRF) and cited answers
  graph.py      LLM entity/relation extraction, SQLite graph, NetworkX export
  classify.py   runtime topic/sensitivity classifier + secret/PII rules
  parse.py      PDF/MD/TXT extraction and chunking
  llm.py        Ollama clients + offline fakes
  db.py         SQLite schema: users, documents, audit, entities, mentions, relations
  ml/           classifier experiments (data, PyTorch models, comparison script)
backend/tests/  pytest suite
frontend/       Next.js UI
reports/        classifier.md (accuracy / F1 comparison)
samples/        demo documents
```

## Classifier report

See [`reports/classifier.md`](reports/classifier.md): TF-IDF + logistic regression vs an MLP on frozen
`qwen3-embedding` vectors vs a fine-tuned small BERT, on AG News, plus the sensitivity classifier.

## Limitations / next steps

- The sensitivity classifier is trained on synthetic template data (no public dataset exists for this); the regex rules are the safety net. Training on real labeled company documents is the obvious next step.
- BM25 is rebuilt per query over the permitted chunks: fine for thousands of chunks, not millions (switch to LanceDB's full-text index with a tag filter at that point).
- Entity extraction looks at the first `RAGVAULT_GRAPH_MAX_CHUNKS` chunks of each document to keep ingest fast, and entity de-duplication is by normalised name only.
- Auth is a simple local user table with JWTs (no SSO); access is per tag, not per document or per group.
