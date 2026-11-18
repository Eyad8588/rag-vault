"""FastAPI app. Run: uv run uvicorn ragvault.api:app --port 8710"""

from __future__ import annotations

import logging
import os
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .auth import ACCESS_TAGS, DEFAULT_CLEARANCES, can, decode_token, hash_password, make_token, verify_password
from .config import Settings
from .llm import FakeLLM, HashEmbedder, OllamaEmbedder, OllamaLLM
from .vault import PermissionDenied, Vault

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class LoginIn(BaseModel):
    username: str
    password: str


class TagIn(BaseModel):
    access_tag: Literal["public", "internal", "restricted"]


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=6, ge=1, le=20)


class UserIn(BaseModel):
    username: str = Field(min_length=2, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=4)
    role: Literal["reader", "editor", "admin"]
    clearances: list[Literal["public", "internal", "restricted"]] | None = None


class UserPatch(BaseModel):
    role: Literal["reader", "editor", "admin"] | None = None
    clearances: list[Literal["public", "internal", "restricted"]] | None = None
    password: str | None = Field(default=None, min_length=4)


def build_vault(settings: Settings | None = None) -> Vault:
    s = settings or Settings()
    if os.environ.get("RAGVAULT_FAKE_MODELS") == "1":
        v = Vault(s, HashEmbedder(), FakeLLM())
    else:
        v = Vault(s, OllamaEmbedder(s.ollama_url, s.embed_model), OllamaLLM(s.ollama_url, s.chat_model))
    v.seed_users()
    return v


def create_app(vault: Vault | None = None, background_graph: bool = True) -> FastAPI:
    app = FastAPI(title="RAG Vault", version="0.1.0")
    app.state.vault = vault or build_vault()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in app.state.vault.s.cors_origins.split(",")],
        allow_methods=["*"], allow_headers=["*"],
    )

    def get_vault() -> Vault:
        return app.state.vault

    def current_user(authorization: str = Header(default=""), v: Vault = Depends(get_vault)) -> dict:
        token = authorization.removeprefix("Bearer ").strip()
        name = decode_token(token, v.s.jwt_secret) if token else None
        user = v.db.get_user(name) if name else None
        if not user:
            raise HTTPException(401, "Not authenticated")
        return user

    def require(perm: str):
        def dep(user: dict = Depends(current_user)) -> dict:
            if not can(user["role"], perm):
                raise HTTPException(403, f"Role '{user['role']}' cannot {perm}")
            return user
        return dep

    def public_user(u: dict) -> dict:
        return {"username": u["username"], "role": u["role"], "clearances": u["clearances"]}

    # ---------------------------------------------------------------- auth

    @app.post("/auth/login")
    def login(body: LoginIn, v: Vault = Depends(get_vault)):
        user = v.db.get_user(body.username)
        if not user or not verify_password(body.password, user["password_hash"]):
            v.db.audit(body.username, None, "login_failed")
            raise HTTPException(401, "Wrong username or password")
        v.db.audit(user["username"], user["role"], "login")
        return {"token": make_token(user["username"], v.s.jwt_secret), "user": public_user(user)}

    @app.get("/auth/me")
    def me(user: dict = Depends(current_user)):
        return public_user(user)

    @app.get("/health")
    def health(v: Vault = Depends(get_vault)):
        return {"ok": True, "chat_model": v.s.chat_model, "embed_model": v.s.embed_model,
                "classifier_loaded": bool(v.classifier.topic and v.classifier.sens)}

    # ---------------------------------------------------------------- documents
    @app.get("/documents")
    def list_documents(user: dict = Depends(current_user), v: Vault = Depends(get_vault)):
        return v.visible_docs(user)

    @app.post("/documents")
    async def upload(background: BackgroundTasks, file: UploadFile = File(...),
                     access_tag: str | None = Form(default=None),
                     user: dict = Depends(require("add_doc")), v: Vault = Depends(get_vault)):
        data = await file.read()
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(413, "File too large (max 25 MB)")
        try:
            doc = v.ingest(user, file.filename or "upload.txt", data, access_tag or None)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if background_graph:
            background.add_task(v.extract_graph, doc["id"])
        else:
            v.extract_graph(doc["id"])
            doc = v.get_doc(doc["id"])
        return doc

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str, user: dict = Depends(current_user), v: Vault = Depends(get_vault)):
        docs = {d["id"]: d for d in v.visible_docs(user)}
        if doc_id not in docs:  # same 404 for "missing" and "not allowed": don't leak existence
            raise HTTPException(404, "Document not found")
        chunks = v.store.chunks_for_doc(doc_id)
        return docs[doc_id] | {"chunks": [{"index": c["chunk_index"], "text": c["text"]} for c in chunks]}

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, user: dict = Depends(require("remove_doc")), v: Vault = Depends(get_vault)):
        try:
            v.remove(user, doc_id)
        except KeyError:
            raise HTTPException(404, "Document not found")
        return {"ok": True}


    @app.patch("/documents/{doc_id}")
    def retag(doc_id: str, body: TagIn, user: dict = Depends(require("remove_doc")), v: Vault = Depends(get_vault)):
        try:
            return v.set_access_tag(user, doc_id, body.access_tag)
        except KeyError:
            raise HTTPException(404, "Document not found")

    # ---------------------------------------------------------------- query / graph

    @app.post("/query")
    def query(body: QueryIn, user: dict = Depends(require("query")), v: Vault = Depends(get_vault)):
        return v.query(user, body.question, body.k)

    @app.get("/graph")
    def graph(user: dict = Depends(current_user), v: Vault = Depends(get_vault)):
        return v.graph(user)

    # ---------------------------------------------------------------- admin


    @app.get("/admin/users")
    def users(user: dict = Depends(require("manage_users")), v: Vault = Depends(get_vault)):
        return v.db.list_users()

    @app.post("/admin/users")
    def create_user(body: UserIn, user: dict = Depends(require("manage_users")), v: Vault = Depends(get_vault)):
        if v.db.get_user(body.username):
            raise HTTPException(409, "User already exists")
        cl = body.clearances if body.clearances is not None else DEFAULT_CLEARANCES[body.role]
        v.db.upsert_user(body.username, hash_password(body.password), body.role, cl)
        v.db.audit(user["username"], user["role"], "create_user", body.username, {"role": body.role, "clearances": cl})
        return public_user(v.db.get_user(body.username))

    @app.patch("/admin/users/{username}")
    def update_user(username: str, body: UserPatch, user: dict = Depends(require("manage_users")),
                    v: Vault = Depends(get_vault)):
        target = v.db.get_user(username)
        if not target:
            raise HTTPException(404, "User not found")
        if username == user["username"] and body.role and body.role != "admin":
            raise HTTPException(400, "You can't remove your own admin role")
        v.db.update_user(username, body.role, body.clearances)
        if body.password:
            v.db.upsert_user(username, hash_password(body.password), body.role or target["role"],
                             body.clearances if body.clearances is not None else target["clearances"])
        v.db.audit(user["username"], user["role"], "update_user", username,
                   {"role": body.role, "clearances": body.clearances, "password_changed": bool(body.password)})
        return public_user(v.db.get_user(username))

    @app.delete("/admin/users/{username}")
    def delete_user(username: str, user: dict = Depends(require("manage_users")), v: Vault = Depends(get_vault)):
        if username == user["username"]:
            raise HTTPException(400, "You can't delete yourself")
        if not v.db.get_user(username):
            raise HTTPException(404, "User not found")
        v.db.delete_user(username)
        v.db.audit(user["username"], user["role"], "delete_user", username)
        return {"ok": True}

    @app.get("/admin/audit")
    def audit(limit: int = 200, action: str | None = None, user: dict = Depends(require("view_audit")),
              v: Vault = Depends(get_vault)):
        return v.db.list_audit(min(limit, 1000), action)

    @app.get("/meta")
    def meta():
        return {"access_tags": list(ACCESS_TAGS), "roles": ["reader", "editor", "admin"]}

    return app


def __getattr__(name: str):
    # Lazily build the app so importing this module (e.g. in tests) doesn't touch Ollama or ./data.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
