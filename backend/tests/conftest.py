"""Tests run fully offline: a hashing embedder and a fake LLM replace Ollama, and every test gets a
fresh data directory. The real PyTorch classifier is replaced by a stub so labels are deterministic."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ragvault.api import create_app
from ragvault.config import Settings
from ragvault.llm import FakeLLM, HashEmbedder
from ragvault.vault import Vault


class StubClassifier:
    topic = sens = True

    def classify(self, chunks, embeddings):
        text = " ".join(chunks).lower()
        sens = "confidential" if "salary" in text or "secret" in text else (
            "internal" if "roadmap" in text else "public")
        return {"topic": "Business", "topic_conf": 0.9, "sensitivity": sens, "sens_conf": 0.9, "reasons": ["stub"]}


@pytest.fixture
def vault(tmp_path):
    s = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    v = Vault(s, HashEmbedder(), FakeLLM(), classifier=StubClassifier())
    v.seed_users()
    return v


@pytest.fixture
def client(vault):
    return TestClient(create_app(vault, background_graph=False))


def login(client, name: str) -> dict:
    r = client.post("/auth/login", json={"username": name, "password": name})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def upload(client, headers, filename: str, text: str, access_tag: str | None = None):
    data = {"access_tag": access_tag} if access_tag else {}
    return client.post("/documents", headers=headers, files={"file": (filename, text.encode(), "text/plain")}, data=data)
