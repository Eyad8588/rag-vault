"""Runtime settings. Everything can be overridden with RAGVAULT_* environment variables."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # rag-vault/


def _env(name: str, default: str) -> str:
    return os.environ.get(f"RAGVAULT_{name}", default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", str(PROJECT_ROOT / "models"))))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "qwen3.5:4b"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "qwen3-embedding:0.6b"))
    jwt_secret: str = field(default_factory=lambda: _env("JWT_SECRET", ""))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "900")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))
    # Max chunks per document sent to the LLM for entity extraction (keeps ingest fast).
    graph_max_chunks: int = field(default_factory=lambda: int(_env("GRAPH_MAX_CHUNKS", "8")))
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "http://localhost:3710,http://127.0.0.1:3710"))

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "ragvault.db"

    @property
    def lancedb_path(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.uploads_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)
        if not self.jwt_secret:  # no secret configured: create a random one once and keep it in data/
            f = self.data_dir / ".jwt_secret"
            if not f.exists():
                f.write_text(secrets.token_hex(32))
                f.chmod(0o600)
            self.jwt_secret = f.read_text().strip()
