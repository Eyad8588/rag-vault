"""Local model clients (Ollama) plus deterministic fakes used by the test-suite."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

import httpx
import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LLM(Protocol):
    def chat(self, messages: list[dict], json_mode: bool = False, temperature: float = 0.2) -> str: ...


class OllamaEmbedder:
    def __init__(self, url: str, model: str, batch_size: int = 32, timeout: float = 300):
        self.url, self.model, self.batch_size = url.rstrip("/"), model, batch_size
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:6000] for t in texts[i : i + self.batch_size]]
            r = self._client.post(f"{self.url}/api/embed", json={"model": self.model, "input": batch,
                                                                  "keep_alive": "10m"})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-8, None)


class OllamaLLM:
    def __init__(self, url: str, model: str, timeout: float = 600):
        self.url, self.model = url.rstrip("/"), model
        self._client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], json_mode: bool = False, temperature: float = 0.2) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,  # qwen3.5 is a thinking model; we want direct answers
            "keep_alive": "10m",
            "options": {"temperature": temperature, "num_ctx": 8192},
        }
        if json_mode:
            body["format"] = "json"
        r = self._client.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return strip_think(text)


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ---------------------------------------------------------------------------
# Fakes: no network, deterministic. Used in tests and when RAGVAULT_FAKE_MODELS=1.
# ---------------------------------------------------------------------------
class HashEmbedder:
    """Bag-of-words hashed into a fixed-size vector. Similar texts get similar vectors."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                arr[i, h % self.dim] += 1.0
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-8, None)


class FakeLLM:
    """Extracts capitalised words as entities; answers by quoting the first source."""

    def chat(self, messages: list[dict], json_mode: bool = False, temperature: float = 0.2) -> str:
        prompt = messages[-1]["content"]
        if json_mode:
            text = prompt.split("TEXT:", 1)[-1]
            names = list(dict.fromkeys(re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text)))[:6]
            ents = [{"name": n, "type": "concept"} for n in names]
            rels = [{"source": a, "target": b, "relation": "related_to"} for a, b in zip(names, names[1:])]
            return json.dumps({"entities": ents, "relations": rels})
        m = re.search(r"\[(\d+)\]", prompt)
        return f"Based on the sources [{m.group(1) if m else 1}]."
