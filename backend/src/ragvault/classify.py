"""Runtime document classifier: topic + sensitivity from the chunk embeddings.

Uses the PyTorch MLP heads trained by `python -m ragvault.ml.compare` (models/*.pt). A regex check for
secrets/PII escalates sensitivity to 'confidential' regardless of the model (fail closed). If the
model files are missing, topic is 'unknown' and sensitivity comes from the regex + keyword rules."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch

from .ml.data import SENSITIVITY_LABELS
from .ml.models import EmbeddingMLP

log = logging.getLogger(__name__)

SECRET_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                       # US SSN
    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),                    # card number
    re.compile(r"\b(?:sk|pk|ghp|xox[bp])[-_][A-Za-z0-9]{16,}\b"),  # API tokens
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bpassword\s*[:=]\s*\S+", re.I),
]
CONFIDENTIAL_WORDS = re.compile(r"\b(strictly confidential|confidential|do not distribute|privileged|"
                                r"eyes only|embargoed)\b", re.I)
INTERNAL_WORDS = re.compile(r"\b(internal only|internal use|for internal|not for publication)\b", re.I)


def _load(path: Path):
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = EmbeddingMLP(ck["in_dim"], len(ck["labels"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


class DocClassifier:
    def __init__(self, models_dir: Path):
        self.topic = _load(models_dir / "topic_mlp.pt")
        self.sens = _load(models_dir / "sensitivity_mlp.pt")
        if not self.topic or not self.sens:
            log.warning("Classifier models missing in %s: run `uv run python -m ragvault.ml.compare`", models_dir)

    def _probs(self, loaded, emb: np.ndarray) -> np.ndarray | None:
        if not loaded or emb.shape[1] != loaded[1]["in_dim"]:
            return None
        with torch.no_grad():
            return torch.softmax(loaded[0](torch.as_tensor(emb, dtype=torch.float32)), dim=1).numpy()

    def classify(self, chunks: list[str], embeddings: np.ndarray) -> dict:
        """Document-level labels. Topic = argmax of the mean chunk probabilities; sensitivity = the most
        sensitive chunk wins (one confidential paragraph makes the document confidential)."""
        out = {"topic": "unknown", "topic_conf": 0.0, "sensitivity": "public", "sens_conf": 0.0, "reasons": []}
        tp = self._probs(self.topic, embeddings)
        if tp is not None:
            mean = tp.mean(0)
            out["topic"], out["topic_conf"] = self.topic[1]["labels"][int(mean.argmax())], float(mean.max())

        level, conf = 0, 0.0
        sp = self._probs(self.sens, embeddings)
        if sp is not None:
            per_chunk = sp.argmax(1)
            level = int(per_chunk.max())
            conf = float(sp[per_chunk == level, level].max())
            out["reasons"].append("model")
        full = "\n".join(chunks)
        if any(p.search(full) for p in SECRET_PATTERNS):
            level, conf = 2, 1.0
            out["reasons"].append("secret/PII pattern")
        elif CONFIDENTIAL_WORDS.search(full) and level < 2:
            level, conf = 2, max(conf, 0.9)
            out["reasons"].append("confidential marker")
        elif INTERNAL_WORDS.search(full) and level < 1:
            level, conf = 1, max(conf, 0.9)
            out["reasons"].append("internal marker")
        out["sensitivity"], out["sens_conf"] = SENSITIVITY_LABELS[level], conf
        return out
