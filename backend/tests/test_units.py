import numpy as np

from ragvault.classify import DocClassifier
from ragvault.graph import extract
from ragvault.parse import chunk_text
from ragvault.search import tokenize


def test_chunking_respects_size_and_covers_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = chunk_text(text, size=500, overlap=80)
    assert len(chunks) > 3
    assert all(len(c) <= 500 + 80 + 2 for c in chunks)
    for i in range(20):
        assert any(f"Paragraph {i}." in c for c in chunks)


def test_secret_patterns_force_confidential(tmp_path):
    clf = DocClassifier(tmp_path)  # no model files -> rules only
    emb = np.zeros((1, 8), dtype=np.float32)
    assert clf.classify(["Release notes for the app."], emb)["sensitivity"] == "public"
    assert clf.classify(["SSN 123-45-6789 on file"], emb)["sensitivity"] == "confidential"
    assert clf.classify(["db password: hunter2"], emb)["sensitivity"] == "confidential"
    assert clf.classify(["For internal use only: roadmap"], emb)["sensitivity"] == "internal"


def test_extract_drops_relations_to_unknown_entities():
    class LLM:
        def chat(self, messages, json_mode=False, temperature=0.0):
            return ('{"entities":[{"name":"Orion Labs","type":"organization"},{"name":"Cairo","type":"place"}],'
                    '"relations":[{"source":"Orion Labs","target":"Cairo","relation":"based in"},'
                    '{"source":"Orion Labs","target":"Mars","relation":"visits"}]}')

    out = extract(LLM(), "Orion Labs is based in Cairo.")
    assert [e["type"] for e in out["entities"]] == ["organization", "concept"]
    assert out["relations"] == [{"source": "Orion Labs", "target": "Cairo", "relation": "based in"}]


def test_tokenize_drops_stopwords():
    assert tokenize("What is the Kestrel salary?") == ["kestrel", "salary"]
