"""Classifier comparison: TF-IDF+LogReg vs MLP on frozen embeddings vs a fine-tuned small transformer.

Run:  uv run python -m ragvault.ml.compare            (from backend/)
      uv run python -m ragvault.ml.compare --quick    (smaller subset, for a smoke test)

Writes reports/classifier.md + reports/classifier_results.json and saves the runtime models
(models/topic_mlp.pt, models/sensitivity_mlp.pt) used by the ingest pipeline.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from ..config import PROJECT_ROOT, Settings
from ..llm import OllamaEmbedder
from .data import SENSITIVITY_LABELS, TOPIC_LABELS, load_ag_news, make_sensitivity_dataset
from .models import best_device, train_mlp

REPORTS = PROJECT_ROOT / "reports"


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
def run_tfidf(xtr, ytr, xte, yte) -> dict:
    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    clf = LogisticRegression(max_iter=2000, C=5.0)
    clf.fit(vec.fit_transform(xtr), ytr)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = clf.predict(vec.transform(xte))
    infer_s = time.perf_counter() - t0
    return metrics(yte, pred) | {"train_s": train_s, "infer_ms_per_1k": 1000 * infer_s / len(xte) * 1000}


def embed_cached(embedder: OllamaEmbedder, texts: list[str], cache: Path) -> tuple[np.ndarray, float]:
    if cache.exists():
        arr = np.load(cache)
        if len(arr) == len(texts):
            return arr, float("nan")
    t0 = time.perf_counter()
    arr = embedder.embed(texts)
    secs = time.perf_counter() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr, secs


def run_mlp(etr, ytr, ete, yte, n_classes) -> tuple[dict, torch.nn.Module]:
    xa, xv, ya, yv = train_test_split(etr, ytr, test_size=0.1, random_state=0, stratify=ytr)
    t0 = time.perf_counter()
    model, hist = train_mlp(xa, ya, xv, yv, n_classes)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(torch.as_tensor(ete)).argmax(1).numpy()
    infer_s = time.perf_counter() - t0
    return metrics(yte, pred) | {"train_s": train_s, "epochs": len(hist),
                                 "infer_ms_per_1k": 1000 * infer_s / len(yte) * 1000}, model


def _layer_stack(model) -> torch.nn.ModuleList:
    """The main stack of transformer blocks (model.model.layers for Qwen, bert.encoder.layer for BERT)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.split(".")[-1] in ("layers", "layer", "h"):
            if best is None or len(mod) > len(best):
                best = mod
    return best


def run_transformer(model_name, xtr, ytr, xte, yte, n_classes, epochs=3, lr=1e-4, bs=32, max_len=128,
                    unfreeze_layers=-1) -> dict:
    """Fine-tune a pretrained transformer with a fresh classification head.

    `unfreeze_layers`: train only the top N transformer blocks + final norm + head (-1 = everything).
    With Qwen3-0.6B on a 16 GB Mac, training the top 6 of 28 blocks keeps memory around 4-5 GB."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    device = best_device()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes,
                                                               dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    n_params = sum(p.numel() for p in model.parameters())
    if unfreeze_layers >= 0:
        for p in model.parameters():
            p.requires_grad = False
        stack = _layer_stack(model)
        for block in list(stack)[len(stack) - unfreeze_layers:]:
            for p in block.parameters():
                p.requires_grad = True
        for name, mod in model.named_modules():
            if name.split(".")[-1] in ("norm", "score", "classifier", "pooler", "ln_f"):
                for p in mod.parameters():
                    p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    model.to(device)
    print(f"  [transformer] {model_name}: {n_params / 1e6:.0f}M params, {n_trainable / 1e6:.1f}M trainable")

    use_amp = device.type in ("mps", "cuda")
    y = torch.as_tensor(ytr)
    steps = epochs * ((len(xtr) + bs - 1) // bs)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    t0 = time.perf_counter()
    step = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr)).tolist()
        running = 0.0
        for i in range(0, len(xtr), bs):
            idx = perm[i : i + bs]
            enc = tok([xtr[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(**enc, labels=y[idx].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += out.loss.item() * len(idx)
            step += 1
            if step % 50 == 0:
                print(f"  [transformer] step {step}/{steps} loss={running / min(i + bs, len(xtr)):.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    if device.type == "mps":
        torch.mps.synchronize()
    train_s = time.perf_counter() - t0

    model.eval()
    preds = []
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, len(xte), 64):
            e = tok(xte[i : i + 64], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            preds.extend(model(**e).logits.float().argmax(1).cpu().tolist())
    infer_s = time.perf_counter() - t0
    del model, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return metrics(yte, preds) | {"train_s": train_s, "epochs": epochs, "params_m": n_params / 1e6,
                                  "trainable_m": n_trainable / 1e6, "unfreeze_layers": unfreeze_layers,
                                  "infer_ms_per_1k": 1000 * infer_s / len(xte) * 1000}


# ---------------------------------------------------------------------------
def save_runtime_model(model, path: Path, labels: list[str], embed_model: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = model.net[0]
    torch.save({"state_dict": model.state_dict(), "in_dim": first.in_features, "labels": labels,
                "embed_model": embed_model, **extra}, path)


def fmt_conf(conf, labels) -> str:
    head = "| true \\ pred | " + " | ".join(labels) + " |\n|" + "---|" * (len(labels) + 1) + "\n"
    return head + "\n".join(f"| **{labels[i]}** | " + " | ".join(str(v) for v in row) + " |"
                            for i, row in enumerate(conf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--transformer", default="google/bert_uncased_L-4_H-256_A-4",
                    help="any HF model id, e.g. Qwen/Qwen3-0.6B with --unfreeze-layers 6 (needs ~5 GB free RAM)")
    ap.add_argument("--transformer-epochs", type=int, default=3)
    ap.add_argument("--transformer-lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze-layers", type=int, default=-1, help="train only the top N blocks; -1 = all")
    ap.add_argument("--skip-transformer", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild classifier.md from classifier_results.json")
    args = ap.parse_args()
    if args.report_only:
        write_report(json.loads((REPORTS / "classifier_results.json").read_text()))
        return
    if args.quick:
        args.n_train, args.n_test, args.transformer_epochs = 1200, 400, 1

    s = Settings()
    s.ensure_dirs()
    cache_dir = s.data_dir / "ml_cache"
    embedder = OllamaEmbedder(s.ollama_url, s.embed_model, batch_size=64)
    results: dict = {"config": vars(args) | {"embed_model": s.embed_model, "device": str(best_device()),
                                             "machine": platform.machine(), "torch": torch.__version__}}

    # ------------------------------ Topic (AG News) ------------------------------
    print(f"Loading AG News subset: {args.n_train} train / {args.n_test} test")
    xtr, ytr, xte, yte = load_ag_news(args.n_train, args.n_test, cache_dir)
    topic = {}
    print("[1/3] TF-IDF + logistic regression")
    topic["tfidf_logreg"] = run_tfidf(xtr, ytr, xte, yte)
    print("   ", {k: round(v, 4) for k, v in topic["tfidf_logreg"].items() if k != "confusion"})

    print(f"[2/3] MLP on frozen {s.embed_model} embeddings (embedding {len(xtr) + len(xte)} texts via Ollama)")
    etr, embed_s1 = embed_cached(embedder, xtr, cache_dir / f"ag_train_{args.n_train}.npy")
    ete, embed_s2 = embed_cached(embedder, xte, cache_dir / f"ag_test_{args.n_test}.npy")
    topic["embedding_mlp"], topic_model = run_mlp(etr, np.array(ytr), ete, np.array(yte), len(TOPIC_LABELS))
    topic["embedding_mlp"]["embed_s"] = (embed_s1 + embed_s2) if not np.isnan(embed_s1 + embed_s2) else None
    print("   ", {k: round(v, 4) for k, v in topic["embedding_mlp"].items() if isinstance(v, float)})
    save_runtime_model(topic_model, s.models_dir / "topic_mlp.pt", TOPIC_LABELS, s.embed_model,
                       {"dataset": "ag_news", "test_accuracy": topic["embedding_mlp"]["accuracy"]})

    if not args.skip_transformer:
        print(f"[3/3] Fine-tuning {args.transformer} on {best_device()}")
        topic["finetuned_transformer"] = run_transformer(args.transformer, xtr, ytr, xte, yte, len(TOPIC_LABELS),
                                                         epochs=args.transformer_epochs, lr=args.transformer_lr,
                                                         bs=32,
                                                         unfreeze_layers=args.unfreeze_layers)
        topic["finetuned_transformer"]["model"] = args.transformer
        print("   ", {k: round(v, 4) for k, v in topic["finetuned_transformer"].items() if isinstance(v, float)})
    results["topic"] = topic

    # --------------------------- Sensitivity (synthetic) ---------------------------
    print("Sensitivity: synthetic templates, test on held-out templates")
    s_xtr, s_ytr = make_sensitivity_dataset(300, seed=0, split="train")
    s_xte, s_yte = make_sensitivity_dataset(100, seed=1, split="test")
    sens = {"tfidf_logreg": run_tfidf(s_xtr, s_ytr, s_xte, s_yte)}
    s_etr = embedder.embed(s_xtr)
    s_ete = embedder.embed(s_xte)
    sens["embedding_mlp"], _ = run_mlp(s_etr, np.array(s_ytr), s_ete, np.array(s_yte), len(SENSITIVITY_LABELS))
    results["sensitivity"] = sens
    print("   ", {m: round(r["macro_f1"], 4) for m, r in sens.items()})
    # Runtime sensitivity model: train on every template.
    a_x, a_y = make_sensitivity_dataset(400, seed=2, split="all")
    a_e = embedder.embed(a_x)
    xa, xv, ya, yv = train_test_split(a_e, np.array(a_y), test_size=0.1, random_state=0, stratify=a_y)
    sens_model, _ = train_mlp(xa, ya, xv, yv, len(SENSITIVITY_LABELS))
    save_runtime_model(sens_model, s.models_dir / "sensitivity_mlp.pt", SENSITIVITY_LABELS, s.embed_model,
                       {"dataset": "synthetic_templates"})

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "classifier_results.json").write_text(json.dumps(results, indent=2))
    write_report(results)
    print(f"Wrote {REPORTS / 'classifier.md'}")


def _ft_name(m: dict | None) -> str:
    if not m:
        return ""
    if m.get("unfreeze_layers", -1) >= 0:
        return (f"Fine-tuned `{m['model']}` (top {m['unfreeze_layers']} blocks + head, "
                f"{m['trainable_m']:.0f}M trainable)")
    return f"Fine-tuned `{m['model']}` (all layers)"


def write_report(r: dict) -> None:
    c, topic, sens = r["config"], r["topic"], r["sensitivity"]
    names = {"tfidf_logreg": "TF-IDF (1-2 gram) + logistic regression",
             "embedding_mlp": f"MLP on frozen `{c['embed_model']}` embeddings",
             "finetuned_transformer": _ft_name(topic.get("finetuned_transformer"))}

    best_acc = max(m["accuracy"] for m in topic.values())
    best_f1 = max(m["macro_f1"] for m in topic.values())

    def b(v, best):
        return f"**{v:.4f}**" if v == best else f"{v:.4f}"

    def row(key, m):
        params = f"{m['params_m']:.1f}M" if "params_m" in m else ("~0.3M (head only)" if key == "embedding_mlp" else "")
        return (f"| {names[key]} | {b(m['accuracy'], best_acc)} | {b(m['macro_f1'], best_f1)} | {m['train_s']:.1f}s "
                f"| {m['infer_ms_per_1k']:.0f} ms | {params} |")

    lines = [
        "# RAG Vault: classifier comparison",
        "",
        "_Generated by `uv run python -m ragvault.ml.compare` (backend/). Re-running overwrites the numbers;_",
        "_the discussion comes from `reports/classifier_discussion.md`._",
        "",
        f"Machine: Apple Silicon ({c['machine']}), PyTorch {c['torch']}, device `{c['device']}`.",
        "",
        "## 1. Topic classification: AG News",
        "",
        f"Public dataset [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (4 classes: "
        f"{', '.join(TOPIC_LABELS)}). Balanced random subset: **{c['n_train']} train / {c['n_test']} test** "
        "(test drawn from the official test split). All three methods use exactly the same texts.",
        "",
        "| Method | Accuracy | Macro-F1 | Train time | Inference / 1k docs | Params |",
        "|---|---|---|---|---|---|",
    ]
    lines += [row(k, m) for k, m in topic.items()]
    emb = topic["embedding_mlp"].get("embed_s")
    lines += [
        "",
        "Train/inference time for the MLP excludes computing the embeddings"
        + (f" (embedding all {c['n_train'] + c['n_test']} texts with Ollama took {emb:.0f}s)." if emb else
           " (embeddings were cached from an earlier run)."),
        "In RAG Vault those embeddings are computed anyway for retrieval, so the classifier itself is almost free.",
        "",
    ]
    for k, m in topic.items():
        lines += [f"<details><summary>Confusion matrix: {names[k]}</summary>", "", fmt_conf(m["confusion"], TOPIC_LABELS),
                  "", "</details>", ""]
    lines += [
        "## 2. Sensitivity classification (synthetic)",
        "",
        "No public dataset labels documents as public / internal / confidential, so this uses a small "
        "template-generated set (`ragvault/ml/data.py`). The test set uses **held-out templates** the model never "
        "saw, so it measures generalisation to new phrasings rather than memorisation. Treat these numbers as a "
        "sanity check, not a benchmark.",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---|---|",
    ]
    lines += [f"| {names[k]} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |" for k, m in sens.items()]
    lines += ["", fmt_conf(sens["embedding_mlp"]["confusion"], SENSITIVITY_LABELS), ""]
    disc = REPORTS / "classifier_discussion.md"
    if disc.exists():
        lines += [disc.read_text().strip(), ""]
    (REPORTS / "classifier.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
