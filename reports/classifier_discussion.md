## 3. Discussion

**Topic.** All three approaches land within 1.5 points of each other on AG News, and the ranking is the expected
one: the fine-tuned transformer is best (macro-F1 0.895), the MLP on frozen `qwen3-embedding` vectors is close
behind (0.888), and TF-IDF + logistic regression is a strong baseline (0.880). The confusion matrices show the same
weak spot for every model: **Business vs Sci/Tech** (tech-company and markets stories), which is where most of the
errors are. Sports is almost never confused.

**Cost matters more than the last point of F1.** The fine-tuned BERT needed ~11 minutes of MPS training and is the
slowest at inference (it runs its own forward pass per document, ~3.7 s per 1k texts here on a busy machine). The
embedding MLP trains in under a minute, has ~0.3M parameters, and at ingest time it re-uses the embeddings RAG
Vault already computes for retrieval, so classification adds almost no latency. That is why **the runtime pipeline
uses the embedding MLP** (`models/topic_mlp.pt`): it gives 99% of the fine-tuned model's F1 for close to zero extra
compute and no second model to keep in memory.

**Sensitivity.** On held-out templates the embedding MLP clearly beats TF-IDF (macro-F1 0.72 vs 0.59): TF-IDF keys
on exact words from the training templates, while embeddings carry over to new phrasings. The errors are
informative: confidential recall is 100% (good: the costly mistake is leaking something confidential), but many
*internal* documents are predicted *public*. Because of that, the runtime classifier does not rely on the model
alone: the **most sensitive chunk decides the document's label** and **regex rules for secrets and PII** (SSNs,
card numbers, API keys, private keys, `password:`) plus explicit markers ("confidential", "internal use only")
escalate the label. Editors can override the tag at upload and admins can re-tag later; every change is audited.

**Limitations and next steps.**
- The sensitivity data is synthetic (42 templates). The best next step is labelling a few hundred real documents
  and re-training; the training code takes any `(text, label)` list.
- Results are from one seed on an 8k/2k subset, so differences under ~1 point are within noise; the full 120k
  training set would raise all three (TF-IDF and the transformer the most).
- A larger fine-tuned model is supported (`--transformer Qwen/Qwen3-0.6B --unfreeze-layers 6`), but on a 16 GB
  Mac shared with an LLM it ran out of memory headroom, so the reported transformer is the 11M-parameter BERT-mini.
