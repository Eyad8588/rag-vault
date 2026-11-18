"""Upload the demo documents in samples/ through the running API (as the `editor` demo user).

Usage: uv run python -m ragvault.load_samples [--api http://localhost:8710] [--user editor --password editor]
"""

from __future__ import annotations

import argparse

import httpx

from .config import PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8710")
    ap.add_argument("--user", default="editor")
    ap.add_argument("--password", default="editor")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.api, timeout=600)
    token = c.post("/auth/login", json={"username": args.user, "password": args.password}).raise_for_status().json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    for f in sorted((PROJECT_ROOT / "samples").iterdir()):
        if f.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        r = c.post("/documents", headers=h, files={"file": (f.name, f.read_bytes())})
        if r.status_code != 200:
            print(f"✗ {f.name}: {r.status_code} {r.text}")
            continue
        d = r.json()
        print(f"✓ {f.name:28s} topic={d['topic']:<9} sensitivity={d['sensitivity']:<12} -> {d['access_tag']}  "
              f"({d['n_chunks']} chunks)")
    print("Entity extraction runs in the background; watch the status column on the Documents page.")


if __name__ == "__main__":
    main()
