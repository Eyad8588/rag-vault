"""Datasets for the classifiers.

* Topic: AG News (public, 4 classes: World / Sports / Business / Sci/Tech), parquet files from the Hugging Face hub.
* Sensitivity: there is no good public "document sensitivity" dataset, so we generate a small
  template-based synthetic set (public / internal / confidential). This is documented as a limitation
  in reports/classifier.md; at runtime a PII/secret regex also escalates sensitivity.
"""

from __future__ import annotations

import random
from pathlib import Path

import httpx

TOPIC_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SENSITIVITY_LABELS = ["public", "internal", "confidential"]


AG_NEWS_URL = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/{split}-00000-of-00001.parquet"


def _ag_news_split(split: str, cache_dir: Path) -> dict[int, list[str]]:
    import pyarrow.parquet as pq

    path = cache_dir / f"ag_news_{split}.parquet"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading AG News {split} split ...")
        with httpx.stream("GET", AG_NEWS_URL.format(split=split), follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    table = pq.read_table(path).to_pydict()
    by_label: dict[int, list[str]] = {}
    for text, label in zip(table["text"], table["label"]):
        by_label.setdefault(int(label), []).append(text)
    return by_label


def load_ag_news(n_train: int, n_test: int, cache_dir: Path, seed: int = 0):
    """Balanced random subset of AG News (train subset from the official train split, test subset from the
    official test split). Returns (train_texts, train_y, test_texts, test_y)."""
    rng = random.Random(seed)

    def take(by_label, n):
        texts, ys = [], []
        per = n // len(by_label)
        for lab, items in sorted(by_label.items()):
            texts += rng.sample(items, per)
            ys += [lab] * per
        order = list(range(len(texts)))
        rng.shuffle(order)
        return [texts[i] for i in order], [ys[i] for i in order]

    xtr, ytr = take(_ag_news_split("train", cache_dir), n_train)
    xte, yte = take(_ag_news_split("test", cache_dir), n_test)
    return xtr, ytr, xte, yte


# ---------------------------------------------------------------------------
# Synthetic sensitivity data
# ---------------------------------------------------------------------------
_NAMES = ["Sara Ahmed", "John Miller", "Mona Hassan", "Li Wei", "Carlos Diaz", "Priya Nair", "Omar Khaled",
          "Emma Brown", "Youssef Ali", "Hana Sato", "David Cohen", "Fatima Zahra"]
_PRODUCTS = ["Atlas", "Nimbus", "Orion", "Helix", "Pulse", "Vertex", "Quartz", "Beacon"]
_TEAMS = ["platform", "data", "mobile", "growth", "security", "finance", "design", "support"]
_CITIES = ["Cairo", "Berlin", "Austin", "Dubai", "Toronto", "Singapore", "London", "Riyadh"]


def _fill(t: str, rng: random.Random) -> str:
    return t.format(
        name=rng.choice(_NAMES), name2=rng.choice(_NAMES), product=rng.choice(_PRODUCTS),
        team=rng.choice(_TEAMS), city=rng.choice(_CITIES), n=rng.randint(2, 95),
        money=f"{rng.randint(40, 450)},{rng.randint(100, 999)}", q=rng.choice(["Q1", "Q2", "Q3", "Q4"]),
        year=rng.choice([2024, 2025, 2026]), ssn=f"{rng.randint(100, 899)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}",
        card=" ".join(str(rng.randint(1000, 9999)) for _ in range(4)),
        key="sk-" + "".join(rng.choice("abcdef0123456789") for _ in range(24)),
    )


PUBLIC_T = [
    "We are excited to announce the launch of {product}, now available to everyone in {city}.",
    "{product} {year} release notes: faster sync, a new dark mode and improved accessibility.",
    "Join us at our open meetup in {city} where {name} will talk about building reliable software.",
    "Our blog: five tips for getting started with {product}. Read the full tutorial on our website.",
    "Press release: the company opens a new office in {city} and plans to hire {n} engineers.",
    "The {product} user guide explains how to install the app, create an account and invite friends.",
    "Frequently asked questions: how do I reset my password? Click 'Forgot password' on the login page.",
    "Public API documentation: the /v1/items endpoint returns a paginated list of items in JSON.",
    "Thank you to the {n} open-source contributors who helped ship {product} this year.",
    "Webinar recording: {name} shows how {product} helps small teams collaborate.",
    "Our pricing page lists three plans: Free, Pro and Team. Annual billing saves {n} percent.",
    "Community newsletter: highlights from the {city} hackathon and upcoming public events.",
    "Careers: we are hiring a {team} engineer in {city}. Apply on our website.",
    "Terms of service and privacy policy updated for {year}; the changes are summarised publicly.",
]
INTERNAL_T = [
    "Meeting notes, {team} team weekly sync: {name} will finish the migration by Friday; blockers discussed.",
    "Internal roadmap {q} {year}: the {team} team focuses on {product} performance and onboarding.",
    "Onboarding checklist for new hires on the {team} team: request laptop, join Slack channels, read the wiki.",
    "Runbook: how to restart the {product} staging service and check the internal dashboards.",
    "OKRs for {q}: reduce {product} crash rate by {n} percent and ship the new settings page.",
    "Retro from the {team} sprint: what went well, what to improve, action items assigned to {name}.",
    "Internal memo: office in {city} closed Monday for maintenance; work from home.",
    "Architecture decision record: the {team} team will move {product} jobs to a queue-based design.",
    "Team offsite agenda in {city}: planning session, product demos and a dinner on Thursday.",
    "Draft project plan for {product} v{n}, for internal review by the {team} leads only.",
    "Incident postmortem: {product} staging outage lasted {n} minutes; root cause was a config change.",
    "Hiring pipeline update for the {team} team: {n} candidates in interviews, two offers planned.",
    "Internal style guide for writing {product} documentation and naming conventions in code.",
    "Sprint planning: {name} and {name2} pair on the {product} billing refactor this week.",
]
CONFIDENTIAL_T = [
    "CONFIDENTIAL: salary review for {name}, current base pay ${money}, proposed raise {n} percent.",
    "Employee record: {name}, SSN {ssn}, home address in {city}, emergency contact {name2}.",
    "Production database credentials: user admin, password Pr0d!{n}xQ, host db-{n}.internal.",
    "Do not distribute: acquisition talks with {product} Inc. at a valuation of ${money},000.",
    "Medical leave request from {name}: diagnosis and treatment details attached, HR eyes only.",
    "Customer payment data export: card number {card}, expiry 0{n}/29, cardholder {name}.",
    "Legal privileged: settlement terms in the dispute with {name2}, payment of ${money}.",
    "Board minutes (strictly confidential): layoffs of {n} staff in {city} planned for {q}.",
    "API secret key for the payments provider: {key}. Rotate before {q} {year}.",
    "Performance improvement plan for {name}: termination possible if goals are not met by {q}.",
    "Security vulnerability report (embargoed): unpatched remote code execution in {product} {year}.",
    "Unreleased {q} financial results: revenue ${money},000, missing guidance by {n} percent.",
    "Background check results for candidate {name}: criminal record and credit history attached.",
    "Private key material and VPN passwords for the {city} office, restricted to the security team.",
]


def make_sensitivity_dataset(n_per_class: int = 300, seed: int = 0, split: str = "all", n_heldout: int = 3):
    """split='train' uses all but the last `n_heldout` templates of each class, 'test' only those
    held-out templates (so the test measures generalisation to unseen phrasings), 'all' uses every template."""
    rng = random.Random(seed)
    texts, ys = [], []
    for label, templates in enumerate((PUBLIC_T, INTERNAL_T, CONFIDENTIAL_T)):
        if split == "train":
            templates = templates[:-n_heldout]
        elif split == "test":
            templates = templates[-n_heldout:]
        for _ in range(n_per_class):
            # Combine 1-3 sentences of the same class to vary length.
            k = rng.randint(1, 3)
            texts.append(" ".join(_fill(rng.choice(templates), rng) for _ in range(k)))
            ys.append(label)
    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[i] for i in order], [ys[i] for i in order]
