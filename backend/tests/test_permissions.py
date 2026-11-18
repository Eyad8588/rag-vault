from conftest import login, upload

SECRET = (
    "Project Falcon salary bands. Engineer Kestrel salary is 240000 dollars. "
    "The Falcon acquisition codename is Osprey and the budget is secret."
)
PUBLIC = "Project Falcon public announcement. Falcon ships a new mobile app called Sparrow next month."


def test_reader_cannot_retrieve_restricted_doc(client, vault):
    ed = login(client, "editor")
    secret = upload(client, ed, "salaries.md", SECRET).json()
    public = upload(client, ed, "launch.md", PUBLIC).json()
    assert secret["access_tag"] == "restricted"  # classifier label -> default access tag
    assert public["access_tag"] == "public"

    rd = login(client, "reader")
    r = client.post("/query", headers=rd, json={"question": "What is the Kestrel salary in Project Falcon?"})
    assert r.status_code == 200
    cited = {c["doc_id"] for c in r.json()["citations"]}
    assert secret["id"] not in cited
    assert all("240000" not in c["text"] for c in r.json()["citations"])
    assert "240000" not in r.json()["answer"]

    # Not in the doc list, not fetchable directly, not in the graph.
    assert secret["id"] not in {d["id"] for d in client.get("/documents", headers=rd).json()}
    assert client.get(f"/documents/{secret['id']}", headers=rd).status_code == 404
    graph_docs = {d["id"] for n in client.get("/graph", headers=rd).json()["nodes"] for d in n["docs"]}
    assert secret["id"] not in graph_docs

    # The editor (cleared for restricted) does get it.
    r = client.post("/query", headers=ed, json={"question": "What is the Kestrel salary in Project Falcon?"})
    assert secret["id"] in {c["doc_id"] for c in r.json()["citations"]}


def test_filter_is_applied_before_retrieval(vault):
    """Even the raw retrievers never see chunks outside the allowed tags."""
    reader = vault.db.get_user("reader")
    editor = vault.db.get_user("editor")
    vault.ingest(editor, "s.md", SECRET.encode(), build_graph=False)
    vault.ingest(editor, "p.md", PUBLIC.encode(), build_graph=False)
    from ragvault.auth import readable_tags

    tags = readable_tags(reader)
    qv = vault.embedder.embed(["Kestrel salary"])[0]
    assert all(h["access_tag"] in tags for h in vault.store.vector_search(qv, tags, k=50))
    assert all(h["access_tag"] in tags for h in vault.store.all_permitted(tags))
    assert vault.store.all_permitted([]) == []


def test_role_enforcement(client):
    rd, ed = login(client, "reader"), login(client, "editor")
    assert upload(client, rd, "x.txt", "hello world").status_code == 403
    doc = upload(client, ed, "x.txt", "hello world").json()
    assert client.delete(f"/documents/{doc['id']}", headers=ed).status_code == 403
    assert client.get("/admin/users", headers=ed).status_code == 403
    assert client.get("/admin/audit", headers=rd).status_code == 403
    assert client.post("/query", json={"question": "hi"}).status_code == 401


def test_admin_can_change_clearance(client):
    ed, ad = login(client, "editor"), login(client, "admin")
    secret = upload(client, ed, "salaries.md", SECRET).json()
    client.patch("/admin/users/reader", headers=ad, json={"clearances": ["public", "internal", "restricted"]})
    rd = login(client, "reader")
    assert secret["id"] in {d["id"] for d in client.get("/documents", headers=rd).json()}


def test_manual_access_tag_override(client):
    ed, rd = login(client, "editor"), login(client, "reader")
    doc = upload(client, ed, "launch.md", PUBLIC, access_tag="restricted").json()
    assert doc["access_tag"] == "restricted" and doc["tag_source"] == "manual"
    assert doc["id"] not in {d["id"] for d in client.get("/documents", headers=rd).json()}
