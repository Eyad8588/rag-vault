from conftest import login, upload

DOC_A = "Nimbus Cloud is built by Orion Labs. Orion Labs is based in Cairo. Nimbus Cloud uses LanceDB."
DOC_B = "Orion Labs hired Mona Hassan. Mona Hassan leads Helix Research in Berlin."


def entity_names(vault):
    return {r["name"] for r in vault.db.query("SELECT name FROM entities")}


def test_removing_doc_removes_chunks_and_graph_nodes(client, vault):
    ed, ad = login(client, "editor"), login(client, "admin")
    a = upload(client, ed, "a.md", DOC_A).json()
    b = upload(client, ed, "b.md", DOC_B).json()
    assert vault.store.count(a["id"]) == a["n_chunks"] > 0
    names = entity_names(vault)
    assert {"Nimbus", "Cairo", "Orion", "Mona", "Helix"} <= names  # FakeLLM: capitalised words
    assert vault.db.one("SELECT COUNT(*) n FROM relations WHERE doc_id=?", (a["id"],))["n"] > 0

    r = client.delete(f"/documents/{a['id']}", headers=ad)
    assert r.status_code == 200

    assert vault.store.count(a["id"]) == 0
    assert vault.store.count(b["id"]) == b["n_chunks"]
    assert vault.db.one("SELECT COUNT(*) n FROM mentions WHERE doc_id=?", (a["id"],))["n"] == 0
    assert vault.db.one("SELECT COUNT(*) n FROM relations WHERE doc_id=?", (a["id"],))["n"] == 0
    names = entity_names(vault)
    assert "Nimbus" not in names and "Cairo" not in names  # only in doc A -> gone
    assert "Orion" in names and "Mona" in names            # shared with / only in doc B -> kept
    graph = client.get("/graph", headers=ad).json()
    assert all(d["id"] != a["id"] for n in graph["nodes"] for d in n["docs"])
    assert client.get(f"/documents/{a['id']}", headers=ad).status_code == 404

    # The removed doc can no longer be retrieved.
    res = client.post("/query", headers=ad, json={"question": "Where is Nimbus Cloud built?"}).json()
    assert a["id"] not in {c["doc_id"] for c in res["citations"]}


def test_audit_log_records_add_remove_query(client):
    ed, ad = login(client, "editor"), login(client, "admin")
    doc = upload(client, ed, "a.md", DOC_A).json()
    client.post("/query", headers=ed, json={"question": "What does Orion Labs build?"})
    client.delete(f"/documents/{doc['id']}", headers=ad)
    log = client.get("/admin/audit", headers=ad).json()
    actions = [(e["action"], e["username"]) for e in log]
    assert ("add_doc", "editor") in actions
    assert ("query", "editor") in actions
    assert ("remove_doc", "admin") in actions
    q = next(e for e in log if e["action"] == "query")
    assert q["details"]["question"] == "What does Orion Labs build?"
    assert doc["id"] in q["details"]["cited_docs"]


def test_query_returns_citations(client):
    ed = login(client, "editor")
    upload(client, ed, "a.md", DOC_A)
    upload(client, ed, "b.md", DOC_B)
    res = client.post("/query", headers=ed, json={"question": "Who leads Helix Research?"}).json()
    assert res["citations"], res
    top = res["citations"][0]
    assert "Helix" in top["text"] and top["title"] and top["n"] == 1
    assert "[1]" in res["answer"]


def test_unsupported_file_rejected(client):
    ed = login(client, "editor")
    r = client.post("/documents", headers=ed, files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
