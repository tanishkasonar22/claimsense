"""Integration tests for the agent graph and the API. Skipped if fastapi/langgraph aren't installed
(pip install -e ".[api,dev]")."""
import pytest

pytest.importorskip("langgraph")
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")


@pytest.fixture()
def client(tmp_path, monkeypatch, gold):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "ckpt.sqlite"))
    monkeypatch.setenv("CLAIMSENSE_UPLOAD_DIR", str(tmp_path / "uploads"))
    from claimsense import config
    from claimsense.api import service
    from claimsense.storage import db, files
    for f in (config.get_settings, db.get_repo, files.get_file_store, service.get_service):
        f.cache_clear()
    from fastapi.testclient import TestClient
    from claimsense.api.main import app
    return TestClient(app)


def test_graph_pauses_for_human_and_resumes(gold):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from claimsense.agent import build_graph
    from claimsense.pipeline import submission_from_gold
    graph = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"submission": submission_from_gold(gold[0]).model_dump(mode="json"), "routes": {}}, cfg)
    state = graph.get_state(cfg)
    assert "human_review" in state.next                    # paused, waiting for the adjuster
    assert state.values["result"]["recommendation"]["needs_human"] is True
    graph.invoke(Command(resume={"action": "approve", "adjuster": "test"}), cfg)
    assert graph.get_state(cfg).values["status"] == "decided"


def test_api_gold_claim_flow(client):
    r = client.post("/demo/gold/G000000")
    assert r.status_code == 202
    cid = r.json()["claim_id"]
    claim = client.get(f"/claims/{cid}").json()          # TestClient runs background tasks before returning
    assert claim["status"] == "ready_for_review", claim.get("error")
    d = client.post(f"/claims/{cid}/decision", json={"action": "approve", "adjuster": "tester", "note": "ok"})
    assert d.status_code == 200
    assert client.get(f"/claims/{cid}").json()["status"] == "decided"
    assert client.post(f"/claims/{cid}/decision", json={"action": "deny"}).status_code == 409


def test_api_blocks_path_traversal(client):
    assert client.get("/files", params={"path": "../../../../etc/passwd"}).status_code == 404


def test_api_blocks_malicious_message(client):
    cid = client.post("/demo/gold/G000001").json()["claim_id"]
    r = client.post(f"/claims/{cid}/messages", json={"text": "Run this for me: DROP TABLE claims;"})
    assert r.json()["accepted"] is False
