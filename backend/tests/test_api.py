"""End-to-end tests for the FastAPI layer using a fake local corpus."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ina_core.store import DEFAULT_HF_PARQUET_FILES


def _make_test_parquet(path: Path, n: int, channel: str, year: int):
    df = pd.DataFrame({
        "source_file": [path.name] * n,
        "title": [f"Inflation {('hausse' if i % 2 == 0 else 'baisse')} {i}" for i in range(n)],
        "_title_norm": [
            f"inflation {('hausse' if i % 2 == 0 else 'baisse')} {i}" for i in range(n)
        ],
        "_date": pd.date_range(f"{year}-01-01", periods=n, freq="MS"),
        "_channel": [channel] * n,
    })
    df.to_parquet(path)


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Build a fake project root with a minimal local snapshot + dictionary."""
    project_root = tmp_path
    snapshot = project_root / "data" / "clean" / "v_test"
    snapshot.mkdir(parents=True)

    # One small parquet per expected filename. They share the same fake data.
    for i, fname in enumerate(DEFAULT_HF_PARQUET_FILES):
        _make_test_parquet(snapshot / fname, n=12, channel=f"chaine_{i}", year=2020 + i)

    (project_root / "data" / "clean" / "CURRENT").write_text("v_test")

    # Seed dictionaries.json with one theme
    dictionary = {
        "inflation": {
            "concept": ["inflation"],
            "context": [],
            "up": ["hausse"],
            "down": ["baisse"],
        }
    }
    (project_root / "dictionaries.json").write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setenv("INA_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("INA_HF_REPO_ID", "test/fake-repo")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return project_root


@pytest.fixture
def client(app_env):
    # Import after env is set so the app builds with the right settings.
    from ina_api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


# ─── Health & metadata ──────────────────────────────────────────────────────


def test_health_returns_ok_when_corpus_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["data_loaded"] is True
    assert body["n_rows"] > 0
    assert body["source"] == "local"


def test_metadata_returns_corpus_summary(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] > 0
    assert body["date_min"] is not None
    assert len(body["channels"]) == len(DEFAULT_HF_PARQUET_FILES)


# ─── Themes CRUD ────────────────────────────────────────────────────────────


def test_list_themes_returns_seeded_inflation(client):
    r = client.get("/themes")
    assert r.status_code == 200
    names = [t["name"] for t in r.json()]
    assert "inflation" in names


def test_get_theme_returns_full_dictionary(client):
    r = client.get("/themes/inflation")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "inflation"
    assert "inflation" in body["concept"]


def test_get_theme_404_when_unknown(client):
    r = client.get("/themes/does_not_exist")
    assert r.status_code == 404


def test_create_theme_then_retrieve(client):
    r = client.post("/themes", json={
        "name": "chomage",
        "concept": ["chomage", "unedic"],
        "up": ["hausse"],
        "down": ["baisse"],
    })
    assert r.status_code == 201
    assert r.json()["name"] == "chomage"

    r = client.get("/themes/chomage")
    assert r.status_code == 200
    assert r.json()["concept"] == ["chomage", "unedic"]


def test_create_existing_theme_returns_409(client):
    r = client.post("/themes", json={"name": "inflation"})
    assert r.status_code == 409


def test_update_theme_replaces_concept(client):
    r = client.put("/themes/inflation", json={"concept": ["ipc"]})
    assert r.status_code == 200
    assert r.json()["concept"] == ["ipc"]


def test_delete_theme_blocks_when_only_one(client):
    r = client.delete("/themes/inflation")
    assert r.status_code == 400  # cannot delete the last theme


def test_delete_theme_when_more_than_one(client):
    client.post("/themes", json={"name": "chomage"})
    r = client.delete("/themes/chomage")
    assert r.status_code == 204
    assert client.get("/themes/chomage").status_code == 404


# ─── Channels & coverage ────────────────────────────────────────────────────


def test_channels_returns_one_per_channel(client):
    r = client.get("/channels")
    assert r.status_code == 200
    assert len(r.json()) == len(DEFAULT_HF_PARQUET_FILES)


def test_coverage_returns_decade_distribution(client):
    r = client.get("/coverage")
    assert r.status_code == 200
    body = r.json()
    # Each row has the expected keys
    if body:
        assert {"channel", "decade", "decade_label", "n", "pct"} <= set(body[0].keys())


# ─── Analysis ───────────────────────────────────────────────────────────────


def test_analysis_basic_flow(client):
    r = client.post("/analysis", json={
        "theme": "inflation",
        "frequency": "monthly",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["kpi"]["n_total"] > 0
    assert isinstance(body["series"], list)
    assert isinstance(body["top_channels"], list)
    assert len(body["descriptive"]) == 12


def test_analysis_unknown_theme_returns_404(client):
    r = client.post("/analysis", json={"theme": "ghost"})
    assert r.status_code == 404


def test_analysis_filter_by_channels(client):
    r = client.post("/analysis", json={
        "theme": "inflation",
        "channels": ["chaine_0"],
    })
    assert r.status_code == 200
    # Only one channel selected → top_channels is at most 1 row
    assert len(r.json()["top_channels"]) == 1


def test_analysis_invalid_dates_rejected(client):
    r = client.post("/analysis", json={
        "theme": "inflation",
        "date_start": "2025-01-01",
        "date_end": "2020-01-01",
    })
    assert r.status_code == 422  # Pydantic validation error


def test_analysis_creates_then_uses_new_theme(client):
    """End-to-end: create theme via API, then analyze it."""
    client.post("/themes", json={
        "name": "test_theme",
        "concept": ["inflation"],
    })
    r = client.post("/analysis", json={"theme": "test_theme"})
    assert r.status_code == 200


def test_analysis_theme_without_concept_returns_400(client):
    client.post("/themes", json={"name": "empty_theme"})
    r = client.post("/analysis", json={"theme": "empty_theme"})
    assert r.status_code == 400


# ─── Export CSV ─────────────────────────────────────────────────────────────


def test_export_csv_returns_csv_attachment(client):
    r = client.get("/export/csv?theme=inflation&frequency=monthly")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    body = r.text
    # Header row must be present
    assert "date" in body.split("\n")[0]
    assert "frequence" in body.split("\n")[0]


def test_export_csv_unknown_theme_returns_404(client):
    r = client.get("/export/csv?theme=ghost")
    assert r.status_code == 404


# ─── Cache invalidation ─────────────────────────────────────────────────────


def test_updating_theme_invalidates_cache(client):
    """After PUT /themes, the next /analysis must reflect new keywords."""
    # First analysis with the seeded keywords
    r1 = client.post("/analysis", json={"theme": "inflation"})
    matched_before = r1.json()["kpi"]["n_matched"]

    # Update with a keyword that won't match anything
    client.put("/themes/inflation", json={"concept": ["xyz_noise_term"]})

    # Second analysis must re-tag and find different counts
    r2 = client.post("/analysis", json={"theme": "inflation"})
    matched_after = r2.json()["kpi"]["n_matched"]
    assert matched_after != matched_before
