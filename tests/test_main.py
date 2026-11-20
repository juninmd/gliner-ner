from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from gliner_ner import main


@pytest.fixture
def fake_model(monkeypatch):
    model = MagicMock()
    model.predict_entities.return_value = [
        {"text": "Lula", "label": "pessoa", "score": 0.98, "start": 0, "end": 4}
    ]
    # Patched before the app starts, so the lifespan's from_pretrained call
    # never touches the network or downloads real model weights in CI.
    monkeypatch.setattr(main.GLiNER, "from_pretrained", lambda *_a, **_k: model)
    yield model


@pytest.fixture
def client(fake_model):
    with TestClient(main.app) as c:
        yield c


def test_health_reports_ok_when_model_loaded(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_extract_returns_entities(client, fake_model):
    res = client.post("/extract", json={"text": "Lula esteve em Franca"})
    assert res.status_code == 200
    body = res.json()
    assert body["entities"][0]["text"] == "Lula"
    assert body["entities"][0]["label"] == "pessoa"
    fake_model.predict_entities.assert_called_once()


def test_extract_rejects_empty_text(client):
    res = client.post("/extract", json={"text": ""})
    assert res.status_code == 422


def test_extract_batch_maps_results_by_id(client, fake_model):
    res = client.post(
        "/extract/batch",
        json={"items": [{"id": "a1", "text": "Lula esteve em Franca"}]},
    )
    assert res.status_code == 200
    body = res.json()
    assert "a1" in body["results"]
    assert body["results"]["a1"][0]["text"] == "Lula"


def test_extract_batch_rejects_more_than_max_size(client):
    items = [{"id": str(i), "text": "x"} for i in range(main.MAX_BATCH_SIZE + 1)]
    res = client.post("/extract/batch", json={"items": items})
    assert res.status_code == 422


def test_extract_returns_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(main, "_model", None)
    res = client.post("/extract", json={"text": "x"})
    assert res.status_code == 503


def test_extract_batch_rejects_duplicate_ids(client, fake_model):
    items = [{"id": "a", "text": "um"}, {"id": "a", "text": "dois"}]
    res = client.post("/extract/batch", json={"items": items})
    assert res.status_code == 422
    fake_model.predict_entities.assert_not_called()


def test_blank_labels_fall_back_to_defaults_and_others_are_trimmed(client, fake_model):
    client.post("/extract", json={"text": "Lula", "labels": ["  ", ""]})
    assert fake_model.predict_entities.call_args.args[1] == main.DEFAULT_LABELS
    client.post("/extract", json={"text": "Lula", "labels": [" pessoa ", "pessoa", "local"]})
    assert fake_model.predict_entities.call_args.args[1] == ["pessoa", "local"]


def test_labels_are_bounded(client):
    too_many = [f"l{i}" for i in range(main.MAX_LABELS + 1)]
    assert client.post("/extract", json={"text": "x", "labels": too_many}).status_code == 422
    too_long = ["a" * (main.MAX_LABEL_LENGTH + 1)]
    assert client.post("/extract", json={"text": "x", "labels": too_long}).status_code == 422


@pytest.mark.parametrize("blank", [" ", "\n\t  "])
def test_extract_rejects_blank_text(client, fake_model, blank):
    res = client.post("/extract", json={"text": blank})
    assert res.status_code == 422
    fake_model.predict_entities.assert_not_called()


def test_extract_batch_rejects_blank_item_text(client, fake_model):
    res = client.post("/extract/batch", json={"items": [{"id": "a", "text": "   "}]})
    assert res.status_code == 422
    fake_model.predict_entities.assert_not_called()
