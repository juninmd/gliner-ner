from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from gliner_ner import main


@pytest.fixture
def fake_model(monkeypatch):
    model = MagicMock()
    lula = {"text": "Lula", "label": "pessoa", "score": 0.98, "start": 0, "end": 4}
    model.predict_entities.return_value = [lula]
    # Batched inference returns one list per input text, in order.
    model.inference.side_effect = lambda texts, *_a, **_k: [
        [lula] if "Lula" in t else [] for t in texts
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
    fake_model.inference.assert_called_once()


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
    fake_model.inference.assert_not_called()


def test_blank_labels_fall_back_to_defaults_and_others_are_trimmed(client, fake_model):
    client.post("/extract", json={"text": "Lula", "labels": ["  ", ""]})
    assert fake_model.inference.call_args.args[1] == main.DEFAULT_LABELS
    client.post("/extract", json={"text": "Lula", "labels": [" pessoa ", "pessoa", "local"]})
    assert fake_model.inference.call_args.args[1] == ["pessoa", "local"]


def test_labels_are_bounded(client):
    too_many = [f"l{i}" for i in range(main.MAX_LABELS + 1)]
    assert client.post("/extract", json={"text": "x", "labels": too_many}).status_code == 422
    too_long = ["a" * (main.MAX_LABEL_LENGTH + 1)]
    assert client.post("/extract", json={"text": "x", "labels": too_long}).status_code == 422


@pytest.mark.parametrize("blank", [" ", "\n\t  "])
def test_extract_rejects_blank_text(client, fake_model, blank):
    res = client.post("/extract", json={"text": blank})
    assert res.status_code == 422
    fake_model.inference.assert_not_called()


def test_extract_batch_rejects_blank_item_text(client, fake_model):
    res = client.post("/extract/batch", json={"items": [{"id": "a", "text": "   "}]})
    assert res.status_code == 422
    fake_model.inference.assert_not_called()


def test_extract_batch_uses_one_batched_model_call_and_keeps_order(client, fake_model):
    items = [{"id": "a", "text": "Lula esteve em Franca"}, {"id": "b", "text": "nada aqui"}]
    res = client.post("/extract/batch", json={"items": items, "labels": ["pessoa"]})
    assert res.status_code == 200
    assert res.json()["results"] == {
        "a": [{"text": "Lula", "label": "pessoa", "score": 0.98, "start": 0, "end": 4}],
        "b": [],
    }
    fake_model.inference.assert_called_once()
    assert fake_model.inference.call_args.args[0] == ["Lula esteve em Franca", "nada aqui"]
    assert fake_model.inference.call_args.kwargs["batch_size"] == main.INFERENCE_BATCH_SIZE
    fake_model.predict_entities.assert_not_called()


def _find_maria(texts, *_a, **_k):
    # Fake model: finds "Maria" in each window it receives, with offsets local to the window.
    out = []
    for t in texts:
        i = t.find("Maria")
        out.append(
            []
            if i < 0
            else [{"text": "Maria", "label": "pessoa", "score": 0.9, "start": i, "end": i + 5}]
        )
    return out


def test_long_text_is_read_to_the_end_in_windows(client, fake_model):
    # GLiNER drops every word past its max_len (384) without error: the end of a long text vanished.
    words = ["ab"] * 900
    words[850] = "Maria"
    text = " ".join(words)
    fake_model.inference.side_effect = _find_maria
    res = client.post("/extract", json={"text": text})
    assert res.status_code == 200
    [found] = res.json()["entities"]
    assert text[found["start"] : found["end"]] == "Maria"
    windows = fake_model.inference.call_args.args[0]
    assert len(windows) > 1
    assert all(len(w.split()) <= main.CHUNK_WORDS for w in windows)


def test_entity_in_window_overlap_is_reported_once(client, fake_model):
    words = ["ab"] * 400
    overlap_word = main.CHUNK_WORDS - main.CHUNK_OVERLAP + 5  # in the first and second window
    words[overlap_word] = "Maria"
    fake_model.inference.side_effect = _find_maria
    res = client.post(
        "/extract/batch",
        json={"items": [{"id": "a", "text": " ".join(words)}, {"id": "b", "text": "Maria"}]},
    )
    results = res.json()["results"]
    assert len(results["a"]) == 1
    assert results["b"] == [
        {"text": "Maria", "label": "pessoa", "score": 0.9, "start": 0, "end": 5}
    ]


@pytest.mark.parametrize(
    ("url", "body"),
    [
        ("/extract", {"text": "Lula esteve em Franca"}),
        ("/extract/batch", {"items": [{"id": "a", "text": "Lula esteve em Franca"}]}),
    ],
)
def test_null_threshold_uses_the_default(client, fake_model, url, body):
    # Clients often serialize an unset option as null; that means "use the default", not 422.
    res = client.post(url, json={**body, "threshold": None})
    assert res.status_code == 200
    assert fake_model.inference.call_args.kwargs["threshold"] == main.DEFAULT_THRESHOLD


def test_zero_threshold_is_kept(client, fake_model):
    res = client.post("/extract", json={"text": "Lula", "threshold": 0})
    assert res.status_code == 200
    assert fake_model.inference.call_args.kwargs["threshold"] == 0
