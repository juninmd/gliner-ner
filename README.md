# gliner-ner

Shared zero-shot NER extraction service ([GLiNER](https://github.com/urchade/GLiNER)) built with FastAPI, for reuse across apps in the cluster.

## Run

```bash
uv sync
PYTHONPATH=src uv run uvicorn gliner_ner.main:app --port 8080
```

Or with Docker: `docker build -t gliner-ner . && docker run -p 8080:8080 gliner-ner`.

The model is downloaded on startup (cache in `HF_HOME`). `/health` reports `loading` until it is ready; extract routes return `503` meanwhile.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status": "ok" \| "loading", "model": "..."}` |
| POST | `/extract` | `{"text", "labels"?, "threshold"?}` -> `{"entities": [{text, label, score, start, end}]}` |
| POST | `/extract/batch` | `{"items": [{"id", "text"}], "labels"?, "threshold"?}` -> `{"results": {id: [entities]}}` |

```bash
curl -s localhost:8080/extract -H 'content-type: application/json' \
  -d '{"text": "Lula esteve em Franca", "labels": ["pessoa", "local"]}'
```

## Configuration (environment)

| Variable | Default | Purpose |
|---|---|---|
| `GLINER_MODEL` | `urchade/gliner_multi-v2.1` | Hugging Face model id |
| `GLINER_DEFAULT_LABELS` | `pessoa,organização,local,empresa` | Labels used when the request omits `labels` |
| `GLINER_DEFAULT_THRESHOLD` | `0.5` | Score threshold when the request omits `threshold` |
| `GLINER_MAX_TEXT_LENGTH` | `5000` | Max characters per text |
| `GLINER_MAX_BATCH_SIZE` | `50` | Max items per batch request |

## Development

```bash
uv sync
uv run ruff check .
PYTHONPATH=src uv run pytest
```
