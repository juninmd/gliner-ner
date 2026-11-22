# AGENTS.md

gliner-ner: a zero-shot named-entity extraction HTTP service on GLiNER and FastAPI. One process loads one model at startup, so several cluster apps can share it. Public docs are in Portuguese and English (`README.md`).

## Stack
- Python >= 3.12, FastAPI, Uvicorn, Pydantic, GLiNER (default model `urchade/gliner_multi-v2.1`), PyTorch (CPU wheels from the `pytorch-cpu` index)
- `uv` for dependencies (`pyproject.toml`, `uv.lock`), Ruff for lint, pytest for tests
- Docker image on `python:3.12-slim`

## Commands
- Install: `uv sync`
- Run locally: `PYTHONPATH=src uv run uvicorn gliner_ner.main:app --port 8080`
- Run in Docker: `docker build -t gliner-ner . && docker run --rm -p 8080:8080 gliner-ner`
- Test: `uv run --frozen pytest -q`
- Lint: `uv run --frozen ruff check .`
- CI (`build.yml`): `uv sync --locked`, then the Ruff and pytest commands above

## Structure
- `src/gliner_ner/main.py`: the FastAPI app, request models, validation and the model
- `src/gliner_ner/__init__.py`: package marker
- `tests/test_main.py`: API tests. The model is mocked, so tests need no download
- `docs/readme/`: README images. `Dockerfile`, `pyproject.toml`, `uv.lock`

## Conventions
- The API contract lives in `README.md` (`/health`, `/extract`, `/extract/batch`). Change the README in the same PR as any request or response change
- Validation uses Pydantic bounds. Limits come from `GLINER_*` env vars, not hardcoded values
- Regenerate `uv.lock` with `uv`. Do not edit it by hand
- Keep `ruff check` clean. Do not suppress rules without a comment that explains why

## Gotchas
- **Model download:** the first start downloads the model from Hugging Face. Set `HF_HOME` to a persistent volume, because the image uses `/tmp/hf-cache`, which is not a volume
- **Mocked tests:** `tests/test_main.py` asserts the response shape only. Scores in the README (`0.98`) are fixture values, not benchmarks
- **Env vars:** `GLINER_MODEL`, `GLINER_DEFAULT_LABELS`, `GLINER_DEFAULT_THRESHOLD`, `GLINER_MAX_TEXT_LENGTH`, `GLINER_MAX_LABELS`, `GLINER_MAX_BATCH_SIZE` (50), `GLINER_INFERENCE_BATCH_SIZE`
- **Offsets:** `start` and `end` are character offsets into the submitted text

<!-- base-actions-main-policy -->
## GitHub Actions — padrão obrigatório (@main)

- Referencie as actions internas sempre pelo branch, nunca por SHA:
  - `uses: juninmd/base-actions/<action>@main`
  - `uses: juninmd/base-actions/.github/workflows/<arquivo>.yml@main`
- Não fixe `@<sha>` em `juninmd/base-actions`: a atualização é centralizada no
  `main` e passa a valer no próximo run.
- PRs cujos checks não iniciam por billing/spending limit são mesclados pelo bot central
  `juninmd/base-actions/.github/workflows/auto-merge-billing-blocked.yml`.
  Para impedir o merge automático de um PR, aplique a label `do-not-merge`.