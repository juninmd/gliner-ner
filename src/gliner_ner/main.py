import os
import re
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from gliner import GLiNER
from pydantic import AfterValidator, BaseModel, Field, field_validator

MODEL_NAME = os.environ.get("GLINER_MODEL", "urchade/gliner_multi-v2.1")
DEFAULT_LABELS = [
    l.strip()
    for l in os.environ.get("GLINER_DEFAULT_LABELS", "pessoa,organização,local,empresa").split(",")
    if l.strip()
]
DEFAULT_THRESHOLD = float(os.environ.get("GLINER_DEFAULT_THRESHOLD", "0.5"))
MAX_TEXT_LENGTH = int(os.environ.get("GLINER_MAX_TEXT_LENGTH", "5000"))
MAX_BATCH_SIZE = int(os.environ.get("GLINER_MAX_BATCH_SIZE", "50"))
MAX_LABELS = int(os.environ.get("GLINER_MAX_LABELS", "50"))
MAX_LABEL_LENGTH = 100
INFERENCE_BATCH_SIZE = int(os.environ.get("GLINER_INFERENCE_BATCH_SIZE", "8"))
# GLiNER silently drops every word past its max_len (384); long texts are read in windows
# well under it (subword tokens and labels also count), overlapping so no entity is cut.
CHUNK_WORDS = 200
CHUNK_OVERLAP = 30
_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")  # same word splitter GLiNER uses

_model: GLiNER | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = GLiNER.from_pretrained(MODEL_NAME)
    yield
    _model = None


app = FastAPI(
    title="gliner-ner",
    description="Shared zero-shot NER extraction service, reusable across apps.",
    lifespan=lifespan,
)


def _clean_labels(labels: list[str] | None) -> list[str] | None:
    """Trim, drop blanks and repeats; an empty result means "use the default labels"."""
    if labels is None:
        return None
    cleaned = list(dict.fromkeys(label.strip() for label in labels if label.strip()))
    if any(len(label) > MAX_LABEL_LENGTH for label in cleaned):
        raise ValueError(f"labels must have at most {MAX_LABEL_LENGTH} characters")
    return cleaned or None


def _non_blank(text: str) -> str:
    # Whitespace-only text has nothing to extract but would still cost a model call.
    if not text.strip():
        raise ValueError("text must not be blank")
    return text


Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LENGTH), AfterValidator(_non_blank)]
Labels = Annotated[list[str] | None, Field(max_length=MAX_LABELS), AfterValidator(_clean_labels)]


class ExtractRequest(BaseModel):
    text: Text
    labels: Labels = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class Entity(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


class ExtractResponse(BaseModel):
    entities: list[Entity]


class BatchItem(BaseModel):
    id: str
    text: Text


class BatchExtractRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    labels: Labels = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("items")
    @classmethod
    def _unique_ids(cls, items: list[BatchItem]) -> list[BatchItem]:
        # Results are keyed by id: a repeat would silently overwrite an earlier item's entities.
        if len({item.id for item in items}) != len(items):
            raise ValueError("items[].id must be unique")
        return items


class BatchExtractResponse(BaseModel):
    results: dict[str, list[Entity]]


def _require_model() -> GLiNER:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _model


def _entities(raw: list[dict]) -> list[Entity]:
    return [
        Entity(
            text=e["text"],
            label=e["label"],
            score=e["score"],
            start=e["start"],
            end=e["end"],
        )
        for e in raw
    ]


def _windows(text: str) -> list[tuple[int, str]]:
    """(offset, piece) windows of at most CHUNK_WORDS words covering the whole text."""
    spans = [m.span() for m in _WORD.finditer(text)]
    if len(spans) <= CHUNK_WORDS:
        return [(0, text)]
    windows = []
    for first in range(0, len(spans), CHUNK_WORDS - CHUNK_OVERLAP):
        part = spans[first : first + CHUNK_WORDS]
        start, end = part[0][0], part[-1][1]
        windows.append((start, text[start:end]))
        if first + CHUNK_WORDS >= len(spans):
            break
    return windows


def _run_batch(
    model: GLiNER, texts: list[str], labels: list[str] | None, threshold: float | None
) -> list[list[Entity]]:
    # One batched forward pass over every window of every text instead of one call per item.
    windows = [_windows(text) for text in texts]
    raw = iter(
        model.inference(
            [piece for item in windows for _, piece in item],
            labels or DEFAULT_LABELS,
            threshold=threshold if threshold is not None else DEFAULT_THRESHOLD,
            batch_size=INFERENCE_BATCH_SIZE,
        )
    )
    results = []
    for item in windows:
        # Overlapping windows can find the same span twice: keep the best score.
        best: dict[tuple[int, int, str], dict] = {}
        for offset, _ in item:
            for e in next(raw):
                key = (e["start"] + offset, e["end"] + offset, e["label"])
                if key not in best or e["score"] > best[key]["score"]:
                    best[key] = {**e, "start": key[0], "end": key[1]}
        results.append(_entities(sorted(best.values(), key=lambda e: (e["start"], e["end"]))))
    return results


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "model": MODEL_NAME}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    model = _require_model()
    [entities] = _run_batch(model, [req.text], req.labels, req.threshold)
    return ExtractResponse(entities=entities)


@app.post("/extract/batch", response_model=BatchExtractResponse)
def extract_batch(req: BatchExtractRequest):
    model = _require_model()
    entities = _run_batch(model, [item.text for item in req.items], req.labels, req.threshold)
    return BatchExtractResponse(
        results={item.id: found for item, found in zip(req.items, entities, strict=True)}
    )
