import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from gliner import GLiNER
from pydantic import AfterValidator, BaseModel, Field, field_validator

MODEL_NAME = os.environ.get("GLINER_MODEL", "urchade/gliner_multi-v2.1")
DEFAULT_LABELS = [
    l.strip()
    for l in os.environ.get(
        "GLINER_DEFAULT_LABELS", "pessoa,organização,local,empresa"
    ).split(",")
    if l.strip()
]
DEFAULT_THRESHOLD = float(os.environ.get("GLINER_DEFAULT_THRESHOLD", "0.5"))
MAX_TEXT_LENGTH = int(os.environ.get("GLINER_MAX_TEXT_LENGTH", "5000"))
MAX_BATCH_SIZE = int(os.environ.get("GLINER_MAX_BATCH_SIZE", "50"))
MAX_LABELS = int(os.environ.get("GLINER_MAX_LABELS", "50"))
MAX_LABEL_LENGTH = 100

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
    threshold: float = Field(default=None, ge=0.0, le=1.0)  # type: ignore[assignment]


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
    threshold: float = Field(default=None, ge=0.0, le=1.0)  # type: ignore[assignment]

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


def _run(model: GLiNER, text: str, labels: list[str] | None, threshold: float | None) -> list[Entity]:
    raw = model.predict_entities(
        text,
        labels or DEFAULT_LABELS,
        threshold=threshold if threshold is not None else DEFAULT_THRESHOLD,
    )
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


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "model": MODEL_NAME}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    model = _require_model()
    return ExtractResponse(entities=_run(model, req.text, req.labels, req.threshold))


@app.post("/extract/batch", response_model=BatchExtractResponse)
def extract_batch(req: BatchExtractRequest):
    model = _require_model()
    results = {
        item.id: _run(model, item.text, req.labels, req.threshold) for item in req.items
    }
    return BatchExtractResponse(results=results)
