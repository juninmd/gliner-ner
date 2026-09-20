FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONPATH=/app/src \
    HF_HOME=/tmp/hf-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

COPY src/ src/

EXPOSE 8080

CMD ["uv", "run", "--no-dev", "--no-editable", "uvicorn", "gliner_ner.main:app", \
     "--host=0.0.0.0", "--port=8080"]
