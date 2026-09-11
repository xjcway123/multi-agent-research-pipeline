FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Install runtime deps first so the layer is cached when only code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code.
COPY app ./app

EXPOSE 8000

# Cloud deploys have no Claude CLI on PATH, so the planner falls back to
# the OpenAI-compatible LLM. Set LLM_API_KEY/LLM_BASE_URL/LLM_MODEL via
# the platform's secrets (e.g. `fly secrets set ...`).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
