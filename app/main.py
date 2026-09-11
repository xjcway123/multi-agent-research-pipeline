from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.observability import LANGFUSE_ENABLED, flush as langfuse_flush
from app.routes import research

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Multi-Agent Research Pipeline",
    description="LangGraph + CrewAI research pipeline with multi-source routing, "
                "SSE streaming, and Langfuse observability.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research.router, prefix="/research", tags=["research"])


@app.on_event("shutdown")
def _flush_observability() -> None:
    """Push any pending Langfuse events before the process exits."""
    if LANGFUSE_ENABLED:
        langfuse_flush()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}
