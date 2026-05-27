"""FastAPI application entry point for Micro-Q."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Micro-Q starting up (LLM available: %s)", settings.llm_available)
    yield
    logger.info("Micro-Q shutting down")


app = FastAPI(
    title="Micro-Q — Autonomous Materials Discovery Pipeline",
    description="A proof-of-concept mimicking Novyte's core product Q: Literature RAG → "
    "Bayesian Optimization → Scientific Reasoning Diary.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "Micro-Q",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "Ingest papers": "POST /api/v1/ingest",
            "Run optimization": "POST /api/v1/optimize",
            "Generate diary": "POST /api/v1/diary",
            "Full pipeline": "POST /api/v1/pipeline",
        },
    }
