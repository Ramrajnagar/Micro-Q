"""FastAPI router for Micro-Q endpoints."""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from src.bayesian_opt.optimizer import BayesianOptimizer
from src.bayesian_opt.search_space import DEFAULT_SPACE
from src.config import settings
from src.paper_qa.agent import build_first_trial_card
from src.paper_qa.ingestion import PaperStore
from src.reasoning.diary import generate_diary_entry

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)


class IngestionRequest(BaseModel):
    query: str = "perovskite solar cell fabrication parameters"
    k: int = 5


class OptimizeRequest(BaseModel):
    n_initial: int = 8
    n_rounds: int = 3
    candidates_per_round: int = 3


class DiaryRequest(BaseModel):
    observations_X: list[list[float]]
    observations_Y: list[list[float]]
    selected_candidates: list[dict[str, Any]]
    rejected_alternatives: list[dict[str, Any]]


class PipelineRequest(BaseModel):
    ingestion_query: str = "perovskite solar cell fabrication parameters"
    n_initial: int = 8
    n_rounds: int = 3
    candidates_per_round: int = 3
    generate_diary: bool = True


def _get_llm():
    if not settings.llm_available:
        raise HTTPException(status_code=503, detail="LLM_API_KEY not configured")
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_endpoint,
        temperature=0.3,
    )


def _get_embeddings():
    if not settings.llm_available:
        raise HTTPException(status_code=503, detail="LLM_API_KEY not configured for embeddings")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_endpoint,
    )


@router.post("/ingest", summary="Ingest papers and extract First Trial Card")
async def ingest_papers(req: IngestionRequest):
    """Run the RAG pipeline: ingest papers, retrieve context, extract a FirstTrialCard."""
    llm = _get_llm()
    embeddings = _get_embeddings()

    store = PaperStore(embedding_fn=embeddings)
    ingested = store.ingest_all_papers()
    if not ingested:
        raise HTTPException(status_code=404, detail="No papers found in data/papers/")

    docs = store.retrieve(req.query, k=req.k)
    if not docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    card = build_first_trial_card(llm, docs, user_query=req.query)
    return {
        "ingested_files": ingested,
        "retrieved_chunks": len(docs),
        "first_trial_card": card.model_dump(mode="json"),
    }


@router.post("/optimize", summary="Run Bayesian optimization loop")
async def optimize(req: OptimizeRequest):
    """Run multi-objective Bayesian optimization on the default material search space."""
    optimizer = BayesianOptimizer(search_space=DEFAULT_SPACE)
    result = optimizer.run_optimization_loop(
        n_initial=req.n_initial,
        n_rounds=req.n_rounds,
        candidates_per_round=req.candidates_per_round,
    )
    return result


@router.post("/diary", summary="Generate scientific reasoning diary entry")
async def diary(req: DiaryRequest):
    """Generate an LLM-powered scientific diary entry explaining optimization choices."""
    llm = _get_llm()
    param_names = [p.name for p in DEFAULT_SPACE.parameters]
    obj_names = [o.name for o in DEFAULT_SPACE.objectives]

    entry = generate_diary_entry(
        llm=llm,
        X=req.observations_X,
        Y=req.observations_Y,
        selected_candidates=req.selected_candidates,
        rejected_alternatives=req.rejected_alternatives,
        param_names=param_names,
        obj_names=obj_names,
    )
    return {"diary_entry": entry}


@router.post("/pipeline", summary="Run the full Micro-Q pipeline end-to-end")
async def run_pipeline(req: PipelineRequest):
    """Run the complete pipeline: ingest → optimize → diary."""
    llm = _get_llm()
    embeddings = _get_embeddings()

    # Step 1: Ingest papers and build first trial card
    store = PaperStore(embedding_fn=embeddings)
    ingested = store.ingest_all_papers()
    docs = store.retrieve(req.ingestion_query, k=5)
    first_trial_card = None
    if docs:
        try:
            card = build_first_trial_card(llm, docs, user_query=req.ingestion_query)
            first_trial_card = card.model_dump(mode="json")
        except Exception as exc:
            logger.warning("First trial card extraction failed: %s", exc)

    # Step 2: Bayesian optimization
    optimizer = BayesianOptimizer(search_space=DEFAULT_SPACE)
    opt_result = optimizer.run_optimization_loop(
        n_initial=req.n_initial,
        n_rounds=req.n_rounds,
        candidates_per_round=req.candidates_per_round,
    )

    # Step 3: Generate diary entry
    diary_entry = None
    if req.generate_diary:
        try:
            # Use the last round's candidates as selected, earlier rounds as rejected
            all_X = opt_result["final_X"]
            all_Y = opt_result["final_Y"]
            last_round = opt_result["rounds"][-1]
            selected = [
                {"params": p, "objectives": o}
                for p, o in zip(last_round["suggested_params"], last_round["suggested_objectives"])
            ]
            rejected = []
            for rnd in opt_result["rounds"][:-1]:
                for p, o in zip(rnd["suggested_params"], rnd["suggested_objectives"]):
                    rejected.append({"params": p, "objectives": o})

            diary_entry = generate_diary_entry(
                llm=llm,
                X=all_X,
                Y=all_Y,
                selected_candidates=selected,
                rejected_alternatives=rejected,
                param_names=[p.name for p in DEFAULT_SPACE.parameters],
                obj_names=[o.name for o in DEFAULT_SPACE.objectives],
            )
        except Exception as exc:
            logger.warning("Diary generation failed: %s", exc)

    # Save results to disk
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pipeline_result.json", "w") as f:
        json.dump({
            "first_trial_card": first_trial_card,
            "optimization": opt_result,
            "diary_entry": diary_entry,
        }, f, indent=2)

    return {
        "ingested_files": list(ingested.keys()),
        "first_trial_card": first_trial_card,
        "optimization": {
            "n_rounds": len(opt_result["rounds"]),
            "n_total_observations": len(opt_result["final_X"]),
            "pareto_frontier": opt_result["pareto_frontier"],
            "rounds": opt_result["rounds"],
        },
        "diary_entry": diary_entry,
    }
