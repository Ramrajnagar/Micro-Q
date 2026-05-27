#!/usr/bin/env python3
"""CLI runner for the full Micro-Q pipeline.

Usage:
    # Full pipeline (requires LLM_API_KEY for RAG + diary):
    export LLM_API_KEY="sk-..."
    python scripts/run_pipeline.py --full

    # Optimization only (no API key needed):
    python scripts/run_pipeline.py --optimize-only

    # With custom rounds:
    python scripts/run_pipeline.py --optimize-only --rounds 5 --candidates 4
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bayesian_opt.optimizer import BayesianOptimizer
from src.bayesian_opt.search_space import DEFAULT_SPACE
from src.config import settings
from src.reasoning.diary import generate_diary_entry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("micro_q_cli")


def run_optimization(args: argparse.Namespace) -> dict:
    """Run the Bayesian optimization loop."""
    logger.info("Starting optimization: %d initial, %d rounds, %d candidates/round",
                args.n_initial, args.rounds, args.candidates)
    optimizer = BayesianOptimizer(search_space=DEFAULT_SPACE)
    result = optimizer.run_optimization_loop(
        n_initial=args.n_initial,
        n_rounds=args.rounds,
        candidates_per_round=args.candidates,
    )
    logger.info("Optimization complete: %d total observations, %d Pareto points",
                len(result["final_X"]), len(result["pareto_frontier"]))
    return result


def run_diary(result: dict) -> str | None:
    """Generate the scientific reasoning diary entry."""
    if not settings.llm_available:
        logger.warning("LLM_API_KEY not set — skipping diary generation")
        return None

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_endpoint,
        temperature=0.3,
    )

    param_names = [p.name for p in DEFAULT_SPACE.parameters]
    obj_names = [o.name for o in DEFAULT_SPACE.objectives]
    all_X = result["final_X"]
    all_Y = result["final_Y"]
    last_round = result["rounds"][-1]
    selected = [
        {"params": p, "objectives": o}
        for p, o in zip(last_round["suggested_params"], last_round["suggested_objectives"])
    ]
    rejected = []
    for rnd in result["rounds"][:-1]:
        for p, o in zip(rnd["suggested_params"], rnd["suggested_objectives"]):
            rejected.append({"params": p, "objectives": o})

    diary = generate_diary_entry(
        llm=llm,
        X=all_X,
        Y=all_Y,
        selected_candidates=selected,
        rejected_alternatives=rejected,
        param_names=param_names,
        obj_names=obj_names,
    )
    return diary


def main():
    parser = argparse.ArgumentParser(description="Micro-Q — Autonomous Materials Discovery Pipeline")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (requires LLM_API_KEY)")
    parser.add_argument("--optimize-only", action="store_true", default=True, help="Run optimization only (default)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of optimization rounds")
    parser.add_argument("--candidates", type=int, default=3, help="Candidates per round")
    parser.add_argument("--n-initial", type=int, default=8, help="Initial random observations")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    # If --full is passed, override the default
    if args.full:
        args.optimize_only = False

    result = run_optimization(args)

    diary_entry = None
    if args.full:
        diary_entry = run_diary(result)

    output = {
        "pipeline": "full" if args.full else "optimization_only",
        "optimization": result,
        "diary_entry": diary_entry,
    }

    output_path = args.output or str(Path(settings.output_dir) / "pipeline_result.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("MICRO-Q PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Rounds completed:     {args.rounds}")
    print(f"  Total observations:   {len(result['final_X'])}")
    print(f"  Pareto frontier size: {len(result['pareto_frontier'])}")
    print(f"  Search space dim:     {len(DEFAULT_SPACE.parameters)}")
    print(f"  Objectives:           {len(DEFAULT_SPACE.objectives)}")
    if diary_entry:
        print(f"\n  Diary entry ({len(diary_entry)} chars):")
        print(f"  {diary_entry[:200]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
