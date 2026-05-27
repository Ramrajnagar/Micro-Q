"""LLM-powered scientific reasoning diary generator."""

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings
from src.reasoning.prompts import DIARY_SYSTEM_PROMPT, DIARY_USER_PROMPT

logger = logging.getLogger(__name__)


def format_obs_table(X, Y, param_names, obj_names) -> str:
    """Format observations into a readable table string."""
    lines = [f"| {' | '.join(param_names)} | {' | '.join(obj_names)} |"]
    lines.append(f"|{'|'.join([' --- ' for _ in range(len(param_names) + len(obj_names))])}|")
    for xi, yi in zip(X, Y):
        x_str = " | ".join(f"{v:.4f}" for v in xi)
        y_str = " | ".join(f"{v:.4f}" for v in yi)
        lines.append(f"| {x_str} | {y_str} |")
    return "\n".join(lines)


def format_candidate_table(candidates, obj_names) -> str:
    """Format candidate suggestions into a table."""
    lines = [f"| Candidate | {' | '.join(obj_names)} |"]
    lines.append(f"|{'|'.join([' --- ' for _ in range(len(obj_names) + 1)])}|")
    for i, cand in enumerate(candidates):
        y_str = " | ".join(f"{v:.4f}" for v in cand["objectives"])
        lines.append(f"| Candidate {i+1} | {y_str} |")
    return "\n".join(lines)


def format_alternatives(alternatives, param_names, obj_names) -> str:
    """Format rejected alternatives into a table."""
    lines = [f"| {' | '.join(param_names)} | {' | '.join(obj_names)} |"]
    lines.append(f"|{'|'.join([' --- ' for _ in range(len(param_names) + len(obj_names) + 1)])}|")
    for i, alt in enumerate(alternatives):
        x_str = " | ".join(f"{v:.4f}" for v in alt["params"])
        y_str = " | ".join(f"{v:.4f}" for v in alt["objectives"])
        lines.append(f"| {x_str} | {y_str} |")
    return "\n".join(lines)


def generate_diary_entry(
    llm: BaseChatModel,
    X: list[list[float]],
    Y: list[list[float]],
    selected_candidates: list[dict],
    rejected_alternatives: list[dict],
    param_names: list[str],
    obj_names: list[str],
    ref_point: list[float] | None = None,
) -> str:
    """Generate a scientific diary entry explaining the optimization choices."""
    obs_table = format_obs_table(X, Y, param_names, obj_names)
    cand_table = format_candidate_table(selected_candidates, obj_names)
    alt_table = format_alternatives(rejected_alternatives, param_names, obj_names)

    ref_str = str(ref_point) if ref_point else "auto-computed from observed data"
    user_message = DIARY_USER_PROMPT.format(
        n_obs=len(X),
        observations=obs_table,
        candidates=cand_table,
        alternatives=alt_table,
        ref_point=ref_str,
    )

    messages = [
        ("system", DIARY_SYSTEM_PROMPT),
        ("human", user_message),
    ]

    response = llm.invoke(messages)
    return response.content.strip()
