"""LangGraph agent that extracts structured FirstTrialCard from papers."""

import json
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from src.paper_qa.models import FirstTrialCard, ExperimentalParameter, Citation

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a materials science research assistant. Your task is to analyze the following
paper excerpts and produce a structured "First Trial Card" that an experimentalist could take into the lab.

Given the paper context below, extract:

1. **Material system** – what class of materials is being studied? (e.g., "Methylammonium lead iodide
   perovskites", "LFP cathode with PEO-based electrolyte")
2. **Experimental parameters** – list every numeric or categorical parameter mentioned (temperature,
   concentration, solvent ratio, annealing time, doping level, etc.). For each, give the value, unit,
   and your confidence (0–1) that this value is clearly stated in the paper.
3. **Target property** – what property is being optimized? (e.g., "power conversion efficiency",
   "ionic conductivity", "tensile strength")
4. **Rationale** – a 1–2 sentence summary of why this formulation was tested.
5. **Citations** – for each parameter, cite the exact sentence that supports it.

Output ONLY valid JSON matching this schema:
{{
  "experiment_name": "<concise name>",
  "material_system": "<material class>",
  "parameters": [
    {{"name": "<param>", "value": <number or string>, "unit": "<unit or null>", "confidence": <0-1 or null>}}
  ],
  "target_property": "<property>",
  "estimated_outcome": "<expected result or null>",
  "rationale": "<summary>",
  "citations": [
    {{"paper_title": "<title>", "authors": "<authors or null>", "year": <year or null>,
      "relevant_sentence": "<exact supporting sentence>", "doi": "<doi or null>"}}
  ],
  "source_papers": ["<filename1>", "<filename2>"]
}}

--- Paper Context ---
{context}

--- End of Paper Context ---

JSON output:"""


def build_first_trial_card(
    llm: BaseChatModel,
    retrieved_docs: list[Document],
    user_query: str = "",
) -> FirstTrialCard:
    """Run the extraction agent using LangChain's structured output."""

    context_parts = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = EXTRACTION_PROMPT.format(context=context)

    response = llm.invoke(prompt)
    raw = response.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("LLM output was not valid JSON: %s\nRaw: %s", exc, raw)
        raise

    params = []
    for p in data.get("parameters", []):
        params.append(ExperimentalParameter(
            name=p["name"],
            value=p["value"],
            unit=p.get("unit"),
            confidence=p.get("confidence"),
        ))

    citations = []
    for c in data.get("citations", []):
        citations.append(Citation(
            paper_title=c.get("paper_title", "Unknown"),
            authors=c.get("authors"),
            year=c.get("year"),
            relevant_sentence=c.get("relevant_sentence", ""),
            doi=c.get("doi"),
        ))

    return FirstTrialCard(
        experiment_name=data.get("experiment_name", "Extracted Experiment"),
        material_system=data.get("material_system", "Unknown"),
        parameters=params,
        target_property=data.get("target_property", ""),
        estimated_outcome=data.get("estimated_outcome"),
        rationale=data.get("rationale", ""),
        citations=citations,
        source_papers=data.get("source_papers", []),
    )
