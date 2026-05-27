from pydantic import BaseModel, Field
from typing import Optional


class ExperimentalParameter(BaseModel):
    name: str
    value: float | str
    unit: Optional[str] = None
    confidence: Optional[float] = None


class Citation(BaseModel):
    paper_title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    relevant_sentence: str
    doi: Optional[str] = None


class FirstTrialCard(BaseModel):
    experiment_name: str
    material_system: str
    parameters: list[ExperimentalParameter]
    target_property: str
    estimated_outcome: Optional[str] = None
    rationale: str
    citations: list[Citation]
    source_papers: list[str]
