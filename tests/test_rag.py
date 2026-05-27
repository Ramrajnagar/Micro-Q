"""Tests for the PaperQA ingestion pipeline."""

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.paper_qa.ingestion import PaperStore
from src.paper_qa.models import ExperimentalParameter, Citation, FirstTrialCard
from src.paper_qa.agent import build_first_trial_card, EXTRACTION_PROMPT


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 128


class FakeChatModel(BaseChatModel):
    """Returns a fixed JSON response for extraction."""

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager=None, **kwargs) -> ChatResult:
        content = """{
            "experiment_name": "Triple-Cation Perovskite Test",
            "material_system": "Cs/FA/MA triple-cation perovskite",
            "parameters": [
                {"name": "annealing_temperature", "value": 100, "unit": "C", "confidence": 0.95},
                {"name": "precursor_concentration", "value": 1.35, "unit": "M", "confidence": 0.9},
                {"name": "solvent_ratio_DMF_DMSO", "value": 4.0, "unit": "v/v", "confidence": 0.85}
            ],
            "target_property": "power conversion efficiency",
            "estimated_outcome": "~21% PCE",
            "rationale": "Triple-cation compositions show improved stability and efficiency.",
            "citations": [
                {"paper_title": "Compositional engineering", "authors": "Saliba et al.",
                 "year": 2016, "relevant_sentence": "Triple-cation perovskite achieves 21.1%.",
                 "doi": "10.1039/C5EE03874J"}
            ],
            "source_papers": ["perovskite_efficiency.txt"]
        }"""
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake"


SAMPLE_TEXT = """Title: Test Paper on Perovskite Solar Cells.
The precursor was annealed at 100°C for 60 minutes, achieving a PCE of 21.1%.
The device showed excellent stability retaining 95% efficiency after 500 hours.
We used a triple-cation composition with Cs, FA, and MA.
The solvent ratio of DMF to DMSO was 4:1 by volume.
Antisolvent treatment with chlorobenzene was applied during spin-coating.
These results demonstrate the effectiveness of compositional engineering.
Further optimization of the annealing temperature may improve performance.
The bandgap of the material was measured at 1.55 eV.
Film thickness was approximately 500 nm as determined by SEM.
XRD analysis confirmed the perovskite crystal structure."""


def test_paper_store_ingest_and_retrieve():
    store = PaperStore(embedding_fn=FakeEmbeddings())
    count = store.ingest_text(SAMPLE_TEXT, metadata={"source": "test.txt"})
    assert count >= 1
    assert store.is_loaded
    docs = store.retrieve("perovskite annealing", k=2)
    assert len(docs) >= 1


def test_ingest_text_file(tmp_path: Path):
    store = PaperStore(embedding_fn=FakeEmbeddings())
    f = tmp_path / "test_paper.txt"
    f.write_text(SAMPLE_TEXT)
    count = store.ingest_paper_file(str(f))
    assert count >= 1


def test_extraction_agent():
    llm = FakeChatModel()
    store = PaperStore(embedding_fn=FakeEmbeddings())
    store.ingest_text(SAMPLE_TEXT, metadata={"source": "test.txt"})
    docs = store.retrieve("perovskite", k=5)
    card = build_first_trial_card(llm, docs)
    assert isinstance(card, FirstTrialCard)
    assert card.material_system == "Cs/FA/MA triple-cation perovskite"
    assert len(card.parameters) == 3
    assert card.parameters[0].name == "annealing_temperature"
    assert len(card.citations) == 1
    assert card.citations[0].paper_title == "Compositional engineering"


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_parameter_model_validation(confidence):
    p = ExperimentalParameter(name="test", value=1.0, unit="eV", confidence=confidence)
    assert p.confidence == confidence


def test_citation_model():
    c = Citation(paper_title="Test", authors="Test", year=2023,
                 relevant_sentence="This is a test.", doi="10.1234/test")
    assert c.doi == "10.1234/test"
