# Micro-Q  Autonomous Materials Discovery Pipeline

> *"I already built a proof-of-concept of what you sell."*

**Micro-Q** is a localized, end-to-end materials discovery pipeline that mimics the core value proposition of **Novyte Materials' product Q**: going from a literature hypothesis to an optimized experimental design with scientific reasoning.

Built for the Novyte Materials.

---

## Architecture

```
                           ┌─────────────────────────────┐
  PDFs/TXT ──────────────▶ │  Component 1: PaperQA       │
   (Perovskite,            │  LangChain RAG + LangGraph   │
    Battery Electrolyte)   │  → Extracts "First Trial     │
                           │    Card" JSON with citations  │
                           └────────────┬────────────────┘
                                        │ structured params
                                        ▼
                           ┌─────────────────────────────┐
                           │  Component 2: Bayesian Opt  │
                           │  BoTorch (PyTorch GP)       │
                           │  → qEHVI acquisition        │
                           │  → Pareto frontier          │
                           │  → Next best 3 experiments  │
                           └────────────┬────────────────┘
                                        │ candidate points
                                        ▼
                           ┌─────────────────────────────┐
                           │  Component 3: Scientific    │
                           │  Diary                      │
                           │  LLM (OpenAI-compatible)    │
                           │  → "Why this formulation?"  │
                           │  → "Why not the others?"    │
                           └─────────────────────────────┘
```

---

## Components

### Component 1: PaperQA Literature Ingestion (The Hypothesis)

- **LangChain** RAG pipeline with **FAISS** vector store
- Ingests 3 sample material science papers (perovskite solar cells, battery electrolytes)
- A **LangGraph-inspired agent** extracts structured experimental parameters into a `FirstTrialCard` (JSON with citations)
- Mimics Novyte's PaperQA2 feature

### Component 2: Bayesian Optimization Loop (The Design Engine)

- **BoTorch** (PyTorch GP) multi-objective optimization
- 4-dimensional search space: precursor ratio, annealing temp, doping concentration, solvent ratio
- 3 objectives: maximize **mechanical strength**, minimize **production cost**, maximize **ionic conductivity**
- **qExpected Hypervolume Improvement (qEHVI)** acquisition function
- Pareto frontier computation
- Claims 65% fewer trials — demonstrated with synthetic objective functions

### Component 3: Scientific Reasoning Diary (The Differentiator)

- LLM-powered diary entries explaining every optimization decision
- "I selected Formulation X because it maximizes the Pareto frontier for strength, sacrificing a 5% increase in cost."
- "I rejected Formulation Y because its uncertainty bounds are too high given current data maturity."
- Uses `langchain-openai` — works with any OpenAI-compatible endpoint

### API Layer

- **FastAPI** backend with full REST API
- CORS-enabled for frontend integration
- Built-in `/docs` (Swagger UI) for interactive testing

---

## Project Structure

```
Micro-Q/
├── src/
│   ├── config.py                  # Environment-based config
│   ├── paper_qa/
│   │   ├── ingestion.py           # RAG pipeline (PaperStore)
│   │   ├── models.py              # Pydantic models (FirstTrialCard, etc.)
│   │   └── agent.py               # LangGraph extraction agent
│   ├── bayesian_opt/
│   │   ├── search_space.py        # Material search space + synthetic objectives
│   │   └── optimizer.py           # BoTorch qEHVI optimization loop
│   ├── reasoning/
│   │   ├── prompts.py             # LLM prompt templates
│   │   └── diary.py               # Diary entry generator
│   └── api/
│       ├── main.py                # FastAPI app entrypoint
│       └── routes.py              # REST endpoints
├── data/
│   ├── papers/                    # Sample material science papers (.txt)
│   └── output/                    # Pipeline output JSON
├── scripts/
│   └── run_pipeline.py            # CLI runner
├── tests/
│   ├── test_rag.py
│   ├── test_bayesian.py
│   └── test_diary.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Python ≥ 3.11
- For LLM features (RAG extraction + diary): an OpenAI-compatible API key

### Install

```bash
cd Micro-Q
pip install -r requirements.txt
```

### Run the Optimization (No API Key Needed)

```bash
python scripts/run_pipeline.py --optimize-only --rounds 3 --candidates 3
```

This runs the full Bayesian optimization loop with synthetic objectives — no external dependencies required.

### Run the Full Pipeline (Requires API Key)

```bash
export LLM_API_KEY="sk-..."
python scripts/run_pipeline.py --full --rounds 3 --candidates 3
```

### Start the API Server

```bash
export LLM_API_KEY="sk-..."
uvicorn src.api.main:app --reload --port 8000
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

### Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `POST` | `/api/v1/ingest` | Ingest papers → extract FirstTrialCard |
| `POST` | `/api/v1/optimize` | Run Bayesian optimization loop |
| `POST` | `/api/v1/diary` | Generate scientific diary entry |
| `POST` | `/api/v1/pipeline` | Run full end-to-end pipeline |

---

## Sample Output

After running the pipeline, check `data/output/pipeline_result.json`. The diary entry will look like:

> *"I have selected Formulation X (precursor_ratio=0.72, annealing_temp=0.58, doping=0.31, solvent=0.44) because it sits on the Pareto frontier with predicted strength of 0.91 and cost of 0.28 — the best trade-off in the current round. I rejected candidate Y (strength=0.67, cost=0.15) because its uncertainty bounds (σ=0.23) exceed our threshold given only 8 initial observations; it is a high-risk exploration point that should be revisited after 2 more rounds of data collection."*

---

## How This Maps to Novyte Materials' Stack

| Novyte Feature | Micro-Q Implementation |
|---|---|
| PaperQA2 (Literature RAG) | LangChain RAG + FAISS vector store |
| Active Learning (Bayesian Optimization) | BoTorch qEHVI multi-objective optimization |
| Multi-Objective Physics Constraints | 3-objective Pareto frontier (strength, cost, conductivity) |
| Scientific Diary | LLM-powered reasoning diary with rejected-alternative analysis |
| LangChain / LangGraph (required skill) | LangChain RAG + agentic extraction |
| FastAPI backend | FastAPI REST API |
| Python / PyTorch | Full Python stack with PyTorch GP |

---

## Your Experience Bridge

already worked on **CellBox** — deep learning + differential equations for cellular response modeling and drug optimization.

- **Surrogate modeling** (CellBox's ODE systems → Micro-Q's Gaussian Processes)
- **High-dimensional optimization** (drug design spaces → material formulation spaces)
- **Scientific ML validation** (biological assays → materials characterization)

Micro-Q is a direct domain adaptation of those same skills from **biological systems** → **materials science**.

---

## License

MIT — built for demonstration purposes.
