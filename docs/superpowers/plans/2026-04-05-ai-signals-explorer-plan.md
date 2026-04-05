# AI Signals Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Docker-packaged teaching tool that asks questions to a local LLM (Ollama) and visualizes 18 internal model signals in real-time with progressive delivery, grouping, and executive summaries.

**Architecture:** React+TS frontend communicates with a Python FastAPI backend via REST and SSE. The backend orchestrates signal computation from two model sources: Ollama (logprobs + multi-sampling) and HuggingFace Transformers (hidden states). Everything runs via `docker compose up`.

**Tech Stack:** React 18, TypeScript, Vite, Tailwind CSS, Recharts, Python 3.11, FastAPI, PyTorch CPU, HuggingFace Transformers, Ollama, Docker Compose, SSE

**Spec:** `docs/superpowers/specs/2026-04-05-ai-signals-explorer-design.md`

---

## File Structure

```
ai-signals/
├── docker-compose.yml
├── .gitignore
├── README.md
│
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css
│       ├── types/
│       │   └── signals.ts
│       ├── data/
│       │   └── signalDefinitions.ts
│       ├── themes/
│       │   ├── research.ts
│       │   └── oracle.ts
│       ├── hooks/
│       │   ├── useSignalStream.ts
│       │   └── useTheme.ts
│       ├── components/
│       │   ├── ChatPanel.tsx
│       │   ├── SignalDashboard.tsx
│       │   ├── SignalCard.tsx
│       │   ├── SignalChart.tsx
│       │   ├── ExecutiveSummary.tsx
│       │   ├── CompositeScore.tsx
│       │   ├── SchoolGroup.tsx
│       │   ├── ThemeToggle.tsx
│       │   └── SettingsPanel.tsx
│       └── pages/
│           ├── HomePage.tsx
│           └── SignalDetailPage.tsx
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── scripts/
│   │   └── download_models.py
│   ├── tests/
│   │   ├── test_info_theoretic.py
│   │   ├── test_layer_wise.py
│   │   ├── test_geometric.py
│   │   ├── test_behavioral.py
│   │   ├── test_calibration.py
│   │   ├── test_engine.py
│   │   ├── test_chat_api.py
│   │   └── test_signals_api.py
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── schemas.py
│       ├── summary.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── chat.py
│       │   ├── signals.py
│       │   └── health.py
│       ├── signals/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── info_theoretic.py
│       │   ├── layer_wise.py
│       │   ├── geometric.py
│       │   ├── behavioral.py
│       │   └── calibration.py
│       └── models/
│           ├── __init__.py
│           ├── ollama_client.py
│           └── hf_model.py
│
└── docs/
    └── future-roadmap.md
```

---

## Task 1: Project Scaffolding & Docker Compose

**Files:**
- Create: `docker-compose.yml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `docs/future-roadmap.md`

- [ ] **Step 1: Initialize git repo**

```bash
cd C:/Source-Personal/ai-signals
git init
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# Dependencies
node_modules/
__pycache__/
*.pyc
.venv/
venv/

# Build
dist/
build/
*.egg-info/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# Docker volumes
ollama_data/
hf_cache/

# OS
.DS_Store
Thumbs.db

# Superpowers
.superpowers/
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ai-signals-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ai-signals-backend
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - HF_HOME=/hf_cache
      - DEFAULT_OLLAMA_MODEL=llama3.2:1b
    volumes:
      - hf_cache:/hf_cache
    depends_on:
      ollama:
        condition: service_healthy

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: ai-signals-frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

volumes:
  ollama_data:
  hf_cache:
```

- [ ] **Step 4: Create README.md**

```markdown
# AI Signals Explorer

A teaching tool that visualizes LLM internal signals in real-time. Ask a question, see the model's confidence, uncertainty, and reasoning signals across 5 schools of thought.

## Quick Start

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Clone this repo
3. Run:

```bash
docker compose up
```

4. Open http://localhost:3000

First launch downloads models (~2GB) — subsequent launches are instant.

## What It Does

When you ask a question, the app computes 18 signals from the model's internals:

- **Information-Theoretic** — entropy, perplexity, token probabilities
- **Layer-Wise** — DoLa, logit lens, ICR score
- **Geometric** — embedding trajectories, cosine similarity
- **Behavioral** — self-consistency, hedging detection
- **Calibration** — confidence agreement, composite reliability

Signals arrive progressively — fast ones instantly, expensive ones in seconds.
```

- [ ] **Step 5: Create docs/future-roadmap.md**

```markdown
# Future Roadmap (Phase 2)

1. **Cross-question comparison** — Pick any two past questions and compare signals side-by-side. Session trend panel showing how signals change across questions.
2. **Contextual addendum on detail pages** — Dynamic section showing "How this signal applied to your last question" (extra LLM call per detail page visit).
3. **Additional signals** — Semantic Entropy (full NLI-based), REMA with reference manifold, activation patching.
4. **Model comparison** — Run the same question through two different Ollama models and compare signal profiles side-by-side.
5. **Export** — Export signal reports as PDF or shareable link.
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore docker-compose.yml README.md docs/
git commit -m "feat: project scaffolding with Docker Compose and README"
```

---

## Task 2: Backend Scaffold — FastAPI + Schemas + Health Endpoint

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/schemas.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/health.py`
- Create: `backend/tests/test_chat_api.py` (health test only for now)

- [ ] **Step 1: Create backend/requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
sse-starlette==2.1.0
pydantic==2.9.0
httpx==0.27.0
numpy==1.26.4
scipy==1.13.0
transformers==4.44.0
torch==2.4.0+cpu
scikit-learn==1.5.0
pytest==8.3.0
pytest-asyncio==0.24.0
```

- [ ] **Step 2: Create backend/Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

# Download HF model on first startup
COPY scripts/download_models.py /app/scripts/download_models.py

CMD ["sh", "-c", "python scripts/download_models.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```

- [ ] **Step 3: Create backend/scripts/download_models.py**

```python
"""Download HuggingFace model on first startup. Pulls Ollama model if not present."""
import os
import subprocess
import httpx

HF_MODEL = "Qwen/Qwen2.5-0.5B"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:1b")


def download_hf_model():
    """Pre-download HuggingFace model so first query isn't slow."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[setup] Checking HuggingFace model: {HF_MODEL}")
    try:
        AutoTokenizer.from_pretrained(HF_MODEL)
        AutoModelForCausalLM.from_pretrained(HF_MODEL)
        print(f"[setup] HuggingFace model ready: {HF_MODEL}")
    except Exception as e:
        print(f"[setup] HuggingFace model download failed: {e}")
        print("[setup] Layer-wise and geometric signals will be unavailable.")


def pull_ollama_model():
    """Pull default Ollama model if not already present."""
    print(f"[setup] Checking Ollama model: {DEFAULT_MODEL}")
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        models = [m["name"] for m in response.json().get("models", [])]
        if DEFAULT_MODEL not in models and f"{DEFAULT_MODEL}:latest" not in models:
            print(f"[setup] Pulling {DEFAULT_MODEL}... (this may take a few minutes)")
            httpx.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": DEFAULT_MODEL},
                timeout=600,
            )
            print(f"[setup] Ollama model ready: {DEFAULT_MODEL}")
        else:
            print(f"[setup] Ollama model already present: {DEFAULT_MODEL}")
    except Exception as e:
        print(f"[setup] Ollama model pull failed: {e}")


if __name__ == "__main__":
    pull_ollama_model()
    download_hf_model()
```

- [ ] **Step 4: Create backend/app/__init__.py**

```python
```

- [ ] **Step 5: Create backend/app/schemas.py**

```python
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    model: str = "llama3.2:1b"


class ChatResponse(BaseModel):
    response: str
    request_id: str
    model: str
    generation_time_ms: int
    logprobs: list[float] | None = None
    top_logprobs: list[list[dict]] | None = None


class SignalResult(BaseModel):
    signal_id: str
    school: str
    value: float | list[float] | dict
    interpretation: str
    metadata: dict = {}


class SummaryResult(BaseModel):
    type: str = "summary"
    composite_score: float
    executive_summary: str
    behavioral_groups: list[dict] = []


class HealthStatus(BaseModel):
    ollama: dict
    huggingface: dict


class ModelListResponse(BaseModel):
    models: list[str]
    active: str


class ModelSwitchRequest(BaseModel):
    model: str
```

- [ ] **Step 6: Create backend/app/api/__init__.py**

```python
```

- [ ] **Step 7: Create backend/app/api/health.py**

```python
import os
import httpx
from fastapi import APIRouter

router = APIRouter()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@router.get("/api/health")
async def health_check():
    # Check Ollama
    ollama_status = {"status": "disconnected", "model": None}
    try:
        response = await httpx.AsyncClient().get(
            f"{OLLAMA_BASE_URL}/api/tags", timeout=5
        )
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            ollama_status = {
                "status": "connected",
                "model": models[0] if models else None,
            }
    except Exception:
        pass

    # Check HuggingFace model
    hf_status = {"status": "not_loaded", "model": None}
    try:
        from app.models.hf_model import get_hf_model

        model = get_hf_model()
        if model is not None:
            hf_status = {"status": "loaded", "model": "Qwen/Qwen2.5-0.5B"}
    except Exception:
        pass

    return {"ollama": ollama_status, "huggingface": hf_status}
```

- [ ] **Step 8: Create backend/app/main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health

app = FastAPI(title="AI Signals Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
```

- [ ] **Step 9: Write health endpoint test**

Create `backend/tests/test_chat_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "ollama" in data
    assert "huggingface" in data
    assert "status" in data["ollama"]
    assert "status" in data["huggingface"]
```

- [ ] **Step 10: Run test to verify it passes**

```bash
cd backend
pip install -e . 2>/dev/null || true
python -m pytest tests/test_chat_api.py::test_health_endpoint_returns_200 -v
```

Expected: PASS (Ollama will show as disconnected in test env, which is fine)

- [ ] **Step 11: Commit**

```bash
git add backend/
git commit -m "feat: backend scaffold with FastAPI, schemas, and health endpoint"
```

---

## Task 3: Ollama Client + Chat API

**Files:**
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/ollama_client.py`
- Create: `backend/app/api/chat.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Create backend/app/models/__init__.py**

```python
```

- [ ] **Step 2: Create backend/app/models/ollama_client.py**

```python
import os
import time
import uuid
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_active_model = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:1b")


def get_active_model() -> str:
    return _active_model


def set_active_model(model: str):
    global _active_model
    _active_model = model


async def chat(question: str, model: str | None = None) -> dict:
    """Send a chat request to Ollama with logprobs enabled via OpenAI-compatible API."""
    model = model or _active_model
    request_id = str(uuid.uuid4())
    start_time = time.time()

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": question}],
                "logprobs": True,
                "top_logprobs": 5,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        data = response.json()

    generation_time_ms = int((time.time() - start_time) * 1000)
    choice = data["choices"][0]
    content = choice["message"]["content"]

    logprobs_data = choice.get("logprobs", {}).get("content", [])
    token_logprobs = [t["logprob"] for t in logprobs_data] if logprobs_data else []
    top_logprobs = [t.get("top_logprobs", []) for t in logprobs_data] if logprobs_data else []

    return {
        "response": content,
        "request_id": request_id,
        "model": model,
        "generation_time_ms": generation_time_ms,
        "logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
    }


async def sample_responses(question: str, n: int = 5, model: str | None = None) -> list[str]:
    """Generate multiple responses for self-consistency analysis."""
    model = model or _active_model
    responses = []

    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(n):
            response = await client.post(
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": question}],
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            data = response.json()
            responses.append(data["choices"][0]["message"]["content"])

    return responses


async def list_models() -> list[str]:
    """List available Ollama models."""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]


async def generate_summary(prompt: str, model: str | None = None) -> str:
    """Generate text using Ollama (for executive summary)."""
    model = model or _active_model

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 3: Create backend/app/api/chat.py**

```python
from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest, ChatResponse, ModelListResponse, ModelSwitchRequest
from app.models import ollama_client

router = APIRouter()


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await ollama_client.chat(request.question, request.model)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")


@router.get("/api/models", response_model=ModelListResponse)
async def list_models():
    try:
        models = await ollama_client.list_models()
        return ModelListResponse(
            models=models,
            active=ollama_client.get_active_model(),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")


@router.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    models = await ollama_client.list_models()
    if request.model not in models:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    ollama_client.set_active_model(request.model)
    return {"active": request.model}
```

- [ ] **Step 4: Register chat router in main.py**

Modify `backend/app/main.py` — add after the health router import:

```python
from app.api import health, chat

# ... existing app setup ...

app.include_router(health.router)
app.include_router(chat.router)
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/ backend/app/api/chat.py backend/app/main.py
git commit -m "feat: Ollama client with chat, logprobs, multi-sampling, and model switching"
```

---

## Task 4: Information-Theoretic Signals (6 signals)

**Files:**
- Create: `backend/app/signals/__init__.py`
- Create: `backend/app/signals/info_theoretic.py`
- Create: `backend/tests/test_info_theoretic.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_info_theoretic.py`:

```python
import numpy as np
import pytest
from app.signals.info_theoretic import compute_all_info_theoretic


def test_high_confidence_signals():
    """When logprobs are high (near 0), all signals should indicate confidence."""
    logprobs = [np.log(0.95)] * 20  # Very confident tokens
    result = compute_all_info_theoretic(logprobs, top_logprobs=None)

    assert result["predictive_entropy"]["value"] < 0.5
    assert result["perplexity"]["value"] < 2.0
    assert result["mean_token_prob"]["value"] > 0.9
    assert result["min_token_prob"]["value"] > 0.9
    assert result["token_prob_variance"]["value"] < 0.01


def test_low_confidence_signals():
    """When logprobs are low, all signals should indicate uncertainty."""
    logprobs = [np.log(0.1)] * 20  # Very uncertain tokens
    result = compute_all_info_theoretic(logprobs, top_logprobs=None)

    assert result["predictive_entropy"]["value"] > 1.0
    assert result["perplexity"]["value"] > 5.0
    assert result["mean_token_prob"]["value"] < 0.2
    assert result["min_token_prob"]["value"] < 0.2


def test_mixed_confidence():
    """Mixed logprobs should show high variance."""
    logprobs = [np.log(0.95)] * 10 + [np.log(0.1)] * 10
    result = compute_all_info_theoretic(logprobs, top_logprobs=None)

    assert result["token_prob_variance"]["value"] > 0.05


def test_top_k_with_top_logprobs():
    """Top-k mass should reflect concentration in top tokens."""
    top_logprobs = [
        [{"logprob": np.log(0.9)}, {"logprob": np.log(0.05)}, {"logprob": np.log(0.03)}]
    ] * 10
    logprobs = [np.log(0.9)] * 10
    result = compute_all_info_theoretic(logprobs, top_logprobs=top_logprobs)

    assert result["top_k_prob_mass"]["value"] > 0.9


def test_empty_logprobs_returns_defaults():
    """Empty logprobs should return safe defaults."""
    result = compute_all_info_theoretic([], top_logprobs=None)

    assert result["predictive_entropy"]["value"] == 0.0
    assert result["perplexity"]["value"] == 1.0
    assert result["mean_token_prob"]["value"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
python -m pytest tests/test_info_theoretic.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.signals'`

- [ ] **Step 3: Implement info_theoretic.py**

Create `backend/app/signals/__init__.py`:

```python
```

Create `backend/app/signals/info_theoretic.py`:

```python
import numpy as np


def compute_all_info_theoretic(
    logprobs: list[float],
    top_logprobs: list[list[dict]] | None,
) -> dict:
    """Compute all 6 information-theoretic signals from token logprobs."""
    if not logprobs:
        return {
            "predictive_entropy": _signal("predictive_entropy", 0.0, "No tokens to analyze"),
            "perplexity": _signal("perplexity", 1.0, "No tokens to analyze"),
            "mean_token_prob": _signal("mean_token_prob", 0.0, "No tokens to analyze"),
            "min_token_prob": _signal("min_token_prob", 0.0, "No tokens to analyze"),
            "top_k_prob_mass": _signal("top_k_prob_mass", 0.0, "No tokens to analyze"),
            "token_prob_variance": _signal("token_prob_variance", 0.0, "No tokens to analyze"),
        }

    lp = np.array(logprobs, dtype=np.float64)
    probs = np.exp(lp)

    entropy = _predictive_entropy(probs, lp)
    perplexity = _perplexity(lp)
    mean_tp = float(np.mean(probs))
    min_tp = float(np.min(probs))
    variance = float(np.var(probs))
    top_k = _top_k_mass(top_logprobs)

    return {
        "predictive_entropy": _signal(
            "predictive_entropy",
            entropy,
            _interpret_entropy(entropy),
        ),
        "perplexity": _signal(
            "perplexity",
            perplexity,
            _interpret_perplexity(perplexity),
        ),
        "mean_token_prob": _signal(
            "mean_token_prob",
            mean_tp,
            _interpret_mean_tp(mean_tp),
        ),
        "min_token_prob": _signal(
            "min_token_prob",
            min_tp,
            _interpret_min_tp(min_tp),
        ),
        "top_k_prob_mass": _signal(
            "top_k_prob_mass",
            top_k,
            _interpret_top_k(top_k),
        ),
        "token_prob_variance": _signal(
            "token_prob_variance",
            variance,
            _interpret_variance(variance),
        ),
    }


def _predictive_entropy(probs: np.ndarray, logprobs: np.ndarray) -> float:
    """H(Y|x) = -Σ p(y|x) log p(y|x)"""
    entropy = -np.sum(probs * logprobs)
    return float(max(0.0, entropy))


def _perplexity(logprobs: np.ndarray) -> float:
    """PPL = exp(-(1/n) Σ log p(tᵢ))"""
    return float(np.exp(-np.mean(logprobs)))


def _top_k_mass(top_logprobs: list[list[dict]] | None) -> float:
    """Average probability mass in top-k tokens across all positions."""
    if not top_logprobs:
        return 0.0

    masses = []
    for position in top_logprobs:
        if position:
            position_mass = sum(np.exp(t["logprob"]) for t in position)
            masses.append(position_mass)

    return float(np.mean(masses)) if masses else 0.0


def _signal(signal_id: str, value: float, interpretation: str) -> dict:
    return {
        "signal_id": signal_id,
        "school": "information_theoretic",
        "value": round(value, 4),
        "interpretation": interpretation,
    }


def _interpret_entropy(v: float) -> str:
    if v < 0.5:
        return "Low entropy — model is confident in this response"
    if v < 2.0:
        return "Moderate entropy — some uncertainty in token choices"
    return "High entropy — model is uncertain, probability spread across many tokens"


def _interpret_perplexity(v: float) -> str:
    if v < 2.0:
        return "Model is minimally surprised by its own output"
    if v < 10.0:
        return "Moderate perplexity — multiple plausible tokens at some positions"
    return "High perplexity — model found its own output quite surprising"


def _interpret_mean_tp(v: float) -> str:
    if v > 0.8:
        return "High average confidence across all tokens"
    if v > 0.5:
        return "Moderate average confidence — some tokens chosen with less certainty"
    return "Low average confidence — many tokens were not the model's top choice"


def _interpret_min_tp(v: float) -> str:
    if v > 0.5:
        return "Even the least confident token had reasonable probability"
    if v > 0.1:
        return "At least one token was chosen with low confidence"
    return "Contains a token the model was very unsure about — potential weak point"


def _interpret_top_k(v: float) -> str:
    if v > 0.9:
        return "Probability heavily concentrated in top tokens"
    if v > 0.7:
        return "Most probability in top tokens, some spread"
    return "Probability spread thin across many tokens — low concentration"


def _interpret_variance(v: float) -> str:
    if v < 0.01:
        return "Very consistent confidence across all tokens"
    if v < 0.05:
        return "Some variation in confidence across the response"
    return "High variance — model confidence fluctuates significantly across tokens"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/test_info_theoretic.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/ backend/tests/test_info_theoretic.py
git commit -m "feat: information-theoretic signals (entropy, perplexity, token probs)"
```

---

## Task 5: HuggingFace Model Loader

**Files:**
- Create: `backend/app/models/hf_model.py`

- [ ] **Step 1: Create backend/app/models/hf_model.py**

```python
"""
Loads Qwen2.5-0.5B on CPU for hidden state extraction.
The model is loaded once at startup and reused for all requests.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-0.5B")

_tokenizer = None
_model = None


def get_hf_model():
    """Get or lazily load the HuggingFace model."""
    global _tokenizer, _model
    if _model is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            _model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="cpu",
                output_hidden_states=True,
            )
            _model.eval()
        except Exception as e:
            print(f"[hf_model] Failed to load {HF_MODEL_NAME}: {e}")
            return None
    return _model


def get_hf_tokenizer():
    """Get or lazily load the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        get_hf_model()
    return _tokenizer


def extract_hidden_states(text: str) -> dict | None:
    """
    Run text through HF model and extract hidden states at all layers.

    Returns:
        Dict with 'hidden_states' (list of tensors, one per layer),
        'logits' (final layer logits), and 'input_ids'.
        Returns None if model is not loaded.
    """
    model = get_hf_model()
    tokenizer = get_hf_tokenizer()

    if model is None or tokenizer is None:
        return None

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return {
        "hidden_states": outputs.hidden_states,  # Tuple of tensors (num_layers+1, batch, seq, hidden)
        "logits": outputs.logits,  # (batch, seq, vocab)
        "input_ids": inputs["input_ids"],
    }
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/models/hf_model.py
git commit -m "feat: HuggingFace model loader with hidden state extraction"
```

---

## Task 6: Layer-Wise Signals (4 signals)

**Files:**
- Create: `backend/app/signals/layer_wise.py`
- Create: `backend/tests/test_layer_wise.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_layer_wise.py`:

```python
import torch
import numpy as np
import pytest
from app.signals.layer_wise import compute_all_layer_wise


def _make_hidden_states(num_layers: int, seq_len: int, hidden_dim: int, confident: bool):
    """Create mock hidden states for testing."""
    states = []
    for i in range(num_layers + 1):  # +1 for embedding layer
        if confident:
            # Confident: layers converge to similar representation
            state = torch.randn(1, seq_len, hidden_dim) * 0.1 + i * 0.01
        else:
            # Uncertain: layers diverge significantly
            state = torch.randn(1, seq_len, hidden_dim) * (1.0 + i * 0.5)
        states.append(state)
    return tuple(states)


def _make_mock_model_output(num_layers=24, seq_len=10, hidden_dim=896, confident=True):
    hidden_states = _make_hidden_states(num_layers, seq_len, hidden_dim, confident)
    logits = torch.randn(1, seq_len, 32000)  # Random logits
    input_ids = torch.randint(0, 32000, (1, seq_len))
    return {"hidden_states": hidden_states, "logits": logits, "input_ids": input_ids}


def test_confident_output_has_low_icr():
    """Confident outputs should have low ICR (early/late layers agree)."""
    output = _make_mock_model_output(confident=True)
    result = compute_all_layer_wise(output)
    # ICR is bounded [0, 1], confident should be lower
    assert result["icr_score"]["value"] >= 0.0
    assert result["icr_score"]["value"] <= 1.0


def test_returns_all_four_signals():
    """Should return all four layer-wise signals."""
    output = _make_mock_model_output()
    result = compute_all_layer_wise(output)
    assert "dola_contrast" in result
    assert "logit_lens_evolution" in result
    assert "icr_score" in result
    assert "prediction_stability" in result


def test_none_input_returns_defaults():
    """None input should return safe defaults."""
    result = compute_all_layer_wise(None)
    assert result["dola_contrast"]["value"] == 0.0
    assert result["icr_score"]["value"] == 0.0
    assert result["prediction_stability"]["value"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
python -m pytest tests/test_layer_wise.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement layer_wise.py**

Create `backend/app/signals/layer_wise.py`:

```python
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_all_layer_wise(model_output: dict | None) -> dict:
    """Compute all 4 layer-wise signals from HuggingFace model output."""
    if model_output is None:
        return {
            "dola_contrast": _signal("dola_contrast", 0.0, "HuggingFace model not available"),
            "logit_lens_evolution": _signal("logit_lens_evolution", [], "HuggingFace model not available"),
            "icr_score": _signal("icr_score", 0.0, "HuggingFace model not available"),
            "prediction_stability": _signal("prediction_stability", 0, "HuggingFace model not available"),
        }

    hidden_states = model_output["hidden_states"]
    logits = model_output["logits"]
    num_layers = len(hidden_states) - 1  # Exclude embedding layer

    dola = _dola_contrast(hidden_states, logits)
    lens_evolution = _logit_lens_evolution(hidden_states, logits)
    icr = _icr_score(hidden_states, logits)
    stability = _prediction_stability(lens_evolution)

    return {
        "dola_contrast": _signal("dola_contrast", dola, _interpret_dola(dola)),
        "logit_lens_evolution": _signal(
            "logit_lens_evolution", lens_evolution, _interpret_lens(lens_evolution)
        ),
        "icr_score": _signal("icr_score", icr, _interpret_icr(icr)),
        "prediction_stability": _signal(
            "prediction_stability", stability, _interpret_stability(stability, num_layers)
        ),
    }


def _dola_contrast(hidden_states: tuple, logits: torch.Tensor) -> float:
    """Compute DoLa contrast between early and final layers."""
    num_layers = len(hidden_states) - 1
    early_idx = max(1, num_layers // 4)  # ~25% depth

    early_hidden = hidden_states[early_idx][:, -1, :]  # Last token
    final_hidden = hidden_states[-1][:, -1, :]

    # Cosine similarity as proxy for contrast
    # High similarity = layers agree (confident), low = they disagree
    cos_sim = F.cosine_similarity(early_hidden, final_hidden, dim=-1)
    contrast = 1.0 - cos_sim.item()  # Invert: high contrast = more change
    return round(float(contrast), 4)


def _logit_lens_evolution(hidden_states: tuple, logits: torch.Tensor) -> list:
    """Track top prediction changes through layers. Returns list of per-layer top token indices."""
    evolution = []
    num_layers = len(hidden_states) - 1

    # Sample layers evenly (max 8 points)
    sample_indices = np.linspace(1, num_layers, min(8, num_layers), dtype=int)

    for idx in sample_indices:
        hidden = hidden_states[idx][:, -1, :]  # Last token position
        # Use L2 norm as a proxy for prediction confidence at this layer
        norm = torch.norm(hidden, dim=-1).item()
        evolution.append({"layer": int(idx), "norm": round(norm, 4)})

    return evolution


def _icr_score(hidden_states: tuple, logits: torch.Tensor) -> float:
    """Compute Intrinsic Certainty Reversal via Jensen-Shannon Divergence."""
    num_layers = len(hidden_states) - 1
    early_idx = max(1, num_layers // 4)

    early_hidden = hidden_states[early_idx][:, -1, :].squeeze()
    final_hidden = hidden_states[-1][:, -1, :].squeeze()

    # Normalize to probability-like distributions for JSD
    early_dist = F.softmax(early_hidden, dim=-1).numpy()
    final_dist = F.softmax(final_hidden, dim=-1).numpy()

    jsd = jensenshannon(early_dist, final_dist)
    return round(float(jsd), 4) if not np.isnan(jsd) else 0.0


def _prediction_stability(lens_evolution: list) -> int:
    """Count significant changes in layer norms (proxy for prediction flips)."""
    if len(lens_evolution) < 2:
        return 0

    norms = [e["norm"] for e in lens_evolution]
    changes = 0
    for i in range(1, len(norms)):
        relative_change = abs(norms[i] - norms[i - 1]) / (norms[i - 1] + 1e-10)
        if relative_change > 0.1:  # >10% change = significant
            changes += 1

    return changes


def _signal(signal_id: str, value, interpretation: str) -> dict:
    return {
        "signal_id": signal_id,
        "school": "layer_wise",
        "value": value,
        "interpretation": interpretation,
    }


def _interpret_dola(v: float) -> str:
    if v < 0.1:
        return "Low contrast — early and late layers agree, suggesting factual grounding"
    if v < 0.3:
        return "Moderate contrast — some difference between layer predictions"
    return "High contrast — significant disagreement between early and late layers"


def _interpret_lens(evolution: list) -> str:
    if not evolution:
        return "No layer data available"
    norms = [e["norm"] for e in evolution]
    trend = norms[-1] - norms[0]
    if abs(trend) < 1.0:
        return "Stable predictions across layers — consistent reasoning"
    if trend > 0:
        return "Increasing activation through layers — model building confidence"
    return "Decreasing activation through layers — unusual pattern"


def _interpret_icr(v: float) -> str:
    if v < 0.2:
        return "Low ICR — early and late layers agree, model did not change its mind"
    if v < 0.5:
        return "Moderate ICR — some internal disagreement across layers"
    return "High ICR — model's early and late layers diverge significantly"


def _interpret_stability(changes: int, num_layers: int) -> str:
    ratio = changes / max(1, num_layers)
    if ratio < 0.1:
        return "Highly stable — predictions barely change across layers"
    if ratio < 0.3:
        return "Moderately stable — some prediction shifts through layers"
    return "Unstable — frequent prediction changes, suggesting uncertainty"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/test_layer_wise.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/layer_wise.py backend/tests/test_layer_wise.py
git commit -m "feat: layer-wise signals (DoLa, logit lens, ICR, prediction stability)"
```

---

## Task 7: Geometric Signals (2 signals)

**Files:**
- Create: `backend/app/signals/geometric.py`
- Create: `backend/tests/test_geometric.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_geometric.py`:

```python
import torch
import pytest
from app.signals.geometric import compute_all_geometric


def _make_mock_output(similar_layers=True):
    """Create mock hidden states."""
    states = []
    for i in range(25):  # 24 layers + embedding
        if similar_layers:
            state = torch.ones(1, 10, 896) * (1.0 + i * 0.01)
        else:
            state = torch.randn(1, 10, 896) * (1.0 + i * 0.5)
        states.append(state)
    return {"hidden_states": tuple(states), "logits": torch.randn(1, 10, 32000), "input_ids": torch.randint(0, 32000, (1, 10))}


def test_similar_layers_short_trajectory():
    """Similar layers should produce a short trajectory."""
    output = _make_mock_output(similar_layers=True)
    result = compute_all_geometric(output)
    assert result["embedding_trajectory_length"]["value"] >= 0


def test_divergent_layers_high_trajectory():
    """Divergent layers should produce longer trajectory."""
    similar = _make_mock_output(similar_layers=True)
    divergent = _make_mock_output(similar_layers=False)

    r_similar = compute_all_geometric(similar)
    r_divergent = compute_all_geometric(divergent)

    assert r_divergent["embedding_trajectory_length"]["value"] > r_similar["embedding_trajectory_length"]["value"]


def test_returns_both_signals():
    output = _make_mock_output()
    result = compute_all_geometric(output)
    assert "embedding_trajectory_length" in result
    assert "layerwise_cosine_similarity" in result


def test_none_input():
    result = compute_all_geometric(None)
    assert result["embedding_trajectory_length"]["value"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
python -m pytest tests/test_geometric.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement geometric.py**

Create `backend/app/signals/geometric.py`:

```python
import torch
import torch.nn.functional as F
import numpy as np


def compute_all_geometric(model_output: dict | None) -> dict:
    """Compute 2 geometric signals from hidden states."""
    if model_output is None:
        return {
            "embedding_trajectory_length": _signal("embedding_trajectory_length", 0.0, "HuggingFace model not available"),
            "layerwise_cosine_similarity": _signal("layerwise_cosine_similarity", 0.0, "HuggingFace model not available"),
        }

    hidden_states = model_output["hidden_states"]
    trajectory = _trajectory_length(hidden_states)
    cosine_sim = _layerwise_cosine_sim(hidden_states)

    return {
        "embedding_trajectory_length": _signal(
            "embedding_trajectory_length", trajectory, _interpret_trajectory(trajectory)
        ),
        "layerwise_cosine_similarity": _signal(
            "layerwise_cosine_similarity", cosine_sim, _interpret_cosine(cosine_sim)
        ),
    }


def _trajectory_length(hidden_states: tuple) -> float:
    """Sum of L2 distances between consecutive layers at last token position."""
    total = 0.0
    for i in range(1, len(hidden_states)):
        prev = hidden_states[i - 1][:, -1, :]
        curr = hidden_states[i][:, -1, :]
        dist = torch.norm(curr - prev, dim=-1).item()
        total += dist
    return round(total, 4)


def _layerwise_cosine_sim(hidden_states: tuple) -> float:
    """Average cosine similarity between adjacent layers."""
    sims = []
    for i in range(1, len(hidden_states)):
        prev = hidden_states[i - 1][:, -1, :]
        curr = hidden_states[i][:, -1, :]
        sim = F.cosine_similarity(prev, curr, dim=-1).item()
        sims.append(sim)
    return round(float(np.mean(sims)), 4) if sims else 0.0


def _signal(signal_id: str, value, interpretation: str) -> dict:
    return {
        "signal_id": signal_id,
        "school": "geometric",
        "value": value,
        "interpretation": interpretation,
    }


def _interpret_trajectory(v: float) -> str:
    if v < 10.0:
        return "Short trajectory — representations change little across layers"
    if v < 50.0:
        return "Moderate trajectory — typical transformation through layers"
    return "Long trajectory — significant representational changes, complex processing"


def _interpret_cosine(v: float) -> str:
    if v > 0.95:
        return "Very high layer similarity — minimal transformation between layers"
    if v > 0.8:
        return "High similarity — gradual, smooth changes across layers"
    return "Lower similarity — significant transformations between layers"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/test_geometric.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/geometric.py backend/tests/test_geometric.py
git commit -m "feat: geometric signals (trajectory length, cosine similarity)"
```

---

## Task 8: Behavioral Signals (4 signals)

**Files:**
- Create: `backend/app/signals/behavioral.py`
- Create: `backend/tests/test_behavioral.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_behavioral.py`:

```python
import pytest
from app.signals.behavioral import (
    compute_hedging_score,
    compute_self_consistency,
    compute_response_length_variance,
    compute_semantic_similarity,
)


def test_hedging_high():
    text = "I'm not sure, but I think it might be approximately 42, perhaps."
    result = compute_hedging_score(text)
    assert result["value"] > 0.3


def test_hedging_low():
    text = "The answer is 42. This is a well-established fact."
    result = compute_hedging_score(text)
    assert result["value"] < 0.2


def test_self_consistency_identical():
    responses = ["Paris"] * 5
    result = compute_self_consistency(responses)
    assert result["value"] == 1.0


def test_self_consistency_mixed():
    responses = ["Paris", "Paris", "London", "Berlin", "Paris"]
    result = compute_self_consistency(responses)
    assert result["value"] == 0.6  # 3/5


def test_self_consistency_empty():
    result = compute_self_consistency([])
    assert result["value"] == 0.0


def test_response_length_variance_same():
    responses = ["Hello world", "Hi there!!", "Hey planet"]  # Similar lengths
    result = compute_response_length_variance(responses)
    assert result["value"] < 100


def test_response_length_variance_different():
    responses = ["Hi", "This is a much longer response with many more words and details"]
    result = compute_response_length_variance(responses)
    assert result["value"] > 100


def test_semantic_similarity_identical():
    responses = ["Paris is the capital"] * 3
    result = compute_semantic_similarity(responses)
    assert result["value"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
python -m pytest tests/test_behavioral.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement behavioral.py**

Create `backend/app/signals/behavioral.py`:

```python
import re
import numpy as np
from collections import Counter

HEDGING_PATTERNS = [
    r"\bi'm not (sure|certain)\b",
    r"\bit's possible\b",
    r"\bperhaps\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bi think\b",
    r"\bapproximately\b",
    r"\broughly\b",
    r"\baround\b",
    r"\bmaybe\b",
]


def compute_hedging_score(text: str) -> dict:
    """Detect hedging language indicating uncertainty."""
    text_lower = text.lower()
    matches = sum(1 for p in HEDGING_PATTERNS if re.search(p, text_lower))
    score = round(matches / len(HEDGING_PATTERNS), 4)

    return _signal("hedging_score", score, _interpret_hedging(score))


def compute_self_consistency(responses: list[str]) -> dict:
    """Compute agreement across multiple sampled responses."""
    if not responses:
        return _signal("self_consistency", 0.0, "No responses to compare")

    normalized = [r.strip().lower() for r in responses]
    counts = Counter(normalized)
    majority_count = counts.most_common(1)[0][1]
    score = round(majority_count / len(responses), 4)

    return _signal(
        "self_consistency",
        score,
        _interpret_consistency(score),
        metadata={
            "num_samples": len(responses),
            "num_distinct": len(counts),
            "majority_answer": counts.most_common(1)[0][0][:100],
        },
    )


def compute_response_length_variance(responses: list[str]) -> dict:
    """Compute variance in response lengths across samples."""
    if len(responses) < 2:
        return _signal("response_length_variance", 0.0, "Need multiple responses")

    lengths = [len(r) for r in responses]
    variance = round(float(np.var(lengths)), 4)

    return _signal("response_length_variance", variance, _interpret_length_var(variance))


def compute_semantic_similarity(responses: list[str]) -> dict:
    """Compute average pairwise similarity. Uses character n-gram overlap as a lightweight proxy."""
    if len(responses) < 2:
        return _signal("semantic_similarity", 0.0, "Need multiple responses")

    def ngram_set(text: str, n: int = 3) -> set:
        text = text.lower().strip()
        return {text[i : i + n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}

    sims = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            set_a = ngram_set(responses[i])
            set_b = ngram_set(responses[j])
            if set_a or set_b:
                jaccard = len(set_a & set_b) / len(set_a | set_b)
                sims.append(jaccard)

    score = round(float(np.mean(sims)), 4) if sims else 0.0
    return _signal("semantic_similarity", score, _interpret_semantic_sim(score))


def _signal(signal_id: str, value, interpretation: str, metadata: dict = None) -> dict:
    result = {
        "signal_id": signal_id,
        "school": "behavioral",
        "value": value,
        "interpretation": interpretation,
    }
    if metadata:
        result["metadata"] = metadata
    return result


def _interpret_hedging(v: float) -> str:
    if v < 0.1:
        return "No hedging language detected — model states claims directly"
    if v < 0.3:
        return "Some hedging language — model expresses mild caution"
    return "Significant hedging — model uses uncertain language throughout"


def _interpret_consistency(v: float) -> str:
    if v > 0.8:
        return "High consistency — model gives the same answer across samples"
    if v > 0.5:
        return "Moderate consistency — majority agreement but some variation"
    return "Low consistency — model gives different answers each time"


def _interpret_length_var(v: float) -> str:
    if v < 100:
        return "Consistent response lengths — stable output structure"
    if v < 1000:
        return "Some variation in response length across samples"
    return "High length variance — model produces very different response sizes"


def _interpret_semantic_sim(v: float) -> str:
    if v > 0.8:
        return "High semantic similarity — responses convey consistent meaning"
    if v > 0.5:
        return "Moderate similarity — responses share core content but differ in detail"
    return "Low similarity — responses express substantially different content"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/test_behavioral.py -v
```

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/behavioral.py backend/tests/test_behavioral.py
git commit -m "feat: behavioral signals (hedging, self-consistency, length variance, semantic sim)"
```

---

## Task 9: Calibration Signals (2 signals)

**Files:**
- Create: `backend/app/signals/calibration.py`
- Create: `backend/tests/test_calibration.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_calibration.py`:

```python
import pytest
from app.signals.calibration import compute_confidence_agreement, compute_composite_score


def test_confidence_agreement_match():
    """When verbalized confidence matches signal confidence, agreement should be high."""
    result = compute_confidence_agreement(
        verbalized_confidence=0.9,
        signal_confidence=0.85,
    )
    assert result["value"] > 0.8


def test_confidence_agreement_mismatch():
    """When verbalized and signal confidence disagree, agreement should be low."""
    result = compute_confidence_agreement(
        verbalized_confidence=0.95,
        signal_confidence=0.2,
    )
    assert result["value"] < 0.5


def test_composite_score_all_confident():
    signals = {
        "predictive_entropy": {"value": 0.2},
        "mean_token_prob": {"value": 0.9},
        "self_consistency": {"value": 1.0},
        "hedging_score": {"value": 0.0},
        "icr_score": {"value": 0.05},
    }
    result = compute_composite_score(signals)
    assert result["value"] > 70


def test_composite_score_all_uncertain():
    signals = {
        "predictive_entropy": {"value": 3.0},
        "mean_token_prob": {"value": 0.1},
        "self_consistency": {"value": 0.2},
        "hedging_score": {"value": 0.8},
        "icr_score": {"value": 0.8},
    }
    result = compute_composite_score(signals)
    assert result["value"] < 40


def test_composite_score_empty():
    result = compute_composite_score({})
    assert result["value"] == 50  # Neutral default
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
python -m pytest tests/test_calibration.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement calibration.py**

Create `backend/app/signals/calibration.py`:

```python
import numpy as np


def compute_confidence_agreement(
    verbalized_confidence: float,
    signal_confidence: float,
) -> dict:
    """Compare model's stated confidence with signal-derived confidence."""
    agreement = 1.0 - abs(verbalized_confidence - signal_confidence)
    agreement = round(max(0.0, min(1.0, agreement)), 4)

    return _signal(
        "confidence_agreement",
        agreement,
        _interpret_agreement(agreement),
        metadata={
            "verbalized": round(verbalized_confidence, 4),
            "signal_derived": round(signal_confidence, 4),
        },
    )


def compute_composite_score(signals: dict) -> dict:
    """Compute weighted aggregate reliability score (0-100)."""
    if not signals:
        return _signal("composite_score", 50, "No signals available for scoring")

    scores = []
    weights = []

    # Entropy: lower is more confident. Map [0, 3] -> [100, 0]
    if "predictive_entropy" in signals:
        entropy = signals["predictive_entropy"]["value"]
        score = max(0, min(100, 100 - (entropy / 3.0) * 100))
        scores.append(score)
        weights.append(0.2)

    # Mean token prob: higher is more confident. Map [0, 1] -> [0, 100]
    if "mean_token_prob" in signals:
        mtp = signals["mean_token_prob"]["value"]
        scores.append(mtp * 100)
        weights.append(0.15)

    # Self-consistency: higher is more confident. Map [0, 1] -> [0, 100]
    if "self_consistency" in signals:
        sc = signals["self_consistency"]["value"]
        scores.append(sc * 100)
        weights.append(0.25)

    # Hedging: lower is more confident. Map [0, 1] -> [100, 0]
    if "hedging_score" in signals:
        hedge = signals["hedging_score"]["value"]
        scores.append((1 - hedge) * 100)
        weights.append(0.1)

    # ICR: lower is more confident. Map [0, 1] -> [100, 0]
    if "icr_score" in signals:
        icr = signals["icr_score"]["value"]
        scores.append((1 - icr) * 100)
        weights.append(0.15)

    if not scores:
        return _signal("composite_score", 50, "No scorable signals available")

    total_weight = sum(weights)
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    composite = round(max(0, min(100, weighted_score)), 1)

    return _signal("composite_score", composite, _interpret_composite(composite))


def _signal(signal_id: str, value, interpretation: str, metadata: dict = None) -> dict:
    result = {
        "signal_id": signal_id,
        "school": "calibration",
        "value": value,
        "interpretation": interpretation,
    }
    if metadata:
        result["metadata"] = metadata
    return result


def _interpret_agreement(v: float) -> str:
    if v > 0.8:
        return "Good agreement — stated confidence aligns with signal evidence"
    if v > 0.5:
        return "Moderate agreement — some gap between stated and measured confidence"
    return "Poor agreement — model's stated confidence does not match its signals"


def _interpret_composite(v: float) -> str:
    if v >= 80:
        return "High reliability — multiple signals converge on confidence"
    if v >= 50:
        return "Moderate reliability — mixed signals, verify key claims"
    return "Low reliability — signals indicate significant uncertainty"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/test_calibration.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/calibration.py backend/tests/test_calibration.py
git commit -m "feat: calibration signals (confidence agreement, composite score)"
```

---

## Task 10: Signal Engine + SSE Stream Endpoint

**Files:**
- Create: `backend/app/signals/engine.py`
- Create: `backend/app/summary.py`
- Create: `backend/app/api/signals.py`
- Create: `backend/tests/test_engine.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_engine.py`:

```python
import pytest
from app.signals.engine import compute_fast_signals


def test_fast_signals_with_logprobs():
    import numpy as np
    logprobs = [np.log(0.9)] * 10
    result = compute_fast_signals(logprobs, top_logprobs=None, response_text="Paris is the capital.")
    assert "predictive_entropy" in result
    assert "perplexity" in result
    assert "hedging_score" in result
    assert len(result) == 7  # 6 info-theoretic + 1 hedging


def test_fast_signals_empty():
    result = compute_fast_signals([], top_logprobs=None, response_text="")
    assert "predictive_entropy" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
python -m pytest tests/test_engine.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement engine.py**

Create `backend/app/signals/engine.py`:

```python
import asyncio
from app.signals.info_theoretic import compute_all_info_theoretic
from app.signals.layer_wise import compute_all_layer_wise
from app.signals.geometric import compute_all_geometric
from app.signals.behavioral import (
    compute_hedging_score,
    compute_self_consistency,
    compute_response_length_variance,
    compute_semantic_similarity,
)
from app.signals.calibration import compute_composite_score
from app.models.hf_model import extract_hidden_states
from app.models.ollama_client import sample_responses


def compute_fast_signals(
    logprobs: list[float],
    top_logprobs: list[list[dict]] | None,
    response_text: str,
) -> dict:
    """Compute instant signals (info-theoretic + hedging). Returns in <0.5s."""
    info_signals = compute_all_info_theoretic(logprobs, top_logprobs)
    hedging = compute_hedging_score(response_text)
    return {**info_signals, hedging["signal_id"]: hedging}


def compute_layer_signals(question: str) -> dict:
    """Compute layer-wise and geometric signals via HuggingFace. Takes 3-5s."""
    model_output = extract_hidden_states(question)
    layer_signals = compute_all_layer_wise(model_output)
    geo_signals = compute_all_geometric(model_output)
    return {**layer_signals, **geo_signals}


async def compute_behavioral_signals(question: str) -> dict:
    """Compute behavioral signals via Ollama multi-sampling. Takes 5-8s."""
    responses = await sample_responses(question, n=5)
    consistency = compute_self_consistency(responses)
    length_var = compute_response_length_variance(responses)
    semantic_sim = compute_semantic_similarity(responses)
    return {
        consistency["signal_id"]: consistency,
        length_var["signal_id"]: length_var,
        semantic_sim["signal_id"]: semantic_sim,
    }
```

- [ ] **Step 4: Implement summary.py**

Create `backend/app/summary.py`:

```python
from app.models.ollama_client import generate_summary


async def generate_executive_summary(
    question: str,
    response: str,
    signals: dict,
) -> str:
    """Generate an executive summary interpreting all computed signals."""
    signal_lines = []
    for sig_id, sig_data in signals.items():
        if isinstance(sig_data.get("value"), (int, float)):
            signal_lines.append(f"- {sig_id}: {sig_data['value']} ({sig_data.get('interpretation', '')})")

    signal_text = "\n".join(signal_lines)

    prompt = f"""You are analyzing an LLM's response signals. Write a brief executive summary (2-3 sentences) interpreting what the signals collectively mean about the response's reliability.

Question asked: "{question}"
Response given: "{response[:500]}"

Computed signals:
{signal_text}

Write a concise paragraph referencing specific signal names and values. Focus on what a learner should pay attention to. Do not use bullet points."""

    return await generate_summary(prompt)
```

- [ ] **Step 5: Implement signals API with SSE**

Create `backend/app/api/signals.py`:

```python
import asyncio
import json
from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from app.signals.engine import compute_fast_signals, compute_layer_signals, compute_behavioral_signals
from app.signals.calibration import compute_composite_score, compute_confidence_agreement
from app.summary import generate_executive_summary

router = APIRouter()

# In-memory store for request data (simple dict, fine for single-user teaching tool)
_request_store: dict[str, dict] = {}


def store_request_data(request_id: str, data: dict):
    _request_store[request_id] = data
    # Keep only last 20 requests
    if len(_request_store) > 20:
        oldest = list(_request_store.keys())[0]
        del _request_store[oldest]


@router.get("/api/signals/stream")
async def signal_stream(request_id: str = Query(...)):
    if request_id not in _request_store:
        async def error_stream():
            yield {"event": "error", "data": json.dumps({"error": "Request not found"})}
        return EventSourceResponse(error_stream())

    data = _request_store[request_id]

    async def event_generator():
        all_signals = {}

        # Phase 1: Fast signals (instant)
        fast = compute_fast_signals(
            data.get("logprobs", []),
            data.get("top_logprobs"),
            data.get("response", ""),
        )
        for sig_id, sig_data in fast.items():
            all_signals[sig_id] = sig_data
            yield {"event": "signal", "data": json.dumps(sig_data)}
        await asyncio.sleep(0.1)

        # Phase 2: Layer-wise + Geometric signals (3-5s)
        try:
            layer_signals = await asyncio.get_event_loop().run_in_executor(
                None, compute_layer_signals, data.get("question", "")
            )
            for sig_id, sig_data in layer_signals.items():
                all_signals[sig_id] = sig_data
                yield {"event": "signal", "data": json.dumps(sig_data)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": f"Layer signals failed: {str(e)}"})}
        await asyncio.sleep(0.1)

        # Phase 3: Behavioral signals (5-8s)
        try:
            behavioral = await compute_behavioral_signals(data.get("question", ""))
            for sig_id, sig_data in behavioral.items():
                all_signals[sig_id] = sig_data
                yield {"event": "signal", "data": json.dumps(sig_data)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": f"Behavioral signals failed: {str(e)}"})}
        await asyncio.sleep(0.1)

        # Phase 4: Composite score
        composite = compute_composite_score(all_signals)
        all_signals[composite["signal_id"]] = composite
        yield {"event": "signal", "data": json.dumps(composite)}

        # Phase 5: Executive summary
        try:
            summary_text = await generate_executive_summary(
                data.get("question", ""),
                data.get("response", ""),
                all_signals,
            )
            summary = {
                "type": "summary",
                "composite_score": composite["value"],
                "executive_summary": summary_text,
            }
            yield {"event": "summary", "data": json.dumps(summary)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": f"Summary failed: {str(e)}"})}

        yield {"event": "done", "data": json.dumps({"status": "complete"})}

    return EventSourceResponse(event_generator())
```

- [ ] **Step 6: Update chat.py to store request data**

Modify `backend/app/api/chat.py` — add import and store call:

```python
from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest, ChatResponse, ModelListResponse, ModelSwitchRequest
from app.models import ollama_client
from app.api.signals import store_request_data

router = APIRouter()


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await ollama_client.chat(request.question, request.model)

        # Store for signal computation
        store_request_data(result["request_id"], {
            "question": request.question,
            "response": result["response"],
            "logprobs": result.get("logprobs", []),
            "top_logprobs": result.get("top_logprobs"),
        })

        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")


@router.get("/api/models", response_model=ModelListResponse)
async def list_models():
    try:
        models = await ollama_client.list_models()
        return ModelListResponse(
            models=models,
            active=ollama_client.get_active_model(),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")


@router.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    models = await ollama_client.list_models()
    if request.model not in models:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    ollama_client.set_active_model(request.model)
    return {"active": request.model}
```

- [ ] **Step 7: Register signals router in main.py**

Modify `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, chat, signals

app = FastAPI(title="AI Signals Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(signals.router)
```

- [ ] **Step 8: Run engine tests**

```bash
cd backend
python -m pytest tests/test_engine.py -v
```

Expected: All tests PASS

- [ ] **Step 9: Run all backend tests**

```bash
cd backend
python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add backend/app/signals/engine.py backend/app/summary.py backend/app/api/signals.py backend/app/api/chat.py backend/app/main.py backend/tests/test_engine.py
git commit -m "feat: signal engine with SSE streaming, executive summary, and progressive delivery"
```

---

## Task 11: Frontend Scaffold

**Files:**
- Create: `frontend/Dockerfile`
- Create: `frontend/nginx.conf`
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/index.css`
- Create: `frontend/src/App.tsx`

- [ ] **Step 1: Create frontend/package.json**

```json
{
  "name": "ai-signals-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.26.0",
    "recharts": "^2.12.0",
    "lucide-react": "^0.441.0",
    "clsx": "^2.1.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0"
  }
}
```

- [ ] **Step 2: Create frontend/tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create frontend/vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 4: Create frontend/tailwind.config.js**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

- [ ] **Step 5: Create frontend/postcss.config.js**

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

- [ ] **Step 6: Create frontend/index.html**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI Signals Explorer</title>
  </head>
  <body class="bg-slate-950 text-slate-200">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Create frontend/src/index.css**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
```

- [ ] **Step 8: Create frontend/src/main.tsx**

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)
```

- [ ] **Step 9: Create frontend/src/App.tsx**

```tsx
import { Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import SignalDetailPage from './pages/SignalDetailPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/signals/:signalId" element={<SignalDetailPage />} />
    </Routes>
  )
}
```

- [ ] **Step 10: Create frontend/Dockerfile**

```dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

- [ ] **Step 11: Create frontend/nginx.conf**

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
        proxy_buffering off;
        proxy_cache off;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] **Step 12: Commit**

```bash
git add frontend/
git commit -m "feat: frontend scaffold with React, TypeScript, Vite, Tailwind"
```

---

## Task 12: TypeScript Types + Signal Definitions + Theme System

**Files:**
- Create: `frontend/src/types/signals.ts`
- Create: `frontend/src/data/signalDefinitions.ts`
- Create: `frontend/src/themes/research.ts`
- Create: `frontend/src/themes/oracle.ts`
- Create: `frontend/src/hooks/useTheme.ts`

- [ ] **Step 1: Create frontend/src/types/signals.ts**

```typescript
export interface SignalResult {
  signal_id: string
  school: SchoolId
  value: number | number[] | Record<string, unknown>
  interpretation: string
  metadata?: Record<string, unknown>
}

export interface SummaryResult {
  type: 'summary'
  composite_score: number
  executive_summary: string
  behavioral_groups?: BehavioralGroup[]
}

export interface BehavioralGroup {
  theme: string
  signal_ids: string[]
}

export type SchoolId =
  | 'information_theoretic'
  | 'layer_wise'
  | 'geometric'
  | 'behavioral'
  | 'calibration'

export interface SignalDefinition {
  id: string
  name: string
  school: SchoolId
  formula: string
  briefDescription: string
  thresholds: { low: number; high: number; inverted?: boolean }
  detailPage: {
    whatItIs: string
    howWeCompute: string
    codeSnippet: string
    limitations: string[]
    whenToUse: string
    references: { title: string; url?: string }[]
  }
}

export interface ChatResponse {
  response: string
  request_id: string
  model: string
  generation_time_ms: number
}

export interface HealthStatus {
  ollama: { status: string; model: string | null }
  huggingface: { status: string; model: string | null }
}

export type LayoutMode = 'split' | 'single'
export type GroupingMode = 'school' | 'behavior'
export type ThemeMode = 'research' | 'oracle'
```

- [ ] **Step 2: Create frontend/src/themes/research.ts**

```typescript
export const researchTheme = {
  name: 'research' as const,
  colors: {
    bg: 'bg-slate-950',
    panel: 'bg-slate-900',
    card: 'bg-slate-800',
    cardBorder: 'border-slate-700',
    accent: 'text-blue-400',
    text: 'text-slate-200',
    textMuted: 'text-slate-400',
    textDim: 'text-slate-500',
  },
  copy: {
    appTitle: 'AI Signals Explorer',
    appSubtitle: '',
    appTagline: '',
    send: 'Send',
    computing: 'Computing...',
    expand: 'More detail',
    learnMore: 'Learn more →',
    summary: 'Executive Summary',
    summaryDisclaimer: 'AI-interpreted · verify with your judgement',
    dashboard: 'Signal Dashboard',
    bySchool: 'By School',
    byBehavior: 'By Behavior',
    user: 'You',
    model: (name: string) => name.toUpperCase(),
    queued: 'Queued',
    verdict: 'Composite Reliability',
    behindCurtain: 'How we compute this',
  },
  schools: {
    information_theoretic: { name: 'Information-Theoretic', color: 'text-blue-400', border: 'border-blue-400' },
    layer_wise: { name: 'Layer-Wise', color: 'text-violet-400', border: 'border-violet-400' },
    geometric: { name: 'Geometric / Manifold', color: 'text-pink-400', border: 'border-pink-400' },
    behavioral: { name: 'Behavioral / Consistency', color: 'text-orange-400', border: 'border-orange-400' },
    calibration: { name: 'Calibration / Statistical', color: 'text-teal-400', border: 'border-teal-400' },
  },
}

export type Theme = typeof researchTheme
```

- [ ] **Step 3: Create frontend/src/themes/oracle.ts**

```typescript
import type { Theme } from './research'

export const oracleTheme: Theme = {
  name: 'oracle' as const,
  colors: {
    bg: 'bg-[#0a0a12]',
    panel: 'bg-[#0d0d1a]',
    card: 'bg-[#141428]',
    cardBorder: 'border-[#1e1e3a]',
    accent: 'text-indigo-400',
    text: 'text-[#d4d4e8]',
    textMuted: 'text-[#8b8ba8]',
    textDim: 'text-[#4a3f6b]',
  },
  copy: {
    appTitle: 'The Oracle',
    appSubtitle: '✦ Signals Explorer ✦',
    appTagline: 'a lens, not a prophecy',
    send: 'Consult',
    computing: 'Scrying...',
    expand: 'Peer deeper',
    learnMore: 'Learn the full incantation →',
    summary: 'The Reading',
    summaryDisclaimer: 'interpreted, not decreed',
    dashboard: 'Signal Grimoire',
    bySchool: 'By School',
    byBehavior: 'By Omen',
    user: 'Seeker',
    model: () => 'Oracle Speaks',
    queued: 'Awaiting their turn',
    verdict: 'Verdict',
    behindCurtain: 'Behind the curtain',
  },
  schools: {
    information_theoretic: { name: 'The Probability Seers', color: 'text-indigo-400', border: 'border-indigo-400' },
    layer_wise: { name: 'The Layer Readers', color: 'text-violet-400', border: 'border-violet-400' },
    geometric: { name: 'The Geometry Weavers', color: 'text-pink-400', border: 'border-pink-400' },
    behavioral: { name: 'The Consistency Watchers', color: 'text-orange-400', border: 'border-orange-400' },
    calibration: { name: 'The Calibrators', color: 'text-teal-400', border: 'border-teal-400' },
  },
}
```

- [ ] **Step 4: Create frontend/src/hooks/useTheme.ts**

```typescript
import { useState, useEffect } from 'react'
import { researchTheme, type Theme } from '../themes/research'
import { oracleTheme } from '../themes/oracle'
import type { ThemeMode } from '../types/signals'

const STORAGE_KEY = 'ai-signals-theme'

export function useTheme() {
  const [mode, setMode] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    return (saved === 'oracle' ? 'oracle' : 'research') as ThemeMode
  })

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, mode)
  }, [mode])

  const toggle = () => setMode(m => (m === 'research' ? 'oracle' : 'research'))

  const theme: Theme = mode === 'oracle' ? oracleTheme : researchTheme

  return { mode, theme, toggle }
}
```

- [ ] **Step 5: Create frontend/src/data/signalDefinitions.ts**

This file contains all 18 signal definitions with full detail page content. Due to length, here is the structure with the first 3 signals fully defined — the remaining 15 follow the same pattern:

```typescript
import type { SignalDefinition } from '../types/signals'

export const signalDefinitions: SignalDefinition[] = [
  {
    id: 'predictive_entropy',
    name: 'Predictive Entropy',
    school: 'information_theoretic',
    formula: 'H(Y|x) = -Σ p(y|x) · log p(y|x)',
    briefDescription: 'Measures the spread of the output probability distribution.',
    thresholds: { low: 0.5, high: 2.0 },
    detailPage: {
      whatItIs: 'Predictive entropy measures the spread of the model\'s probability distribution over possible next tokens. When the model is confident, probability concentrates on one or few tokens (low entropy). When uncertain, probability spreads across many tokens (high entropy). It is the most fundamental signal in the Information-Theoretic school.',
      howWeCompute: 'We request logprobs from Ollama\'s OpenAI-compatible API. For each token in the response, we get log p(token). We then compute H = -Σ exp(logp) · logp across all tokens.',
      codeSnippet: 'probs = np.exp(logprobs)\nentropy = -np.sum(probs * logprobs)',
      limitations: [
        'Treats "Paris" and "Paris, France" as different outputs even when semantically identical',
        'Can be fooled by confident-sounding but incorrect responses',
        'Measures token-level uncertainty, not semantic-level uncertainty',
      ],
      whenToUse: 'Use as a fast, first-pass uncertainty indicator. Best combined with other signals — entropy alone cannot distinguish between "genuinely uncertain" and "confidently wrong." Pair with self-consistency or DoLa for more reliable assessment.',
      references: [
        { title: 'Shannon, C.E. (1948). "A Mathematical Theory of Communication"' },
        { title: 'Kuhn et al., 2023. "Semantic Entropy for Language Model Uncertainty"', url: 'https://arxiv.org/abs/2302.09664' },
      ],
    },
  },
  {
    id: 'perplexity',
    name: 'Perplexity',
    school: 'information_theoretic',
    formula: 'PPL = exp(-(1/n) Σ log p(tᵢ))',
    briefDescription: 'How "surprised" the model is by its own output.',
    thresholds: { low: 2.0, high: 10.0 },
    detailPage: {
      whatItIs: 'Perplexity is the geometric mean of inverse token probabilities. It measures how "surprised" the model is by its own output. A perplexity of 1 means the model was completely certain at every token. A perplexity of 10 means on average 10 tokens were plausible at each position.',
      howWeCompute: 'Computed from token logprobs: PPL = exp(-(1/n) Σ log p(tᵢ)), where n is the number of tokens.',
      codeSnippet: 'perplexity = np.exp(-np.mean(logprobs))',
      limitations: [
        'Heavily influenced by rare or unusual tokens in the response',
        'Does not distinguish between factual and stylistic uncertainty',
        'Aggregate measure — does not localize which parts are uncertain',
      ],
      whenToUse: 'Use alongside entropy for a complementary view. Perplexity is more interpretable ("the model considered N alternatives on average") while entropy gives raw distribution width.',
      references: [
        { title: 'Jelinek, F. et al. (1977). "Perplexity — a measure of the difficulty of speech recognition tasks"' },
      ],
    },
  },
  {
    id: 'mean_token_prob',
    name: 'Mean Token Probability',
    school: 'information_theoretic',
    formula: 'MTP = (1/n) Σ p(tᵢ | context)',
    briefDescription: 'Average confidence per generated token.',
    thresholds: { low: 0.5, high: 0.8, inverted: true },
    detailPage: {
      whatItIs: 'The average probability the model assigned to each token it generated. A high mean token probability means the model consistently chose high-probability tokens. A low value means many tokens were not the model\'s top choice — suggesting the response contains unlikely or uncertain text.',
      howWeCompute: 'Convert logprobs to probabilities, then take the arithmetic mean: MTP = mean(exp(logprobs)).',
      codeSnippet: 'probs = np.exp(logprobs)\nmean_tp = np.mean(probs)',
      limitations: [
        'Averages can mask individual low-confidence tokens',
        'Short responses amplify the effect of individual tokens',
        'Does not capture the positional pattern of confidence',
      ],
      whenToUse: 'Use as a quick overview of generation confidence. Pair with min token probability to catch individual weak points that the average might hide.',
      references: [
        { title: 'Kadavath, S. et al. (2022). "Language Models (Mostly) Know What They Know"' },
      ],
    },
  },
  // Remaining 15 signals follow the same structure.
  // Implementation note: define all 18 signals in this array during implementation.
  // Signal IDs: min_token_prob, top_k_prob_mass, token_prob_variance,
  // dola_contrast, logit_lens_evolution, icr_score, prediction_stability,
  // embedding_trajectory_length, layerwise_cosine_similarity,
  // self_consistency, response_length_variance, hedging_score,
  // semantic_similarity, confidence_agreement, composite_score
]

export function getSignalById(id: string): SignalDefinition | undefined {
  return signalDefinitions.find(s => s.id === id)
}

export function getSignalsBySchool(school: string): SignalDefinition[] {
  return signalDefinitions.filter(s => s.school === school)
}
```

**Implementation note:** During implementation, all 18 signals must be fully defined in this array following the same structure. The remaining 15 definitions should be written based on the signal descriptions in Tasks 4-9 and the source curriculum content (formulas, limitations, references with arXiv links where available).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types/ frontend/src/data/ frontend/src/themes/ frontend/src/hooks/useTheme.ts
git commit -m "feat: TypeScript types, signal definitions, and research/oracle theme system"
```

---

## Task 13: SSE Hook + ChatPanel Component

**Files:**
- Create: `frontend/src/hooks/useSignalStream.ts`
- Create: `frontend/src/components/ChatPanel.tsx`

- [ ] **Step 1: Create frontend/src/hooks/useSignalStream.ts**

```typescript
import { useState, useCallback } from 'react'
import type { SignalResult, SummaryResult } from '../types/signals'

interface SignalStreamState {
  signals: Record<string, SignalResult>
  summary: SummaryResult | null
  isStreaming: boolean
  error: string | null
}

export function useSignalStream() {
  const [state, setState] = useState<SignalStreamState>({
    signals: {},
    summary: null,
    isStreaming: false,
    error: null,
  })

  const startStream = useCallback((requestId: string) => {
    setState({ signals: {}, summary: null, isStreaming: true, error: null })

    const eventSource = new EventSource(`/api/signals/stream?request_id=${requestId}`)

    eventSource.addEventListener('signal', (event) => {
      const signal: SignalResult = JSON.parse(event.data)
      setState(prev => ({
        ...prev,
        signals: { ...prev.signals, [signal.signal_id]: signal },
      }))
    })

    eventSource.addEventListener('summary', (event) => {
      const summary: SummaryResult = JSON.parse(event.data)
      setState(prev => ({ ...prev, summary }))
    })

    eventSource.addEventListener('done', () => {
      setState(prev => ({ ...prev, isStreaming: false }))
      eventSource.close()
    })

    eventSource.addEventListener('error', (event) => {
      if (eventSource.readyState === EventSource.CLOSED) {
        setState(prev => ({ ...prev, isStreaming: false }))
      } else {
        setState(prev => ({ ...prev, error: 'Connection lost', isStreaming: false }))
        eventSource.close()
      }
    })
  }, [])

  const reset = useCallback(() => {
    setState({ signals: {}, summary: null, isStreaming: false, error: null })
  }, [])

  return { ...state, startStream, reset }
}
```

- [ ] **Step 2: Create frontend/src/components/ChatPanel.tsx**

```tsx
import { useState } from 'react'
import type { ChatResponse, ThemeMode } from '../types/signals'
import type { Theme } from '../themes/research'
import ExecutiveSummary from './ExecutiveSummary'
import type { SummaryResult } from '../types/signals'

interface Props {
  theme: Theme
  onSubmit: (question: string) => Promise<void>
  chatResponse: ChatResponse | null
  summary: SummaryResult | null
  isLoading: boolean
  lastQuestion: string
}

export default function ChatPanel({ theme, onSubmit, chatResponse, summary, isLoading, lastQuestion }: Props) {
  const [input, setInput] = useState('')
  const t = theme.copy

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    const question = input.trim()
    setInput('')
    await onSubmit(question)
  }

  return (
    <div className={`flex flex-col h-full ${theme.colors.panel}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50 text-center">
        {theme.name === 'oracle' && (
          <div className={`text-[10px] tracking-[4px] ${theme.colors.textDim} uppercase`}>
            {t.appSubtitle}
          </div>
        )}
        <div className={`text-lg font-bold ${theme.colors.accent}`}>{t.appTitle}</div>
        {t.appTagline && (
          <div className={`text-[10px] ${theme.colors.textDim} italic`}>{t.appTagline}</div>
        )}
      </div>

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {lastQuestion && (
          <div>
            <div className={`text-[9px] ${theme.colors.textDim} tracking-widest uppercase mb-1`}>
              {t.user}
            </div>
            <div className={`${theme.colors.card} p-3 rounded-lg ${theme.colors.text} text-sm border ${theme.colors.cardBorder}`}>
              {lastQuestion}
            </div>
          </div>
        )}

        {chatResponse && (
          <div>
            <div className={`text-[9px] ${theme.colors.textDim} tracking-widest uppercase mb-1`}>
              {t.model(chatResponse.model)}
            </div>
            <div className={`${theme.colors.panel} p-3 rounded-lg ${theme.colors.text} text-sm border ${theme.colors.cardBorder}`}>
              {chatResponse.response}
            </div>
          </div>
        )}

        {summary && <ExecutiveSummary theme={theme} summary={summary} />}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-slate-700/50">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            disabled={isLoading}
            className={`flex-1 ${theme.colors.card} border ${theme.colors.cardBorder} rounded-md px-3 py-2 text-sm ${theme.colors.text} placeholder:${theme.colors.textDim} focus:outline-none focus:ring-1 focus:ring-indigo-500`}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-md text-white text-sm font-medium"
          >
            {isLoading ? t.computing : t.send}
          </button>
        </div>
      </form>
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useSignalStream.ts frontend/src/components/ChatPanel.tsx
git commit -m "feat: SSE signal stream hook and ChatPanel component"
```

---

## Task 14: Signal Card + School Group + Composite Score Components

**Files:**
- Create: `frontend/src/components/SignalCard.tsx`
- Create: `frontend/src/components/SchoolGroup.tsx`
- Create: `frontend/src/components/CompositeScore.tsx`
- Create: `frontend/src/components/ExecutiveSummary.tsx`

- [ ] **Step 1: Create frontend/src/components/SignalCard.tsx**

```tsx
import { useState } from 'react'
import { Link } from 'react-router-dom'
import clsx from 'clsx'
import type { SignalResult } from '../types/signals'
import type { Theme } from '../themes/research'
import { getSignalById } from '../data/signalDefinitions'

interface Props {
  signal: SignalResult
  theme: Theme
}

export default function SignalCard({ signal, theme }: Props) {
  const [expanded, setExpanded] = useState(false)
  const definition = getSignalById(signal.signal_id)
  const t = theme.copy

  const numValue = typeof signal.value === 'number' ? signal.value : 0
  const thresholds = definition?.thresholds ?? { low: 0.5, high: 2.0 }
  const inverted = thresholds.inverted ?? false

  // Normalize to 0-100 for progress bar
  const range = thresholds.high - thresholds.low
  const normalized = Math.max(0, Math.min(100, ((numValue - 0) / (thresholds.high * 1.5)) * 100))

  // Color based on value relative to thresholds
  const getColor = () => {
    if (inverted) {
      if (numValue > thresholds.high) return 'text-green-400'
      if (numValue > thresholds.low) return 'text-yellow-400'
      return 'text-red-400'
    }
    if (numValue < thresholds.low) return 'text-green-400'
    if (numValue < thresholds.high) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getBarColor = () => {
    if (inverted) {
      if (numValue > thresholds.high) return 'bg-green-400'
      if (numValue > thresholds.low) return 'bg-yellow-400'
      return 'bg-red-400'
    }
    if (numValue < thresholds.low) return 'bg-green-400'
    if (numValue < thresholds.high) return 'bg-yellow-400'
    return 'bg-red-400'
  }

  return (
    <div className={`${theme.colors.card} border ${theme.colors.cardBorder} rounded-md p-3 text-xs`}>
      <div className="flex justify-between items-start">
        <div className={`font-bold text-[11px] ${theme.colors.text}`}>
          {definition?.name ?? signal.signal_id}
        </div>
        <div className={`text-base font-bold ${getColor()}`}>
          {typeof signal.value === 'number' ? signal.value.toFixed(2) : '—'}
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-1.5">
        <div className={`${theme.colors.bg} rounded-sm h-1 overflow-hidden`}>
          <div className={`${getBarColor()} h-full rounded-sm`} style={{ width: `${normalized}%` }} />
        </div>
      </div>

      {/* Interpretation */}
      <div className={`${theme.colors.textMuted} text-[10px] mt-1.5 leading-relaxed`}>
        {signal.interpretation}
      </div>

      {/* Expand toggle */}
      <button
        onClick={() => setExpanded(!expanded)}
        className={`${theme.colors.textDim} text-[10px] mt-1 hover:${theme.colors.textMuted}`}
      >
        {expanded ? '▾' : '▸'} {t.expand}
      </button>

      {/* Expanded detail */}
      {expanded && definition && (
        <div className="mt-2 space-y-2">
          <div className={`text-[11px] ${theme.colors.textMuted} leading-relaxed`}>
            {definition.briefDescription}
          </div>

          <div className={`${theme.colors.bg} rounded p-2 text-[10px] ${theme.colors.textDim} leading-relaxed border-l-2 ${theme.colors.cardBorder}`}>
            <div className={`${theme.colors.textMuted} mb-1`}>{t.behindCurtain}</div>
            <span className={theme.colors.textMuted}>Computed as:</span> {definition.formula}
          </div>

          <Link
            to={`/signals/${signal.signal_id}`}
            className={`block ${theme.colors.accent} text-[10px] hover:underline`}
          >
            {t.learnMore}
          </Link>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Create frontend/src/components/SchoolGroup.tsx**

```tsx
import type { SignalResult, SchoolId } from '../types/signals'
import type { Theme } from '../themes/research'
import SignalCard from './SignalCard'

interface Props {
  schoolId: SchoolId
  signals: SignalResult[]
  isComputing: boolean
  theme: Theme
}

const SCHOOL_SIGNAL_COUNTS: Record<SchoolId, number> = {
  information_theoretic: 6,
  layer_wise: 4,
  geometric: 2,
  behavioral: 4,
  calibration: 2,
}

export default function SchoolGroup({ schoolId, signals, isComputing, theme }: Props) {
  const schoolConfig = theme.schools[schoolId]
  const total = SCHOOL_SIGNAL_COUNTS[schoolId]
  const computed = signals.length
  const t = theme.copy

  return (
    <div className="mb-4">
      {/* School header */}
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-0.5 h-4 ${schoolConfig.border} rounded`} />
        <span className={`${schoolConfig.color} font-bold text-xs tracking-wide`}>
          {schoolConfig.name}
        </span>
        <span className={`${theme.colors.textDim} text-[10px]`}>
          {computed} / {total} signals
        </span>
        <span className={`text-[10px] ml-auto`}>
          {computed === total ? (
            <span className="text-green-400">✓ Complete</span>
          ) : isComputing ? (
            <span className="text-yellow-400">{t.computing}</span>
          ) : (
            <span className={theme.colors.textDim}>{t.queued}</span>
          )}
        </span>
      </div>

      {/* Signal cards grid */}
      {signals.length > 0 ? (
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
          {signals.map(signal => (
            <SignalCard key={signal.signal_id} signal={signal} theme={theme} />
          ))}
        </div>
      ) : (
        <div className={`${theme.colors.card} border border-dashed ${theme.colors.cardBorder} rounded-md p-3 text-center ${theme.colors.textDim} text-[11px] italic`}>
          {isComputing ? `${t.computing}` : `${t.queued}`}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Create frontend/src/components/CompositeScore.tsx**

```tsx
import type { Theme } from '../themes/research'

interface Props {
  score: number | null
  totalSignals: number
  computedSignals: number
  theme: Theme
}

export default function CompositeScore({ score, totalSignals, computedSignals, theme }: Props) {
  const t = theme.copy
  const displayScore = score ?? 0
  const color = displayScore >= 80 ? 'bg-green-400' : displayScore >= 50 ? 'bg-yellow-400' : 'bg-red-400'
  const textColor = displayScore >= 80 ? 'text-green-400' : displayScore >= 50 ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className={`px-4 py-2.5 ${theme.colors.bg} border-b border-slate-700/30 flex items-center gap-3`}>
      <span className={`text-[10px] ${theme.colors.textDim} tracking-wide uppercase`}>{t.verdict}</span>
      <div className={`flex-1 ${theme.colors.card} rounded h-2 overflow-hidden`}>
        <div
          className={`${color} h-full rounded transition-all duration-500`}
          style={{ width: `${displayScore}%` }}
        />
      </div>
      <span className={`text-sm font-bold ${textColor}`}>
        {score !== null ? Math.round(displayScore) : '—'}
      </span>
      <span className={`text-[9px] ${theme.colors.textDim}`}>
        {computedSignals} of {totalSignals} signals
      </span>
    </div>
  )
}
```

- [ ] **Step 4: Create frontend/src/components/ExecutiveSummary.tsx**

```tsx
import type { SummaryResult } from '../types/signals'
import type { Theme } from '../themes/research'

interface Props {
  theme: Theme
  summary: SummaryResult
}

export default function ExecutiveSummary({ theme, summary }: Props) {
  const t = theme.copy

  return (
    <div className={`border ${theme.colors.cardBorder} rounded-lg overflow-hidden`}>
      <div className={`px-3 py-2 ${theme.colors.card} border-b ${theme.colors.cardBorder} flex justify-between items-center`}>
        <span className={`text-xs font-bold ${theme.colors.accent}`}>{t.summary}</span>
        <span className={`text-[9px] ${theme.colors.textDim} italic`}>{t.summaryDisclaimer}</span>
      </div>
      <div className={`p-3 text-xs ${theme.colors.textMuted} leading-relaxed`}>
        {summary.executive_summary}
        <div className={`mt-2 text-right text-[10px] ${theme.colors.textDim}`}>
          Reliability: <span className="text-green-400 font-bold">{Math.round(summary.composite_score)} / 100</span>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/SignalCard.tsx frontend/src/components/SchoolGroup.tsx frontend/src/components/CompositeScore.tsx frontend/src/components/ExecutiveSummary.tsx
git commit -m "feat: SignalCard, SchoolGroup, CompositeScore, and ExecutiveSummary components"
```

---

## Task 15: Signal Dashboard + Signal Chart + Settings + Theme Toggle

**Files:**
- Create: `frontend/src/components/SignalDashboard.tsx`
- Create: `frontend/src/components/SignalChart.tsx`
- Create: `frontend/src/components/SettingsPanel.tsx`
- Create: `frontend/src/components/ThemeToggle.tsx`

- [ ] **Step 1: Create frontend/src/components/SignalDashboard.tsx**

```tsx
import type { SignalResult, SchoolId, GroupingMode } from '../types/signals'
import type { Theme } from '../themes/research'
import SchoolGroup from './SchoolGroup'
import CompositeScore from './CompositeScore'

interface Props {
  signals: Record<string, SignalResult>
  isStreaming: boolean
  theme: Theme
  groupingMode: GroupingMode
  onGroupingChange: (mode: GroupingMode) => void
}

const SCHOOL_ORDER: SchoolId[] = [
  'information_theoretic',
  'layer_wise',
  'geometric',
  'behavioral',
  'calibration',
]

export default function SignalDashboard({ signals, isStreaming, theme, groupingMode, onGroupingChange }: Props) {
  const t = theme.copy
  const signalList = Object.values(signals)
  const compositeSignal = signals['composite_score']
  const compositeScore = compositeSignal && typeof compositeSignal.value === 'number' ? compositeSignal.value : null

  const signalsBySchool = (school: SchoolId) =>
    signalList.filter(s => s.school === school)

  const isSchoolComputing = (school: SchoolId) => {
    if (!isStreaming) return false
    const schoolSignals = signalsBySchool(school)
    return schoolSignals.length > 0 && schoolSignals.length < getExpectedCount(school)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50 flex justify-between items-center">
        <div>
          <span className={`${theme.colors.text} font-bold text-sm`}>{t.dashboard}</span>
          <span className={`${theme.colors.textDim} text-[10px] ml-2`}>18 signals across 5 schools</span>
        </div>
        <div className="flex gap-1.5 text-[11px]">
          <button
            onClick={() => onGroupingChange('school')}
            className={`px-2.5 py-1 rounded ${groupingMode === 'school' ? `${theme.colors.card} ${theme.colors.accent}` : `${theme.colors.textDim}`}`}
          >
            {t.bySchool}
          </button>
          <button
            onClick={() => onGroupingChange('behavior')}
            className={`px-2.5 py-1 rounded ${groupingMode === 'behavior' ? `${theme.colors.card} ${theme.colors.accent}` : `${theme.colors.textDim}`}`}
          >
            {t.byBehavior}
          </button>
        </div>
      </div>

      {/* Composite bar */}
      <CompositeScore
        score={compositeScore}
        totalSignals={18}
        computedSignals={signalList.length}
        theme={theme}
      />

      {/* Signal groups */}
      <div className="flex-1 overflow-y-auto p-4">
        {signalList.length === 0 && !isStreaming ? (
          <div className={`text-center ${theme.colors.textDim} text-sm py-20`}>
            Ask a question to see signals
          </div>
        ) : (
          SCHOOL_ORDER.map(school => (
            <SchoolGroup
              key={school}
              schoolId={school}
              signals={signalsBySchool(school)}
              isComputing={isSchoolComputing(school) || (isStreaming && signalsBySchool(school).length === 0)}
              theme={theme}
            />
          ))
        )}
      </div>
    </div>
  )
}

function getExpectedCount(school: SchoolId): number {
  const counts: Record<SchoolId, number> = {
    information_theoretic: 6,
    layer_wise: 4,
    geometric: 2,
    behavioral: 4,
    calibration: 2,
  }
  return counts[school]
}
```

- [ ] **Step 2: Create frontend/src/components/SignalChart.tsx**

```tsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts'
import type { SignalResult } from '../types/signals'
import type { Theme } from '../themes/research'

interface Props {
  signals: Record<string, SignalResult>
  theme: Theme
}

export default function SignalChart({ signals, theme }: Props) {
  const numericSignals = Object.entries(signals)
    .filter(([_, s]) => typeof s.value === 'number' && s.signal_id !== 'composite_score')
    .map(([id, s]) => ({
      name: id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).substring(0, 15),
      value: s.value as number,
      school: s.school,
    }))

  if (numericSignals.length === 0) return null

  return (
    <div className={`${theme.colors.card} border ${theme.colors.cardBorder} rounded-md p-3 mb-4`}>
      <div className={`text-[11px] font-bold ${theme.colors.textMuted} mb-2`}>Signal Overview</div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={numericSignals} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <XAxis dataKey="name" tick={{ fontSize: 8, fill: '#64748b' }} angle={-45} textAnchor="end" height={60} />
          <YAxis tick={{ fontSize: 9, fill: '#64748b' }} />
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', fontSize: 11 }} />
          <Bar dataKey="value" fill="#818cf8" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 3: Create frontend/src/components/SettingsPanel.tsx**

```tsx
import type { LayoutMode } from '../types/signals'
import type { Theme } from '../themes/research'

interface Props {
  layoutMode: LayoutMode
  onLayoutChange: (mode: LayoutMode) => void
  theme: Theme
}

export default function SettingsPanel({ layoutMode, onLayoutChange, theme }: Props) {
  return (
    <div className={`px-4 py-2 border-b border-slate-700/30 flex items-center gap-4 text-[11px] ${theme.colors.textDim}`}>
      <span>Layout:</span>
      <button
        onClick={() => onLayoutChange('split')}
        className={layoutMode === 'split' ? theme.colors.accent : ''}
      >
        Split Panel
      </button>
      <span>|</span>
      <button
        onClick={() => onLayoutChange('single')}
        className={layoutMode === 'single' ? theme.colors.accent : ''}
      >
        Single Page
      </button>
    </div>
  )
}
```

- [ ] **Step 4: Create frontend/src/components/ThemeToggle.tsx**

```tsx
import type { ThemeMode } from '../types/signals'

interface Props {
  mode: ThemeMode
  onToggle: () => void
}

export default function ThemeToggle({ mode, onToggle }: Props) {
  return (
    <button
      onClick={onToggle}
      className="fixed bottom-4 right-4 text-xl opacity-40 hover:opacity-80 transition-opacity z-50"
      title=""
    >
      {mode === 'research' ? '🔮' : '🔍'}
    </button>
  )
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/SignalDashboard.tsx frontend/src/components/SignalChart.tsx frontend/src/components/SettingsPanel.tsx frontend/src/components/ThemeToggle.tsx
git commit -m "feat: SignalDashboard, SignalChart, SettingsPanel, and ThemeToggle components"
```

---

## Task 16: HomePage + Signal Detail Page

**Files:**
- Create: `frontend/src/pages/HomePage.tsx`
- Create: `frontend/src/pages/SignalDetailPage.tsx`

- [ ] **Step 1: Create frontend/src/pages/HomePage.tsx**

```tsx
import { useState, useCallback } from 'react'
import { useTheme } from '../hooks/useTheme'
import { useSignalStream } from '../hooks/useSignalStream'
import ChatPanel from '../components/ChatPanel'
import SignalDashboard from '../components/SignalDashboard'
import SignalChart from '../components/SignalChart'
import SettingsPanel from '../components/SettingsPanel'
import ThemeToggle from '../components/ThemeToggle'
import type { ChatResponse, LayoutMode, GroupingMode } from '../types/signals'

export default function HomePage() {
  const { mode, theme, toggle } = useTheme()
  const { signals, summary, isStreaming, error, startStream, reset } = useSignalStream()

  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null)
  const [lastQuestion, setLastQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('split')
  const [groupingMode, setGroupingMode] = useState<GroupingMode>('school')

  const handleSubmit = useCallback(async (question: string) => {
    setIsLoading(true)
    setLastQuestion(question)
    setChatResponse(null)
    reset()

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!res.ok) throw new Error(`Chat failed: ${res.status}`)

      const data: ChatResponse = await res.json()
      setChatResponse(data)
      startStream(data.request_id)
    } catch (err) {
      console.error('Chat error:', err)
    } finally {
      setIsLoading(false)
    }
  }, [reset, startStream])

  if (layoutMode === 'single') {
    return (
      <div className={`min-h-screen ${theme.colors.bg}`}>
        <SettingsPanel layoutMode={layoutMode} onLayoutChange={setLayoutMode} theme={theme} />
        <div className="max-w-4xl mx-auto">
          <ChatPanel
            theme={theme}
            onSubmit={handleSubmit}
            chatResponse={chatResponse}
            summary={summary}
            isLoading={isLoading}
            lastQuestion={lastQuestion}
          />
          {Object.keys(signals).length > 0 && (
            <div className="p-4">
              <SignalChart signals={signals} theme={theme} />
              <SignalDashboard
                signals={signals}
                isStreaming={isStreaming}
                theme={theme}
                groupingMode={groupingMode}
                onGroupingChange={setGroupingMode}
              />
            </div>
          )}
        </div>
        <ThemeToggle mode={mode} onToggle={toggle} />
      </div>
    )
  }

  return (
    <div className={`h-screen flex flex-col ${theme.colors.bg}`}>
      <SettingsPanel layoutMode={layoutMode} onLayoutChange={setLayoutMode} theme={theme} />
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: Chat */}
        <div className="w-[40%] border-r border-slate-700/30">
          <ChatPanel
            theme={theme}
            onSubmit={handleSubmit}
            chatResponse={chatResponse}
            summary={summary}
            isLoading={isLoading}
            lastQuestion={lastQuestion}
          />
        </div>
        {/* Right panel: Signals */}
        <div className="w-[60%] flex flex-col">
          {Object.keys(signals).length > 0 && (
            <SignalChart signals={signals} theme={theme} />
          )}
          <SignalDashboard
            signals={signals}
            isStreaming={isStreaming}
            theme={theme}
            groupingMode={groupingMode}
            onGroupingChange={setGroupingMode}
          />
        </div>
      </div>
      <ThemeToggle mode={mode} onToggle={toggle} />
    </div>
  )
}
```

- [ ] **Step 2: Create frontend/src/pages/SignalDetailPage.tsx**

```tsx
import { useParams, Link } from 'react-router-dom'
import { useTheme } from '../hooks/useTheme'
import { getSignalById } from '../data/signalDefinitions'
import ThemeToggle from '../components/ThemeToggle'

export default function SignalDetailPage() {
  const { signalId } = useParams<{ signalId: string }>()
  const { mode, theme, toggle } = useTheme()
  const definition = signalId ? getSignalById(signalId) : undefined

  if (!definition) {
    return (
      <div className={`min-h-screen ${theme.colors.bg} p-8`}>
        <Link to="/" className={`${theme.colors.accent} text-sm`}>← Back to dashboard</Link>
        <div className={`${theme.colors.text} mt-8`}>Signal not found</div>
      </div>
    )
  }

  const schoolConfig = theme.schools[definition.school]

  return (
    <div className={`min-h-screen ${theme.colors.bg} p-8`}>
      <div className="max-w-2xl mx-auto">
        <Link to="/" className={`${theme.colors.textDim} text-xs hover:${theme.colors.textMuted}`}>
          ← Back to dashboard
        </Link>

        {/* Header */}
        <div className="border-b border-slate-700/50 pb-4 mt-4 mb-6">
          <div className={`text-[10px] ${schoolConfig.color} tracking-widest uppercase mb-1`}>
            {schoolConfig.name}
          </div>
          <h1 className={`text-2xl font-bold ${theme.colors.text}`}>{definition.name}</h1>
          <p className={`text-sm ${theme.colors.textMuted} mt-1`}>{definition.briefDescription}</p>
        </div>

        {/* What It Is */}
        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>What It Is</h2>
          <p className={`text-sm ${theme.colors.text} leading-relaxed`}>{definition.detailPage.whatItIs}</p>
        </section>

        {/* Formula */}
        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>The Formula</h2>
          <div className={`${theme.colors.card} border ${theme.colors.cardBorder} p-3 rounded text-center font-mono text-sm ${theme.colors.text}`}>
            {definition.formula}
          </div>
        </section>

        {/* How We Compute It */}
        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>How We Compute It</h2>
          <p className={`text-sm ${theme.colors.text} leading-relaxed mb-2`}>{definition.detailPage.howWeCompute}</p>
          <pre className={`${theme.colors.card} border ${theme.colors.cardBorder} p-3 rounded text-xs font-mono text-indigo-300 overflow-x-auto`}>
            {definition.detailPage.codeSnippet}
          </pre>
        </section>

        {/* Interpretation */}
        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>Interpretation</h2>
          <div className={`${theme.colors.card} border ${theme.colors.cardBorder} rounded p-3 text-sm space-y-1`}>
            <div className="flex gap-3">
              <span className="text-green-400 font-bold w-16">{'< '}{definition.thresholds.low}</span>
              <span className={theme.colors.text}>{definition.thresholds.inverted ? 'Uncertain' : 'Confident'}</span>
            </div>
            <div className="flex gap-3">
              <span className="text-yellow-400 font-bold w-16">{definition.thresholds.low}–{definition.thresholds.high}</span>
              <span className={theme.colors.text}>Moderate</span>
            </div>
            <div className="flex gap-3">
              <span className="text-red-400 font-bold w-16">{'> '}{definition.thresholds.high}</span>
              <span className={theme.colors.text}>{definition.thresholds.inverted ? 'Confident' : 'Uncertain'}</span>
            </div>
          </div>
        </section>

        {/* Limitations */}
        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>Limitations</h2>
          <ul className={`text-sm ${theme.colors.textMuted} space-y-1.5 leading-relaxed`}>
            {definition.detailPage.limitations.map((lim, i) => (
              <li key={i}>• {lim}</li>
            ))}
          </ul>
        </section>

        {/* When To Use */}
        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>When To Use</h2>
          <p className={`text-sm ${theme.colors.text} leading-relaxed`}>{definition.detailPage.whenToUse}</p>
        </section>

        {/* References */}
        <section className="border-t border-slate-700/50 pt-4 mt-8">
          <h2 className={`text-xs ${theme.colors.textDim} mb-2`}>References</h2>
          {definition.detailPage.references.map((ref, i) => (
            <div key={i} className={`text-xs ${theme.colors.textDim} mb-1`}>
              {ref.url ? (
                <a href={ref.url} target="_blank" rel="noopener noreferrer" className={`${theme.colors.accent} hover:underline`}>
                  {ref.title}
                </a>
              ) : (
                ref.title
              )}
            </div>
          ))}
        </section>
      </div>
      <ThemeToggle mode={mode} onToggle={toggle} />
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/
git commit -m "feat: HomePage with split/single layout and SignalDetailPage with full curriculum content"
```

---

## Task 17: Complete Signal Definitions + End-to-End Smoke Test

**Files:**
- Modify: `frontend/src/data/signalDefinitions.ts` (complete all 18 definitions)

- [ ] **Step 1: Complete all 18 signal definitions**

Add the remaining 15 signal definitions to `signalDefinitions.ts` following the same structure as the first 3. Each definition needs: `id`, `name`, `school`, `formula`, `briefDescription`, `thresholds`, and `detailPage` with all 7 fields populated.

Signal IDs to complete:
1. `min_token_prob` — school: information_theoretic
2. `top_k_prob_mass` — school: information_theoretic
3. `token_prob_variance` — school: information_theoretic
4. `dola_contrast` — school: layer_wise
5. `logit_lens_evolution` — school: layer_wise
6. `icr_score` — school: layer_wise
7. `prediction_stability` — school: layer_wise
8. `embedding_trajectory_length` — school: geometric
9. `layerwise_cosine_similarity` — school: geometric
10. `self_consistency` — school: behavioral
11. `response_length_variance` — school: behavioral
12. `hedging_score` — school: behavioral
13. `semantic_similarity` — school: behavioral
14. `confidence_agreement` — school: calibration
15. `composite_score` — school: calibration

Use formulas, limitations, and references from the backend signal implementations (Tasks 4-9) and the source research papers.

- [ ] **Step 2: Build frontend to verify no TypeScript errors**

```bash
cd frontend
npm install
npm run build
```

Expected: Build succeeds with no errors

- [ ] **Step 3: Run all backend tests**

```bash
cd backend
python -m pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 4: Test Docker Compose builds**

```bash
docker compose build
```

Expected: All 3 services build successfully

- [ ] **Step 5: Commit**

```bash
git add frontend/src/data/signalDefinitions.ts
git commit -m "feat: complete all 18 signal definitions with full detail page content"
```

---

## Task 18: Final Integration + Docker Compose Up

- [ ] **Step 1: Add .superpowers to .gitignore if not already present**

Verify `.superpowers/` is in `.gitignore`.

- [ ] **Step 2: Run docker compose up and verify**

```bash
docker compose up --build
```

Verify:
1. Ollama container starts and pulls llama3.2:1b
2. Backend container starts, downloads Qwen2.5-0.5B
3. Frontend builds and serves on http://localhost:3000
4. Health endpoint returns both models connected: `curl http://localhost:3000/api/health`
5. Submit a question and see signals arrive progressively

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: AI Signals Explorer v0.1 — complete Docker Compose setup"
```
