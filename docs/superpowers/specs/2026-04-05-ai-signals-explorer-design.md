# AI Signals Explorer — Design Specification

## Overview

A teaching and portfolio showcase tool that lets users ask questions to a local LLM (via Ollama) and visualizes the model's internal signals in real-time. The app makes LLM observability concepts tangible — extracting, displaying, grouping, and interpreting 18 signals across 5 schools of thought, with progressive delivery, interactive charts, and per-response executive summaries.

**Primary purpose:** Teaching/demo + portfolio showcase for AI learners
**Core insight:** LLMs are not black boxes. Their internal states are readable. This tool makes that readable.

---

## Architecture

### Three-Layer System

```
┌─────────────────────┐     SSE/REST     ┌─────────────────────┐            ┌─────────────────────┐
│   Frontend          │ ◄──────────────► │   Backend           │ ◄────────► │   Model Layer       │
│   React + TS + Vite │                  │   Python FastAPI     │            │                     │
│   Tailwind + Recharts│                  │                     │            │   Ollama (1B)       │
│                     │                  │   Signal Engine      │            │   HuggingFace (0.5B)│
└─────────────────────┘                  └─────────────────────┘            └─────────────────────┘
```

- **Frontend**: React + TypeScript + Vite + Tailwind + Recharts
- **Backend**: Python FastAPI + SSE (sse-starlette) for progressive signal delivery
- **Model Layer**: Ollama (Llama 3.2 1B Q4, default — user can switch models in settings) for chat/logprobs/multi-sampling + HuggingFace (Qwen2.5-0.5B on CPU) for hidden state extraction

### Data Flow

1. User submits question → `POST /api/chat` → Ollama generates response with logprobs
2. Response returned to frontend immediately
3. Backend kicks off signal computation and streams results via `GET /api/signals/stream` (SSE)
4. Fast signals arrive first (entropy, perplexity ~instant) → layer-wise (~3-5s) → behavioral (~5-8s)
5. Executive summary generated last (after all signals computed) via Ollama

### Packaging & Deployment

**Docker Compose** — single command startup, zero friction:

```
1. Install Docker Desktop (one-time)
2. git clone the repo
3. docker compose up
4. Open http://localhost:3000
```

- Frontend: built to static files, served by nginx container
- Backend: Python container with all dependencies baked in (including PyTorch CPU-only)
- Ollama: runs as a Docker service (CPU-only)
- Ollama model: auto-pulled on first startup into a persistent Docker volume
- HuggingFace model: auto-downloaded on first startup into a persistent Docker volume
- First launch: ~5-10 min (model downloads). Subsequent launches: seconds.
- A "Downloading models... first time only" progress screen shows in the UI during initial setup.

---

## Dependencies

### Prerequisites

- Docker Desktop (only requirement on user's machine)

### Frontend (npm — inside Docker)

| Package | Purpose |
|---------|---------|
| react, react-dom | UI framework |
| typescript | Type safety |
| vite | Build tool & dev server |
| react-router-dom | Signal detail page routing |
| tailwindcss | Utility-first CSS |
| recharts | Signal charts (bar, radar, line) |
| lucide-react | Icons |
| clsx | Conditional class names |

### Backend (pip — baked into Docker image)

| Package | Purpose |
|---------|---------|
| fastapi | Web framework |
| uvicorn | ASGI server |
| sse-starlette | Server-Sent Events |
| pydantic | Data validation |
| httpx | Async HTTP client for Ollama API |
| numpy | Numerical computation |
| scipy | Jensen-Shannon Divergence (ICR) |
| transformers | HuggingFace model loading |
| torch (CPU-only) | Tensor operations (~800MB, baked in) |
| scikit-learn | NearestNeighbors (geometric), probes |

### Models (downloaded at runtime, persisted in Docker volumes)

| Model | Source | Size | Purpose |
|-------|--------|------|---------|
| llama3.2:1b (default) | Ollama (auto-pulled) | ~1GB | Chat + logprobs + multi-sampling |
| Qwen/Qwen2.5-0.5B | HuggingFace (auto-downloaded) | ~1GB | Hidden state extraction (24 layers) |

Users can switch the Ollama model via settings (e.g., to llama3.2:3b or any other installed model).

### Docker Image Size

| Component | Size |
|-----------|------|
| Python slim base | ~120MB |
| PyTorch CPU-only | ~800MB |
| transformers + tokenizers | ~300MB |
| scipy + scikit-learn + numpy | ~100MB |
| FastAPI + other deps | ~30MB |
| Frontend (nginx + static) | ~30MB |
| **Total image (no models)** | **~1.4-1.8GB** |

### Estimated Runtime Resource Usage

- RAM: ~6-8 GB total (Ollama 1B: ~2GB, Qwen 0.5B: ~2GB, Python+containers: ~2GB)
- Disk: ~4 GB total (models: ~2GB, Docker image: ~1.8GB)
- GPU: not used (all inference on CPU)

---

## Signal Catalog

18 signals across 5 schools, computed progressively:

### School 1: Information-Theoretic (6 signals) — via Ollama logprobs

| # | Signal | Formula | Speed | Interpretation |
|---|--------|---------|-------|----------------|
| 1 | Predictive Entropy | H = -Σ p·log(p) | Instant | Spread of output distribution |
| 2 | Perplexity | PPL = exp(-(1/n) Σ log p(tᵢ)) | Instant | How "surprised" the model is by its own output |
| 3 | Mean Token Probability | MTP = (1/n) Σ p(tᵢ) | Instant | Average confidence per token |
| 4 | Min Token Probability | min(p(tᵢ)) | Instant | Weakest link — least confident token |
| 5 | Top-k Probability Mass | Pₖ = Σᵢ₌₁ᵏ p(tᵢ) | Instant | Concentration in top tokens |
| 6 | Token Probability Variance | Var(p(tᵢ)) | Instant | Consistency of confidence across response |

### School 2: Layer-Wise (4 signals) — via HuggingFace hidden states

| # | Signal | Formula | Speed | Interpretation |
|---|--------|---------|-------|----------------|
| 7 | DoLa Contrast Score | softmax(log(q_final / q_early)) | 3-5s | Factual knowledge amplification |
| 8 | Logit Lens Evolution | Track top prediction per layer | 3-5s | How predictions form across layers |
| 9 | ICR Score | JSD(p_early ‖ p_final) | 3-5s | Did the model "change its mind" across layers |
| 10 | Prediction Stability | Count of top-prediction flips | 3-5s | Layer-to-layer prediction consistency |

### School 3: Geometric/Manifold (2 signals) — via HuggingFace hidden states

| # | Signal | Formula | Speed | Interpretation |
|---|--------|---------|-------|----------------|
| 11 | Embedding Trajectory Length | Σ ‖hₜ - hₜ₋₁‖₂ | 3-5s | Distance traveled through representation space |
| 12 | Layer-wise Cosine Similarity | cos(hₗ, hₗ₊₁) | 3-5s | How much each layer changes the representation |

### School 4: Behavioral/Consistency (4 signals) — via Ollama multi-sampling + text

| # | Signal | Formula | Speed | Interpretation |
|---|--------|---------|-------|----------------|
| 13 | Self-Consistency Score | max_a \|{r : answer(r) = a}\| / N | 5-8s | Agreement across 5 sampled responses |
| 14 | Response Length Variance | Var(len(rᵢ)) | 5-8s | Consistent length = confident |
| 15 | Hedging Score | Regex pattern matching | Instant | Linguistic uncertainty markers |
| 16 | Semantic Similarity | avg cosine_sim(embed(rᵢ), embed(rⱼ)) | 5-8s | Proxy for semantic entropy |

### School 5: Calibration/Statistical (2 signals) — derived

| # | Signal | Formula | Speed | Interpretation |
|---|--------|---------|-------|----------------|
| 17 | Confidence-Uncertainty Agreement | Verbalized vs signal confidence | 1-2s | Miscalibration detection |
| 18 | Composite Reliability Score | Weighted aggregate (0-100) | After all | Single "bottom line" metric |

### Progressive Delivery Timeline

```
0-0.5s  →  7 signals  (entropy, perplexity, MTP, min-TP, top-k, variance, hedging)
1-5s    →  7 signals  (DoLa, logit lens, ICR, stability, trajectory, cosine sim, confidence agreement)
5-10s   →  3 signals  (self-consistency, length variance, semantic similarity)
Last    →  Composite score + executive summary
```

The progressive delivery naturally teaches the efficiency-accuracy tradeoff — fast signals are cheap approximations, slow signals provide deeper insight.

---

## UI Design

### Layout Modes

Two layout modes, toggled via settings:

1. **Split-panel (default):** Left panel = chat (question input + response + executive summary). Right panel = signal dashboard (grouped cards + charts + composite score bar).
2. **Single-page:** Linear scroll — question input at top, response below, then signals grouped by school. Same progressive reveal behavior.

Both modes share the same signal components; only the layout container changes.

### Signal Grouping

Two grouping modes, toggled in the dashboard header:

1. **By School (default):** Signals grouped under their school (Information-Theoretic, Layer-Wise, etc.) with color-coded headers.
2. **By Behavior:** Dynamic grouping based on what signals are saying for this specific response. Uses an LLM call to interpret patterns (e.g., "These 5 signals all indicate high confidence", "These 3 disagree"). Toggle labeled "By Omen" in Oracle mode.

### Signal Cards

Each signal renders as a card containing:
- Signal name + numerical value (prominent)
- Mini progress bar showing where the value falls in its range
- One-line contextual interpretation (what this means for *this* response)
- Expandable "More detail" section showing:
  - Brief explanation of the signal
  - "How we compute this" — formula + data source
  - Caveat (e.g., "Confidence ≠ correctness")
  - "Learn more →" link to the full detail page

### Signal Detail Pages

Separate route (`/signals/:signalId`) with full content per signal:

1. **What It Is** — plain English explanation
2. **The Formula** — mathematical notation
3. **How We Compute It** — implementation approach + code snippet
4. **Interpretation** — threshold table (green/yellow/red)
5. **Charts/Visualizations** — relevant graphs from source research (performance AUROC charts, Pareto frontiers, reliability diagrams where applicable)
6. **Limitations** — honest caveats
7. **When To Use** — practical guidance
8. **References** — links to research papers (arXiv links where available, otherwise author/title citations only — no internal curriculum references)

All content is pre-written (static). 18 pages total.

### Executive Summary

A per-response paragraph that synthesizes all computed signals into a readable narrative with inline references to specific signals. Example:

> The model responded with **high confidence** (entropy: 0.3, self-consistency: 9/10) and **stable layer agreement** (ICR: 0.08). However, the claim of *987 AD* shows **moderate hesitation** in deeper layers (DoLa: +8%). The date may warrant your own verification. Overall reliability: **High (87/100)**.

Signal names in the summary are clickable — they highlight/scroll to the corresponding signal card.

Generated by Ollama after all signals are computed. Labeled as AI-interpreted.

### Composite Reliability Bar

Horizontal bar at the top of the signal dashboard:
- 0-100 score (weighted aggregate of all signals)
- Color gradient (red → yellow → green)
- Progress counter ("14 of 18 signals computed")

---

## Theme System

### Research Mode (Default)

Professional, clean dashboard aesthetic:
- Dark slate palette (#0f172a, #1e293b, #334155)
- Technical language (Dashboard, Computing, Expand, Send, Summary)

### Oracle Mode (Easter Egg)

Mystical-but-deadpan theme. The humor is in the contrast between rigorous signal analysis and mystical framing:
- Deep indigo/purple palette (#0a0a12, #141428, #1e1e3a)
- Subtle glow effects on cards
- Renamed UI chrome (signal names stay technical):

| Research Mode | Oracle Mode |
|--------------|-------------|
| Dashboard | Signal Grimoire |
| Computing... | Scrying... |
| Expand | Peer deeper |
| Send | Consult |
| Summary | The Reading |
| By Behavior | By Omen |
| User | Seeker |
| Model response | Oracle Speaks |
| How we compute this | Behind the curtain |
| Learn more | Learn the full incantation |
| Queued | Awaiting their turn |

**Signal names stay technical.** "Predictive Entropy" is still "Predictive Entropy". The mysticism is in the chrome, not the content.

**School names in Oracle mode:**
- Information-Theoretic → The Probability Seers
- Layer-Wise → The Layer Readers
- Geometric/Manifold → The Geometry Weavers
- Behavioral/Consistency → The Consistency Watchers
- Calibration/Statistical → The Calibrators

**Discovery:** A small 🔮 in the bottom-right corner — no label, no tooltip. Click → smooth CSS transition to Oracle mode → icon becomes 🔍 to switch back. State persists in localStorage.

**Oracle mode identity:**
- Title: "The Oracle"
- Subtitle: "✦ Signals Explorer ✦"
- Tagline: "a lens, not a prophecy"

### Implementation

Theme config lives in `themes/research.ts` and `themes/oracle.ts` — colors, labels, copy. Components read from `useTheme()` hook. Same components, different config.

---

## Transparency Philosophy

The app presents itself as a **lens, not an oracle**. Transparency is baked in gently — empowering, not trust-breaking.

### Three Layers

**1. Signal card level:**
Each expanded card shows "How we compute this" and research basis. Learner sees: raw value → interpretation → methodology. Not a warning — just honest provenance.

**2. Executive summary level:**
Small label: *"AI-interpreted · verify with your judgement"* (Research) or *"interpreted, not decreed"* (Oracle). Honest framing, not scary disclaimer.

**3. App-wide level:**
Research mode: implicit in professional design. Oracle mode tagline: *"a lens, not a prophecy."*

### Tone

- NOT: "WARNING: Do not trust these results."
- YES: "Here's the value. Here's what research says it means. Here's how we computed it. Your judgement completes the picture."

Signals are proxies, not oracles. Confidence ≠ correctness. Multiple signals disagreeing is itself informative. These are facts presented as educational context, not disclaimers.

---

## Project Structure

```
ai-signals/
├── docker-compose.yml
├── README.md
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── App.tsx
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
│       ├── pages/
│       │   ├── HomePage.tsx
│       │   └── SignalDetailPage.tsx
│       ├── hooks/
│       │   ├── useSignalStream.ts
│       │   └── useTheme.ts
│       ├── data/
│       │   └── signalDefinitions.ts
│       ├── themes/
│       │   ├── research.ts
│       │   └── oracle.ts
│       └── types/
│           └── signals.ts
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── api/
│       │   ├── chat.py
│       │   ├── signals.py
│       │   └── health.py
│       ├── signals/
│       │   ├── engine.py
│       │   ├── info_theoretic.py
│       │   ├── layer_wise.py
│       │   ├── geometric.py
│       │   ├── behavioral.py
│       │   └── calibration.py
│       ├── models/
│       │   ├── ollama_client.py
│       │   └── hf_model.py
│       ├── summary.py
│       └── schemas.py
│
└── docs/
    └── future-roadmap.md
```

---

## API Endpoints

### POST /api/chat

Request:
```json
{
  "question": "What is the capital of France?",
  "model": "llama3.2:1b"
}
```

Response:
```json
{
  "response": "Paris is the capital of France...",
  "request_id": "uuid",
  "model": "llama3.2:1b",
  "generation_time_ms": 1200
}
```

### GET /api/signals/stream?request_id={id}

SSE stream. Each event:
```json
{
  "signal_id": "predictive_entropy",
  "school": "information_theoretic",
  "value": 0.3,
  "interpretation": "Low entropy — model is confident in this response",
  "metadata": {
    "per_token_values": [0.1, 0.4, 0.2],
    "threshold_low": 0.5,
    "threshold_high": 2.0
  }
}
```

Final event:
```json
{
  "type": "summary",
  "composite_score": 87,
  "executive_summary": "The model responded with high confidence...",
  "behavioral_groups": [
    {
      "theme": "High confidence signals",
      "signal_ids": ["predictive_entropy", "self_consistency", "mean_token_prob"]
    }
  ]
}
```

### GET /api/health

```json
{
  "ollama": { "status": "connected", "model": "llama3.2:1b" },
  "huggingface": { "status": "loaded", "model": "Qwen/Qwen2.5-0.5B" }
}
```

### GET /api/models

Returns available Ollama models for the settings model picker:
```json
{
  "models": ["llama3.2:1b", "llama3.2:3b", "mistral:7b"],
  "active": "llama3.2:1b"
}
```

### POST /api/models/switch

```json
{
  "model": "llama3.2:3b"
}
```

---

## Future Roadmap (Phase 2)

Tracked in `docs/future-roadmap.md`:

1. **Cross-question comparison** — Pick any two past questions and compare signals side-by-side. Session trend panel showing how signals change across questions.
2. **Contextual addendum on detail pages** — Dynamic section showing "How this signal applied to your last question" (extra LLM call per detail page visit).
3. **Additional signals** — Semantic Entropy (full NLI-based), REMA with reference manifold, activation patching.
4. **Model comparison** — Run the same question through two different Ollama models and compare signal profiles side-by-side.
5. **Export** — Export signal reports as PDF or shareable link.

---

## Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Ollama + HF dual model | Yes | Ollama for UX (fast chat), HF for depth (hidden states). Best signal coverage without killing laptop. |
| Default Ollama model | llama3.2:1b | Smaller (~1GB), sufficient for teaching demos. User can switch to larger models in settings. |
| HF model | Qwen2.5-0.5B | Smallest model with meaningful layer depth (24 layers). Smaller models lack enough layers for DoLa/ICR. |
| CPU-only inference | Yes | User's MX450 (2GB VRAM) too small. 32GB RAM handles CPU inference fine. |
| PyTorch baked in Docker | Yes | ~800MB but reliable. Runtime pip install is fragile and slow. One-time image pull. |
| SSE over WebSocket | SSE | Simpler, one-way stream sufficient. Better FastAPI support. |
| Docker Compose with Ollama | Yes | Zero-friction. User installs Docker, runs one command. |
| Static detail pages | Yes | Consistent quality, no extra LLM calls. Dynamic addendum deferred to Phase 2. |
| Oracle mode as easter egg | Yes | Adds personality without compromising professional first impression. |
| 18 signals (not 147) | Yes | Covers 5 of 7 schools with real computable signals. Mechanistic + Representation Engineering require full weight access or are too expensive. |
| Detail page references | Paper citations only | No internal curriculum references. ArXiv links where available, author/title otherwise. |
| Detail page charts | Include from research | Performance AUROC, Pareto frontiers, reliability diagrams where source research provides them. |
