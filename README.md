# Living Weather: A Multi-Agent Weather System Built on Graceful Degradation

Kaggle 5-Day AI Agents Intensive — Vibe Coding Capstone Project

A 5-agent system that fetches real weather data, never goes silent when an API fails, refuses to send safety alerts based on unreliable data, and calibrates its own trustworthiness over time. Built entirely with local Python, FastAPI, Redis, and PostgreSQL — no cloud platform dependency.

**Full writeup (architecture, design decisions, evidence):** [Kaggle Writeup — link to be added]
**Interactive demo (runs the core agents live, no Docker required):** [Kaggle Notebook — living-weather-multi-agent-system]

## Architecture

| Agent | Responsibility |
|---|---|
| **Master Agent** | Orchestrates the other agents in sequence. |
| **DI-Agent** | Fetches real weather data from Open-Meteo. Wrapped in a circuit breaker. |
| **LLM-Agent** | Seasonal fallback, activated when DI-Agent's confidence drops below 0.5. |
| **AS-Agent** | Checks thresholds and issues alerts through a Policy Server — never automatically on low-confidence data. |
| **FL-Agent** | Records predictions, compares them against actual outcomes with a source-specific tolerance band. |
| **RI-Agent** *(optional)* | Plans a route between two points; generates a recommendation with an LLM and judges its own output with a second LLM call. |

```
User query
   │
   ▼
Master Agent ── resolves location → coordinates
   │
   ▼
DI-Agent ── tries real API (circuit breaker protected)
   │
   ├─ confidence ≥ 0.5 ───────────────► use DI-Agent's reading
   │
   └─ confidence < 0.5 ──► LLM-Agent ► use seasonal estimate (flagged low-confidence)
   │
   ▼
AS-Agent ── threshold check
   │
   ├─ confidence ≥ 0.5 and threshold breached → Policy Server → alert
   └─ confidence < 0.5 and threshold breached → flagged "requires human review", NO auto-alert
   │
   ▼
FL-Agent ── records the prediction for future calibration
```

## Project structure

```
app/
├── core/
│   ├── models.py              Shared Pydantic data models
│   ├── policy_server.py       Structural + semantic gating (Policy Server)
│   ├── policies.yaml          Role/environment permission rules
│   ├── evaluation_engine.py   Tolerance-band scoring + LLM-as-judge
│   ├── redis_circuit_breaker.py   Redis-backed circuit breaker (shared across workers)
│   ├── db.py                  PostgreSQL access layer (raw SQL via psycopg)
│   ├── schema.sql             PostgreSQL table definition
│   └── context_resolver.py    Placeholder resolution + PII masking
├── agents/
│   ├── master/agent.py
│   ├── di_agent/agent.py
│   ├── llm_agent/agent.py
│   ├── as_agent/agent.py
│   ├── fl_agent/agent.py
│   └── ri_agent/agent.py
└── api/
    └── main.py                FastAPI endpoints (/health, /weather, /route)

docker-compose.yml              Redis + PostgreSQL containers (local dev only)
```

## Running it locally

Requires Python 3.10+, Docker Desktop.

```bash
# 1. Install dependencies
pip install fastapi uvicorn httpx pydantic pyyaml redis "psycopg[binary]" google-genai

# 2. Start Redis + PostgreSQL
docker compose up -d

# 3. Initialize the database schema
python -m app.core.db

# 4. Run the API
uvicorn app.api.main:app --reload --port 8000
```

Then visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI, or:

```bash
curl -X POST http://127.0.0.1:8000/weather -H "Content-Type: application/json" -d "{\"location\": \"İzmir\"}"
curl -X POST http://127.0.0.1:8000/route -H "Content-Type: application/json" -d "{\"origin\": \"İzmir\", \"destination\": \"Bodrum\"}"
```

Each agent can also be run standalone to see its own test scenarios:

```bash
python -m app.agents.di_agent.agent
python -m app.agents.llm_agent.agent
python -m app.agents.as_agent.agent
python -m app.agents.fl_agent.agent
python -m app.agents.master.agent
python -m app.agents.ri_agent.agent
```

`RI-Agent`'s LLM features (route text generation, LLM-as-judge evaluation) require a `GEMINI_API_KEY` environment variable. Without it, the system falls back to a deterministic, rule-based recommendation and a neutral evaluation score — it does not crash.

## Why no Cloud Run deployment

The original plan included deploying this to Google Cloud Run. It was dropped deliberately: it isn't a requirement for this capstone (the official deliverables are a writeup, a demo video, and a code link — not a live deployment), and it would have worked against this project's longer-term goal of integrating into a self-hosted personal assistant without depending on any single cloud or model provider.

## Notable design decisions

See the full writeup for details, but briefly:
- **Graceful degradation, proven at three layers**: data ingestion (circuit breaker), reasoning (seasonal fallback), and evaluation (RI-Agent's text generation and judging can each fail independently without crashing the system).
- **Least-privilege Policy Server**: AS-Agent gets a purpose-built role that can only call `send_severe_weather_alert` and `get_weather_forecast` — nothing else.
- **No automatic alerts on low-confidence data** — flagged for human review instead. This is the project's clearest safety decision.
- **Source-specific tolerance bands**: the same 10°C prediction error scores 0.0 for Open-Meteo (a real measurement) but 1.88 for the seasonal LLM-Agent estimate (held to a fairer standard).
- **LLM-as-judge caught real "context hallucination"**: a fluent, confident-sounding LLM-generated route recommendation scored 1.0/5.0 because it gave no actionable advice — the judge wasn't fooled by fluency. After a one-line prompt fix, the same scenario scored 4.0/5.0, then 5.0/5.0 live in the Kaggle Notebook.
