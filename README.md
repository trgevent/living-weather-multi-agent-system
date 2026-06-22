# Living Weather: A Multi-Agent Weather System Built on Graceful Degradation
Kaggle 5-Day AI Agents Intensive — Vibe Coding Capstone Project

A 5-agent system that fetches real weather data, never goes silent when an API fails, refuses to send safety alerts based on unreliable data, and calibrates its own trustworthiness over time. Built entirely with local Python, FastAPI, Redis, and PostgreSQL — no cloud platform dependency.

Full writeup (architecture, design decisions, evidence): [Kaggle Writeup — link to be added]
Interactive demo (runs the core agents live, no Docker required): [Kaggle Notebook — living-weather-multi-agent-system]

## The Problem

Weather data sources fail. APIs go down, rate-limit, or return garbage. Most weather apps either crash, show a blank screen, or — worse — confidently display stale data without telling you. For a weather system specifically, the stakes are concrete: a wrong "no severe weather" reading isn't a cosmetic bug, it's a safety failure.

Living Weather treats "the data source is unreliable right now" as a first-class state the system reasons about, not an exception to catch and ignore — built to demonstrate, with working code, how production-grade patterns (graceful degradation, policy-gated actions, evaluation beyond binary tests) apply directly to this kind of failure surface.

## Architecture
| Agent | Responsibility |
|---|---|
| Master Agent | Orchestrates the other agents in sequence. |
| DI-Agent | Fetches real weather data from Open-Meteo. Wrapped in a circuit breaker. |
| LLM-Agent | Seasonal fallback, activated when DI-Agent's confidence drops below 0.5. |
| AS-Agent | Checks thresholds and issues alerts through a Policy Server — never automatically on low-confidence data. |
| FL-Agent | Records predictions, compares them against actual outcomes with a source-specific tolerance band. |
| RI-Agent (optional) | Plans a route between two points; generates a recommendation with an LLM and judges its own output with a second LLM call. |
| PA-Agent (optional) | Generates personalized clothing/health/activity advice from a weather reading, using placeholder resolution ([[LOCATION]], [[TEMP_C]]) — rule-based, no LLM call. |
| MC-Agent (optional) | Compares the current micro-climate between two free-form coordinate pairs (not tied to any fixed city table); generates a comparison with an LLM and judges its own output, same pattern as RI-Agent. |

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
│   ├── ri_agent/agent.py
│   ├── pa_agent/agent.py
│   └── mc_agent/agent.py
└── api/
    └── main.py                FastAPI endpoints (/health, /weather, /route, /advisory, /microclimate)

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
curl -X POST http://127.0.0.1:8000/advisory -H "Content-Type: application/json" -d "{\"location\": \"İzmir\", \"user_name\": \"Levent\"}"
curl -X POST http://127.0.0.1:8000/microclimate -H "Content-Type: application/json" -d "{\"point_a\": \"İzmir\", \"point_b\": \"Bodrum\"}"
```

Each agent can also be run standalone to see its own test scenarios:

```bash
python -m app.agents.di_agent.agent
python -m app.agents.llm_agent.agent
python -m app.agents.as_agent.agent
python -m app.agents.fl_agent.agent
python -m app.agents.master.agent
python -m app.agents.ri_agent.agent
python -m app.agents.pa_agent.agent
python -m app.agents.mc_agent.agent
```

RI-Agent's and MC-Agent's LLM features (text generation, LLM-as-judge evaluation) require a `GEMINI_API_KEY` environment variable. Without it, the system falls back to a deterministic, rule-based recommendation and a neutral evaluation score — it does not crash.

## Why no Cloud Run deployment
The original plan included deploying this to Google Cloud Run. It was dropped deliberately: it isn't a requirement for this capstone (the official deliverables are a writeup, a demo video, and a code link — not a live deployment), and it would have worked against this project's longer-term goal of integrating into a self-hosted personal assistant without depending on any single cloud or model provider.

## Notable design decisions
See the full writeup for details, but briefly:

- **Graceful degradation, proven at three layers**: data ingestion (circuit breaker), reasoning (seasonal fallback), and evaluation (RI-Agent's and MC-Agent's text generation and judging can each fail independently without crashing the system).
- **Least-privilege Policy Server**: AS-Agent gets a purpose-built role that can only call `send_severe_weather_alert` and `get_weather_forecast` — nothing else.
- **No automatic alerts on low-confidence data** — flagged for human review instead. This is the project's clearest safety decision.
- **Source-specific tolerance bands**: the same 10°C prediction error scores 0.0 for Open-Meteo (a real measurement) but 1.88 for the seasonal LLM-Agent estimate (held to a fairer standard).
- **LLM-as-judge caught real "context hallucination"**: a fluent, confident-sounding LLM-generated route recommendation scored 1.0/5.0 because it gave no actionable advice — the judge wasn't fooled by fluency. After fixing a task-description/rubric inconsistency, the same scenario scored 5.0/5.0 live in the Kaggle Notebook, and the same fix independently improved MC-Agent's score from 2.0/5.0 to 5.0/5.0 — the same root cause, found and fixed once, verified across two agents.
- **Free-form coordinates, not a fixed lookup table**: MC-Agent takes raw `(latitude, longitude)` pairs rather than depending on any hardcoded city table, so the same capability can be reused later in other location-aware projects without rework.