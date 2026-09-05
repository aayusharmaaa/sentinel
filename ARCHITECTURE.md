# Architecture

Sentinel is a two-speed merchant-trust system: cheap payment telemetry watches
everyone; expensive channels (vision, agent investigation, graph, voice) run
only when triage earns the spend. A fusion gate turns channel disagreements into
a response ladder (`T0`–`T4`). No single channel can hold settlements.

The demo API and console are **read-only over pipeline artifacts**. Scoring
happens offline; the live surface serves cards, metrics, and fixtures from
`artifacts/`.

---

## System overview

```mermaid
flowchart TB
  subgraph Data["Data generation"]
    GEN["sentinel.data.generate"]
    ART_IN["merchants.csv · features · scripts"]
  end

  subgraph Pipeline["Offline pipeline"]
    TEL["Telemetry · GBDT + baselines"]
    TRI["Triage · 4 cheap triggers"]
    AGT["Agent · 4 tools · ≤8 steps"]
    VIS["Vision · homepage + checkout"]
    GR["Graph · rarity + synchrony"]
    VO["Voice verify · gated ≥ T3"]
    GATE["Fusion gate · T0–T4"]
    CARD["Evidence card + narrative"]
  end

  subgraph Artifacts["artifacts/"]
    DEC["decisions.csv"]
    CARDS["cards/*.json"]
    EVAL["evaluation.json"]
    FIX["fixtures/*.html"]
  end

  subgraph Surfaces["Demo surfaces"]
    API["FastAPI · thin over disk"]
    LAND["/ · landing"]
    CON["/console · review queue"]
  end

  GEN --> ART_IN --> TEL --> TRI
  GR --> TRI
  TRI -->|selected| AGT
  AGT --> VIS
  VIS --> GATE
  AGT --> GATE
  GR --> GATE
  GATE -->|tier ≥ VERIFY| VO --> GATE
  GATE --> CARD
  CARD --> CARDS
  GATE --> DEC
  DEC --> EVAL
  CARDS --> API
  DEC --> API
  EVAL --> API
  FIX --> API
  API --> LAND
  API --> CON
```

---

## How a merchant is decided

```mermaid
flowchart LR
  M[Merchant window] --> T[Cheap telemetry]
  T --> Q{Triage?}
  Q -->|no| T0[T0 Monitor]
  Q -->|yes| I[Bounded investigation]
  I --> C[Channel verdicts]
  C --> G{Two-channel gate}
  G -->|storefront + ≥1| T4[T4 Restrict]
  G -->|storefront only| T3[T3 Verify]
  G -->|telemetry / net only| T2[T2 Review]
  G -->|weak| T1[T1 Recheck]
  T2 & T3 & T4 --> E[Evidence card]
  E --> UI[Analyst console]
```

Only **T4** stops money, and only when the storefront channel dissents **plus**
at least one independent corroborator. T3 exists so a single-channel
disagreement has somewhere to go that is not a freeze.

---

## Modules

| Area | Path | Role |
|------|------|------|
| **API** | `sentinel/api/` | Serves landing/console HTML and JSON over disk artifacts. Does not re-score. |
| **Pipeline** | `sentinel/pipeline.py` | Score → triage → investigate → fuse → decide → write cards. |
| **Telemetry** | `sentinel/telemetry/` | Deterministic features, category baselines, GBDT + attribution. |
| **Triage** | `sentinel/triage.py` | Four cheap triggers that earn vision/agent spend. |
| **Vision** | `sentinel/vision/` | Storefront fixtures + VLM/simulator; homepage and checkout. |
| **Agent** | `sentinel/agent/` | Bounded investigation loop over four tools. |
| **Graph** | `sentinel/graph/` | Rarity-weighted shared-infra graph; cohesion + synchrony. |
| **Fusion** | `sentinel/fusion/` | Channel verdicts + response ladder; two-channel hold rule. |
| **Voice** | `sentinel/voice/` | Simulated verification call; only on cases already ≥ T3. |
| **Narrative** | `sentinel/narrative/` | Structured-features-only summary and case copilot. |
| **Data / eval** | `sentinel/data/`, `sentinel/eval/` | Synthetic population; held-out metrics → `evaluation.json`. |
| **Console** | `console/` | Landing overview + analyst review UI. |

---

## Runtime artifacts

| Artifact | Used for |
|----------|----------|
| `artifacts/cards/*.json` | Console queue and case evidence |
| `artifacts/decisions.csv` | Landing population / threshold interactives |
| `artifacts/evaluation.json` | Headline metrics, ablations, economics |
| `artifacts/fixtures/*.html` | Storefront surfaces opened beside findings |
| `artifacts/throughput.json` | Cost / funnel narrative on landing |

Large regenerated inputs (`features_raw.csv`, `merchants.csv`, …) are not
required to run the demo console once cards and evaluation are present.

---

## Request flows

### Landing (`/`)

1. Browser loads `console/landing.html`
2. Client fetches `GET /api/landing` → `decisions.csv` + `evaluation.json`
3. Optional storefront embeds via `GET /api/storefront/{vertical}/{surface}`

### Console (`/console`)

1. Browser loads `console/index.html`
2. `GET /api/queue` lists `artifacts/cards/*.json`
3. Open case → `GET /api/card/{mid}`; storefront → fixtures; ask → `/api/ask`
4. Analyst ruling → `POST /api/decision` → `analyst_decisions.json`

The API never recomputes tiers. Missing artifacts → HTTP 503 with “run pipeline
/ eval first.”

---

## Offline vs live

| Mode | Behaviour |
|------|-----------|
| **Default demo** | Pipeline already run; vision uses an error-model simulator; agent uses a heuristic planner; console is artifact-backed. |
| **Live** (`ANTHROPIC_API_KEY`) | Vision/narrative/copilot can call real models; **re-run the pipeline** to refresh cards. The HTTP surface stays artifact-backed. |

Entry for hosts that look for `main:app` (local uvicorn, Vercel): `main.py`
re-exports `app` from `sentinel.api.app`.
