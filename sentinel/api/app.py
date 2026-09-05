"""The analyst console.

Serves the queue and the evidence cards produced by a pipeline run. It is a
thin layer over artefacts on disk, deliberately: the console must not be able
to compute anything the pipeline did not, or the thing an analyst sees stops
being the thing the audit trail records.

Three interface obligations that are not decoration:

  * The narrative states its provenance in the interface. A reviewer needs to
    know, without asking anyone, that the summary was written from structured
    features and never from the merchant's own page copy. If that is only true
    in a docstring, it is not a control.

  * The case copilot answers questions from the same structured-features
    allowlist. Conversational convenience must not open a channel from scraped
    page text into what the analyst reads before deciding.

  * Ground truth is shown as ground truth. These cards come from a synthetic
    corpus, and the console labels the label as something that does not exist
    at decision time. A demo that quietly shows the answer next to the
    prediction teaches the wrong thing about what the system knows.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from sentinel.config import ARTIFACTS, ROOT

app = FastAPI(title="Sentinel", version="1.0")
CONSOLE = ROOT / "console"


def _cards_dir() -> Path:
    d = ARTIFACTS / "cards"
    if not d.exists():
        raise HTTPException(
            503,
            "No evidence cards on disk. Run `python -m sentinel.pipeline` first.",
        )
    return d


@app.get("/")
def index() -> FileResponse:
    """The landing page. The console is the working surface behind it."""
    return FileResponse(CONSOLE / "landing.html")


@app.get("/theme.css")
def theme_css() -> FileResponse:
    """The shared design system for the landing page and the console."""
    return FileResponse(CONSOLE / "theme.css", media_type="text/css")


@app.get("/console")
def console() -> FileResponse:
    return FileResponse(CONSOLE / "index.html")


@app.get("/api/landing")
def landing_data() -> JSONResponse:
    """Real held-out population behind the landing page's interactive panels."""
    from sentinel.api.landing import build_landing_payload

    if not (ARTIFACTS / "evaluation.json").exists():
        raise HTTPException(
            503,
            "Run `python -m sentinel.pipeline` and `python -m sentinel.eval.harness` first.",
        )
    return JSONResponse(build_landing_payload())


@app.get("/standalone")
def standalone() -> FileResponse:
    """The publishable single-file build, for checking it before publishing."""
    path = ARTIFACTS / "sentinel.html"
    if not path.exists():
        raise HTTPException(503, "Run `python scripts/build_web.py` first.")
    return FileResponse(path, media_type="text/html")


@app.get("/showcase")
def showcase() -> FileResponse:
    """The generated showcase page, built by scripts/build_showcase.py."""
    path = ARTIFACTS / "showcase.html"
    if not path.exists():
        raise HTTPException(503, "Run `python scripts/build_showcase.py` first.")
    return FileResponse(path, media_type="text/html")


@app.get("/api/queue")
def queue() -> JSONResponse:
    """The analyst queue, highest tier first."""
    from sentinel.narrative.copilot import queue_brief

    rows = []
    for path in sorted(_cards_dir().glob("*.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        d = card["decision"]
        brief = queue_brief(card)
        rows.append({
            "merchant_id": card["merchant_id"],
            "declared_category": card["declared_category"],
            "tier": d["tier"],
            "tier_name": d["tier_name"],
            "settlements_held": d["settlements_held"],
            "dissenting_channels": d["dissenting_channels"],
            "second_channel_source": d["second_channel_source"],
            "telemetry_score": round(card["telemetry"]["score"], 3),
            "vision_vertical": (
                next(iter(card["vision"].values()))["vertical"]
                if card.get("vision") else None
            ),
            "steps_used": card["investigation"].get("steps_used", 0),
            "injection_flagged": any(
                v.get("injection", {}).get("present")
                for v in (card.get("vision") or {}).values()
            ),
            "brief": brief["brief"],
            "urgency": brief["urgency"],
        })
    urgency_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    rows.sort(
        key=lambda r: (
            -r["tier"],
            -urgency_rank.get(r["urgency"], 0),
            -r["telemetry_score"],
        )
    )
    return JSONResponse(rows)


@app.post("/api/ask")
def ask_case(body: dict) -> JSONResponse:
    """Ask a grounded question about one evidence card.

    Answers use the same structured-features allowlist as the narrative
    summariser. The console must never send page copy through this path.
    """
    from sentinel.narrative.copilot import ask, guided_steps, suggest_for

    mid = str(body.get("merchant_id", "")).strip()
    question = str(body.get("question", "")).strip()
    if not mid:
        raise HTTPException(400, "merchant_id is required.")
    path = _cards_dir() / f"{mid}.json"
    if not path.exists():
        raise HTTPException(404, f"No evidence card for {mid}.")
    card = json.loads(path.read_text(encoding="utf-8"))
    if not question:
        return JSONResponse({
            "answer": (
                "Ask anything about this case — why it was queued, what each "
                "channel found, or whether a settlement hold is justified."
            ),
            "provenance": "structured-features-only (prompt)",
            "suggestions": suggest_for(card),
            "intent": "empty",
            "guide": guided_steps(card),
        })
    reply = ask(card, question)
    out = reply.to_dict()
    out["guide"] = guided_steps(card)
    return JSONResponse(out)


@app.get("/api/guide/{merchant_id}")
def case_guide(merchant_id: str) -> JSONResponse:
    """Guided review checklist for one case."""
    from sentinel.narrative.copilot import guided_steps, suggest_for

    path = _cards_dir() / f"{merchant_id}.json"
    if not path.exists():
        raise HTTPException(404, f"No evidence card for {merchant_id}.")
    card = json.loads(path.read_text(encoding="utf-8"))
    return JSONResponse({
        "steps": guided_steps(card),
        "suggestions": suggest_for(card),
    })


@app.get("/api/card/{merchant_id}")
def card(merchant_id: str) -> JSONResponse:
    path = _cards_dir() / f"{merchant_id}.json"
    if not path.exists():
        raise HTTPException(404, f"No evidence card for {merchant_id}.")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/evaluation")
def evaluation() -> JSONResponse:
    path = ARTIFACTS / "evaluation.json"
    if not path.exists():
        raise HTTPException(
            503, "No evaluation on disk. Run `python -m sentinel.eval.harness`."
        )
    # evaluation.json legitimately contains NaN (precision is undefined when a
    # policy holds nobody). json.loads accepts it; JSONResponse will not.
    from sentinel.api.landing import clean

    return JSONResponse(clean(json.loads(path.read_text(encoding="utf-8"))))


@app.get("/api/throughput")
def throughput() -> JSONResponse:
    path = ARTIFACTS / "throughput.json"
    if not path.exists():
        raise HTTPException(503, "No throughput report. Run the pipeline first.")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


DECISIONS = ARTIFACTS / "analyst_decisions.json"


def _decisions() -> dict:
    if DECISIONS.exists():
        try:
            return json.loads(DECISIONS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


@app.get("/api/decisions")
def list_decisions() -> JSONResponse:
    """What the analyst has already ruled on."""
    return JSONResponse(_decisions())


@app.post("/api/decision")
def record_decision(body: dict) -> JSONResponse:
    """Record an analyst's ruling on a case.

    The console is only useful if it closes the loop: a queue you cannot clear
    is a report, not a workflow. Rulings are appended to disk so the queue
    survives a reload, and the record keeps who decided what and when, because
    that is the half of the audit trail the model does not write.
    """
    import datetime

    mid = str(body.get("merchant_id", "")).strip()
    action = str(body.get("action", "")).strip()
    if not mid or action not in {"uphold", "release", "more_info", "reopen"}:
        raise HTTPException(400, "merchant_id and a valid action are required.")

    store = _decisions()
    if action == "reopen":
        store.pop(mid, None)
    else:
        store[mid] = {
            "action": action,
            "note": str(body.get("note", ""))[:500],
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    DECISIONS.write_text(json.dumps(store, indent=2), encoding="utf-8")
    return JSONResponse({"ok": True, "merchant_id": mid, "state": store.get(mid)})


@app.get("/api/storefront/{vertical}/{surface}")
def storefront(vertical: str, surface: str) -> FileResponse:
    """The rendered fixture the storefront channel actually looked at.

    An analyst reading a finding like "deposit and withdraw as the two primary
    actions" should be able to open the page and check it. A finding nobody can
    check against the surface it came from is not evidence.
    """
    path = ARTIFACTS / "fixtures" / f"{vertical}__{surface}.html"
    if not path.exists():
        raise HTTPException(404, "No such storefront fixture.")
    return FileResponse(path, media_type="text/html")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
