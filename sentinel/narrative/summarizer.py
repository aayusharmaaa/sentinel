"""The analyst rationale.

An LLM writes the narrative, from the structured feature vector only, never
from raw page text.

The reason is narrow and worth stating precisely. If scraped page content
reaches the summariser, a merchant can write copy on their own site designed to
influence what the compliance narrative says about them -- not to change the
verdict, which the gate protects, but to change the sentence a human reads
before deciding whether to act on it. That is a softer target and a more useful
one. Keeping the generator strictly downstream of structured features closes
the path entirely, because there is no channel from the page into the text.

Two things make that a real guarantee rather than a promise:

  * The input is built by an allowlist. `build_summary_input` names every field
    that may reach the model, and nothing else can arrive by accident later
    when someone adds a field upstream.
  * `assert_no_page_text` runs before the call and fails loudly if any string
    from the rendered page appears in the payload. A guarantee nothing checks
    is a guarantee that quietly stops holding.

The console states the provenance in the interface, so a reviewer knows what
the summary was and was not built from without having to trust this docstring.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from sentinel.fusion.gate import LADDER, Decision, Tier

MODEL = "claude-opus-5"

SYSTEM = """\
You write the analyst rationale for a payment aggregator's merchant monitoring \
system. Your reader is a compliance analyst who may be about to hold a real \
business's settlements, which is the same as holding their payroll.

You will receive ONLY structured findings: channel verdicts, feature deviations \
against a category baseline, and an investigation trace. You will never receive \
the merchant's page copy, and you must not speculate about anything outside \
what you are given.

Write four to six sentences, plain prose, no headings or bullets.

State what the merchant declared and what each channel found. Name the specific \
features and their deviations. Where the channels disagree, say so plainly \
rather than resolving it. If the recommendation is a settlement hold, state \
what the second corroborating channel was, because a single channel is never \
sufficient. If a channel is structurally blind to this kind of case, say that \
too.

Do not overstate. "Consistent with" is not "proves". The analyst decides; you \
are describing evidence, not reaching a verdict. Never recommend an action \
beyond the tier you are given.
"""


@dataclass
class Narrative:
    text: str
    provenance: str
    inputs_used: list[str]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "provenance": self.provenance,
            "inputs_used": self.inputs_used,
        }


# Fields permitted to reach the summariser. Adding one is a deliberate act.
ALLOWED_FIELDS = [
    "merchant_id",
    "declared_category",
    "onboarded_days_ago",
    "tier",
    "tier_name",
    "channel_verdicts",
    "top_feature_deviations",
    "investigation_steps",
    "stop_reason",
    "injection_flagged",
    "second_channel_source",
    # Added deliberately: the contradiction engine's structured output. Still
    # no page text and no call audio -- only the extracted claim and the
    # disagreements between views.
    "contradictions",
    "evidence_agreement",
    "model_confidence",
    "action_confidence",
    "network",
    "statement",
]


def build_summary_input(
    decision: Decision,
    feature_deviations: list[dict],
    investigation: dict,
    onboarded_days_ago: int,
    injection_flagged: bool,
    assessment: dict | None = None,
    network: dict | None = None,
    voice: dict | None = None,
) -> dict:
    """Assemble the summariser's input from structured findings only."""
    return {
        "merchant_id": decision.merchant_id,
        "declared_category": decision.declared_category,
        "onboarded_days_ago": onboarded_days_ago,
        "tier": int(decision.tier),
        "tier_name": LADDER[decision.tier]["name"],
        "channel_verdicts": [c.to_dict() for c in decision.channels],
        "top_feature_deviations": [
            {
                "feature": d["feature"],
                "value": d["value"],
                "category_baseline": d["category_baseline"],
                "sigma": d["z"],
            }
            for d in feature_deviations[:6]
        ],
        "investigation_steps": [
            {"step": s["step"], "tool": s["tool"], "returned": s["returned"]}
            for s in investigation.get("steps", [])
        ],
        "stop_reason": investigation.get("stop_reason"),
        "injection_flagged": injection_flagged,
        "second_channel_source": decision.second_channel_source,
        "contradictions": [
            {"between": c["between"], "statement": c["statement"],
             "severity": c["severity"]}
            for c in (assessment or {}).get("contradictions", [])
        ],
        "evidence_agreement": (assessment or {}).get("evidence_agreement", 0),
        "model_confidence": (assessment or {}).get("model_confidence", 0.0),
        "action_confidence": (assessment or {}).get("action_confidence", "NONE"),
        "network": {
            "in_cluster": bool((network or {}).get("in_cluster")),
            "size": (network or {}).get("size", 0),
            "synchrony": (network or {}).get("synchrony", 0.0),
            "shared_kinds": (network or {}).get("shared_kinds", []),
        },
        "statement": {
            "status": (voice or {}).get("status", "not_called"),
            "claimed_business": (voice or {}).get("claimed_business"),
            "consistent": (voice or {}).get("consistent"),
        },
    }


class PageTextLeak(AssertionError):
    """Raised when page-derived text reaches the summariser input."""


def assert_no_page_text(payload: dict, page_corpus: list[str]) -> None:
    """Fail loudly if any rendered page string appears in the summariser input.

    Runs on every call, including in production. The cost is a substring scan;
    the thing it buys is that the provenance claim on the evidence card stays
    true even after someone adds a field upstream without reading this file.
    """
    unexpected = set(payload) - set(ALLOWED_FIELDS)
    if unexpected:
        raise PageTextLeak(
            f"Fields not on the summariser allowlist: {sorted(unexpected)}"
        )

    flat = _flatten(payload).lower()
    for chunk in page_corpus:
        # Compare on distinctive spans; short common words would match anything.
        for span in _distinctive_spans(chunk):
            if span in flat:
                raise PageTextLeak(
                    f"Page-derived text reached the summariser input: {span!r}"
                )


def _flatten(obj) -> str:
    if isinstance(obj, dict):
        return " ".join(_flatten(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return " ".join(_flatten(v) for v in obj)
    return str(obj)


def _distinctive_spans(text: str, min_words: int = 6) -> list[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [
        " ".join(words[i: i + min_words])
        for i in range(0, max(len(words) - min_words + 1, 0), min_words)
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def summarise(
    payload: dict,
    page_corpus: list[str] | None = None,
    live: bool | None = None,
) -> Narrative:
    """Produce the analyst rationale."""
    assert_no_page_text(payload, page_corpus or [])
    use_live = bool(os.environ.get("ANTHROPIC_API_KEY")) if live is None else live
    if use_live:
        return _summarise_live(payload)
    return _summarise_offline(payload)


def _summarise_live(payload: dict) -> Narrative:
    import anthropic
    import json

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[{
            "role": "user",
            "content": (
                "Structured findings for one merchant:\n\n"
                + json.dumps(payload, indent=2)
            ),
        }],
    )
    text = " ".join(b.text for b in response.content if b.type == "text").strip()
    return Narrative(text, "structured-features-only (live model)", ALLOWED_FIELDS)


def _summarise_offline(payload: dict) -> Narrative:
    """Deterministic rationale, from the same allowlisted payload.

    Used when no model credential is present so the pipeline is complete
    without one. It reads the same fields in the same order the prompt asks
    for, which is what keeps the two modes comparable.
    """
    dec = payload["declared_category"]
    tier = payload["tier"]
    verdicts = {c["channel"]: c for c in payload["channel_verdicts"]}
    devs = payload["top_feature_deviations"]

    parts = [
        f"Merchant {payload['merchant_id']} declared {dec} at onboarding "
        f"{payload['onboarded_days_ago']} days ago."
    ]

    tel = verdicts.get("telemetry", {})
    if devs:
        top = ", ".join(
            f"{d['feature']} at {d['sigma']:+.1f} sigma "
            f"(observed {d['value']:.3g} against a category baseline of "
            f"{d['category_baseline']:.3g})"
            for d in devs[:3]
        )
        parts.append(
            f"Telemetry {'dissents' if tel.get('verdict') == 'DISSENT' else 'agrees'}: "
            f"the largest deviations from the {dec} baseline are {top}."
        )

    vis = verdicts.get("storefront", {})
    if vis.get("verdict") == "BLIND":
        parts.append(
            "The storefront was not classified, so no independent view of the "
            "business is available yet."
        )
    elif vis:
        parts.append(vis["statement"])

    infra = verdicts.get("infrastructure", {})
    if infra.get("verdict") == "DISSENT":
        parts.append(infra["statement"])

    if payload.get("injection_flagged"):
        parts.append(
            "The storefront also carries text addressed to an automated "
            "reviewer, which was recorded as a suspicion signal and not acted "
            "on as an instruction."
        )

    net = payload.get("network") or {}
    if net.get("in_cluster") and net.get("size", 0) > 1:
        parts.append(
            f"It is linked to {net['size'] - 1} other merchant account"
            f"{'s' if net['size'] > 2 else ''} by rare shared "
            f"{', '.join(k.replace('_', ' ') for k in net.get('shared_kinds', []))}"
            + (f", and those accounts are moving in step "
               f"(synchrony {net['synchrony']:.2f})."
               if net.get("synchrony", 0) >= 0.35 else ", though they are not "
               "moving in step.")
        )

    st = payload.get("statement") or {}
    if st.get("status") == "answered" and st.get("consistent") is False:
        parts.append(
            f"Asked directly, the merchant described a "
            f"{str(st.get('claimed_business')).replace('_', ' ')} business, which "
            f"is not what it declared."
        )
    elif st.get("status") == "answered" and st.get("consistent") is True:
        parts.append("Asked directly, the merchant restated its declared business.")
    elif st.get("status") in ("no_answer", "declined"):
        parts.append(
            "A verification call was placed and produced no usable statement, "
            "which counts neither for nor against the merchant."
        )

    contradictions = payload.get("contradictions") or []
    if contradictions:
        parts.append(
            f"In total {len(contradictions)} independent view"
            f"{'s' if len(contradictions) > 1 else ''} of this merchant "
            f"contradict its declaration."
        )

    steps = payload.get("investigation_steps", [])
    if steps:
        parts.append(
            f"The investigation used {len(steps)} of a permitted 8 steps and "
            f"stopped because {payload.get('stop_reason', 'unknown').replace('_', ' ')}."
        )

    tier = payload["tier"]
    if tier == int(Tier.RESTRICT):
        parts.append(
            "Recommended tier restricts settlements, corroborated by the "
            f"{payload.get('second_channel_source')} channel; the storefront "
            "alone would not have been sufficient."
        )
    elif tier == int(Tier.VERIFY):
        parts.append(
            "Recommended tier is a merchant verification call. Settlements "
            "continue: one dissenting view is enough to ask, not to hold."
        )
    elif tier == int(Tier.REVIEW):
        parts.append(
            "Recommended tier is analyst review with settlements continuing, "
            "because only one view dissents."
        )
    elif tier == int(Tier.RECHECK):
        parts.append(
            "Recommended tier is an automatic recheck. No analyst time is spent "
            "and the merchant sees nothing."
        )
    else:
        parts.append("No action recommended; the available views agree.")

    if payload.get("model_confidence", 0) >= 0.9 and payload.get("evidence_agreement", 0) < 2:
        parts.append(
            f"A model is {payload['model_confidence']:.2f} confident, but only "
            f"{payload['evidence_agreement']} independent view dissents -- model "
            f"confidence is not evidence agreement, and the action reflects the "
            f"latter."
        )

    return Narrative(
        " ".join(parts),
        "structured-features-only (deterministic, no model credential present)",
        ALLOWED_FIELDS,
    )
