"""Interactive case copilot for the analyst console.

The pain point is narrow: evidence cards are dense, and an analyst's time is
spent figuring out what matters before they can decide. The copilot answers
questions about one card from the same structured-features allowlist the
narrative summariser uses — never from scraped page copy — so the control
that protects the written summary also protects the conversational path.

Offline answers are deterministic intent routing over the card. Live mode
(when ANTHROPIC_API_KEY is set) asks Claude with the same allowlisted payload.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from sentinel.fusion.gate import LADDER, Tier
from sentinel.narrative.summarizer import (
    assert_no_page_text,
    build_summary_input,
)

MODEL = "claude-opus-5"

SYSTEM = """\
You are the Sentinel case copilot for a payment-aggregator compliance analyst.
You answer questions about ONE merchant evidence card.

Rules:
- Use ONLY the structured findings in the JSON payload. Never invent channels,
  surfaces, or numbers that are not present.
- Never recommend holding payouts unless the card's tier is Restricted (4) and
  a second corroborating channel is named. One dissenting channel is never enough.
- Prefer plain language. Name channels and features when they matter.
- If the question asks for something outside the card, say you cannot see it.
- Keep answers to 3–6 short sentences unless the analyst asks for a decision memo.
- Do not quote or speculate about page copy; you only see structured findings.
"""


@dataclass
class CopilotReply:
    answer: str
    provenance: str
    suggestions: list[str]
    intent: str

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "provenance": self.provenance,
            "suggestions": self.suggestions,
            "intent": self.intent,
        }


def dissents_human(channels: list[str]) -> list[str]:
    names = {
        "telemetry": "payments",
        "storefront": "website",
        "scripts": "page code",
        "infrastructure": "shared setup",
        "statement": "the merchant's statement",
        "page_integrity": "page tampering",
    }
    return [names.get(c, c) for c in channels]


def _payload_from_card(card: dict) -> dict:
    """Rebuild the allowlisted summariser payload from a stored evidence card."""
    from sentinel.fusion.gate import ChannelVerdict, Decision

    d = card["decision"]
    channels = [
        ChannelVerdict(
            channel=c["channel"],
            dissents=(c.get("verdict") == "DISSENT"),
            confidence=float(c.get("confidence") or 0),
            statement=c.get("statement") or "",
            blind=(c.get("verdict") == "BLIND"),
        )
        for c in d.get("channels", [])
    ]
    decision = Decision(
        merchant_id=card["merchant_id"],
        declared_category=card["declared_category"],
        tier=Tier(int(d["tier"])),
        channels=channels,
        justification=d.get("justification") or "",
        dissenting_channels=list(d.get("dissenting_channels") or []),
        second_channel_source=d.get("second_channel_source"),
        notes=list(d.get("notes") or []),
        assessment=card.get("assessment"),
    )
    return build_summary_input(
        decision=decision,
        feature_deviations=list((card.get("telemetry") or {}).get("deviations") or []),
        investigation=card.get("investigation") or {},
        onboarded_days_ago=int(card.get("onboarded_days_ago") or 0),
        injection_flagged=any(
            (v.get("injection") or {}).get("present")
            for v in (card.get("vision") or {}).values()
        ),
        assessment=card.get("assessment"),
        network=card.get("network"),
        voice=card.get("voice"),
    )


def queue_brief(card: dict) -> dict:
    """One-line priority brief for the sidebar queue."""
    d = card["decision"]
    tier = int(d["tier"])
    declared = str(card["declared_category"]).replace("_", " ")
    vision = card.get("vision") or {}
    seen = None
    for surface, v in vision.items():
        vert = v.get("vertical")
        if vert and vert != card["declared_category"] and vert != "unknown":
            seen = (surface, str(vert).replace("_", " "), float(v.get("confidence") or 0))
            break
    dissent = list(d.get("dissenting_channels") or [])
    if tier >= 4 and d.get("second_channel_source"):
        why = f"Hold candidate - website + {d['second_channel_source']} both dissent."
        urgency = "critical"
    elif tier == 3:
        why = "Ask the merchant - website disagrees; settlements still running."
        urgency = "high"
    elif seen:
        why = f"Checkout/homepage reads as {seen[1]}; payments may still look normal."
        urgency = "medium"
    elif dissent:
        why = f"{', '.join(dissents_human(dissent))} disagree with declared {declared}."
        urgency = "medium"
    else:
        why = f"Queued for review as a declared {declared} merchant."
        urgency = "low"

    return {
        "brief": why,
        "urgency": urgency,
        "preview": ((card.get("narrative") or {}).get("text") or "")[:180],
    }


def suggest_for(card: dict) -> list[str]:
    out = [
        "Why is this recommended?",
        "Which channels disagree?",
        "Is it safe to release?",
    ]
    if card.get("vision"):
        out.append("What does the website show?")
    if (card.get("network") or {}).get("in_cluster"):
        out.append("Who is it linked to?")
    if card.get("voice"):
        out.append("What did the merchant say on the call?")
    out.append("Draft my decision note")
    seen: set[str] = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:6]


def ask(card: dict, question: str, live: bool | None = None) -> CopilotReply:
    """Answer a question about one evidence card."""
    q = (question or "").strip()
    if not q:
        return CopilotReply(
            "Ask anything about this case — why it was queued, what each channel "
            "found, or whether a settlement hold is justified.",
            "structured-features-only (prompt)",
            suggest_for(card),
            "empty",
        )

    payload = _payload_from_card(card)
    assert_no_page_text(payload, [])

    use_live = bool(os.environ.get("ANTHROPIC_API_KEY")) if live is None else live
    intent = classify_intent(q)

    if use_live and intent != "walkthrough":
        try:
            return _ask_live(payload, q, card, intent)
        except Exception:
            pass

    return _ask_offline(card, payload, q, intent)


def classify_intent(q: str) -> str:
    t = q.lower()
    if re.search(r"\b(draft|memo|note|write|document)\b", t):
        return "memo"
    if re.search(r"\b(walk|guide|step|tour|explain (the )?case|how (do|should) i)\b", t):
        return "walkthrough"
    if re.search(r"\b(safe to release|clear|release|false positive|wrong)\b", t):
        return "release"
    if re.search(r"\b(why|recommend|tier|action|hold|restrict|freeze)\b", t):
        return "why"
    if re.search(r"\b(website|storefront|checkout|homepage|page|vertical)\b", t):
        return "vision"
    if re.search(r"\b(channel|dissent|disagree|contradict|agree)\b", t):
        return "channels"
    if re.search(r"\b(payment|telemetry|feature|sigma|velocity|refund|chargeback)\b", t):
        return "payments"
    if re.search(r"\b(network|link|cluster|ring|shared|infrastructure|who is it)\b", t):
        return "network"
    if re.search(r"\b(call|voice|statement|said|merchant said)\b", t):
        return "voice"
    if re.search(r"\b(second channel|two.channel|gate)\b", t):
        return "gate"
    return "general"


def _plain(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _ask_offline(card: dict, payload: dict, q: str, intent: str) -> CopilotReply:
    d = card["decision"]
    tier = int(d["tier"])
    tier_name = LADDER[Tier(tier)]["name"]
    declared = str(card["declared_category"]).replace("_", " ")
    mid = card["merchant_id"]
    dissent = list(d.get("dissenting_channels") or [])
    second = d.get("second_channel_source")

    if intent == "why":
        just = (d.get("justification") or "the gate found dissenting views.").rstrip(".")
        if just and just[0].isupper():
            just = just[0].lower() + just[1:]
        answer = f"{mid} is recommended for {tier_name} because {just}. "
        if tier >= 4 and second:
            answer += (
                f"Settlements are held only because {second} corroborates the "
                f"storefront — a single channel would never be enough."
            )
        elif tier == 3:
            answer += "Settlements continue; one dissenting view is enough to ask, not to hold."
        elif tier == 2:
            answer += "Settlements continue while you review."
        else:
            answer += "No hold is in force."

    elif intent == "release":
        if tier >= 4:
            answer = (
                f"Releasing {mid} would unfreeze settlements that two independent views "
                f"say should stay held"
                + (f" (website + {second})" if second else "")
                + ". Only clear it if you have evidence those readings are wrong — "
                "for example a misclassified storefront you have checked yourself. "
                "Model confidence is not a reason to overturn a two-channel hold."
            )
        elif tier == 3:
            answer = (
                f"Payouts are already running. Clearing {mid} means you are satisfied "
                "the website finding does not need a merchant call. That is a lighter "
                "call than a hold, but still leave a note if you skip verification."
            )
        else:
            answer = (
                f"{mid} is not on a settlement hold. Marking it clear is appropriate if "
                "the remaining dissent (usually payments alone) looks like a legitimate "
                "shape for this category — which is common."
            )

    elif intent == "vision":
        vis = card.get("vision") or {}
        if not vis:
            answer = "No storefront classification was recorded on this card."
        else:
            bits = []
            for surface, v in vis.items():
                bits.append(
                    f"The {surface} reads as "
                    f"{str(v.get('vertical', 'unknown')).replace('_', ' ')} "
                    f"({float(v.get('confidence') or 0):.0%} confidence) against a "
                    f"declared {declared} business."
                )
                findings = v.get("findings") or []
                if findings:
                    f0 = findings[0]
                    bits.append(
                        f"Example finding: {f0.get('observation', '')} "
                        f"→ {f0.get('implication', '')}"
                    )
            if any((v.get("injection") or {}).get("present") for v in vis.values()):
                bits.append(
                    "Page-tampering text aimed at an automated reviewer was flagged as "
                    "a suspicion signal only — never as an instruction."
                )
            answer = " ".join(bits)

    elif intent == "channels":
        lines = []
        for c in d.get("channels") or []:
            lines.append(
                f"{c['channel']}: {c['verdict']} — {c.get('statement', '')}"
            )
        if dissent:
            answer = (
                f"{len(dissent)} channel(s) dissent "
                f"({', '.join(dissents_human(dissent))}). "
                + " ".join(lines)
            )
        else:
            answer = "No channel currently dissents. " + " ".join(lines)

    elif intent == "payments":
        tel = card.get("telemetry") or {}
        score = tel.get("score")
        devs = tel.get("deviations") or []
        bits = []
        if score is not None:
            bits.append(
                f"Payment score for this declared {declared} merchant is "
                f"{float(score):.2f}."
            )
        else:
            bits.append("No telemetry score is on the card.")
        if devs:
            top = ", ".join(
                f"{str(x['feature']).replace('_', ' ')} at {x['z']:+.1f}σ"
                for x in devs[:3]
            )
            bits.append(f"Largest deviations from the category baseline: {top}.")
        bits.append(
            "Odd payments alone never freeze settlements — the gate requires the "
            "website plus a second independent view."
        )
        answer = " ".join(bits)

    elif intent == "network":
        net = card.get("network") or {}
        if not net.get("in_cluster"):
            answer = (
                "This merchant is not in a rare-shared-infrastructure cluster "
                "on this card."
            )
        else:
            kinds = ", ".join(
                str(k).replace("_", " ") for k in (net.get("shared_kinds") or [])
            ) or "shared signals"
            size = int(net.get("size") or 1)
            syn = float(net.get("synchrony") or 0)
            answer = (
                f"Linked to {max(size - 1, 0)} other account(s) via rare {kinds}. "
                f"Synchrony is {syn:.2f}"
                + (
                    " — those accounts are moving in step, which independent "
                    "businesses rarely do."
                    if syn >= 0.35
                    else " — linked, but not clearly moving in step."
                )
            )

    elif intent == "voice":
        voice = card.get("voice") or {}
        status = voice.get("status") or "not_called"
        if status != "answered":
            answer = (
                f"The verification call status is {status.replace('_', ' ')}. "
                "That counts neither for nor against the merchant."
            )
        elif voice.get("consistent") is True:
            answer = "Asked directly, the merchant restated its declared business."
        elif voice.get("consistent") is False:
            claimed = str(voice.get("claimed_business") or "something else").replace("_", " ")
            answer = (
                f"Asked directly, the merchant described a {claimed} business, "
                f"which contradicts the declared {declared} category."
            )
            if voice.get("contradiction"):
                answer += f" {voice['contradiction']}"
        else:
            answer = "The call was answered but produced no clear account of the business."

    elif intent == "gate":
        answer = (
            "The two-channel gate freezes payouts only when the website reading "
            "and at least one other independent view both contradict the declaration. "
        )
        if tier >= 4 and second:
            answer += f"On this card, the second channel is {second}."
        elif tier < 4:
            answer += (
                f"Here, {len(dissent)} channel(s) dissent"
                + (f" ({', '.join(dissents_human(dissent))})" if dissent else "")
                + f", so the recommended tier stays {tier_name} without a hold."
            )
        else:
            answer += "A hold is already recommended."

    elif intent == "memo":
        answer = _draft_memo(card)

    elif intent == "walkthrough":
        answer = _walkthrough(card)

    else:
        narr = (card.get("narrative") or {}).get("text")
        if narr:
            answer = narr
        else:
            answer = (
                f"{mid} declared {declared}. Recommended: {tier_name}. "
                f"Dissenting: {', '.join(dissents_human(dissent)) or 'none'}."
            )

    return CopilotReply(
        _plain(answer),
        "structured-features-only (deterministic copilot)",
        suggest_for(card),
        intent,
    )


def _draft_memo(card: dict) -> str:
    d = card["decision"]
    tier = int(d["tier"])
    mid = card["merchant_id"]
    declared = str(card["declared_category"]).replace("_", " ")
    dissent = dissents_human(list(d.get("dissenting_channels") or []))
    action = {
        4: "UPHOLD restriction — keep settlements held pending review.",
        3: "PROCEED to merchant verification call; do not hold settlements.",
        2: "CONTINUE review with settlements running; no merchant contact yet.",
    }.get(tier, "NO ACTION beyond monitoring.")

    lines = [
        f"Decision note — {mid}.",
        f"Declared category: {declared}. Recommended tier: {LADDER[Tier(tier)]['name']}.",
        f"Dissenting views: {', '.join(dissent) or 'none'}.",
    ]
    if d.get("second_channel_source"):
        lines.append(f"Second channel for any hold: {d['second_channel_source']}.")
    lines.append(f"Gate rationale: {(d.get('justification') or '').strip()}")
    a = card.get("assessment") or {}
    if a:
        lines.append(
            f"Evidence agreement: {a.get('evidence_agreement')} dissenting independent "
            f"view(s); action confidence: {a.get('action_confidence')}; "
            f"model confidence: {float(a.get('model_confidence') or 0):.2f} "
            f"(informational only)."
        )
    lines.append(f"Proposed analyst action: {action}")
    lines.append(
        "I reviewed the structured channel findings and the storefront fixtures "
        "where available. Provenance: structured features only — no page copy."
    )
    return " ".join(lines)


def _walkthrough(card: dict) -> str:
    d = card["decision"]
    mid = card["merchant_id"]
    declared = str(card["declared_category"]).replace("_", " ")
    steps = [
        f"1. Start with the story: {mid} signed up as {declared}.",
        "2. Check who it looks like now — the five identity readings.",
        "3. Open the website panel and verify findings against the real fixture pages.",
        "4. Skim payment deviations only if you need a second channel or a release case.",
        "5. Read the gate rule: holds need website + one other view.",
        f"6. Decide: {LADDER[Tier(int(d['tier']))]['name']} — confirm, clear, or ask for more info.",
    ]
    return " ".join(steps)


def _ask_live(payload: dict, question: str, card: dict, intent: str) -> CopilotReply:
    import json

    import anthropic

    client = anthropic.Anthropic()
    extra = ""
    if intent == "memo":
        extra = (
            "\nThe analyst asked for a decision memo. Write a short paste-ready note "
            "in plain prose (no bullets) that a compliance file could keep."
        )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                "Structured findings for one merchant:\n\n"
                + json.dumps(payload, indent=2)
                + "\n\nAnalyst question:\n"
                + question
                + extra
            ),
        }],
    )
    text = " ".join(b.text for b in response.content if b.type == "text").strip()
    return CopilotReply(
        text,
        "structured-features-only (live model copilot)",
        suggest_for(card),
        intent,
    )


def guided_steps(card: dict) -> list[dict]:
    """Interactive guided-review checklist for the console."""
    d = card["decision"]
    declared = str(card["declared_category"]).replace("_", " ")
    mid = card["merchant_id"]
    vis = card.get("vision") or {}
    dissent = list(d.get("dissenting_channels") or [])

    steps = [
        {
            "id": "story",
            "title": "Read the story",
            "detail": (
                f"{mid} declared {declared}. Confirm the recommended action "
                "makes sense at a glance."
            ),
            "ask": "Why is this recommended?",
        },
        {
            "id": "identity",
            "title": "Check the five readings",
            "detail": (
                "Look for where declared, payments, website, network, and "
                "statement disagree."
            ),
            "ask": "Which channels disagree?",
        },
    ]
    if vis:
        steps.append({
            "id": "website",
            "title": "Open the website",
            "detail": "Verify at least one storefront finding against the fixture page.",
            "ask": "What does the website show?",
        })
    if "telemetry" in dissent or int(d["tier"]) >= 4:
        steps.append({
            "id": "payments",
            "title": "Scan payment deviations",
            "detail": "Only needed when payments are a dissenting or corroborating channel.",
            "ask": "What do the payments show?",
        })
    steps.append({
        "id": "gate",
        "title": "Apply the gate",
        "detail": (
            "Holds require website + a second independent view. "
            "Model confidence alone never holds money."
        ),
        "ask": "Is it safe to release?",
    })
    steps.append({
        "id": "decide",
        "title": "Record your decision",
        "detail": (
            "Confirm the action, clear the merchant, or request more information "
            "— then leave a note."
        ),
        "ask": "Draft my decision note",
    })
    return steps
