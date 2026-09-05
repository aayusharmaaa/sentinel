"""The contradiction engine.

The question this system asks is not "what is the risk score". It is:

    Where do this merchant's five realities disagree?

A merchant has one declared identity and four observed ones. Each is gathered a
different way, and none of them can be forged into agreement with the others by
the same act:

    DECLARED        what the merchant told us at onboarding
    PAYMENTS        what its money actually does
    STOREFRONT      what its customers actually see
    NETWORK         who it is actually connected to
    STATEMENT       what it says when asked directly

A merchant that has quietly become a different business cannot keep all five
aligned. It can repaint the homepage, but the checkout still has to take money
the new way. It can restate the declared category on a phone call, but its
payout account is still shared with the operator's other shells.

That is the whole idea, and it produces something better than a score: a list of
specific disagreements, each naming the two views that disagree and what each
one says. An analyst can check any of them.

Three confidences, kept apart
-----------------------------
The most important thing in this file is that these are not the same number:

    model confidence    how sure the models are
    evidence agreement  how many independent views concur
    action confidence   whether that is enough to justify doing something

A single model at 0.99 is not evidence agreement, and neither is enough on its
own to freeze a business's money. Collapsing them into one score is how
monitoring systems end up holding merchants on the strength of one loud channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinel.config import PROHIBITED_VERTICALS

# The five views. Order is the order they appear on the evidence card.
VIEWS = ["declared", "payments", "storefront", "network", "statement"]

VIEW_LABEL = {
    "declared": "Declared at onboarding",
    "payments": "What its payments do",
    "storefront": "What its website shows",
    "network": "Who it is connected to",
    "statement": "What the merchant says",
}

# How much a disagreement between two views is worth. A storefront that shows a
# different business is the strongest single contradiction because it is the
# hardest to produce by accident; a payments-only disagreement is the weakest
# because so many honest things cause it.
SEVERITY = {
    ("declared", "storefront"): 1.00,
    ("declared", "statement"): 0.85,
    ("declared", "network"): 0.60,
    ("declared", "payments"): 0.45,
    ("storefront", "statement"): 0.70,
    ("payments", "storefront"): 0.55,
}


@dataclass
class View:
    """One reading of what business this merchant is."""

    name: str
    value: str | None            # the vertical this view implies, if any
    confidence: float
    detail: str
    available: bool = True

    def to_dict(self) -> dict:
        return {
            "view": self.name,
            "label": VIEW_LABEL[self.name],
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "detail": self.detail,
            "available": self.available,
        }


@dataclass
class Contradiction:
    left: str
    right: str
    severity: float
    statement: str

    def to_dict(self) -> dict:
        return {
            "between": [VIEW_LABEL[self.left], VIEW_LABEL[self.right]],
            "views": [self.left, self.right],
            "severity": round(self.severity, 3),
            "statement": self.statement,
        }


@dataclass
class IdentityAssessment:
    merchant_id: str
    declared: str
    views: list[View]
    contradictions: list[Contradiction]
    model_confidence: float
    evidence_agreement: int          # independent views that dissent
    independent_channels: int        # views that were actually gathered
    contradiction_severity: str      # NONE | LOW | MEDIUM | HIGH
    action_confidence: str           # NONE | WATCH | REVIEW | VERIFY | RESTRICT
    summary: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "merchant_id": self.merchant_id,
            "declared": self.declared,
            "views": [v.to_dict() for v in self.views],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "model_confidence": round(self.model_confidence, 3),
            "evidence_agreement": self.evidence_agreement,
            "independent_channels": self.independent_channels,
            "contradiction_severity": self.contradiction_severity,
            "action_confidence": self.action_confidence,
            "summary": self.summary,
            "notes": self.notes,
        }


def build_views(
    *,
    declared: str,
    telemetry_score: float,
    telemetry_dissent: bool,
    telemetry_detail: str,
    vision_vertical: str | None,
    vision_confidence: float,
    vision_surface: str | None,
    network: dict,
    voice: dict | None,
) -> list[View]:
    """Assemble the merchant's five identities from the channel outputs."""
    views = [
        View("declared", declared, 1.0,
             f"Onboarded as a {declared.replace('_', ' ')} business."),
        View(
            "payments",
            None if not telemetry_dissent else "not_" + declared,
            telemetry_score,
            telemetry_detail,
        ),
    ]

    if vision_vertical is None:
        views.append(View("storefront", None, 0.0,
                          "Not examined this cycle -- nothing triggered the "
                          "expensive channel.", available=False))
    else:
        views.append(View(
            "storefront", vision_vertical, vision_confidence,
            f"The {vision_surface} surface presents a "
            f"{vision_vertical.replace('_', ' ')} interface."
        ))

    if network and network.get("in_cluster"):
        views.append(View("network", "cluster", network.get("score", 0.0),
                          network.get("statement", "")))
    else:
        views.append(View("network", None, 0.0,
                          network.get("statement", "No rare shared identities.")
                          if network else "Not examined."))

    if not voice or voice.get("status") != "answered" or voice.get("consistent") is None:
        detail = {
            None: "No verification call placed.",
            "no_answer": "Verification call placed; the merchant did not answer.",
            "declined": "Merchant declined the call and asked for a human.",
            "answered": "Merchant answered but did not clearly describe its business.",
        }.get(voice.get("status") if voice else None, "No verification call placed.")
        views.append(View("statement", None, 0.0, detail, available=False))
    else:
        views.append(View(
            "statement", voice.get("claimed_business"),
            voice.get("confidence", 0.0),
            (voice.get("contradiction")
             or f"Merchant describes itself as "
                f"{str(voice.get('claimed_business')).replace('_', ' ')}, "
                f"consistent with its declaration."),
        ))
    return views


def find_contradictions(views: list[View], declared: str) -> list[Contradiction]:
    """Every place two independent views of this merchant disagree."""
    by = {v.name: v for v in views}
    out: list[Contradiction] = []

    store = by["storefront"]
    if store.available and store.value not in (None, "unknown") and store.value != declared:
        prohibited = store.value in PROHIBITED_VERTICALS
        out.append(Contradiction(
            "declared", "storefront", SEVERITY[("declared", "storefront")],
            f"Declared {declared.replace('_', ' ')}; the live page is a "
            f"{store.value.replace('_', ' ')} interface."
            + (" That vertical cannot be approved on this platform." if prohibited else ""),
        ))

    pay = by["payments"]
    if pay.value is not None:
        out.append(Contradiction(
            "declared", "payments", SEVERITY[("declared", "payments")],
            f"Declared {declared.replace('_', ' ')}; the payment behaviour is not "
            f"typical of that category.",
        ))

    net = by["network"]
    if net.value == "cluster" and net.confidence >= 0.45:
        out.append(Contradiction(
            "declared", "network", SEVERITY[("declared", "network")],
            "Declared an independent business; it is linked to other merchant "
            "accounts by rare shared identities and moves with them.",
        ))

    st = by["statement"]
    if st.available and st.value and st.value != declared:
        out.append(Contradiction(
            "declared", "statement", SEVERITY[("declared", "statement")],
            f"Declared {declared.replace('_', ' ')}; the merchant itself describes "
            f"a {str(st.value).replace('_', ' ')} business.",
        ))
        if store.available and store.value not in (None, "unknown") and store.value != st.value:
            out.append(Contradiction(
                "storefront", "statement", SEVERITY[("storefront", "statement")],
                f"The page reads {store.value.replace('_', ' ')}; the merchant says "
                f"{str(st.value).replace('_', ' ')}. Two accounts of the same "
                f"business that do not match each other.",
            ))
    elif st.available and st.value == declared and store.available \
            and store.value not in (None, "unknown") and store.value != declared:
        out.append(Contradiction(
            "storefront", "statement", SEVERITY[("storefront", "statement")],
            f"The merchant restated its declared {declared.replace('_', ' ')} "
            f"business on the call, while its live page presents "
            f"{store.value.replace('_', ' ')}.",
        ))

    out.sort(key=lambda c: -c.severity)
    return out


def assess(
    merchant_id: str,
    declared: str,
    views: list[View],
    contradictions: list[Contradiction],
) -> IdentityAssessment:
    """Turn views and contradictions into three separate confidences."""
    by = {v.name: v for v in views}

    # 1. Model confidence -- how sure the models themselves are. Deliberately
    #    the max of the model-driven channels, because this is the number people
    #    reach for, and keeping it separate makes it obvious that it alone
    #    decides nothing.
    model_confidence = max(
        by["payments"].confidence if by["payments"].value else 0.0,
        by["storefront"].confidence if by["storefront"].value not in (None, "unknown")
        and by["storefront"].value != declared else 0.0,
    )

    # 2. Evidence agreement -- how many independent views dissent from the
    #    declaration. This is a count, not a probability, and that is the point.
    dissenting = {c.left for c in contradictions} | {c.right for c in contradictions}
    dissenting.discard("declared")
    evidence_agreement = len(dissenting)
    independent = sum(1 for v in views if v.name != "declared" and v.available)

    top = contradictions[0].severity if contradictions else 0.0
    severity = ("HIGH" if top >= 0.85 and evidence_agreement >= 2
                else "MEDIUM" if top >= 0.6
                else "LOW" if contradictions else "NONE")

    # 3. Action confidence -- what the evidence justifies *doing*. The storefront
    #    is required for anything above review, and two independent dissents are
    #    required before restriction is even on the table.
    store_dissents = any(c.left == "storefront" or c.right == "storefront"
                         for c in contradictions)
    if store_dissents and evidence_agreement >= 2:
        action = "RESTRICT"
    elif store_dissents and evidence_agreement == 1:
        action = "VERIFY"
    elif evidence_agreement >= 2:
        action = "VERIFY"
    elif evidence_agreement == 1:
        action = "REVIEW"
    elif by["payments"].confidence >= 0.35:
        action = "WATCH"
    else:
        action = "NONE"

    notes = []
    if not by["storefront"].available:
        notes.append("The storefront has not been examined, so no restriction is "
                     "available on this evidence however strong the rest looks.")
    if not by["statement"].available:
        notes.append("No usable merchant statement. A call was either not placed, "
                     "not answered, or gave no clear account of the business.")
    if model_confidence >= 0.9 and evidence_agreement < 2:
        notes.append(f"A model is {model_confidence:.2f} confident, but only "
                     f"{evidence_agreement} independent view dissents. Model "
                     f"confidence is not evidence agreement.")

    if not contradictions:
        summary = (f"All available views agree that this is a "
                   f"{declared.replace('_', ' ')} business.")
    else:
        parts = [VIEW_LABEL[v].lower() for v in
                 sorted(dissenting, key=lambda x: VIEWS.index(x))]
        summary = (f"{len(contradictions)} contradiction"
                   f"{'s' if len(contradictions) > 1 else ''} across "
                   f"{evidence_agreement} independent view"
                   f"{'s' if evidence_agreement > 1 else ''} "
                   f"({', '.join(parts)}).")

    return IdentityAssessment(
        merchant_id=merchant_id,
        declared=declared,
        views=views,
        contradictions=contradictions,
        model_confidence=model_confidence,
        evidence_agreement=evidence_agreement,
        independent_channels=independent,
        contradiction_severity=severity,
        action_confidence=action,
        summary=summary,
        notes=notes,
    )
