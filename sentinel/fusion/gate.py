"""The two-channel gate and the response ladder.

The scoring rule is the product's central claim, so it is written as a rule
rather than a threshold on a blended score:

    No single channel can trigger a settlement hold.

Tier 3 requires the storefront channel to dissent from the declaration AND a
second, independent channel to dissent as well. Independent evidence types
agreeing is a far stronger signal than either being loud, and blending them
into one number destroys exactly the information that makes the pair
trustworthy -- you can no longer tell two-channels-at-0.6 from
one-channel-at-1.0-and-one-at-0.2, and those are completely different cases.

This rule is what protects the legitimate skill-gaming platform whose telemetry
is nearly indistinguishable from a laundering front's. Its round-value share is
68%, its repeat velocity is five times retail, its hours peak at 2am, and every
one of those is normal for what it actually is. Its storefront agrees with its
declaration. One channel dissents. It does not get held.

Two deliberate asymmetries:

  Storefront dissent is necessary for Tier 3, and telemetry dissent alone never
  is. That is not because vision is more reliable -- it is because telemetry
  dissent has a large population of innocent explanations and storefront
  dissent has very few. A furniture shop whose payments moved is a business
  that changed. A furniture shop whose checkout is a betting interface is not.

  Infrastructure can substitute for telemetry as the second channel, which is
  an extension of the stated rule and is reported separately in the evaluation
  for exactly that reason. Ring members are constructed to sit below the
  single-merchant telemetry threshold, so without this substitution the ring
  archetype can never reach Tier 3 and the funnel case is unreachable by
  design. The substitution is only permitted alongside storefront dissent, so
  it can never hold a merchant on infrastructure alone -- otherwise every
  business on a popular host becomes collateral.

Single-channel dissent still produces action, just lesser action. That is
correct for a licensing violation, which vision can see and telemetry
structurally cannot: a pharmacy operating without registration has the same
payment shape as one with it. Waiting for a second channel that can never
arrive would mean never acting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from sentinel.config import PROHIBITED_VERTICALS, Economics, Thresholds


class Tier(IntEnum):
    MONITOR = 0
    RECHECK = 1
    REVIEW = 2
    VERIFY = 3
    RESTRICT = 4


LADDER = {
    Tier.MONITOR: {
        "name": "Monitor",
        "action": "No action. The merchant stays on the standard sweep.",
        "analyst_time": False,
        "settlements_held": False,
        "merchant_contact": False,
    },
    Tier.RECHECK: {
        "name": "Automatic recheck",
        "action": (
            "Requeued for an accelerated recheck on a timer. No analyst time, "
            "no merchant contact, nothing the merchant ever sees."
        ),
        "analyst_time": False,
        "settlements_held": False,
        "merchant_contact": False,
    },
    Tier.REVIEW: {
        "name": "Analyst review",
        "action": (
            "Enters the analyst queue with a full evidence card. Settlements "
            "continue normally while the case is worked."
        ),
        "analyst_time": True,
        "settlements_held": False,
        "merchant_contact": False,
    },
    Tier.VERIFY: {
        "name": "Merchant verification",
        "action": (
            "An automated verification call asks the merchant to describe its "
            "own business. Settlements continue. The merchant may decline and "
            "speak to a person instead."
        ),
        "analyst_time": True,
        "settlements_held": False,
        "merchant_contact": True,
    },
    Tier.RESTRICT: {
        "name": "Restricted pending review",
        "action": (
            "Analyst review with settlements held pending outcome. Requires the "
            "storefront to dissent plus at least one other independent view."
        ),
        "analyst_time": True,
        "settlements_held": True,
        "merchant_contact": True,
    },
}

# The tier at and above which a merchant's money stops moving. Everything below
# it is reversible and invisible to the merchant, which is why the ladder has
# four rungs before it rather than one.
FIRST_HOLDING_TIER = Tier.RESTRICT


@dataclass
class ChannelVerdict:
    channel: str
    dissents: bool
    confidence: float
    statement: str
    # Set when a channel structurally cannot see a given abuse type. Saying so
    # is more useful than reporting a confident "no finding".
    blind: bool = False

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "verdict": "DISSENT" if self.dissents else ("BLIND" if self.blind else "AGREES"),
            "confidence": round(self.confidence, 4),
            "statement": self.statement,
        }


@dataclass
class Decision:
    merchant_id: str
    declared_category: str
    tier: Tier
    channels: list[ChannelVerdict]
    justification: str
    dissenting_channels: list[str]
    second_channel_source: str | None = None
    estimated_cost_inr: float = 0.0
    notes: list[str] = field(default_factory=list)
    assessment: dict | None = None

    def to_dict(self) -> dict:
        return {
            "merchant_id": self.merchant_id,
            "declared_category": self.declared_category,
            "tier": int(self.tier),
            "tier_name": LADDER[self.tier]["name"],
            "action": LADDER[self.tier]["action"],
            "settlements_held": LADDER[self.tier]["settlements_held"],
            "channels": [c.to_dict() for c in self.channels],
            "dissenting_channels": self.dissenting_channels,
            "second_channel_source": self.second_channel_source,
            "justification": self.justification,
            "notes": self.notes,
            "assessment": self.assessment,
        }


def build_channels(
    *,
    declared_category: str,
    telemetry_score: float,
    vision_vertical: str | None,
    vision_confidence: float,
    vision_surface: str | None,
    injection_present: bool,
    infra_linked: int = 0,
    infra_rarity_total: float = 0.0,
    network: dict | None = None,
    voice: dict | None = None,
    script_conflicts: dict[str, str] | None = None,
    thresholds: Thresholds,
) -> list[ChannelVerdict]:
    """Turn raw channel outputs into dissent verdicts with readable statements."""
    tel_dissent = telemetry_score >= thresholds.telemetry_dissent
    telemetry = ChannelVerdict(
        channel="telemetry",
        dissents=tel_dissent,
        confidence=telemetry_score,
        statement=(
            f"Payment behaviour is atypical for a {declared_category} merchant "
            f"(score {telemetry_score:.2f} against the category baseline)."
            if tel_dissent
            else f"Payment behaviour is within the normal range for a "
                 f"{declared_category} merchant (score {telemetry_score:.2f})."
        ),
    )

    if vision_vertical is None:
        storefront = ChannelVerdict(
            "storefront", False, 0.0,
            "Storefront not classified. Telemetry did not clear the triage gate, "
            "so no expensive call was made.",
            blind=True,
        )
    else:
        vis_dissent = (
            vision_vertical != declared_category
            and vision_vertical != "unknown"
            and vision_confidence >= thresholds.vision_dissent_confidence
        )
        prohibited = vision_vertical in PROHIBITED_VERTICALS
        storefront = ChannelVerdict(
            channel="storefront",
            dissents=vis_dissent,
            confidence=vision_confidence,
            statement=(
                f"The {vision_surface} surface presents a {vision_vertical} "
                f"interface at {vision_confidence:.2f} confidence, against a "
                f"declared category of {declared_category}."
                + (" This vertical cannot be approved on this platform."
                   if prohibited else "")
                if vis_dissent
                else f"The {vision_surface} surface is consistent with the "
                     f"declared {declared_category} category "
                     f"({vision_confidence:.2f} confidence)."
            ),
        )

    # The infrastructure channel now speaks for the relationship graph. A
    # cluster is only a dissent when it is both tightly linked by rare shared
    # identities and moving in synchrony -- either alone has an innocent
    # explanation, and acting on either alone freezes shopping malls.
    net = network or {}
    infra_dissent = bool(net.get("in_cluster")) and net.get("score", 0.0) >= 0.55
    infrastructure = ChannelVerdict(
        channel="infrastructure",
        dissents=infra_dissent,
        confidence=float(net.get("score", 0.0)),
        statement=net.get(
            "statement",
            "No rare shared identities linking this merchant to any other.",
        ),
    )

    # Third-party scripts are a separate observation from the rendered page:
    # different data, gathered a different way, and not something a merchant can
    # restyle. A betting front can rewrite every word of its copy and still has
    # to load an odds feed for the product to work. It is a weaker signal than
    # the storefront -- integrations get left behind, partners ship widgets --
    # which is why it corroborates and never leads.
    script_conflicts = script_conflicts or {}
    conflicting = {k: v for k, v in script_conflicts.items() if v != declared_category}
    scripts = ChannelVerdict(
        channel="scripts",
        dissents=bool(conflicting),
        confidence=min(0.5 + 0.2 * len(conflicting), 0.95) if conflicting else 0.0,
        statement=(
            "Third-party scripts belonging to another vertical are loaded: "
            + "; ".join(f"{k} is specific to {v}" for k, v in conflicting.items())
            + "."
            if conflicting
            else "No third-party scripts specific to another vertical."
        ),
    )

    # The merchant's own account of itself. This is the only channel where the
    # subject knows it is being measured, so it is scored asymmetrically: a
    # merchant that contradicts its own declaration is strong evidence, and a
    # merchant that restates it is almost none -- restating is what an honest
    # merchant and a dishonest one both do.
    v = voice or {}
    statement_dissents = (
        v.get("status") == "answered" and v.get("consistent") is False
    )
    statement = ChannelVerdict(
        channel="statement",
        dissents=statement_dissents,
        confidence=float(v.get("confidence", 0.0)) if statement_dissents else 0.0,
        statement=(
            v.get("contradiction")
            or {
                "answered": "The merchant restated its declared business, which is "
                            "what an honest merchant and a dishonest one both do.",
                "no_answer": "A verification call was placed and not answered. That "
                             "counts neither for nor against the merchant.",
                "declined": "The merchant declined the call and asked for a person. "
                            "That is their right and it is not evidence.",
            }.get(v.get("status"), "No verification call placed.")
        ),
        blind=v.get("status") in (None, "no_answer", "declined"),
    )

    channels = [telemetry, storefront, scripts, infrastructure, statement]
    if injection_present:
        channels.append(
            ChannelVerdict(
                channel="page_integrity",
                dissents=False,  # never a gating channel on its own
                confidence=0.0,
                statement=(
                    "Storefront carries text addressed to an automated reviewer. "
                    "Recorded as a suspicion signal and never acted on as an "
                    "instruction. Does not count towards the two-channel gate."
                ),
            )
        )
    return channels


def decide(
    merchant_id: str,
    declared_category: str,
    channels: list[ChannelVerdict],
    telemetry_score: float,
    thresholds: Thresholds,
    assessment=None,
    storefront_available: bool = True,
) -> Decision:
    """Apply the gate. This function is the product.

    The rule is unchanged and is still the whole point: the storefront must
    dissent, and one other independent view must dissent with it, before a
    merchant's money stops. What the identity assessment adds is a rung between
    "look at this" and "stop their settlements" -- asking the merchant.
    """
    by_name = {c.channel: c for c in channels}
    tel = by_name["telemetry"]
    vis = by_name["storefront"]
    infra = by_name["infrastructure"]
    scripts = by_name.get(
        "scripts", ChannelVerdict("scripts", False, 0.0, "Not gathered.")
    )
    statement = by_name.get(
        "statement", ChannelVerdict("statement", False, 0.0, "Not gathered.")
    )

    dissenting = [c.channel for c in channels if c.dissents]
    notes: list[str] = []
    second_source: str | None = None

    # Ordered by how hard the corroboration is to fake. A merchant telling you
    # itself that it changed business is the strongest of these, and it is also
    # the only one it volunteered.
    corroborators = [
        ("the merchant's own statement", statement.dissents),
        ("telemetry", tel.dissents),
        ("scripts", scripts.dissents),
        ("infrastructure", infra.dissents),
    ]
    second = next((name for name, ok in corroborators if ok), None)

    if vis.dissents and second:
        tier = Tier.RESTRICT
        second_source = second
        justification = (
            "Two independent views of this merchant disagree with its "
            "declaration. The storefront presents a different business, and "
            f"the {second} channel agrees. Neither could have produced the "
            "other: a merchant cannot change its payment behaviour without "
            "changing its actual business, and cannot change its interface "
            "without changing what customers see."
        )
        if second == "the merchant's own statement":
            notes.append(
                "The corroborating view is the merchant's own account of itself. "
                "It was asked a neutral question about what it sells and "
                "described a different business from the one it declared."
            )
        elif second != "telemetry":
            notes.append(
                f"The corroborating view is {second} rather than telemetry. "
                "This is the rented-account shape: the genuine store keeps "
                "trading, so the payment mixture stays near the category "
                "baseline and telemetry alone never crosses its threshold."
            )
    elif vis.dissents:
        tier = Tier.VERIFY
        justification = (
            "The storefront disagrees with the declaration and nothing else "
            "does. That is enough to ask the merchant directly, and not enough "
            "to touch their money. It is also the correct outcome for a "
            "licensing or disclosure violation, which the storefront can see "
            "and telemetry structurally cannot: an unregistered pharmacy has "
            "the same payment shape as a registered one."
        )
    elif tel.dissents and vis.blind and storefront_available:
        tier = Tier.RECHECK
        justification = (
            "Payment behaviour has drifted and the storefront has not been "
            "examined yet. Queued for a storefront look rather than sent to an "
            "analyst, because payment drift alone has too many innocent "
            "explanations to justify anyone's time."
        )
    elif tel.dissents:
        tier = Tier.REVIEW
        justification = (
            "Payment behaviour has drifted but the storefront agrees with the "
            "declaration. Settlements continue and will not be held on this "
            "evidence. A legitimate business whose payment shape resembles an "
            "illicit one is common; one whose storefront is a different "
            "business is not."
        )
        notes.append(
            "Payment-only dissent never reaches a settlement hold. This is the "
            "rule that protects merchants whose honest behaviour looks anomalous."
        )
    elif infra.dissents and telemetry_score >= thresholds.triage_gate:
        tier = Tier.REVIEW
        justification = (
            "This merchant is linked to other accounts by rare shared identities, "
            "those accounts are moving in step, and its own payment behaviour is "
            "elevated. Individually it is unremarkable -- which is what a funnel "
            "member is designed to be -- so the case for looking at it is the "
            "company it keeps. Settlements continue: a network link is a reason "
            "to investigate, never on its own a reason to hold money."
        )
        notes.append(
            "Surfaced by the relationship graph rather than by anything about "
            "this merchant alone."
        )
    elif infra.dissents or scripts.dissents:
        tier = Tier.RECHECK
        justification = (
            "A supporting channel dissents with nothing else agreeing. Requeued "
            "for a storefront look. Neither shared infrastructure nor a stray "
            "script holds anyone on its own: co-location is not evidence of a "
            "common business."
        )
    elif telemetry_score >= thresholds.triage_gate:
        tier = Tier.RECHECK
        justification = (
            "Payment behaviour is elevated but below the dissent threshold and "
            "no other view objects. Recheck on a timer. No analyst time and "
            "nothing the merchant ever sees."
        )
    else:
        tier = Tier.MONITOR
        justification = "All available views agree with the declaration."

    if assessment is not None:
        notes.extend(assessment.notes)
        # The assessment can never escalate past what the gate allows; it is a
        # second opinion that is only permitted to pull the tier down.
        if assessment.action_confidence in ("NONE", "WATCH") and tier >= Tier.RESTRICT:
            tier = Tier.VERIFY
            notes.append(
                "Contradiction engine judged the evidence insufficient to "
                "restrict, so the tier was reduced. The gate and the assessment "
                "must both agree before money stops."
            )

    return Decision(
        merchant_id=merchant_id,
        declared_category=declared_category,
        tier=tier,
        channels=channels,
        justification=justification,
        dissenting_channels=dissenting,
        second_channel_source=second_source,
        notes=notes,
        assessment=assessment.to_dict() if assessment is not None else None,
    )


# ---------------------------------------------------------------------------
# What a decision costs
# ---------------------------------------------------------------------------

def decision_cost(
    tier: Tier,
    is_actually_diverged: bool,
    daily_gmv_inr: float,
    days_since_divergence: int,
    econ: Economics,
) -> dict:
    """Price one decision in rupees, split into the costs it actually creates.

    Holding a legitimate merchant's settlements is holding their payroll. A
    system that treats that as a rounding error in its precision number has
    misunderstood the job, so it is priced here as an explicit line rather than
    absorbed into an F1 score.
    """
    spec = LADDER[tier]
    analyst = econ.analyst_review_cost_inr if spec["analyst_time"] else 0.0

    false_hold = 0.0
    if spec["settlements_held"] and not is_actually_diverged:
        frozen = daily_gmv_inr * econ.hold_days
        lifetime_margin = (
            daily_gmv_inr * econ.take_rate * econ.merchant_remaining_lifetime_days
        )
        false_hold = (
            frozen * econ.hold_harm_rate
            + econ.hold_fixed_cost_inr
            # Expected lost lifetime margin from the merchants who leave. This
            # term scales with merchant size, which is the point: wrongly
            # holding a large legitimate merchant is the most expensive single
            # mistake the system can make, and a count of false positives
            # cannot see that.
            + econ.churn_probability_after_hold * lifetime_margin
        )

    missed = 0.0
    if is_actually_diverged and tier < Tier.REVIEW:
        exposure = daily_gmv_inr * max(days_since_divergence, 1)
        missed = (
            econ.network_fine_inr
            + exposure * econ.chargeback_liability_rate
            + exposure * econ.enforcement_freeze_rate
        )

    return {
        "analyst_cost": round(analyst, 2),
        "false_hold_cost": round(false_hold, 2),
        "missed_abuse_cost": round(missed, 2),
        "total": round(analyst + false_hold + missed, 2),
    }
