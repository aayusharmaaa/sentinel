"""Central configuration. Everything tunable lives here so the evaluation
harness can sweep a parameter without editing pipeline code."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FIXTURES = ROOT / "sentinel" / "vision" / "fixtures"

# ---------------------------------------------------------------------------
# Declared categories. Each is a category a merchant can select at onboarding.
# The whole telemetry channel is relative to these: we never ask "is this
# merchant unusual", we ask "is this merchant unusual FOR WHAT IT CLAIMS".
# ---------------------------------------------------------------------------
CATEGORIES = [
    "home_furnishing",
    "electronics_retail",
    "apparel",
    "skill_gaming",
    "subscription_box",
    "pharmacy",
    "travel_agency",
    "digital_services",
]

# Verticals a storefront can actually be classified into by the vision channel.
VERTICALS = CATEGORIES + [
    "sports_betting",
    "unlicensed_lending",
    "crypto_exchange",
    "adult_content",
    "unregistered_pharmacy",
    "unknown",
]

# Verticals that can never be approved on this aggregator.
PROHIBITED_VERTICALS = {
    "sports_betting",
    "unlicensed_lending",
    "crypto_exchange",
    "adult_content",
    "unregistered_pharmacy",
}


@dataclass(frozen=True)
class Economics:
    """Cost model used to price false positives and false negatives in rupees.

    These are the numbers that make the precision/recall trade-off legible to
    someone who has to sign off on it. They are deliberately conservative and
    every one of them is stated, not hidden inside a scoring constant.
    """

    # A settlement hold freezes a merchant's payouts while a case is worked.
    # Median working days a Tier 3 case is open before resolution.
    hold_days: float = 4.0
    # Working-capital harm as a fraction of the frozen volume. Not the full
    # frozen amount -- most of it is eventually released.
    hold_harm_rate: float = 0.06
    # Flat operational cost of a hold: support escalation, comms, reinstatement.
    hold_fixed_cost_inr: float = 12_000.0
    # The dominant cost of a wrong hold, and the one it is easiest to leave out.
    # A legitimate business whose payouts stopped for four days without warning
    # often does not stay. Omitting this makes missed abuse look an order of
    # magnitude worse than a false hold, which would tell you to hold everything
    # -- and a cost model that recommends holding everything is not a cost model,
    # it is a rounding error with a conclusion attached.
    churn_probability_after_hold: float = 0.22
    # Aggregator take rate, and the remaining lifetime over which it is earned.
    take_rate: float = 0.02
    merchant_remaining_lifetime_days: float = 1_095.0
    # Fully loaded cost of one analyst review (Tier 2 and Tier 3 both incur it).
    analyst_review_cost_inr: float = 900.0

    # A missed illicit merchant costs the aggregator on four lines.
    # Card network non-compliance exposure, amortised per undetected merchant.
    network_fine_inr: float = 120_000.0
    # Chargeback liability the aggregator eats after the merchant disappears,
    # as a fraction of the volume processed after divergence began.
    chargeback_liability_rate: float = 0.11
    # Expected value of funds frozen in the aggregator's pool by enforcement,
    # as a fraction of post-divergence volume.
    enforcement_freeze_rate: float = 0.18

    # Cost of the expensive channel. Vision classification is the only
    # per-merchant external spend; this is what the two-speed design buys down.
    vision_call_cost_inr: float = 2.20
    # An automated verification call: telephony, speech processing and the
    # small share that escalates to a human when the merchant declines.
    voice_call_cost_inr: float = 14.0


@dataclass(frozen=True)
class Thresholds:
    """Fusion thresholds. Tuned on the training split only -- never on test."""

    # Telemetry dissent: category-relative anomaly score above this means the
    # payment behaviour does not match the declared category.
    telemetry_dissent: float = 0.62
    # Vision dissent: storefront classified as a vertical incompatible with the
    # declaration, at or above this confidence.
    vision_dissent_confidence: float = 0.55
    # Triage gate: telemetry score above this earns an expensive vision call.
    triage_gate: float = 0.35
    # Agent stops early once fused confidence clears this.
    agent_confidence_stop: float = 0.90


@dataclass(frozen=True)
class Settings:
    seed: int = 7
    n_merchants: int = 4000
    days: int = 90
    abuse_prevalence: float = 0.035  # realistic base rate, not a balanced toy set
    test_fraction: float = 0.30
    economics: Economics = field(default_factory=Economics)
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def live_mode(self) -> bool:
        """True when a real model key is present. Everything degrades to
        deterministic cached fixtures when it is not, so the repo is runnable
        end to end -- including the full evaluation -- with no credentials."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))


def stable_hash(*parts: object) -> int:
    """A hash that survives process restarts.

    Python randomises `hash()` for strings on every interpreter start unless
    PYTHONHASHSEED is pinned. Anything that seeds behaviour from `hash()` is
    therefore not reproducible across runs -- which is fatal here, because the
    system's claim is that a decision can be re-derived months later and come
    out identical. Every seed, every deterministic assignment, and every
    fixture selection goes through this instead.
    """
    raw = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


SETTINGS = Settings()
