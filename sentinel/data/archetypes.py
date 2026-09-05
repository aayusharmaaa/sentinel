"""Behavioural profiles for legitimate categories and abuse archetypes.

The design rule for this file: every abuse archetype must be detectable by a
specific channel and invisible to another, and every category must contain at
least one legitimate profile that mimics an abuse signature. If the synthetic
population does not contain merchants that single-signal rules get wrong, the
evaluation cannot demonstrate that the two-channel gate is worth anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BehaviourProfile:
    """Parameters that drive one merchant's transaction stream."""

    # Share of transactions priced at an exact multiple of Rs 100.
    round_share: float
    # Log-normal ticket size (rupees).
    ticket_mu: float
    ticket_sigma: float
    # Hour-of-day activity: mixture of gaussians over 0..23, (centre, spread, weight).
    hour_peaks: Sequence[tuple[float, float, float]]
    # Distinct transactions per unique payment instrument per day.
    repeat_velocity: float
    # Daily transaction count at t=0, and multiplicative growth per day.
    base_daily_txns: float
    growth_per_day: float
    refund_rate: float
    chargeback_rate: float
    # Whether a spike, if present, decays back down (viral sale) or holds (pivot).
    spike_decays: bool = True

    def hour_pdf(self) -> np.ndarray:
        hours = np.arange(24)
        pdf = np.zeros(24)
        for centre, spread, weight in self.hour_peaks:
            d = np.minimum(np.abs(hours - centre), 24 - np.abs(hours - centre))
            pdf += weight * np.exp(-0.5 * (d / spread) ** 2)
        return pdf / pdf.sum()


# ---------------------------------------------------------------------------
# Legitimate category profiles.
# ---------------------------------------------------------------------------
# Note skill_gaming and subscription_box: these are legitimate businesses whose
# payment telemetry is close to indistinguishable from a laundering front. They
# are the reason absolute thresholds cannot be used, and the reason the gate
# requires a second, independent channel before settlements are held.

LEGIT_PROFILES: dict[str, BehaviourProfile] = {
    "home_furnishing": BehaviourProfile(
        round_share=0.14, ticket_mu=8.1, ticket_sigma=0.75,
        hour_peaks=[(13, 3.5, 1.0), (20, 2.5, 0.8)],
        repeat_velocity=1.12, base_daily_txns=11, growth_per_day=0.0012,
        refund_rate=0.045, chargeback_rate=0.0022,
    ),
    "electronics_retail": BehaviourProfile(
        round_share=0.19, ticket_mu=8.6, ticket_sigma=0.9,
        hour_peaks=[(12, 3.0, 0.9), (21, 2.8, 1.0)],
        repeat_velocity=1.18, base_daily_txns=19, growth_per_day=0.0016,
        refund_rate=0.061, chargeback_rate=0.0034,
    ),
    "apparel": BehaviourProfile(
        round_share=0.22, ticket_mu=7.5, ticket_sigma=0.7,
        hour_peaks=[(14, 4.0, 1.0), (21, 3.0, 0.9)],
        repeat_velocity=1.31, base_daily_txns=28, growth_per_day=0.0018,
        refund_rate=0.098, chargeback_rate=0.0029,
    ),
    # Legitimate skill gaming: round deposits, high repeat, late-night skew.
    # Telemetry signature overlaps heavily with a betting front. By design.
    "skill_gaming": BehaviourProfile(
        round_share=0.68, ticket_mu=6.3, ticket_sigma=0.55,
        hour_peaks=[(22, 3.2, 1.0), (1, 2.6, 0.75), (16, 3.0, 0.4)],
        repeat_velocity=4.6, base_daily_txns=66, growth_per_day=0.0031,
        refund_rate=0.021, chargeback_rate=0.0071,
    ),
    # Legitimate subscription commerce: very high repeat velocity, flat hours.
    "subscription_box": BehaviourProfile(
        round_share=0.41, ticket_mu=6.9, ticket_sigma=0.35,
        hour_peaks=[(9, 5.5, 1.0), (18, 4.5, 0.7)],
        repeat_velocity=3.9, base_daily_txns=38, growth_per_day=0.0022,
        refund_rate=0.034, chargeback_rate=0.0041,
    ),
    "pharmacy": BehaviourProfile(
        round_share=0.16, ticket_mu=6.7, ticket_sigma=0.6,
        hour_peaks=[(11, 3.5, 1.0), (19, 3.0, 0.95)],
        repeat_velocity=1.74, base_daily_txns=30, growth_per_day=0.0014,
        refund_rate=0.028, chargeback_rate=0.0019,
    ),
    "travel_agency": BehaviourProfile(
        round_share=0.11, ticket_mu=9.3, ticket_sigma=1.05,
        hour_peaks=[(11, 3.0, 1.0), (17, 3.5, 0.85)],
        repeat_velocity=1.05, base_daily_txns=8, growth_per_day=0.0009,
        refund_rate=0.121, chargeback_rate=0.0052,
    ),
    "digital_services": BehaviourProfile(
        round_share=0.37, ticket_mu=7.1, ticket_sigma=0.8,
        hour_peaks=[(15, 5.0, 1.0), (22, 3.5, 0.8)],
        repeat_velocity=2.1, base_daily_txns=23, growth_per_day=0.0026,
        refund_rate=0.052, chargeback_rate=0.0046,
    ),
}


# ---------------------------------------------------------------------------
# Post-divergence profiles: what the merchant's payments actually look like
# once the real business is no longer the declared one.
# ---------------------------------------------------------------------------

DIVERGED_PROFILES: dict[str, BehaviourProfile] = {
    "sports_betting": BehaviourProfile(
        round_share=0.79, ticket_mu=6.4, ticket_sigma=0.5,
        hour_peaks=[(23, 2.8, 1.0), (2, 2.4, 0.9), (19, 2.2, 0.55)],
        repeat_velocity=6.9, base_daily_txns=81, growth_per_day=0.0125,
        refund_rate=0.014, chargeback_rate=0.0165, spike_decays=False,
    ),
    "unlicensed_lending": BehaviourProfile(
        round_share=0.86, ticket_mu=7.9, ticket_sigma=0.42,
        hour_peaks=[(10, 4.5, 1.0), (16, 4.0, 0.9)],
        repeat_velocity=1.9, base_daily_txns=56, growth_per_day=0.0168,
        refund_rate=0.006, chargeback_rate=0.0243, spike_decays=False,
    ),
    "crypto_exchange": BehaviourProfile(
        round_share=0.72, ticket_mu=9.1, ticket_sigma=0.95,
        hour_peaks=[(3, 6.0, 1.0), (14, 6.0, 0.95)],
        repeat_velocity=5.2, base_daily_txns=47, growth_per_day=0.0141,
        refund_rate=0.009, chargeback_rate=0.0128, spike_decays=False,
    ),
    "adult_content": BehaviourProfile(
        round_share=0.63, ticket_mu=6.6, ticket_sigma=0.45,
        hour_peaks=[(0, 3.0, 1.0), (23, 2.5, 0.95)],
        repeat_velocity=3.4, base_daily_txns=59, growth_per_day=0.0092,
        refund_rate=0.018, chargeback_rate=0.0311, spike_decays=False,
    ),
    # A pharmacy without registration sells the same basket at the same prices
    # to the same people at the same hours as one with it. The telemetry is
    # identical on purpose: this archetype exists to prove that the telemetry
    # channel has a blind spot no amount of feature engineering closes.
    "unregistered_pharmacy": BehaviourProfile(
        round_share=0.16, ticket_mu=6.7, ticket_sigma=0.6,
        hour_peaks=[(11, 3.5, 1.0), (19, 3.0, 0.95)],
        repeat_velocity=1.74, base_daily_txns=30, growth_per_day=0.0014,
        refund_rate=0.028, chargeback_rate=0.0019,
    ),
}


@dataclass(frozen=True)
class Archetype:
    """An abuse pattern, defined by which channel can see it."""

    name: str
    # Categories a merchant using this archetype plausibly declares.
    declares: Sequence[str]
    # What the business actually becomes.
    becomes: str
    # Fraction of post-divergence volume that is the illicit business
    # 1.0 means a full pivot; below 1.0 the genuine store keeps running.
    illicit_share: float
    # Which rendered surface reveals the real business.
    tell_surface: str
    telemetry_visible: bool
    vision_visible: bool
    notes: str


ABUSE_ARCHETYPES: list[Archetype] = [
    Archetype(
        name="pivot_prohibited",
        declares=["home_furnishing", "apparel", "electronics_retail", "digital_services"],
        becomes="sports_betting",
        illicit_share=1.0,
        tell_surface="homepage",
        telemetry_visible=True,
        vision_visible=True,
        notes=(
            "Honest onboarding, then a full pivot into a prohibited vertical on "
            "the same merchant ID. Both channels dissent. The clean Tier 3 case."
        ),
    ),
    Archetype(
        name="rented_account",
        declares=["electronics_retail", "home_furnishing", "apparel", "travel_agency"],
        becomes="unlicensed_lending",
        illicit_share=0.62,
        tell_surface="checkout",
        telemetry_visible=True,
        vision_visible=True,
        notes=(
            "The genuine storefront keeps trading while the same MID processes "
            "for a second business. Telemetry is a mixture, so the deviation is "
            "diluted; the homepage still classifies clean. Only the checkout "
            "surface reveals it. This is the case the agent exists for."
        ),
    ),
    Archetype(
        name="funnel_ring",
        declares=["digital_services", "apparel", "subscription_box", "electronics_retail"],
        becomes="crypto_exchange",
        illicit_share=0.85,
        tell_surface="checkout",
        telemetry_visible=False,
        vision_visible=True,
        notes=(
            "Several small accounts funnelling to one beneficiary. Each member "
            "is individually unremarkable and sits below the single-merchant "
            "telemetry threshold; the ring is only visible as shared "
            "infrastructure across members."
        ),
    ),
    Archetype(
        name="licensing_violation",
        declares=["pharmacy"],
        becomes="unregistered_pharmacy",
        illicit_share=1.0,
        tell_surface="homepage",
        telemetry_visible=False,
        vision_visible=True,
        notes=(
            "A pharmacy operating without registration has exactly the payment "
            "shape of one with it. Telemetry cannot see this and never will. "
            "Vision can. This is why single-channel dissent must still produce "
            "action, just lesser action than a settlement hold."
        ),
    ),
]

ARCHETYPES_BY_NAME = {a.name: a for a in ABUSE_ARCHETYPES}


# ---------------------------------------------------------------------------
# Legitimate-but-alarming variants. These are the false-positive traps.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardNegative:
    name: str
    applies_to: Sequence[str]
    notes: str


HARD_NEGATIVES: list[HardNegative] = [
    HardNegative(
        "viral_spike",
        ["apparel", "electronics_retail", "home_furnishing", "subscription_box"],
        "A genuine sale or press mention. Volume slope spikes hard, then decays. "
        "Slope alone flags it; slope plus peak shape does not.",
    ),
    HardNegative(
        "festival_season",
        ["apparel", "electronics_retail", "home_furnishing", "travel_agency"],
        "Seasonal step change in volume and a shifted hour histogram. Looks like "
        "a pivot to a rule that only reads volume trend.",
    ),
    HardNegative(
        "price_point_change",
        ["digital_services", "subscription_box", "apparel"],
        "Merchant moves to a flat Rs 499 or Rs 999 SKU. Round-value share jumps "
        "to near 1.0 overnight with no change of business whatsoever.",
    ),
    HardNegative(
        "shared_platform",
        ["digital_services", "subscription_box", "apparel", "pharmacy"],
        "Uses the same popular hosting ASN and checkout plugin as hundreds of "
        "others. Naive infrastructure clustering links them into a fake ring.",
    ),
]

HARD_NEGATIVES_BY_NAME = {h.name: h for h in HARD_NEGATIVES}
