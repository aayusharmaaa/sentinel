"""Synthetic merchant population and payment streams.

Honesty note, stated here because it is the first thing a reviewer should ask:
this is generated data. It cannot prove Sentinel works on Razorpay's real book.
What it can do, and what it is built to do, is hold the design claims to a
falsifiable standard:

  1. Abuse archetypes are generated with a declared channel visibility
     (see archetypes.py). licensing_violation is drawn from a distribution
     identical to legitimate pharmacy, so any telemetry model that "detects"
     it is reading noise. funnel_ring members are drawn below the
     single-merchant threshold, so recall on them must come from the
     infrastructure signal or not at all.
  2. Legitimate merchants include four hard-negative variants whose telemetry
     mimics abuse. A model that scores 1.0 on this set has memorised the
     generator, not learned the task.
  3. Prevalence is 3.5%, not 50%. Precision numbers on a balanced toy set are
     meaningless for a system whose cost function is dominated by false
     positives.

The generator writes labels for archetype and channel-visibility so the
evaluation can report recall broken out by what was structurally detectable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.config import ARTIFACTS, CATEGORIES, SETTINGS, stable_hash
from sentinel.data.archetypes import (
    ABUSE_ARCHETYPES,
    DIVERGED_PROFILES,
    HARD_NEGATIVES,
    LEGIT_PROFILES,
    BehaviourProfile,
)

# Hosting ASNs. Deliberately heavy-tailed: the top three carry most of the
# legitimate population, which is what makes naive shared-ASN clustering
# useless and forces the infrastructure signal to weight by rarity.
POPULAR_ASNS = ["AS16509", "AS13335", "AS14061", "AS24940", "AS394161"]
LONGTAIL_ASNS = [f"AS{n}" for n in range(45000, 45140)]

POPULAR_SCRIPT_BUNDLES = [
    "gtm-core@2.14", "razorpay-checkout@1.9", "hotjar@6.1",
    "fbpixel@2.9", "clarity@0.7",
]
RARE_SCRIPT_BUNDLES = [f"bundle-{h:04x}" for h in range(0x1000, 0x10A0)]

# Scripts that only appear on a genuinely different kind of business. The agent
# looks for these with list_third_party_scripts.
VERTICAL_SCRIPTS = {
    "sports_betting": ["oddsfeed-live@3.2", "geo-fence-sdk@1.1"],
    "unlicensed_lending": ["kyc-lite@0.9", "collections-dialer@2.0"],
    "crypto_exchange": ["chain-rates@4.4", "wallet-connect@2.11"],
    "adult_content": ["age-gate@1.3"],
    # unregistered_pharmacy deliberately has none. A prescription-upload widget
    # is what a LICENSED pharmacy loads; an unregistered one whose entire offer
    # is "no prescription needed" has no distinguishing script at all. Giving it
    # one would hand the corroborating channel a tell that does not exist, and
    # would let licensing violations reach a settlement hold on evidence the
    # real world does not provide -- which is exactly the case the design says
    # should stop at analyst review.
}


@dataclass
class Merchant:
    merchant_id: str
    declared_category: str
    onboarded_day: int          # days before the observation window opened
    scale: float                # volume multiplier, long-tailed
    label: int                  # 1 = business has diverged from declaration
    archetype: str              # abuse archetype, or "legit"
    hard_negative: str          # legit-but-alarming variant, or "none"
    true_vertical: str
    divergence_day: int         # -1 when the merchant never diverges
    illicit_share: float
    tell_surface: str           # which rendered surface reveals it
    telemetry_visible: int      # ground truth: can telemetry see this at all
    vision_visible: int
    ring_id: str
    asn: str
    ip_block: str
    payout_fingerprint: str
    script_bundle: str
    # Identities the relationship graph joins on. A payout fingerprint is the
    # strongest of these -- two merchants settling to the same beneficiary is a
    # far rarer coincidence than two merchants on the same host -- and the graph
    # weights them accordingly rather than treating every shared attribute alike.
    beneficiary_id: str
    phone_fp: str
    device_fp: str
    # Which storefront fixture this merchant renders. Used by the vision channel.
    fixture_homepage: str
    fixture_checkout: str
    # Whether the storefront carries text addressed at an automated classifier.
    injection_present: int


def _stable_hash(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _blend_profiles(base: BehaviourProfile, other: BehaviourProfile, w: float) -> BehaviourProfile:
    """Mixture used for rented accounts, where the genuine store keeps trading."""
    def mix(a: float, b: float) -> float:
        return (1 - w) * a + w * b

    return BehaviourProfile(
        round_share=mix(base.round_share, other.round_share),
        ticket_mu=mix(base.ticket_mu, other.ticket_mu),
        ticket_sigma=mix(base.ticket_sigma, other.ticket_sigma),
        hour_peaks=list(base.hour_peaks) + [
            (c, s, wt * w / max(1 - w, 1e-6)) for (c, s, wt) in other.hour_peaks
        ],
        repeat_velocity=mix(base.repeat_velocity, other.repeat_velocity),
        base_daily_txns=mix(base.base_daily_txns, other.base_daily_txns),
        growth_per_day=mix(base.growth_per_day, other.growth_per_day),
        refund_rate=mix(base.refund_rate, other.refund_rate),
        chargeback_rate=mix(base.chargeback_rate, other.chargeback_rate),
        spike_decays=other.spike_decays,
    )


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def build_population(rng: np.random.Generator, n: int) -> list[Merchant]:
    merchants: list[Merchant] = []
    n_abuse = int(round(n * SETTINGS.abuse_prevalence))

    # Rings consume several merchant slots each.
    ring_specs: list[tuple[str, int]] = []
    remaining_ring_budget = int(n_abuse * 0.30)
    ring_idx = 0
    while remaining_ring_budget > 5:
        size = int(rng.integers(5, 9))
        size = min(size, remaining_ring_budget)
        ring_specs.append((f"RING-{ring_idx:02d}", size))
        remaining_ring_budget -= size
        ring_idx += 1

    ring_member_total = sum(s for _, s in ring_specs)
    solo_abuse = n_abuse - ring_member_total

    def identities(is_ring: bool, ring_id: str, seq: int, payout: str
                   ) -> tuple[str, str, str]:
        """Beneficiary, contact phone and device fingerprint.

        Ring members share a beneficiary by definition -- that is what makes
        them a ring rather than four unrelated shops -- and often reuse a
        contact number or a device. Legitimate merchants share none of these,
        but a small share of them do reuse a phone: an accountant or an agency
        that registers several genuine businesses. Without those the phone edge
        would be a perfect signal, and a perfect edge is not one an evaluation
        can learn anything from.
        """
        if is_ring:
            ben = f"BEN-{_stable_hash('ben', ring_id)}"
            phone = (f"PH-{_stable_hash('ph', ring_id)}"
                     if rng.random() < 0.72 else f"PH-{_stable_hash('ph', seq)}")
            dev = (f"DEV-{_stable_hash('dev', ring_id)}"
                   if rng.random() < 0.55 else f"DEV-{_stable_hash('dev', seq)}")
            return ben, phone, dev
        ben = f"BEN-{_stable_hash('ben', payout)}"
        # ~3% of legitimate merchants share a filing agent's phone number.
        phone = (f"PH-agent{int(rng.integers(0, 14))}"
                 if rng.random() < 0.03 else f"PH-{_stable_hash('ph', seq)}")
        dev = f"DEV-{_stable_hash('dev', seq)}"
        return ben, phone, dev

    def infra(is_ring: bool, ring_id: str, seq: int) -> tuple[str, str, str, str]:
        if is_ring:
            # Ring members share a rare tuple. Not the popular ASN: a ring that
            # hid behind the most common host would be invisible to this signal,
            # and pretending otherwise would flatter the evaluation.
            return (
                LONGTAIL_ASNS[stable_hash(ring_id) % len(LONGTAIL_ASNS)],
                f"103.{stable_hash(ring_id) % 254}."
                f"{(stable_hash(ring_id) // 254) % 254}.0/24",
                f"payout-{_stable_hash(ring_id)}",
                RARE_SCRIPT_BUNDLES[stable_hash(ring_id) % len(RARE_SCRIPT_BUNDLES)],
            )
        if rng.random() < 0.62:
            asn = POPULAR_ASNS[int(rng.integers(0, len(POPULAR_ASNS)))]
        else:
            asn = LONGTAIL_ASNS[int(rng.integers(0, len(LONGTAIL_ASNS)))]
        bundle = (
            POPULAR_SCRIPT_BUNDLES[int(rng.integers(0, len(POPULAR_SCRIPT_BUNDLES)))]
            if rng.random() < 0.75
            else RARE_SCRIPT_BUNDLES[int(rng.integers(0, len(RARE_SCRIPT_BUNDLES)))]
        )
        return (
            asn,
            f"{int(rng.integers(11, 220))}.{int(rng.integers(0, 254))}."
            f"{int(rng.integers(0, 254))}.0/24",
            f"payout-{_stable_hash('solo', seq, rng.integers(0, 10**9))}",
            bundle,
        )

    seq = 0

    def make(
        category: str,
        label: int,
        archetype: str,
        hard_negative: str,
        true_vertical: str,
        divergence_day: int,
        illicit_share: float,
        tell_surface: str,
        tel_vis: int,
        vis_vis: int,
        ring_id: str,
        is_ring: bool,
    ) -> Merchant:
        nonlocal seq
        seq += 1
        asn, ip_block, payout, bundle = infra(is_ring, ring_id, seq)
        beneficiary, phone_fp, device_fp = identities(is_ring, ring_id, seq, payout)
        if hard_negative == "shared_platform":
            # The false-positive trap for infrastructure clustering: hundreds of
            # unrelated legitimate merchants on one host and one plugin build.
            asn, bundle = POPULAR_ASNS[0], POPULAR_SCRIPT_BUNDLES[1]
            ip_block = "13.234.0.0/16"
            # These merchants genuinely look connected: same host, same plugin
            # build, same filing agent. They are the graph's false-positive trap.
            phone_fp = "PH-agent0"
        # Merchants that have diverged put adversarial copy on the page with
        # some probability -- text addressed at an automated reviewer.
        injection = int(label == 1 and rng.random() < 0.34)
        return Merchant(
            merchant_id=f"MID{100000 + seq}",
            declared_category=category,
            onboarded_day=int(rng.integers(40, 900)),
            scale=float(np.clip(np.exp(rng.normal(0.0, 0.62)), 0.22, 4.5)),
            label=label,
            archetype=archetype,
            hard_negative=hard_negative,
            true_vertical=true_vertical,
            divergence_day=divergence_day,
            illicit_share=illicit_share,
            tell_surface=tell_surface,
            telemetry_visible=tel_vis,
            vision_visible=vis_vis,
            ring_id=ring_id,
            asn=asn,
            ip_block=ip_block,
            payout_fingerprint=payout,
            script_bundle=bundle,
            beneficiary_id=beneficiary,
            phone_fp=phone_fp,
            device_fp=device_fp,
            fixture_homepage=(
                true_vertical if (label == 1 and tell_surface == "homepage") else category
            ),
            fixture_checkout=(true_vertical if label == 1 else category),
            injection_present=injection,
        )

    # --- ring members ---
    for ring_id, size in ring_specs:
        arch = next(a for a in ABUSE_ARCHETYPES if a.name == "funnel_ring")
        for _ in range(size):
            cat = str(rng.choice(list(arch.declares)))
            merchants.append(
                make(
                    cat, 1, arch.name, "none", arch.becomes,
                    int(rng.integers(25, 74)), arch.illicit_share, arch.tell_surface,
                    int(arch.telemetry_visible), int(arch.vision_visible), ring_id, True,
                )
            )

    # --- solo abusers ---
    solo_archs = [a for a in ABUSE_ARCHETYPES if a.name != "funnel_ring"]
    weights = np.array([0.42, 0.34, 0.24])  # pivot, rented, licensing
    for _ in range(max(solo_abuse, 0)):
        arch = solo_archs[int(rng.choice(len(solo_archs), p=weights))]
        cat = str(rng.choice(list(arch.declares)))
        merchants.append(
            make(
                cat, 1, arch.name, "none", arch.becomes,
                int(rng.integers(20, 78)), arch.illicit_share, arch.tell_surface,
                int(arch.telemetry_visible), int(arch.vision_visible), "none", False,
            )
        )

    # --- legitimate population ---
    # Category mix is skewed towards ordinary retail, with a meaningful slice of
    # the two categories whose honest telemetry looks like abuse.
    cat_p = np.array([0.17, 0.19, 0.20, 0.09, 0.08, 0.09, 0.08, 0.10])
    cat_p = cat_p / cat_p.sum()
    for _ in range(n - len(merchants)):
        cat = str(rng.choice(CATEGORIES, p=cat_p))
        eligible = [h.name for h in HARD_NEGATIVES if cat in h.applies_to]
        hn = "none"
        if eligible and rng.random() < 0.30:
            hn = str(rng.choice(eligible))
        merchants.append(
            make(cat, 0, "legit", hn, cat, -1, 0.0, "none", 0, 0, "none", False)
        )

    rng.shuffle(merchants)  # type: ignore[arg-type]
    return merchants


# ---------------------------------------------------------------------------
# Transaction stream
# ---------------------------------------------------------------------------

ROUND_TICKETS = np.array([100, 200, 300, 500, 1000, 2000, 5000, 10000], dtype=float)


def _personalise(p: BehaviourProfile, rng: np.random.Generator) -> BehaviourProfile:
    """Give one merchant its own profile, drawn around the category's.

    This is the most consequential function in the generator and it was missing
    from the first version, which drew every merchant in a category from
    identical parameters. The effect was a category baseline with almost no
    spread, so a legitimate merchant sat at zero sigma and any diverged merchant
    sat at forty, and the telemetry model separated them perfectly. That is not
    a detector working. That is a detector reading the generator.

    Real merchants inside one category vary enormously: two furniture shops have
    different price points, different repeat rates, different peak hours. The
    spread here is what makes a z-score against the category baseline mean
    anything, and it is what allows a legitimate merchant to land in the tail --
    which is the case the whole two-channel gate exists to handle.
    """
    def lg(sigma: float) -> float:
        return float(np.exp(rng.normal(0.0, sigma)))

    return BehaviourProfile(
        round_share=float(np.clip(p.round_share + rng.normal(0, 0.115), 0.01, 0.97)),
        ticket_mu=float(p.ticket_mu + rng.normal(0, 0.38)),
        ticket_sigma=float(max(p.ticket_sigma * lg(0.22), 0.15)),
        hour_peaks=[
            (float(c + rng.normal(0, 1.7)), float(max(sp * lg(0.26), 1.2)),
             float(max(w * lg(0.24), 0.05)))
            for c, sp, w in p.hour_peaks
        ],
        repeat_velocity=float(max(p.repeat_velocity * lg(0.30), 1.0)),
        base_daily_txns=p.base_daily_txns,
        growth_per_day=float(p.growth_per_day + rng.normal(0, 0.0016)),
        refund_rate=float(np.clip(p.refund_rate * lg(0.42), 0.0, 0.40)),
        chargeback_rate=float(np.clip(p.chargeback_rate * lg(0.48), 0.0, 0.09)),
        spike_decays=p.spike_decays,
    )


# The five feature families the telemetry channel measures. An abuser does not
# move all of them, which is the point of splitting them out.
FEATURE_FAMILIES = ("pricing", "hours", "velocity", "growth", "risk")


def _partial_diverge(
    base: BehaviourProfile,
    target: BehaviourProfile,
    rng: np.random.Generator,
    share: float,
) -> BehaviourProfile:
    """Blend towards the illicit profile on some feature families, not all.

    A merchant that pivots does not repaint every dimension of its payment
    behaviour at once. It might take on the new business's ticket structure
    while keeping its existing customer base's hours, or acquire the new hours
    and keep pricing. Diverging on all five families simultaneously produces an
    abuse population that no real operator resembles and that any model
    separates trivially.

    Each abuser here moves on two to five families, chosen at random, blended by
    `share`. The ones that move on two are genuinely hard, and they are supposed
    to be: some of them should be missed, and the evaluation should say so.
    """
    k = int(rng.integers(2, len(FEATURE_FAMILIES) + 1))
    moved = set(rng.choice(FEATURE_FAMILIES, size=k, replace=False).tolist())

    def mix(a: float, b: float, family: str) -> float:
        return (1 - share) * a + share * b if family in moved else a

    hours = (
        list(base.hour_peaks) + [
            (c, sp, w * share / max(1 - share, 1e-6)) for c, sp, w in target.hour_peaks
        ]
        if "hours" in moved else list(base.hour_peaks)
    )
    return BehaviourProfile(
        round_share=mix(base.round_share, target.round_share, "pricing"),
        ticket_mu=mix(base.ticket_mu, target.ticket_mu, "pricing"),
        ticket_sigma=mix(base.ticket_sigma, target.ticket_sigma, "pricing"),
        hour_peaks=hours,
        repeat_velocity=mix(base.repeat_velocity, target.repeat_velocity, "velocity"),
        base_daily_txns=base.base_daily_txns,
        growth_per_day=mix(base.growth_per_day, target.growth_per_day, "growth"),
        refund_rate=mix(base.refund_rate, target.refund_rate, "risk"),
        chargeback_rate=mix(base.chargeback_rate, target.chargeback_rate, "risk"),
        spike_decays=target.spike_decays,
    )


def simulate_stream(m: Merchant, rng: np.random.Generator, days: int) -> pd.DataFrame:
    """Generate one merchant's transactions over the observation window."""
    # Every merchant gets its own profile drawn around its category's, so the
    # category baseline has real spread and a z-score means something.
    base = _personalise(LEGIT_PROFILES[m.declared_category], rng)
    post = None
    if m.label == 1 and m.archetype == "licensing_violation":
        # Structurally invisible to telemetry, and generated that way rather
        # than merely described that way. An unregistered pharmacy sells the
        # same basket at the same prices to the same people at the same hours as
        # a registered one; the only difference is a licence it does not hold.
        # Post-divergence behaviour is byte-for-byte the pre-divergence
        # behaviour, so any telemetry model that scores these above chance is
        # reading noise, and the evaluation will show that it does not.
        post = None
    elif m.label == 1:
        target = _personalise(DIVERGED_PROFILES[m.true_vertical], rng)
        if m.archetype == "funnel_ring":
            # Ring members are individually unremarkable by construction. Each
            # account carries a small slice of the funnel, which is the entire
            # reason a ring is built this way rather than run through one
            # merchant ID. Recall on these has to come from the infrastructure
            # signal; if it came from telemetry the archetype would be pointless.
            share = float(rng.uniform(0.10, 0.32))
        else:
            # How much of the illicit business shows in the payment stream, and
            # how careful the operator is about it. Beta(2.2, 2.0) puts most
            # operators in the middle rather than at a full, obvious swap.
            share = float(np.clip(m.illicit_share * rng.beta(2.2, 2.0) * 1.35, 0.15, 1.0))
        post = _partial_diverge(base, target, rng, share)
    # Post-divergence volume is anchored to this merchant's own trading level,
    # not to the archetype's absolute scale. A merchant that pivots does not
    # suddenly acquire the volume of a large betting site; it processes what its
    # new business brings in, starting from where it was. Keying off the
    # archetype's base rate would leave a 7x step change in raw transaction
    # count that any rule would catch, and that is not the problem worth solving.
    post_multiplier = (
        1.0 if m.archetype == "licensing_violation"
        else float(rng.uniform(1.02, 1.6)) if m.archetype == "funnel_ring"
        else float(rng.uniform(1.05, 2.4))
    )

    # Hard-negative modifiers, applied to a legitimate merchant.
    spike_day = int(rng.integers(30, days - 12)) if m.hard_negative == "viral_spike" else -1
    season_start = int(rng.integers(35, days - 20)) if m.hard_negative == "festival_season" else -1
    repricing_day = int(rng.integers(30, days - 15)) if m.hard_negative == "price_point_change" else -1

    rows_day: list[int] = []
    rows_hour: list[int] = []
    rows_amt: list[float] = []
    rows_inst: list[int] = []
    rows_refund: list[int] = []
    rows_cb: list[int] = []

    instrument_pool_size = 0
    instrument_high_water = 0

    for d in range(days):
        diverged = m.label == 1 and d >= m.divergence_day
        prof = post if (diverged and post is not None) else base

        if diverged and post is not None:
            age = d - m.divergence_day
            growth = min(prof.growth_per_day, 0.007)
            count = (
                base.base_daily_txns * m.scale * post_multiplier
                * ((1 + base.growth_per_day) ** m.divergence_day)
                * ((1 + growth) ** age)
            )
        else:
            count = prof.base_daily_txns * m.scale * ((1 + prof.growth_per_day) ** d)
        round_share = prof.round_share
        hour_pdf = prof.hour_pdf()

        if spike_day >= 0 and d >= spike_day:
            # Genuine viral spike: sharp rise, exponential decay back to trend.
            age = d - spike_day
            count *= 1.0 + 5.5 * np.exp(-age / 4.5)
        if season_start >= 0 and d >= season_start:
            count *= 2.3
            hour_pdf = 0.6 * hour_pdf + 0.4 * BehaviourProfile(
                0, 0, 0, [(21, 3.0, 1.0), (12, 3.0, 0.7)], 0, 0, 0, 0, 0
            ).hour_pdf()
        if repricing_day >= 0 and d >= repricing_day:
            round_share = 0.96

        count = max(int(rng.poisson(max(count, 0.5))), 0)
        if count == 0:
            continue

        # Payment instruments. repeat_velocity = txns per unique instrument per day.
        uniq = max(int(round(count / max(prof.repeat_velocity, 1e-6))), 1)
        # A pool that grows over time; repeat payers are drawn from the existing pool.
        instrument_pool_size = max(instrument_pool_size, uniq * 3)
        new_ids = np.arange(instrument_high_water, instrument_high_water + uniq)
        instrument_high_water += uniq
        # Reuse a share of yesterday's instruments so velocity is a real property
        # of the stream rather than a parameter written into a column.
        reuse_lo = max(0, instrument_high_water - instrument_pool_size)
        pool = np.arange(reuse_lo, instrument_high_water)
        chosen_uniq = rng.choice(pool, size=min(uniq, len(pool)), replace=False)
        inst = rng.choice(chosen_uniq, size=count, replace=True)

        hours = rng.choice(24, size=count, p=hour_pdf)
        is_round = rng.random(count) < round_share
        amounts = np.where(
            is_round,
            rng.choice(ROUND_TICKETS, size=count),
            np.round(rng.lognormal(prof.ticket_mu, prof.ticket_sigma, size=count), 2),
        )
        if repricing_day >= 0 and d >= repricing_day:
            amounts = np.where(is_round, rng.choice([499.0, 999.0]), amounts)

        refunds = (rng.random(count) < prof.refund_rate).astype(int)
        cbs = (rng.random(count) < prof.chargeback_rate).astype(int)

        rows_day.append(d)
        rows_hour.append(hours)          # type: ignore[arg-type]
        rows_amt.append(amounts)         # type: ignore[arg-type]
        rows_inst.append(inst)           # type: ignore[arg-type]
        rows_refund.append(refunds)      # type: ignore[arg-type]
        rows_cb.append(cbs)              # type: ignore[arg-type]

    if not rows_day:
        return pd.DataFrame(
            columns=["day", "hour", "amount", "instrument", "refund", "chargeback"]
        )

    day_col = np.concatenate([np.full(len(h), d) for d, h in zip(rows_day, rows_hour)])
    return pd.DataFrame(
        {
            "day": day_col,
            "hour": np.concatenate(rows_hour),
            "amount": np.concatenate(rows_amt),
            "instrument": np.concatenate(rows_inst),
            "refund": np.concatenate(rows_refund),
            "chargeback": np.concatenate(rows_cb),
        }
    )


def scripts_for(m: Merchant, surface: str, rng: np.random.Generator | None = None) -> list[str]:
    """Third-party scripts present on a rendered surface.

    A small share of legitimate merchants carry a vertical-specific script for
    innocent reasons: a partner widget, an abandoned integration, a template
    that shipped with it. Without them the script signal would have perfect
    precision, and a corroborating channel that is never wrong is not a channel
    that can be honestly evaluated.
    """
    scripts = [m.script_bundle, "razorpay-checkout@1.9"]
    if m.label == 0 and rng is not None and rng.random() < 0.018:
        stray = sorted(VERTICAL_SCRIPTS.keys())[int(rng.integers(0, len(VERTICAL_SCRIPTS)))]
        scripts += VERTICAL_SCRIPTS[stray][:1]
    if m.label == 1 and surface == m.tell_surface:
        scripts += VERTICAL_SCRIPTS.get(m.true_vertical, [])
    elif m.label == 1 and surface == "checkout":
        # The illicit business always transacts on checkout, whatever the
        # homepage shows.
        scripts += VERTICAL_SCRIPTS.get(m.true_vertical, [])
    return sorted(set(scripts))


def main(out_dir: Path | None = None) -> None:
    out = out_dir or ARTIFACTS
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SETTINGS.seed)

    print(f"Building population of {SETTINGS.n_merchants} merchants...")
    population = build_population(rng, SETTINGS.n_merchants)
    mdf = pd.DataFrame([asdict(m) for m in population])
    mdf.to_csv(out / "merchants.csv", index=False)

    from sentinel.telemetry.features import extract_features

    print(f"Simulating {SETTINGS.days}-day payment streams and extracting features...")
    feature_rows = []
    demo_streams = {}
    demo_ids = _pick_demo_merchants(mdf)
    for i, m in enumerate(population):
        if i and i % 400 == 0:
            print(f"  {i}/{len(population)}")
        stream = simulate_stream(m, rng, SETTINGS.days)
        feats = extract_features(stream, m.declared_category, SETTINGS.days)
        feats["merchant_id"] = m.merchant_id
        feature_rows.append(feats)
        if m.merchant_id in demo_ids:
            demo_streams[m.merchant_id] = stream

    fdf = pd.DataFrame(feature_rows)
    fdf.to_csv(out / "features_raw.csv", index=False)

    if demo_streams:
        demo = pd.concat(
            [df.assign(merchant_id=mid) for mid, df in demo_streams.items()],
            ignore_index=True,
        )
        demo.to_csv(out / "transactions_demo.csv.gz", index=False, compression="gzip")

    scripts = {
        m.merchant_id: {s: scripts_for(m, s, rng) for s in ("homepage", "checkout")}
        for m in population
    }
    (out / "scripts.json").write_text(json.dumps(scripts), encoding="utf-8")

    summary = {
        "n_merchants": len(population),
        "n_diverged": int(mdf.label.sum()),
        "prevalence": round(float(mdf.label.mean()), 4),
        "by_archetype": mdf[mdf.label == 1].archetype.value_counts().to_dict(),
        "hard_negatives": mdf[mdf.label == 0].hard_negative.value_counts().to_dict(),
        "telemetry_blind_cases": int(
            ((mdf.label == 1) & (mdf.telemetry_visible == 0)).sum()
        ),
        "demo_merchants": sorted(demo_ids),
    }
    (out / "population_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _pick_demo_merchants(mdf: pd.DataFrame) -> set[str]:
    """One showcase merchant per archetype plus the hard negatives that matter."""
    picks: set[str] = set()
    for arch in ["pivot_prohibited", "rented_account", "funnel_ring", "licensing_violation"]:
        sub = mdf[mdf.archetype == arch]
        if len(sub):
            picks.update(sub.merchant_id.head(2).tolist())
    for cat in ["skill_gaming", "subscription_box"]:
        sub = mdf[(mdf.label == 0) & (mdf.declared_category == cat)]
        if len(sub):
            picks.add(sub.merchant_id.iloc[0])
    for hn in ["viral_spike", "price_point_change", "shared_platform"]:
        sub = mdf[mdf.hard_negative == hn]
        if len(sub):
            picks.add(sub.merchant_id.iloc[0])
    return picks


if __name__ == "__main__":
    main()
