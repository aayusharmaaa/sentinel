"""What earns a merchant the expensive channel.

You cannot screenshot and vision-classify millions of merchants daily. The API
cost alone rules it out. So the expensive layer runs on triggers, and this
module is the list of them.

Four triggers, and the reason there are four rather than one is worth stating,
because building only the first is an easy mistake that quietly makes a whole
class of abuse unreachable:

  telemetry_drift        The cheap channel dissents. Catches merchants whose
                         payment behaviour moved.

  volume_shape_change    The shape of the volume changed even though the level
                         did not drift enough to score. A step that held is a
                         different object from a spike that decayed, and the
                         difference is visible before the level does anything
                         a threshold would notice.

  high_volume_rotation   A slow rotation over the largest accounts, regardless
                         of any signal. This is the trigger that catches the
                         cases telemetry structurally cannot see -- a pharmacy
                         operating without registration has exactly the payment
                         shape of one with it, so it will never drift, never
                         change shape, and never generate an external signal.
                         Without a rotation it is invisible forever, and no
                         amount of work on the telemetry model changes that.
                         Rotation is also the only trigger that bounds worst-case
                         exposure: it puts a ceiling on how long a large account
                         can go unlooked-at.

  external_signal        A chargeback spike, or shared infrastructure with an
                         already-flagged merchant. Evidence that arrived from
                         outside the payment stream.

The rotation trigger is priced deliberately. It spends money on merchants that
nothing is wrong with, and that spend is the premium paid for covering the blind
spot. The evaluation reports recall and cost per trigger so the premium is a
number someone can argue with rather than a design assumption nobody checked.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TriageConfig:
    telemetry_gate: float = 0.35
    # Volume shape: a step that held, rather than a spike that decayed.
    persistence_step: float = 2.2
    peak_ratio_floor: float = 3.0
    # Rotation: fraction of the book in the "large account" tier, and how many
    # cycles it takes to cover all of them once.
    high_volume_quantile: float = 0.80
    rotation_cycles: int = 8
    rotation_cycle: int = 0
    # A second, slower rotation over the ENTIRE book, large or small. Without
    # it the coverage guarantee only applies to merchants big enough to be worth
    # rotating, and a small merchant with no telemetry tell -- an unregistered
    # pharmacy doing Rs 30,000 a day -- is never looked at by anything, ever.
    # At 90 cycles this is a quarterly sweep costing one ninetieth of the book
    # per day, which is the price of being able to say that no merchant goes
    # unexamined indefinitely.
    full_book_rotation_cycles: int = 90
    # Steady-state mode: treat the rotation as having covered everyone once,
    # which is what happens over one full rotation period. Used to report
    # detection within a rotation window alongside single-cycle numbers.
    full_coverage_horizon: bool = False
    # External: chargeback rate this many multiples above the category median.
    chargeback_multiple: float = 3.0


@dataclass
class TriageResult:
    selected: np.ndarray                 # boolean mask over merchants
    reasons: list[list[str]]             # per-merchant trigger names
    counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "selected": int(self.selected.sum()),
            "by_trigger": self.counts,
        }


def _rotation_slice(merchant_id: str, cycles: int, cycle: int) -> bool:
    """Deterministic assignment of a merchant to a rotation cycle.

    Hash-based rather than random so the same merchant lands in the same cycle
    on every rerun. A rotation that reshuffles each run has no coverage
    guarantee at all -- some accounts would be looked at twice and others never.
    """
    h = int(hashlib.sha256(merchant_id.encode()).hexdigest()[:8], 16)
    return h % cycles == cycle


def select(
    merchants: pd.DataFrame,
    features: pd.DataFrame,
    telemetry_scores: np.ndarray,
    flagged_ids: set[str] | None = None,
    config: TriageConfig | None = None,
    network_ids: set[str] | None = None,
) -> TriageResult:
    """Decide which merchants earn a storefront classification this cycle."""
    cfg = config or TriageConfig()
    n = len(merchants)
    reasons: list[list[str]] = [[] for _ in range(n)]
    flagged_ids = flagged_ids or set()

    # 1. telemetry drift
    drift = telemetry_scores >= cfg.telemetry_gate

    # 2. volume shape change: a step that held, not a spike that decayed
    persistence = features["post_peak_persistence"].to_numpy(dtype=float)
    peak = features["peak_ratio"].to_numpy(dtype=float)
    shape = (persistence >= cfg.persistence_step) & (peak >= cfg.peak_ratio_floor)

    # 3. slow rotation over the largest accounts
    gmv = features["daily_gmv_recent"].to_numpy(dtype=float)
    threshold = float(np.quantile(gmv, cfg.high_volume_quantile))
    big = gmv >= threshold
    if cfg.full_coverage_horizon:
        # Over one full rotation period every merchant is reached exactly once.
        rotation = np.ones(n, dtype=bool)
    else:
        in_cycle = np.array([
            _rotation_slice(str(mid), cfg.rotation_cycles, cfg.rotation_cycle)
            for mid in merchants.merchant_id
        ])
        whole_book = np.array([
            _rotation_slice(str(mid), cfg.full_book_rotation_cycles, cfg.rotation_cycle)
            for mid in merchants.merchant_id
        ])
        rotation = (big & in_cycle) | whole_book

    # 4. external signal: chargeback spike, or infrastructure shared with a
    #    merchant already flagged
    cb = features["chargeback_rate"].to_numpy(dtype=float)
    cat_median = (
        features.assign(_cb=cb)
        .groupby("declared_category")["_cb"].transform("median")
        .to_numpy(dtype=float)
    )
    cb_spike = cb >= np.maximum(cat_median * cfg.chargeback_multiple, 1e-6)

    # A merchant sitting in a tight, synchronised cluster earns a storefront
    # look on the strength of its company alone. This is the trigger that
    # reaches a funnel member: individually it is unremarkable by construction,
    # so nothing about the merchant itself will ever select it.
    in_network = np.zeros(n, dtype=bool)
    if network_ids:
        in_network = merchants.merchant_id.astype(str).isin(network_ids).to_numpy()

    linked = np.zeros(n, dtype=bool)
    if flagged_ids:
        flagged = merchants[merchants.merchant_id.isin(flagged_ids)]
        rare_payouts = set(flagged.payout_fingerprint.astype(str))
        rare_blocks = set(flagged.ip_block.astype(str))
        linked = (
            merchants.payout_fingerprint.astype(str).isin(rare_payouts)
            | merchants.ip_block.astype(str).isin(rare_blocks)
        ).to_numpy() & ~merchants.merchant_id.isin(flagged_ids).to_numpy()
    external = cb_spike | linked

    for name, mask in (
        ("telemetry_drift", drift),
        ("volume_shape_change", shape),
        ("high_volume_rotation", rotation),
        ("external_signal", external),
        ("network_cluster", in_network),
    ):
        for i in np.where(mask)[0]:
            reasons[int(i)].append(name)

    selected = drift | shape | rotation | external | in_network
    counts = {
        "telemetry_drift": int(drift.sum()),
        "volume_shape_change": int(shape.sum()),
        "high_volume_rotation": int(rotation.sum()),
        "external_signal": int(external.sum()),
        "network_cluster": int(in_network.sum()),
        "union": int(selected.sum()),
    }
    return TriageResult(selected=selected, reasons=reasons, counts=counts)
