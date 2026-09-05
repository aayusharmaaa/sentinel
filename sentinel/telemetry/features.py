"""Deterministic feature extraction over the payment stream.

No model runs here and none should. This layer is the cheap half of the
two-speed design: it executes on every merchant, every window, using data the
aggregator already holds. No external call, no page fetch, nothing that costs
money per merchant. It is what makes the expensive channel affordable.

It is also the layer that has to rerun identically. A settlement hold gets
challenged; when it does, the features behind it must reproduce bit for bit
from the same transaction window months later.

Five feature families, as designed:
  round-value share, repeat-payer velocity, hour concentration,
  volume trend (with peak shape retained), refund and chargeback mix.

Each is computed three ways: over the full window, over the recent window, and
over the prior window. The differences are the drift features. This matters
more than it looks. A merchant that has always had 68% round-value share is a
different object from one that moved to 68% last month, and a system that only
scores a static profile cannot tell them apart -- which is the whole failure
mode of scoring merchants at onboarding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RECENT_WINDOW = 30
PRIOR_WINDOW = 30

HOUR_COLS = [f"h{i:02d}" for i in range(24)]
HOUR_RECENT_COLS = [f"hr{i:02d}" for i in range(24)]

# Feature names the model consumes, in a fixed order so that importances line
# up with labels on the evidence card without a lookup that can drift.
SCALAR_FEATURES = [
    "round_share_all",
    "round_share_recent",
    "round_share_drift",
    "velocity_all",
    "velocity_recent",
    "velocity_drift",
    "slope_log_volume",
    "peak_ratio",
    "post_peak_persistence",
    "refund_rate",
    "chargeback_rate",
    "chargeback_drift",
    "ticket_entropy",
    "night_share",
]

# Human-readable labels for the evidence card. An analyst holding a merchant's
# settlements should never have to read a feature name like "hour_kl".
FEATURE_LABELS = {
    "round_share_all": "Round-value share (full window)",
    "round_share_recent": "Round-value share (last 30d)",
    "round_share_drift": "Round-value share drift",
    "velocity_all": "Repeat-payer velocity (full window)",
    "velocity_recent": "Repeat-payer velocity (last 30d)",
    "velocity_drift": "Repeat-payer velocity drift",
    "slope_log_volume": "Volume trend slope",
    "peak_ratio": "Peak-to-median volume ratio",
    "post_peak_persistence": "Post-peak persistence",
    "refund_rate": "Refund rate",
    "chargeback_rate": "Chargeback rate",
    "chargeback_drift": "Chargeback rate drift",
    "ticket_entropy": "Ticket-size entropy",
    "night_share": "Night-hours share (00:00-05:00)",
    "hour_kl": "Hour-of-day divergence from category",
}

FEATURE_EXPLAIN = {
    "round_share_all": "Proportion of payments at an exact multiple of Rs 100.",
    "round_share_recent": "The same measure over the last 30 days only.",
    "round_share_drift": "Recent minus prior window. Separates a merchant that "
                         "has always priced in round numbers from one that "
                         "started last month.",
    "velocity_all": "Transactions per unique payment instrument per day.",
    "velocity_recent": "The same measure over the last 30 days only.",
    "velocity_drift": "Change in repeat-payer velocity between windows.",
    "slope_log_volume": "Fitted daily growth rate of transaction count.",
    "peak_ratio": "Largest single day relative to the median day.",
    "post_peak_persistence": "Volume after the peak divided by volume before it. "
                             "Near 1.0 means a spike that decayed -- a sale. "
                             "Well above 1.0 means a step change that held.",
    "refund_rate": "Share of transactions refunded.",
    "chargeback_rate": "Share of transactions charged back.",
    "chargeback_drift": "Change in chargeback rate between windows.",
    "ticket_entropy": "Shannon entropy of the ticket-size distribution. Low "
                      "entropy means a handful of fixed amounts.",
    "night_share": "Share of payments between midnight and 05:00.",
    "hour_kl": "KL divergence of this merchant's hour-of-day histogram from the "
               "baseline for its declared category.",
}


def _round_share(amounts: np.ndarray) -> float:
    if amounts.size == 0:
        return 0.0
    cents = np.rint(amounts * 100).astype(np.int64)
    return float(np.mean(cents % 10_000 == 0))


def _velocity(df: pd.DataFrame) -> float:
    """Distinct transactions per unique payment instrument per active day."""
    if df.empty:
        return 0.0
    per_day = df.groupby("day").agg(
        txns=("amount", "size"), uniq=("instrument", "nunique")
    )
    per_day = per_day[per_day.uniq > 0]
    if per_day.empty:
        return 0.0
    return float((per_day.txns / per_day.uniq).mean())


def _hour_hist(df: pd.DataFrame) -> np.ndarray:
    hist = np.zeros(24)
    if df.empty:
        return np.full(24, 1 / 24)
    counts = df.hour.value_counts()
    for h, c in counts.items():
        hist[int(h)] = float(c)
    total = hist.sum()
    return hist / total if total > 0 else np.full(24, 1 / 24)


def _ticket_entropy(amounts: np.ndarray) -> float:
    if amounts.size == 0:
        return 0.0
    # Log-spaced bins: ticket sizes span orders of magnitude.
    bins = np.geomspace(10, 200_000, 24)
    counts, _ = np.histogram(np.clip(amounts, 10, 200_000), bins=bins)
    p = counts / counts.sum() if counts.sum() else np.ones(len(counts)) / len(counts)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _volume_shape(daily: pd.Series, days: int) -> tuple[float, float, float]:
    """Fitted growth slope, peak ratio, and post-peak persistence.

    Persistence is the feature that keeps a legitimate viral sale out of the
    review queue. Both a sale and a pivot produce a steep slope; only the pivot
    holds its new level once the peak has passed.
    """
    counts = np.zeros(days)
    for d, c in daily.items():
        if 0 <= int(d) < days:
            counts[int(d)] = float(c)
    active = counts > 0
    if active.sum() < 5:
        return 0.0, 0.0, 1.0

    x = np.arange(days)[active]
    y = np.log1p(counts[active])
    slope = float(np.polyfit(x, y, 1)[0])

    median = float(np.median(counts[active]))
    peak_ratio = float(counts.max() / median) if median > 0 else 0.0

    p = int(counts.argmax())
    before = counts[max(0, p - 21): max(0, p - 6)]
    after = counts[min(days - 1, p + 7): min(days, p + 22)]
    if before.size == 0 or after.size == 0 or before.mean() <= 0:
        persistence = 1.0
    else:
        persistence = float(after.mean() / before.mean())
    return slope, peak_ratio, float(np.clip(persistence, 0.0, 25.0))


def extract_features(stream: pd.DataFrame, declared_category: str, days: int) -> dict:
    """Compute the full feature record for one merchant over one window."""
    recent = stream[stream.day >= days - RECENT_WINDOW]
    prior = stream[
        (stream.day >= days - RECENT_WINDOW - PRIOR_WINDOW)
        & (stream.day < days - RECENT_WINDOW)
    ]

    amounts_all = stream.amount.to_numpy() if not stream.empty else np.array([])
    rs_all = _round_share(amounts_all)
    rs_recent = _round_share(recent.amount.to_numpy() if not recent.empty else np.array([]))
    rs_prior = _round_share(prior.amount.to_numpy() if not prior.empty else np.array([]))

    v_all, v_recent, v_prior = _velocity(stream), _velocity(recent), _velocity(prior)

    hist_all = _hour_hist(stream)
    hist_recent = _hour_hist(recent)

    daily = stream.groupby("day").size() if not stream.empty else pd.Series(dtype=float)
    slope, peak_ratio, persistence = _volume_shape(daily, days)

    def rate(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if not df.empty else 0.0

    cb_all, cb_recent, cb_prior = (
        rate(stream, "chargeback"), rate(recent, "chargeback"), rate(prior, "chargeback")
    )

    gmv = float(stream.amount.sum()) if not stream.empty else 0.0
    gmv_recent = float(recent.amount.sum()) if not recent.empty else 0.0

    record = {
        "declared_category": declared_category,
        "n_txns": int(len(stream)),
        "gmv": gmv,
        "daily_gmv_recent": gmv_recent / max(RECENT_WINDOW, 1),
        "round_share_all": rs_all,
        "round_share_recent": rs_recent,
        "round_share_prior": rs_prior,
        "round_share_drift": rs_recent - rs_prior,
        "velocity_all": v_all,
        "velocity_recent": v_recent,
        "velocity_prior": v_prior,
        "velocity_drift": v_recent - v_prior,
        "slope_log_volume": slope,
        "peak_ratio": peak_ratio,
        "post_peak_persistence": persistence,
        "refund_rate": rate(stream, "refund"),
        "chargeback_rate": cb_all,
        "chargeback_drift": cb_recent - cb_prior,
        "ticket_entropy": _ticket_entropy(amounts_all),
        "night_share": float(hist_all[0:6].sum()),
    }
    record.update({c: float(hist_all[i]) for i, c in enumerate(HOUR_COLS)})
    record.update({c: float(hist_recent[i]) for i, c in enumerate(HOUR_RECENT_COLS)})
    return record
