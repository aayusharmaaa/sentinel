"""Per-category baselines.

This is the single most important choice in the telemetry channel, so it gets
its own module and its own argument.

The question is never "is this merchant unusual". It is "is this merchant
unusual for what it claims to be". A skill-gaming platform at 68% round-value
share is ordinary. A home furnishing store at 71% is not. An absolute threshold
flags both, and the one it should not have flagged is a real business whose
payroll now depends on how fast a queue moves.

So every scalar feature is expressed as a robust z-score against the baseline
for the merchant's *declared* category, and the hour histogram is expressed as
a KL divergence from that category's baseline histogram.

Two properties this module has to hold:

  * Baselines are fitted on the training split only, and only on merchants
    labelled legitimate. Fitting on the full population would let the abusers
    pull their own category's baseline towards themselves, which inflates
    every metric downstream and would be invisible in the results.
  * Robust statistics, not mean and standard deviation. A category containing
    one merchant doing a hundred times the volume of the rest should not have
    its baseline dragged by that merchant.

`build_matrix(..., mode="absolute")` exists so the evaluation can run the
counterfactual: the same model, the same data, thresholds absolute rather than
category-relative. The difference in false-positive cost between those two runs
is the argument for this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.telemetry.features import HOUR_COLS, HOUR_RECENT_COLS, SCALAR_FEATURES

EPS = 1e-9

# Features that enter the model as category-relative deviations, on top of the
# scalar z-scores.
DERIVED_FEATURES = ["hour_kl", "hour_kl_recent", "hour_kl_drift"]

MODEL_FEATURES = [f"z_{f}" for f in SCALAR_FEATURES] + DERIVED_FEATURES


@dataclass
class CategoryBaseline:
    category: str
    n: int
    centre: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    hour_pdf: list[float] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "category": self.category,
            "n": self.n,
            "centre": self.centre,
            "scale": self.scale,
            "hour_pdf": self.hour_pdf,
        }


class BaselineSet:
    """Baselines for every declared category."""

    def __init__(self, baselines: dict[str, CategoryBaseline]):
        self.baselines = baselines
        self._global = self._pool(baselines)

    @staticmethod
    def _pool(baselines: dict[str, CategoryBaseline]) -> CategoryBaseline:
        """Fallback for a category with too few legitimate merchants to fit.

        Falling back to a pooled baseline is the honest failure mode: it makes
        the merchant look more normal than a bespoke baseline would, so a thin
        category produces under-detection rather than a wave of false holds.
        """
        if not baselines:
            return CategoryBaseline("__pooled__", 0)
        centre, scale = {}, {}
        for f in SCALAR_FEATURES:
            centre[f] = float(np.median([b.centre.get(f, 0.0) for b in baselines.values()]))
            scale[f] = float(np.median([b.scale.get(f, 1.0) for b in baselines.values()]))
        pdfs = np.array([b.hour_pdf for b in baselines.values() if b.hour_pdf])
        pooled_pdf = pdfs.mean(axis=0) if len(pdfs) else np.full(24, 1 / 24)
        return CategoryBaseline("__pooled__", 0, centre, scale, pooled_pdf.tolist())

    def get(self, category: str) -> CategoryBaseline:
        b = self.baselines.get(category)
        if b is None or b.n < 25:
            return self._global
        return b

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({k: v.to_json() for k, v in self.baselines.items()}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BaselineSet":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls({k: CategoryBaseline(**v) for k, v in raw.items()})


def fit_baselines(features: pd.DataFrame, legit_mask: pd.Series) -> BaselineSet:
    """Fit one baseline per declared category from legitimate training merchants."""
    train = features[legit_mask]
    baselines: dict[str, CategoryBaseline] = {}
    for cat, grp in train.groupby("declared_category"):
        centre, scale = {}, {}
        for f in SCALAR_FEATURES:
            vals = grp[f].to_numpy(dtype=float)
            med = float(np.median(vals))
            # Normalised IQR: a robust standard-deviation equivalent.
            iqr = float(np.subtract(*np.percentile(vals, [75, 25])))
            centre[f] = med
            scale[f] = max(iqr / 1.349, 1e-3)
        hour = grp[HOUR_COLS].to_numpy(dtype=float)
        pdf = hour.mean(axis=0)
        pdf = pdf / max(pdf.sum(), EPS)
        baselines[str(cat)] = CategoryBaseline(
            category=str(cat), n=int(len(grp)), centre=centre, scale=scale,
            hour_pdf=pdf.tolist(),
        )
    return BaselineSet(baselines)


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q), row-wise, with both distributions smoothed."""
    p = p + EPS
    q = q + EPS
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return (p * np.log(p / q)).sum(axis=1)


def build_matrix(
    features: pd.DataFrame,
    baselines: BaselineSet,
    mode: str = "relative",
) -> pd.DataFrame:
    """Turn raw features into the model matrix.

    mode="relative": every feature is a deviation from the merchant's declared
    category. This is the design.

    mode="absolute": raw feature values, no category context. Present only so
    the evaluation can measure what category-relative scoring is worth.
    """
    if mode == "absolute":
        out = features[SCALAR_FEATURES].copy()
        flat = np.full(24, 1 / 24)
        hour = features[HOUR_COLS].to_numpy(dtype=float)
        hour_r = features[HOUR_RECENT_COLS].to_numpy(dtype=float)
        out["hour_kl"] = _kl(hour, np.tile(flat, (len(features), 1)))
        out["hour_kl_recent"] = _kl(hour_r, np.tile(flat, (len(features), 1)))
        out["hour_kl_drift"] = out["hour_kl_recent"] - out["hour_kl"]
        out.columns = MODEL_FEATURES
        return out

    cats = features["declared_category"].to_numpy()
    centres = np.array(
        [[baselines.get(c).centre.get(f, 0.0) for f in SCALAR_FEATURES] for c in cats]
    )
    scales = np.array(
        [[baselines.get(c).scale.get(f, 1.0) for f in SCALAR_FEATURES] for c in cats]
    )
    raw = features[SCALAR_FEATURES].to_numpy(dtype=float)
    z = (raw - centres) / np.maximum(scales, 1e-6)
    # Clip: a merchant 40 standard deviations out and one 400 out are the same
    # decision, and leaving the tail unbounded lets one feature dominate a split.
    z = np.clip(z, -12, 12)

    baseline_pdfs = np.array([baselines.get(c).hour_pdf or [1 / 24] * 24 for c in cats])
    hour = features[HOUR_COLS].to_numpy(dtype=float)
    hour_r = features[HOUR_RECENT_COLS].to_numpy(dtype=float)

    out = pd.DataFrame(z, columns=[f"z_{f}" for f in SCALAR_FEATURES], index=features.index)
    out["hour_kl"] = _kl(hour, baseline_pdfs)
    out["hour_kl_recent"] = _kl(hour_r, baseline_pdfs)
    out["hour_kl_drift"] = out["hour_kl_recent"] - out["hour_kl"]
    return out[MODEL_FEATURES]


def deviation_report(
    feature_row: pd.Series, baselines: BaselineSet
) -> list[dict]:
    """Per-feature comparison against category baseline, for the evidence card.

    This is what an analyst actually reads: the merchant's value, what is normal
    for the category it declared, and how far out it sits. A score without this
    is not actionable, and a compliance function cannot sign off on it.
    """
    cat = str(feature_row["declared_category"])
    b = baselines.get(cat)
    rows = []
    for f in SCALAR_FEATURES:
        val = float(feature_row[f])
        med = float(b.centre.get(f, 0.0))
        sc = float(b.scale.get(f, 1.0))
        z = float(np.clip((val - med) / max(sc, 1e-6), -12, 12))
        rows.append(
            {
                "feature": f,
                "value": round(val, 5),
                "category_baseline": round(med, 5),
                "category_spread": round(sc, 5),
                "z": round(z, 3),
                "direction": "above" if z > 0 else "below",
            }
        )
    hour = feature_row[HOUR_COLS].to_numpy(dtype=float).reshape(1, -1)
    pdf = np.array(b.hour_pdf or [1 / 24] * 24).reshape(1, -1)
    rows.append(
        {
            "feature": "hour_kl",
            "value": round(float(_kl(hour, pdf)[0]), 5),
            "category_baseline": 0.0,
            "category_spread": 1.0,
            "z": round(float(_kl(hour, pdf)[0]), 3),
            "direction": "above",
        }
    )
    return rows
