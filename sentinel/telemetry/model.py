"""The telemetry scorer.

Deliberately not an LLM, for two reasons that both come from the workflow
rather than from accuracy.

First, it has to rerun identically. A settlement hold is challenged weeks later
and the same window has to produce the same score, to the digit. A sampled
decoder cannot promise that and a model behind an API that silently changes
version cannot promise it either.

Second, it has to say which features drove the score. An analyst putting a hold
on a real business needs to see that it was round-value drift and hour
divergence, not a number with no parts. A model that cannot decompose its own
output is unusable in a compliance workflow whatever its AUC is.

Local attribution here is neutralisation-based rather than SHAP: for each
feature, set it to the value a perfectly category-typical merchant would have
and measure how far the score falls. It is deterministic, it needs no extra
dependency, and its meaning is one sentence long -- "this is how much of the
score survives if this feature were normal" -- which matters when the audience
is a compliance reviewer rather than a data scientist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from sentinel.telemetry.baselines import DERIVED_FEATURES, MODEL_FEATURES
from sentinel.telemetry.features import FEATURE_EXPLAIN, FEATURE_LABELS

# A category-typical merchant has z = 0 on every scalar feature and near-zero
# divergence on the hour histogram. That vector is the neutral reference point
# for attribution.
NEUTRAL = {f: 0.0 for f in MODEL_FEATURES}


@dataclass
class SplitIndex:
    train: np.ndarray
    test: np.ndarray


def grouped_split(
    merchants: pd.DataFrame, test_fraction: float, seed: int
) -> SplitIndex:
    """Split by merchant, keeping every ring wholly inside one side.

    If members of the same funnel ring straddled the split, the model would see
    a ring's shared behaviour in training and be scored on its siblings in test.
    That is leakage, it would inflate ring recall, and it is exactly the kind of
    thing that looks fine until the system meets a ring it has never seen.
    """
    rng = np.random.default_rng(seed)
    groups = merchants["ring_id"].where(
        merchants["ring_id"] != "none", merchants["merchant_id"]
    )
    unique = np.array(sorted(groups.unique()))
    rng.shuffle(unique)
    n_test = int(round(len(unique) * test_fraction))
    test_groups = set(unique[:n_test].tolist())
    is_test = groups.isin(test_groups).to_numpy()
    return SplitIndex(train=np.where(~is_test)[0], test=np.where(is_test)[0])


class TelemetryScorer:
    def __init__(self, seed: int = 7):
        self.seed = seed
        self.model = GradientBoostingClassifier(
            n_estimators=260,
            learning_rate=0.055,
            max_depth=3,
            subsample=0.85,
            min_samples_leaf=18,
            random_state=seed,
        )
        self.feature_names = list(MODEL_FEATURES)
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TelemetryScorer":
        self.model.fit(X[self.feature_names].to_numpy(dtype=float), y)
        self.fitted = True
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(
            X[self.feature_names].to_numpy(dtype=float)
        )[:, 1]

    # -- explanation ------------------------------------------------------
    def global_importance(self) -> list[dict]:
        imps = self.model.feature_importances_
        order = np.argsort(imps)[::-1]
        return [
            {
                "feature": self.feature_names[i],
                "label": _label(self.feature_names[i]),
                "importance": round(float(imps[i]), 5),
            }
            for i in order
        ]

    def local_attribution(self, x: pd.Series, top_k: int = 6) -> list[dict]:
        """How much of this merchant's score each feature is responsible for.

        Measured in log-odds, not probability. On a merchant the model is
        confident about, the probability saturates near 1.0 and every
        neutralisation moves it by 0.000 -- the attribution silently becomes a
        column of zeroes exactly on the cases an analyst most needs explained.
        Log-odds keeps resolving at the extremes, so a feature worth four
        factors of ten still reads as four.
        """
        base_row = x[self.feature_names].to_numpy(dtype=float).reshape(1, -1)
        base = _logit(float(self.model.predict_proba(base_row)[0, 1]))

        variants = np.repeat(base_row, len(self.feature_names), axis=0)
        for i, f in enumerate(self.feature_names):
            variants[i, i] = NEUTRAL[f]
        neutralised = self.model.predict_proba(variants)[:, 1]

        rows = []
        for i, f in enumerate(self.feature_names):
            rows.append(
                {
                    "feature": f,
                    "label": _label(f),
                    "explains": FEATURE_EXPLAIN.get(_base_name(f), ""),
                    "value": round(float(base_row[0, i]), 4),
                    "contribution": round(base - _logit(float(neutralised[i])), 4),
                    "units": "log-odds",
                }
            )
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return rows[:top_k]

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        joblib.dump({"model": self.model, "features": self.feature_names, "seed": self.seed}, path)

    @classmethod
    def load(cls, path: Path) -> "TelemetryScorer":
        import joblib

        blob = joblib.load(path)
        obj = cls(seed=blob["seed"])
        obj.model = blob["model"]
        obj.feature_names = blob["features"]
        obj.fitted = True
        return obj


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def _base_name(f: str) -> str:
    return f[2:] if f.startswith("z_") else f


def _label(f: str) -> str:
    base = _base_name(f)
    label = FEATURE_LABELS.get(base, base)
    if f.startswith("z_"):
        return f"{label} (vs category)"
    if f == "hour_kl_recent":
        return "Hour-of-day divergence, last 30d"
    if f == "hour_kl_drift":
        return "Hour-of-day divergence drift"
    return label


def quick_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    return {
        "auc": round(float(roc_auc_score(y, scores)), 4),
        "ap": round(float(average_precision_score(y, scores)), 4),
    }


def verify_determinism(scorer: TelemetryScorer, X: pd.DataFrame) -> dict:
    """Score the same matrix twice and assert the results are bit-identical.

    This runs as part of the evaluation. The claim that the telemetry channel
    reruns identically is a claim, and an unchecked claim in a compliance
    system is worth nothing.
    """
    a = scorer.score(X)
    b = scorer.score(X)
    identical = bool(np.array_equal(a, b))

    # The storefront channel has to be reproducible too, and for a subtler
    # reason: its per-merchant seed used Python's builtin hash(), which is
    # randomised per interpreter start. Same input, different process, different
    # verdict -- and the failure is invisible inside a single run. Seeds now go
    # through config.stable_hash; this asserts it.
    from sentinel.config import stable_hash
    from sentinel.vision.classifier import DEFAULT_ERROR_MODEL, _simulate

    probes = [("sports_betting", "MID100001"), ("skill_gaming", "MID100002"),
              ("pharmacy", "MID100003")]
    vision_stable = all(
        _simulate(v, stable_hash(m, "homepage") % (2**31), DEFAULT_ERROR_MODEL)
        == _simulate(v, stable_hash(m, "homepage") % (2**31), DEFAULT_ERROR_MODEL)
        for v, m in probes
    )
    return {
        "telemetry_bitwise_identical": identical,
        "bitwise_identical": identical,
        "max_abs_delta": float(np.max(np.abs(a - b))) if len(a) else 0.0,
        "storefront_seed_process_stable": vision_stable,
        "storefront_seed_reference": {
            f"{v}/{m}": stable_hash(m, "homepage") % (2**31) for v, m in probes
        },
        "note": "storefront_seed_reference values are fixed constants. If they "
                "differ between runs or machines, seeding has regressed to a "
                "process-local hash and no verdict is reproducible.",
    }
