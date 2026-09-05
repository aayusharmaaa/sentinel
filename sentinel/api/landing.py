"""Data for the landing page.

The interactive panels on the landing page are driven by the real held-out
population, not by illustrative numbers. That is the point of building it this
way: when a visitor drags the threshold slider and watches legitimate merchants
start getting frozen, those are the actual merchants from the actual evaluation,
scored by the actual model.

Everything here is derived from `artifacts/` after a pipeline and evaluation
run. Nothing is computed fresh, so the page cannot show a number the evaluation
did not produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.config import ARTIFACTS, SETTINGS


def clean(obj):
    """Replace NaN/Inf with null so the payload is valid JSON.

    NaN is meaningful in this data -- precision is genuinely undefined when a
    policy holds nobody -- but it is not JSON. `json.dumps` emits a bare `NaN`
    that most parsers accept and the spec does not, so this only surfaced when
    Starlette serialised it with allow_nan=False.
    """
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return clean(float(obj))
    return obj


def _round(xs, n=4):
    return [round(float(x), n) for x in xs]


def build_landing_payload(artifacts: Path = ARTIFACTS) -> dict:
    decisions = pd.read_csv(artifacts / "decisions.csv")
    ev = json.loads((artifacts / "evaluation.json").read_text(encoding="utf-8"))
    test = decisions[decisions.in_test_split].reset_index(drop=True)

    diss = test["dissenting_channels"].fillna("")
    storefront_dissent = diss.str.contains("storefront").astype(int)
    # All four gating channels, so the page can reproduce a counterfactual
    # policy exactly rather than approximating it from two of them.
    telemetry_dissent = diss.str.contains("telemetry").astype(int)
    scripts_dissent = diss.str.contains("scripts").astype(int)
    infra_dissent = diss.str.contains("infrastructure").astype(int)
    # A merchant that never cleared triage has no storefront verdict at all.
    # The gated policy cannot freeze it, and pretending otherwise would make the
    # interactive lie in the flattering direction.
    looked_at = test["triage_reasons"].fillna("").ne("").astype(int)

    cats = sorted(test.declared_category.unique().tolist())
    arches = sorted(test.archetype.unique().tolist())

    population = {
        "score": _round(test.telemetry_score, 5),
        "label": [int(x) for x in test.label],
        "tier": [int(x) for x in test.tier],
        "cat": [cats.index(c) for c in test.declared_category],
        "arch": [arches.index(a) for a in test.archetype],
        "gmv": [int(round(float(g))) for g in test.daily_gmv_recent],
        "vis": [int(x) for x in storefront_dissent],
        "tel": [int(x) for x in telemetry_dissent],
        "scr": [int(x) for x in scripts_dissent],
        "inf": [int(x) for x in infra_dissent],
        "net": _round(test.get("network_score", pd.Series([0.0] * len(test))), 4),
        "con": [int(x) for x in test.get("contradictions", pd.Series([0] * len(test)))],
        "seen": [int(x) for x in looked_at],
        "mid": test.merchant_id.tolist(),
    }

    h = ev["headline"]
    econ = SETTINGS.economics

    # The showcase case, walked step by step.
    case = None
    cards = artifacts / "cards"
    if cards.exists():
        preferred = cards / "MID100057.json"
        pool = [preferred] if preferred.exists() else sorted(cards.glob("*.json"))
        for p in pool:
            c = json.loads(p.read_text(encoding="utf-8"))
            if len(c.get("vision") or {}) >= 2:
                case = {
                    "merchant_id": c["merchant_id"],
                    "declared": c["declared_category"],
                    "onboarded": c["onboarded_days_ago"],
                    "tier": c["decision"]["tier"],
                    "tier_name": c["decision"]["tier_name"],
                    "second": c["decision"]["second_channel_source"],
                    "justification": c["decision"]["justification"],
                    "channels": c["decision"]["channels"],
                    "steps": c["investigation"]["steps"],
                    "stop": c["investigation"]["stop_reason"],
                    "budget": c["investigation"]["step_budget"],
                    "cost": c["investigation"]["total_cost_inr"],
                    "vision": c["vision"],
                    "deviations": c["telemetry"]["deviations"][:6],
                    "narrative": c["narrative"]["text"],
                    "assessment": c.get("assessment"),
                    "network": c.get("network"),
                    "voice": c.get("voice"),
                }
                break

    merchants = pd.read_csv(artifacts / "merchants.csv")
    feats = pd.read_csv(artifacts / "features_raw.csv").drop(columns=["declared_category"])
    full = merchants.merge(feats, on="merchant_id")
    ccols = ["round_share_all", "velocity_all", "night_share", "chargeback_rate"]
    collision = {
        "skill_gaming": _round(full[(full.label == 0) & (full.declared_category == "skill_gaming")][ccols].median()),
        "betting": _round(full[full.archetype == "pivot_prohibited"][ccols].median()),
        "furnishing": _round(full[(full.label == 0) & (full.declared_category == "home_furnishing")][ccols].median()),
        "labels": ["Round-value share", "Repeat-payer velocity",
                   "Night-hours share", "Chargeback rate"],
    }

    payload = {
        "categories": cats,
        "archetypes": arches,
        "population": population,
        "headline": {
            "test_n": h["test_set"]["merchants"],
            "test_pos": h["test_set"]["diverged"],
            "prevalence": h["test_set"]["prevalence"],
            "hold": h["single_cycle"]["settlement_hold_tier3"],
            "hold_rot": h["one_rotation_period"]["settlement_hold_tier3"],
            "action": h["single_cycle"]["any_analyst_action_tier2plus"],
            "action_rot": h["one_rotation_period"]["any_analyst_action_tier2plus"],
        },
        "ablation": ev["policy_ablations"],
        "modality": ev.get("modality_ablation"),
        "contradiction_quality": ev.get("contradiction_quality"),
        "voice_channel": ev.get("voice_channel"),
        "archetype_recall": ev["recall_by_archetype_one_rotation"],
        "sensitivity": ev["vision_sensitivity"],
        "rule": ev["baseline_ablation"]["absolute_threshold_rule"],
        "throughput": ev["throughput"],
        # The whole book, not the held-out split: the funnel on the landing
        # page describes one sweep of every merchant, and mixing the two
        # populations in one picture would overstate what a sweep costs.
        "book": {
            "tiers": {str(int(k)): int(v) for k, v in
                      decisions.tier.value_counts().sort_index().items()},
            "n": int(len(decisions)),
        },
        "cache": ev["cache_economics"],
        "injection": ev["injection_robustness"]["detector"],
        "determinism": ev["determinism"],
        "collision": collision,
        "case": case,
        "economics": {
            "analyst_review_cost_inr": econ.analyst_review_cost_inr,
            "hold_days": econ.hold_days,
            "hold_harm_rate": econ.hold_harm_rate,
            "hold_fixed_cost_inr": econ.hold_fixed_cost_inr,
            "churn_probability_after_hold": econ.churn_probability_after_hold,
            "take_rate": econ.take_rate,
            "merchant_remaining_lifetime_days": econ.merchant_remaining_lifetime_days,
            "network_fine_inr": econ.network_fine_inr,
            "chargeback_liability_rate": econ.chargeback_liability_rate,
            "enforcement_freeze_rate": econ.enforcement_freeze_rate,
        },
        "thresholds": {
            "telemetry_dissent": SETTINGS.thresholds.telemetry_dissent,
            "triage_gate": SETTINGS.thresholds.triage_gate,
        },
        "days": SETTINGS.days,
    }
    return clean(payload)
