"""End-to-end run.

The shape of this function is the argument for the whole system, so it is worth
reading in order rather than as a list of stages.

Every merchant gets the cheap channel. Feature extraction over the payment
stream costs nothing per merchant beyond compute on data the aggregator already
holds -- no external call, no page fetch, no model. That is what lets it run on
everyone, every window, which is the only way to catch a business that changed
after approval rather than at it.

A small fraction of those clear the triage gate and earn the expensive channel.
Vision classification of a rendered storefront is the only per-merchant
external spend in the system, and it is gated behind a signal that costs
nothing to compute.

A smaller fraction again get a full bounded investigation, because their
channels disagree and something has to decide what to look at next.

That inversion -- cheap signal gates expensive signal -- is what makes this
deployable at a few million merchants rather than a demo on fifty. The
throughput block written at the end of this run reports the actual funnel
widths and the actual rupee cost, because a claim about affordability that
isn't measured is just a claim.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.agent.loop import HeuristicPlanner, investigate
from sentinel.agent.tools import Toolbox
from sentinel.config import ARTIFACTS, SETTINGS
from sentinel.fusion.contradiction import assess, build_views, find_contradictions
from sentinel.fusion.gate import Tier, build_channels, decide, decision_cost
from sentinel.graph.network import build_graph, graph_summary, network_risk
from sentinel.voice.verify import verify as voice_verify
from sentinel.narrative.summarizer import build_summary_input, summarise
from sentinel.telemetry.baselines import build_matrix, fit_baselines
from sentinel.telemetry.model import TelemetryScorer, grouped_split, quick_metrics
from sentinel.triage import TriageConfig, select as triage_select
from sentinel.vision.cache import VerdictCache
from sentinel.vision.classifier import StorefrontClassifier, VisionErrorModel


@dataclass
class PipelineResult:
    merchants: pd.DataFrame
    decisions: pd.DataFrame
    cards: dict[str, dict]
    throughput: dict
    split: object
    scorer: TelemetryScorer
    baselines: object
    matrix: pd.DataFrame
    toolbox: Toolbox = field(repr=False, default=None)


def run(
    artifacts: Path = ARTIFACTS,
    error_model: VisionErrorModel | None = None,
    harden_against_injection: bool = True,
    build_cards: bool = True,
    verbose: bool = True,
    enable_graph: bool = True,
    enable_voice: bool = True,
    enable_vision: bool = True,
    triage_config: TriageConfig | None = None,
    persist_cache: bool = True,
) -> PipelineResult:
    t0 = time.time()
    merchants = pd.read_csv(artifacts / "merchants.csv")
    features = pd.read_csv(artifacts / "features_raw.csv")
    scripts = json.loads((artifacts / "scripts.json").read_text(encoding="utf-8"))

    # Align feature rows to merchant rows.
    features = merchants[["merchant_id"]].merge(features, on="merchant_id", how="left")
    assert len(features) == len(merchants)

    # --- split -----------------------------------------------------------
    split = grouped_split(merchants, SETTINGS.test_fraction, SETTINGS.seed)

    # --- baselines, fitted on training legitimate merchants only ---------
    legit_train = pd.Series(False, index=features.index)
    legit_train.iloc[split.train] = merchants.iloc[split.train].label.to_numpy() == 0
    baselines = fit_baselines(features, legit_train)
    baselines.save(artifacts / "baselines.json")

    matrix = build_matrix(features, baselines, mode="relative")

    # --- telemetry scorer -------------------------------------------------
    y = merchants.label.to_numpy()
    scorer = TelemetryScorer(seed=SETTINGS.seed)
    scorer.fit(matrix.iloc[split.train], y[split.train])
    scores = scorer.score(matrix)
    if verbose:
        m = quick_metrics(y[split.test], scores[split.test])
        print(f"  telemetry channel (held out): AUC {m['auc']}  AP {m['ap']}")

    # --- expensive channel, gated ----------------------------------------
    # persist_cache=False gives a cold cache, which is the conservative cost
    # to report: it prices a cycle as if nothing had ever been classified
    # before. Production carries the cache across cycles and pays less.
    cache = VerdictCache(path=artifacts / "vision_cache.json" if persist_cache else None)
    classifier = StorefrontClassifier(
        cache=cache,
        error_model=error_model,
        live=False if error_model is not None else None,
        harden_against_injection=harden_against_injection,
    )
    toolbox = Toolbox(
        merchants=merchants,
        features=features,
        scripts=scripts,
        baselines=baselines,
        classifier=classifier,
        vision_call_cost_inr=SETTINGS.economics.vision_call_cost_inr,
    )

    # --- relationship graph ----------------------------------------------
    # Built once over the whole book: a merchant's network risk is a property
    # of the population, not of the merchant, so it cannot be computed one row
    # at a time the way every other channel can.
    graph = build_graph(merchants, features, scores) if enable_graph else None
    if verbose and graph is not None:
        print(f"  graph: {graph_summary(graph)}")

    gate = SETTINGS.thresholds.triage_gate
    network_ids = set()
    if graph is not None:
        for cl in graph.clusters:
            # Only clusters that are both tightly linked and moving together.
            if cl.cohesion >= 12.0 and cl.synchrony >= 0.30:
                network_ids.update(cl.members)
    triage = triage_select(
        merchants, features, scores,
        config=triage_config or TriageConfig(telemetry_gate=gate),
        network_ids=network_ids,
    )
    triaged = np.where(triage.selected)[0]
    if verbose:
        print(f"  triage: {len(triaged)}/{len(merchants)} merchants earn the "
              f"expensive channel ({len(triaged)/len(merchants):.1%})")
        print(f"    by trigger: {triage.counts}")

    test_ids = set(split.test.tolist())
    rows = []
    cards: dict[str, dict] = {}
    total_cost = 0.0
    voice_spend = 0.0
    investigations = 0
    voice_calls = 0

    for i in range(len(merchants)):
        m = merchants.iloc[i]
        score = float(scores[i])
        inv = None

        net = (network_risk(graph, str(m.merchant_id))
               if graph is not None else {"in_cluster": False, "score": 0.0})

        if triage.selected[i] and enable_vision:
            inv = investigate(
                merchant_id=str(m.merchant_id),
                declared_category=str(m.declared_category),
                telemetry_score=score,
                toolbox=toolbox,
                network_score=float(net.get("score", 0.0)),
                planner=HeuristicPlanner(),
                confidence_stop=SETTINGS.thresholds.agent_confidence_stop,
            )
            investigations += 1
            total_cost += inv.total_cost_inr

        vision_vertical = vision_conf = vision_surface = None
        injection_present = False
        if inv and inv.vision:
            # The dissenting surface is what the decision turns on. A clean
            # homepage next to a dissenting checkout is a rented account, and
            # reporting the homepage would describe the wrong business.
            dissent = [
                (s, v) for s, v in inv.vision.items()
                if v["vertical"] != m.declared_category and v["vertical"] != "unknown"
            ]
            surface, v = (dissent[0] if dissent else list(inv.vision.items())[-1])
            vision_vertical, vision_conf, vision_surface = (
                v["vertical"], float(v["confidence"]), surface
            )
            injection_present = any(
                x.get("injection", {}).get("present") for x in inv.vision.values()
            )

        script_conflicts: dict[str, str] = {}
        if inv:
            for s in inv.steps:
                if s.tool == "list_third_party_scripts":
                    r = toolbox.list_third_party_scripts(
                        str(m.merchant_id), s.args["surface"]
                    )
                    script_conflicts.update(r.payload.get("vertical_hints", {}))

        def _channels(voice_ev):
            return build_channels(
                declared_category=str(m.declared_category),
                telemetry_score=score,
                vision_vertical=vision_vertical,
                vision_confidence=vision_conf or 0.0,
                vision_surface=vision_surface,
                injection_present=injection_present,
                network=net,
                voice=voice_ev,
                script_conflicts=script_conflicts,
                thresholds=SETTINGS.thresholds,
            )

        def _assessment(voice_ev):
            views = build_views(
                declared=str(m.declared_category),
                telemetry_score=score,
                telemetry_dissent=score >= SETTINGS.thresholds.telemetry_dissent,
                telemetry_detail=(
                    f"Payment behaviour scores {score:.2f} against the "
                    f"{m.declared_category} baseline."),
                vision_vertical=vision_vertical,
                vision_confidence=vision_conf or 0.0,
                vision_surface=vision_surface,
                network=net,
                voice=voice_ev,
            )
            return assess(str(m.merchant_id), str(m.declared_category), views,
                          find_contradictions(views, str(m.declared_category)))

        # First pass, without asking the merchant anything.
        channels = _channels(None)
        provisional = decide(str(m.merchant_id), str(m.declared_category),
                             channels, score, SETTINGS.thresholds, _assessment(None),
                             storefront_available=enable_vision)

        # A verification call is only placed on cases already heading for
        # action. Calling everyone would be both expensive and intrusive, and
        # the merchants worth asking are exactly the ones whose observed
        # identities already disagree.
        voice_ev = None
        if enable_voice and provisional.tier >= Tier.VERIFY:
            v = voice_verify(
                merchant_id=str(m.merchant_id),
                declared_category=str(m.declared_category),
                true_vertical=str(m.true_vertical),
                diverged=bool(m.label),
                live=False,
            )
            voice_ev = v.to_dict()
            voice_calls += 1
            voice_spend += SETTINGS.economics.voice_call_cost_inr

        assessment = _assessment(voice_ev)
        decision = decide(str(m.merchant_id), str(m.declared_category),
                          _channels(voice_ev), score, SETTINGS.thresholds,
                          assessment, storefront_available=enable_vision)

        days_since = (
            SETTINGS.days - int(m.divergence_day) if int(m.divergence_day) >= 0 else 0
        )
        daily_gmv = float(features.iloc[i]["daily_gmv_recent"])
        cost = decision_cost(
            decision.tier, bool(m.label), daily_gmv, days_since, SETTINGS.economics
        )

        rows.append({
            "merchant_id": m.merchant_id,
            "declared_category": m.declared_category,
            "label": int(m.label),
            "archetype": m.archetype,
            "hard_negative": m.hard_negative,
            "telemetry_visible": int(m.telemetry_visible),
            "telemetry_score": score,
            "vision_vertical": vision_vertical,
            "vision_confidence": vision_conf,
            "vision_surface": vision_surface,
            "injection_present": injection_present,
            "infra_linked": net.get("size", 0) - 1 if net.get("in_cluster") else 0,
            "network_score": round(float(net.get("score", 0.0)), 4),
            "network_synchrony": float(net.get("synchrony", 0.0)),
            "voice_status": (voice_ev or {}).get("status", "not_called"),
            "voice_consistent": (voice_ev or {}).get("consistent"),
            "contradictions": len(assessment.contradictions),
            "evidence_agreement": assessment.evidence_agreement,
            "action_confidence": assessment.action_confidence,
            "tier": int(decision.tier),
            "dissenting_channels": ";".join(decision.dissenting_channels),
            "second_channel": decision.second_channel_source,
            "steps_used": len(inv.steps) if inv else 0,
            "stop_reason": inv.stop_reason if inv else "not_triaged",
            "investigation_cost_inr": inv.total_cost_inr if inv else 0.0,
            "daily_gmv_recent": daily_gmv,
            "triage_reasons": ";".join(triage.reasons[i]),
            "in_test_split": bool(i in test_ids),
            **{f"cost_{k}": v for k, v in cost.items()},
        })

        # Cards are built for held-out merchants only. A gradient-boosted tree
        # memorises its training positives, so a card for a training merchant
        # shows a telemetry score the model could not have produced on a
        # merchant it had never seen. Putting one in front of an analyst -- or
        # in a demo -- misrepresents what the system actually knows.
        if build_cards and decision.tier >= Tier.REVIEW and i in test_ids:
            card_extra = {"network": net, "voice": voice_ev,
                          "assessment": assessment.to_dict()}
            cards[str(m.merchant_id)] = _build_card(
                m, decision, inv, features.iloc[i], baselines, scorer,
                matrix.iloc[i], injection_present, card_extra,
            )

    decisions = pd.DataFrame(rows)
    cache.flush()

    throughput = {
        "merchants_scored_cheap_channel": len(merchants),
        "cleared_triage": int(len(triaged)),
        "triage_rate": round(len(triaged) / len(merchants), 4),
        "triage_by_trigger": triage.counts,
        "full_investigations": investigations,
        "investigation_rate": round(investigations / len(merchants), 4),
        "vision_calls_made": classifier.calls_made,
        "vision_cache": cache.stats.to_dict(),
        "vision_spend_inr": round(total_cost, 2),
        "voice_calls": voice_calls,
        "voice_spend_inr": round(voice_spend, 2),
        "graph": graph_summary(graph) if graph is not None else None,
        "spend_per_merchant_inr": round(
            (total_cost + voice_spend) / max(len(merchants), 1), 4),
        "wall_clock_seconds": round(time.time() - t0, 2),
    }

    if verbose:
        print(f"  investigations: {investigations}  vision calls: {classifier.calls_made}"
              f"  cache hit rate: {cache.stats.hit_rate:.1%}")
        print(f"  spend: Rs {total_cost:,.2f} "
              f"({throughput['spend_per_merchant_inr']:.4f} per merchant)")

    return PipelineResult(
        merchants=merchants, decisions=decisions, cards=cards,
        throughput=throughput, split=split, scorer=scorer,
        baselines=baselines, matrix=matrix, toolbox=toolbox,
    )


def _build_card(
    m, decision, inv, feature_row, baselines, scorer, matrix_row,
    injection_present, extra: dict | None = None
) -> dict:
    """The evidence card.

    Everything an analyst needs to act, and everything a regulator needs to ask
    how a decision was reached and get an answer.
    """
    from sentinel.telemetry.baselines import deviation_report
    from sentinel.vision.storefronts import STOREFRONTS

    devs = deviation_report(feature_row, baselines)
    devs.sort(key=lambda r: abs(r["z"]), reverse=True)
    inv_dict = inv.to_dict() if inv else {"steps": [], "stop_reason": "not_triaged"}

    # The summariser sees only the structured payload. This corpus is every
    # string that appeared on the merchant's rendered page; the guard uses it to
    # prove none of it reached the generator.
    page_corpus = []
    for vert in {str(m.fixture_homepage), str(m.fixture_checkout)}:
        sf = STOREFRONTS.get(vert)
        if sf:
            page_corpus += [e.text for e in sf.elements] + [sf.brand, sf.tagline]

    payload = build_summary_input(
        decision=decision,
        feature_deviations=devs,
        investigation=inv_dict,
        onboarded_days_ago=int(m.onboarded_day),
        injection_flagged=bool(injection_present),
        assessment=(extra or {}).get("assessment"),
        network=(extra or {}).get("network"),
        voice=(extra or {}).get("voice"),
    )
    narrative = summarise(payload, page_corpus=page_corpus, live=False)

    vision_block = None
    if inv and inv.vision:
        vision_block = {s: v for s, v in inv.vision.items()}

    return {
        "merchant_id": str(m.merchant_id),
        "declared_category": str(m.declared_category),
        "onboarded_days_ago": int(m.onboarded_day),
        "decision": decision.to_dict(),
        "telemetry": {
            "score": float(scorer.score(matrix_row.to_frame().T)[0]),
            "deviations": devs[:8],
            "attribution": scorer.local_attribution(matrix_row),
        },
        "vision": vision_block,
        "network": (extra or {}).get("network"),
        "voice": (extra or {}).get("voice"),
        "assessment": (extra or {}).get("assessment"),
        "investigation": inv_dict,
        "narrative": narrative.to_dict(),
        "ground_truth": {
            "label": int(m.label),
            "archetype": str(m.archetype),
            "hard_negative": str(m.hard_negative),
            "note": "Present because this is a synthetic evaluation corpus. "
                    "Never available at decision time.",
        },
    }


def main() -> None:
    print("Running Sentinel pipeline...")
    # Cold cache, so the artefacts describe one honest cycle. With a warm cache
    # every vision call in the evidence-card traces reads as free, which is true
    # of the cached run and misleading about what an investigation costs.
    result = run(persist_cache=False)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Materialise the storefront fixtures so the console can show an analyst the
    # page a verdict came from. A finding nobody can check against the surface
    # it was drawn from is not evidence.
    from sentinel.vision.storefronts import write_fixtures

    write_fixtures(ARTIFACTS / "fixtures")
    result.decisions.to_csv(ARTIFACTS / "decisions.csv", index=False)
    (ARTIFACTS / "throughput.json").write_text(
        json.dumps(result.throughput, indent=2), encoding="utf-8"
    )
    cards_dir = ARTIFACTS / "cards"
    cards_dir.mkdir(exist_ok=True)
    # Clear first. Cards left behind by an earlier run are stale evidence for a
    # decision that no longer exists, and an analyst console showing one is
    # worse than one showing nothing.
    for old_card in cards_dir.glob("*.json"):
        old_card.unlink()
    for mid, card in result.cards.items():
        (cards_dir / f"{mid}.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"\nWrote {len(result.cards)} evidence cards to {cards_dir}")
    print(json.dumps(result.throughput, indent=2))


if __name__ == "__main__":
    main()
