"""The agent's tools.

Four of them, deliberately. Each one costs something different and returns
something different, and the agent's job is to choose which cost to pay next
given what it already knows.

  classify_storefront(surface)   -- expensive. An external model call against a
                                    rendered page. This is the only tool that
                                    spends money per invocation.
  list_third_party_scripts        -- cheap. A page fetch, no model. Corroborates
                                    or contradicts a storefront verdict.
  query_transaction_features      -- free. Data already held.
  find_shared_infrastructure      -- cheap. A join across the merchant book.

Every call and every return is logged. That log is simultaneously the audit
trail a compliance function requires and the evidence that the investigation
was bounded -- both of which someone will eventually ask for.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sentinel.config import stable_hash
from sentinel.telemetry.baselines import BaselineSet, deviation_report
from sentinel.vision.classifier import StorefrontClassifier, VisionVerdict
from sentinel.vision.storefronts import INJECTION_VARIANTS

# Scripts that only make sense on a business of a particular kind. Finding one
# on a page that classified as something else is corroboration; finding one on a
# page that classified consistently is a contradiction worth surfacing.
SCRIPT_VERTICAL_HINTS = {
    "oddsfeed-live": "sports_betting",
    "geo-fence-sdk": "sports_betting",
    "kyc-lite": "unlicensed_lending",
    "collections-dialer": "unlicensed_lending",
    "chain-rates": "crypto_exchange",
    "wallet-connect": "crypto_exchange",
    "age-gate": "adult_content",
    "rx-upload": "unregistered_pharmacy",
}


@dataclass
class ToolResult:
    tool: str
    args: dict
    ok: bool
    payload: dict
    cost_inr: float = 0.0
    # One line an analyst reads without opening the payload.
    summary: str = ""


@dataclass
class Toolbox:
    merchants: pd.DataFrame
    features: pd.DataFrame
    scripts: dict
    baselines: BaselineSet
    classifier: StorefrontClassifier
    vision_call_cost_inr: float = 2.20
    _idf: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._build_idf()

    # -- infrastructure rarity -------------------------------------------
    def _build_idf(self) -> None:
        """Inverse document frequency over infrastructure attributes.

        Without this the signal is worthless. Most legitimate merchants sit on
        one of five hosting ASNs and ship one of five analytics bundles; naive
        clustering links hundreds of unrelated businesses into a fake ring and
        the analyst queue fills with merchants whose only crime is using a
        popular host. Weighting by rarity means a shared attribute only counts
        for as much as it is unusual.
        """
        n = max(len(self.merchants), 1)
        for col in ("asn", "ip_block", "payout_fingerprint", "script_bundle"):
            counts = Counter(self.merchants[col].astype(str))
            for value, c in counts.items():
                self._idf[f"{col}={value}"] = math.log(n / c)

    def _row(self, merchant_id: str) -> pd.Series:
        m = self.merchants[self.merchants.merchant_id == merchant_id]
        if m.empty:
            raise KeyError(merchant_id)
        return m.iloc[0]

    # -- tools ------------------------------------------------------------
    def classify_storefront(self, merchant_id: str, surface: str = "homepage") -> ToolResult:
        """Render a surface of the merchant's site and classify it. Expensive."""
        try:
            m = self._row(merchant_id)
        except KeyError:
            return ToolResult("classify_storefront", {"surface": surface}, False,
                              {"error": "unknown merchant"}, 0.0, "Merchant not found.")

        rendered = m["fixture_homepage"] if surface == "homepage" else m["fixture_checkout"]
        injected = (
            INJECTION_VARIANTS[stable_hash(merchant_id) % len(INJECTION_VARIANTS)]
            if int(m["injection_present"]) and surface == "homepage"
            else None
        )
        verdict: VisionVerdict = self.classifier.classify(
            merchant_id=merchant_id,
            surface=surface,
            rendered_vertical=str(rendered),
            declared_category=str(m["declared_category"]),
            injected_text=injected,
            seed=stable_hash(merchant_id, surface) % (2**31),
        )
        cost = 0.0 if verdict.cached else self.vision_call_cost_inr
        agrees = verdict.vertical == m["declared_category"]
        summary = (
            f"{surface} classifies as {verdict.vertical} "
            f"(confidence {verdict.confidence:.2f}), which "
            f"{'matches' if agrees else 'does not match'} the declared "
            f"{m['declared_category']}."
        )
        if verdict.injection.get("present"):
            summary += " Page also carries text addressed to an automated reviewer."
        return ToolResult(
            "classify_storefront", {"surface": surface}, True,
            verdict.to_dict(), cost, summary,
        )

    def list_third_party_scripts(self, merchant_id: str, surface: str = "checkout") -> ToolResult:
        """Enumerate third-party scripts on a surface. Cheap corroboration."""
        entry = self.scripts.get(merchant_id, {})
        found = entry.get(surface, [])
        hints = {}
        for s in found:
            base = s.split("@")[0]
            if base in SCRIPT_VERTICAL_HINTS:
                hints[s] = SCRIPT_VERTICAL_HINTS[base]
        summary = (
            f"{len(found)} scripts on {surface}."
            + (
                " Vertical-specific: "
                + ", ".join(f"{k} implies {v}" for k, v in hints.items())
                if hints
                else " None are vertical-specific."
            )
        )
        return ToolResult(
            "list_third_party_scripts", {"surface": surface}, True,
            {"scripts": found, "vertical_hints": hints}, 0.0, summary,
        )

    def query_transaction_features(self, merchant_id: str, top_k: int = 5) -> ToolResult:
        """Return the merchant's features against its declared category baseline."""
        f = self.features[self.features.merchant_id == merchant_id]
        if f.empty:
            return ToolResult("query_transaction_features", {}, False,
                              {"error": "no features"}, 0.0, "No telemetry.")
        rows = deviation_report(f.iloc[0], self.baselines)
        rows.sort(key=lambda r: abs(r["z"]), reverse=True)
        top = rows[:top_k]
        worst = top[0] if top else None
        summary = (
            f"Largest deviation from the {f.iloc[0]['declared_category']} baseline: "
            f"{worst['feature']} at {worst['z']:+.1f} sigma."
            if worst else "No deviations."
        )
        return ToolResult(
            "query_transaction_features", {"top_k": top_k}, True,
            {"deviations": top, "declared_category": str(f.iloc[0]["declared_category"])},
            0.0, summary,
        )

    def find_shared_infrastructure(
        self, merchant_id: str, min_weight: float = 6.0
    ) -> ToolResult:
        """Find merchants sharing rare infrastructure with this one."""
        try:
            me = self._row(merchant_id)
        except KeyError:
            return ToolResult("find_shared_infrastructure", {}, False,
                              {"error": "unknown merchant"}, 0.0, "Merchant not found.")

        cols = ("asn", "ip_block", "payout_fingerprint", "script_bundle")
        scores: dict[str, float] = {}
        shared: dict[str, list[str]] = {}
        for col in cols:
            value = str(me[col])
            weight = self._idf.get(f"{col}={value}", 0.0)
            if weight <= 0:
                continue
            peers = self.merchants[
                (self.merchants[col].astype(str) == value)
                & (self.merchants.merchant_id != merchant_id)
            ].merchant_id.tolist()
            for p in peers:
                scores[p] = scores.get(p, 0.0) + weight
                shared.setdefault(p, []).append(f"{col}={value}")

        linked = [
            {"merchant_id": p, "weight": round(w, 2), "shared": shared[p]}
            for p, w in sorted(scores.items(), key=lambda kv: -kv[1])
            if w >= min_weight
        ]
        total = round(sum(l["weight"] for l in linked), 2)
        if linked:
            summary = (
                f"{len(linked)} merchants share rare infrastructure "
                f"(rarity-weighted total {total}). Shared attributes: "
                + ", ".join(sorted({a.split('=')[0] for l in linked for a in l['shared']}))
                + "."
            )
        else:
            common = [
                c for c in cols
                if self._idf.get(f"{c}={me[c]}", 0.0) > 0
                and self._idf[f"{c}={me[c]}"] < min_weight
            ]
            summary = (
                "No rare infrastructure shared with any other merchant."
                + (
                    f" Attributes shared with many merchants ({', '.join(common)}) "
                    "were discounted as common-platform noise."
                    if common else ""
                )
            )
        return ToolResult(
            "find_shared_infrastructure", {"min_weight": min_weight}, True,
            {"linked": linked[:12], "n_linked": len(linked), "rarity_total": total},
            0.0, summary,
        )

    # -- dispatch ---------------------------------------------------------
    def call(self, name: str, merchant_id: str, **kwargs) -> ToolResult:
        fn = {
            "classify_storefront": self.classify_storefront,
            "list_third_party_scripts": self.list_third_party_scripts,
            "query_transaction_features": self.query_transaction_features,
            "find_shared_infrastructure": self.find_shared_infrastructure,
        }.get(name)
        if fn is None:
            return ToolResult(name, kwargs, False, {"error": "no such tool"}, 0.0,
                              "Unknown tool.")
        return fn(merchant_id, **kwargs)


TOOL_SCHEMAS = [
    {
        "name": "classify_storefront",
        "description": (
            "Render one surface of the merchant's website and classify what kind "
            "of business the interface is built for. EXPENSIVE: this is an "
            "external model call. Surfaces: 'homepage' (the marketing surface) "
            "or 'checkout' (where the customer actually pays). A merchant "
            "running a second business on the same merchant ID will usually keep "
            "the homepage clean, so a clean homepage is weak evidence and a "
            "clean checkout is strong evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"surface": {"type": "string", "enum": ["homepage", "checkout"]}},
            "required": ["surface"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_third_party_scripts",
        "description": (
            "List third-party scripts loaded on a surface. Cheap. Some scripts "
            "only make sense for one kind of business, so this corroborates or "
            "contradicts a storefront classification without paying for another."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"surface": {"type": "string", "enum": ["homepage", "checkout"]}},
            "required": ["surface"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_transaction_features",
        "description": (
            "Return this merchant's payment-behaviour features expressed as "
            "deviations from the baseline for its DECLARED category. Free. "
            "A merchant cannot influence this without changing its actual "
            "business, which makes it the check on anything the storefront says."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"top_k": {"type": "integer"}},
            "required": ["top_k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_shared_infrastructure",
        "description": (
            "Find other merchants sharing rare infrastructure with this one: "
            "hosting range, payout fingerprint, script bundle. Cheap. Weighted "
            "by rarity, so sharing a popular host counts for almost nothing and "
            "sharing a payout fingerprint counts for a great deal. Use when a "
            "merchant looks individually unremarkable but you suspect it is one "
            "of several accounts funnelling to a common beneficiary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"min_weight": {"type": "number"}},
            "required": ["min_weight"],
            "additionalProperties": False,
        },
    },
]
