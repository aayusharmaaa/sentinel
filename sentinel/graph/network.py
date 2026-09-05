"""The merchant relationship graph.

The single-merchant channels answer "is this account behaving oddly". That
question has no answer for a funnel: the whole point of splitting a laundering
operation across eight small accounts is that no individual account is odd. The
evaluation measures this directly -- ring members are generated below the
single-merchant telemetry threshold, and telemetry recall on them is chance.

So this layer asks a different question: **is this merchant suspicious by
itself, or because of the company it keeps?**

How it is built
---------------
Merchants and the identities they use form a bipartite graph. Projecting it to
merchant-to-merchant edges gives a relationship network, but a naive projection
is useless: most legitimate merchants sit on one of five hosting ASNs and ship
one of five analytics bundles, so a naive graph is one giant component and every
merchant is "linked" to hundreds of others.

Two things fix that.

  Rarity. Each shared identity contributes ``log(N / merchants_sharing_it)``.
  A host used by 900 merchants contributes almost nothing; a payout account used
  by six contributes a great deal.

  Type weight. Not all coincidences are equal. Two businesses settling to the
  same bank beneficiary is a much stronger claim of common control than two
  businesses on the same cloud provider, and the weights say so out loud rather
  than burying the judgement in a learned parameter.

Communities are then connected components over edges above a weight floor.
That is a deliberate choice over a modularity optimiser: components are
explainable to an analyst in one sentence ("these four accounts are linked by a
shared beneficiary and a shared payout fingerprint"), they are stable under
small graph changes, and a compliance function can audit why two merchants ended
up in the same cluster. A Louvain partition would be harder to defend and no
more accurate on a graph this sparse.

The signal that only exists here
--------------------------------
`synchrony` is the feature that justifies the whole layer. Independent
businesses drift independently. A ring turns on together, so its members' drift
vectors point the same way at the same time. That is not computable for a single
merchant at all -- it only exists as a property of a group -- which is what
makes the graph a genuinely separate channel rather than a restatement of
telemetry.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Identity columns the graph joins on, and how much a shared value is worth
# relative to its rarity. A shared beneficiary is the strongest claim of common
# control; a shared cloud host is the weakest.
IDENTITY_WEIGHTS = {
    "beneficiary_id": 2.0,
    "payout_fingerprint": 1.8,
    "phone_fp": 1.2,
    "device_fp": 1.1,
    "ip_block": 0.9,
    "script_bundle": 0.5,
    "asn": 0.25,
}

# An identity shared by more than this many merchants is infrastructure, not a
# relationship. Skipped entirely rather than merely down-weighted, because a
# popular host would otherwise create millions of near-zero edges.
MAX_SHARERS = 60

# Edge weight below which two merchants are not considered related at all.
EDGE_FLOOR = 6.0

# Drift features whose direction defines "moving in synchrony".
DRIFT_FEATURES = [
    "round_share_drift", "velocity_drift", "chargeback_drift", "slope_log_volume",
]


@dataclass
class Cluster:
    members: list[str]
    edges: int
    total_weight: float
    shared_kinds: list[str]
    cohesion: float          # mean edge weight per possible pair
    synchrony: float         # mean pairwise cosine of drift vectors, 0..1
    mean_telemetry: float
    max_telemetry: float

    def to_dict(self) -> dict:
        return {
            "members": self.members,
            "size": len(self.members),
            "edges": self.edges,
            "total_weight": round(self.total_weight, 2),
            "shared_kinds": self.shared_kinds,
            "cohesion": round(self.cohesion, 3),
            "synchrony": round(self.synchrony, 3),
            "mean_telemetry": round(self.mean_telemetry, 4),
            "max_telemetry": round(self.max_telemetry, 4),
        }


@dataclass
class MerchantGraph:
    edges: dict[str, dict[str, float]] = field(default_factory=dict)
    shared_by: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    clusters: list[Cluster] = field(default_factory=list)
    cluster_of: dict[str, int] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)

    # -- queries used by the agent and the pipeline ------------------------
    def neighbours(self, merchant_id: str) -> list[tuple[str, float, list[str]]]:
        out = []
        for other, w in self.edges.get(merchant_id, {}).items():
            key = (merchant_id, other) if merchant_id < other else (other, merchant_id)
            out.append((other, w, self.shared_by.get(key, [])))
        out.sort(key=lambda r: -r[1])
        return out

    def cluster_for(self, merchant_id: str) -> Cluster | None:
        i = self.cluster_of.get(merchant_id)
        return self.clusters[i] if i is not None else None


def build_graph(
    merchants: pd.DataFrame,
    features: pd.DataFrame,
    telemetry_scores: np.ndarray | None = None,
) -> MerchantGraph:
    """Project shared identities into a rarity-weighted merchant graph."""
    n = max(len(merchants), 1)
    g = MerchantGraph()
    ids = merchants.merchant_id.astype(str).tolist()

    # Bucket merchants by each identity value, skipping values so common they
    # describe infrastructure rather than a relationship.
    buckets: list[tuple[str, float, list[str]]] = []
    for col, type_w in IDENTITY_WEIGHTS.items():
        if col not in merchants.columns:
            continue
        groups = defaultdict(list)
        for mid, val in zip(ids, merchants[col].astype(str)):
            groups[val].append(mid)
        for val, members in groups.items():
            g.idf[f"{col}={val}"] = math.log(n / max(len(members), 1))
            if len(members) < 2 or len(members) > MAX_SHARERS:
                continue
            weight = type_w * math.log(n / len(members))
            buckets.append((col, weight, members))

    for col, weight, members in buckets:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                g.edges.setdefault(a, {})[b] = g.edges.setdefault(a, {}).get(b, 0.0) + weight
                g.edges.setdefault(b, {})[a] = g.edges.setdefault(b, {}).get(a, 0.0) + weight
                key = (a, b) if a < b else (b, a)
                g.shared_by.setdefault(key, []).append(col)

    _find_clusters(g, merchants, features, telemetry_scores)
    return g


def _find_clusters(
    g: MerchantGraph,
    merchants: pd.DataFrame,
    features: pd.DataFrame,
    telemetry_scores: np.ndarray | None,
) -> None:
    """Connected components over edges above the floor."""
    strong: dict[str, set[str]] = defaultdict(set)
    for a, nbrs in g.edges.items():
        for b, w in nbrs.items():
            if w >= EDGE_FLOOR:
                strong[a].add(b)
                strong[b].add(a)

    feat = features.set_index("merchant_id") if "merchant_id" in features.columns else features
    score_of = {}
    if telemetry_scores is not None:
        score_of = dict(zip(merchants.merchant_id.astype(str), telemetry_scores))

    seen: set[str] = set()
    for start in strong:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in strong[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(comp) < 2:
            continue

        comp.sort()
        pairs, total, kinds = 0, 0.0, set()
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                w = g.edges.get(comp[i], {}).get(comp[j], 0.0)
                if w >= EDGE_FLOOR:
                    pairs += 1
                    total += w
                    key = (comp[i], comp[j]) if comp[i] < comp[j] else (comp[j], comp[i])
                    kinds.update(g.shared_by.get(key, []))

        possible = len(comp) * (len(comp) - 1) / 2
        cohesion = total / possible if possible else 0.0
        scores = [float(score_of.get(m, 0.0)) for m in comp]

        cl = Cluster(
            members=comp,
            edges=pairs,
            total_weight=total,
            shared_kinds=sorted(kinds),
            cohesion=cohesion,
            synchrony=_synchrony(comp, feat),
            mean_telemetry=float(np.mean(scores)) if scores else 0.0,
            max_telemetry=float(np.max(scores)) if scores else 0.0,
        )
        idx = len(g.clusters)
        g.clusters.append(cl)
        for m in comp:
            g.cluster_of[m] = idx


def _synchrony(members: list[str], feat: pd.DataFrame) -> float:
    """Mean pairwise cosine similarity of member drift vectors.

    Independent businesses drift in independent directions, so an honest
    cluster averages near zero. A ring that switched on together points the same
    way. This is the one feature in the whole system that cannot be computed for
    a single merchant, which is exactly why the graph is its own channel.
    """
    rows = []
    for m in members:
        if m not in feat.index:
            continue
        try:
            v = feat.loc[m, DRIFT_FEATURES].to_numpy(dtype=float)
        except KeyError:
            return 0.0
        if np.all(np.isfinite(v)):
            rows.append(v)
    if len(rows) < 2:
        return 0.0

    M = np.array(rows)
    # Scale each feature so one large-magnitude column cannot dominate the angle.
    sd = M.std(axis=0)
    sd[sd < 1e-9] = 1.0
    M = M / sd
    norms = np.linalg.norm(M, axis=1)
    keep = norms > 1e-9
    M, norms = M[keep], norms[keep]
    if len(M) < 2:
        return 0.0

    U = M / norms[:, None]
    sims = U @ U.T
    iu = np.triu_indices(len(U), k=1)
    return float(np.clip(np.mean(sims[iu]), 0.0, 1.0))


def network_risk(g: MerchantGraph, merchant_id: str) -> dict:
    """Per-merchant network risk, with the reason attached.

    Deliberately conservative. A cluster is only risky when it is *both* tightly
    linked by rare identities *and* moving together. Either alone has an
    innocent explanation -- a franchise group settles to one beneficiary; a
    seasonal category drifts together every festival -- and acting on either
    alone is how a monitoring system starts freezing shopping malls.
    """
    cl = g.cluster_for(merchant_id)
    if cl is None:
        return {
            "in_cluster": False, "score": 0.0, "size": 0, "synchrony": 0.0,
            "cohesion": 0.0, "shared_kinds": [], "members": [],
            "statement": "Not linked to any other merchant by a rare shared identity.",
        }

    # Rare-identity strength, squashed so a very tight pair cannot outrank a
    # coordinated group of six on cohesion alone.
    tightness = min(cl.cohesion / 18.0, 1.0)
    size_term = min((len(cl.members) - 1) / 6.0, 1.0)
    score = float(np.clip(0.55 * tightness + 0.15 * size_term + 0.30 * cl.synchrony, 0, 1))

    kinds = ", ".join(k.replace("_", " ") for k in cl.shared_kinds)
    statement = (
        f"Linked to {len(cl.members) - 1} other merchant"
        f"{'s' if len(cl.members) > 2 else ''} by rare shared "
        f"{kinds}. "
        + (
            f"Their payment behaviour is also moving in step "
            f"(synchrony {cl.synchrony:.2f}), which independent businesses do not do."
            if cl.synchrony >= 0.35
            else "Their payment behaviour is not moving in step, so the link alone "
                 "is not evidence of common operation."
        )
    )
    return {
        "in_cluster": True,
        "score": score,
        "size": len(cl.members),
        "synchrony": round(cl.synchrony, 3),
        "cohesion": round(cl.cohesion, 3),
        "shared_kinds": cl.shared_kinds,
        "members": [m for m in cl.members if m != merchant_id][:12],
        "statement": statement,
    }


def graph_summary(g: MerchantGraph) -> dict:
    sizes = [len(c.members) for c in g.clusters]
    return {
        "merchants_with_edges": len(g.edges),
        "clusters": len(g.clusters),
        "clustered_merchants": sum(sizes),
        "largest_cluster": max(sizes) if sizes else 0,
        "median_cluster": int(np.median(sizes)) if sizes else 0,
        "mean_synchrony": round(float(np.mean([c.synchrony for c in g.clusters])), 3)
        if g.clusters else 0.0,
    }
