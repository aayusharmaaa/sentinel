"""Content-addressed cache for storefront classification.

Classification is rerun far more often than pages change. A merchant on the
high-volume rotation gets looked at weekly; most of them have not touched their
checkout page in months. Keying the cache on the merchant would re-pay for every
one of those; keying it on the hash of the rendered image pays once per distinct
page and never again until the page actually changes.

The hit rate this produces is reported in the evaluation as a cost line, not as
a footnote, because it is a large part of what makes the expensive channel
affordable at scale.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path


def image_hash(payload: bytes) -> str:
    """Content hash of a rendered surface. Live mode hashes PNG bytes; offline
    mode hashes the rendered HTML, which is the same idea one layer up."""
    return hashlib.sha256(payload).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class VerdictCache:
    path: Path | None = None
    _store: dict[str, dict] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.path and self.path.exists():
            try:
                self._store = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # A corrupt cache must never take the pipeline down with it.
                # Losing the cache costs money; failing the run costs coverage.
                self._store = {}

    def get(self, key: str) -> dict | None:
        with self._lock:
            hit = self._store.get(key)
            if hit is None:
                self.stats.misses += 1
            else:
                self.stats.hits += 1
            return hit

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._store[key] = value
            self.stats.writes += 1

    def flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self.path.write_text(json.dumps(self._store), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._store)
