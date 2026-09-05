"""Build a standalone, publishable copy of the landing page.

The served page at `/` fetches its data from the API. This produces the same
page with the evaluation payload and the storefront fixtures inlined, so it
works as a single file with no server behind it.

Everything in the output comes from `artifacts/` after a pipeline and
evaluation run, which is the point: the page is generated, so it cannot drift
away from the system it describes.

    python scripts/build_web.py    ->  artifacts/sentinel.html
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.api.landing import build_landing_payload
from sentinel.config import ARTIFACTS, ROOT

SRC = ROOT / "console" / "landing.html"
OUT = ARTIFACTS / "sentinel.html"


def main() -> None:
    if not (ARTIFACTS / "evaluation.json").exists():
        sys.exit("Run `python -m sentinel.pipeline` and "
                 "`python -m sentinel.eval.harness` first.")

    payload = build_landing_payload()
    html = SRC.read_text(encoding="utf-8")

    # No server behind the published file, so the shared stylesheet is inlined
    # rather than linked.
    theme = (ROOT / "console" / "theme.css").read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="/theme.css">',
        "<style>" + theme + "</style>",
    )

    # Only the fixtures the page actually renders, so the file stays small.
    wanted: set[tuple[str, str]] = {
        ("skill_gaming", "homepage"), ("sports_betting", "homepage"),
    }
    case = payload.get("case")
    if case:
        for surface, v in case["vision"].items():
            wanted.add((v["vertical"], surface))

    fixtures = {}
    for vertical, surface in sorted(wanted):
        p = ARTIFACTS / "fixtures" / f"{vertical}__{surface}.html"
        if p.exists():
            fixtures[f"{vertical}__{surface}"] = p.read_text(encoding="utf-8")

    boot = (
        "<script>window.__SENTINEL__="
        + json.dumps(payload, separators=(",", ":"))
        + ";window.__FIXTURES__="
        + json.dumps(fixtures, separators=(",", ":"))
        + ";</script>\n"
    )
    # Inject before the page's own script so both globals exist when it runs.
    marker = "<script>\nconst $ = id => document.getElementById(id);"
    assert marker in html, "landing.html shape changed; update the injection point"
    html = html.replace(marker, boot + marker, 1)

    # The standalone copy has no server, so every console link has nowhere to
    # go. Rewrite href and label together: a CTA reading "Review cases with AI"
    # that lands on a section would be a promise the published file cannot keep.
    # Matched on the href rather than the label, so rewording a button upstream
    # cannot silently reintroduce a dead link -- and the assert below is the
    # backstop if the markup shape changes again.
    html = re.sub(
        r'href="/console(?:\?[^"]*)?"\s*>\s*[^<]*',
        'href="#solution">See how it works ',
        html,
    )
    leftover = html.count('href="/console')
    assert leftover == 0, f"{leftover} console links left in the standalone build"

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, "
          f"{len(fixtures)} fixtures inlined)")


if __name__ == "__main__":
    main()
