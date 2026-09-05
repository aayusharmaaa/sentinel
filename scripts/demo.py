"""Guided walk through the cases that carry the design.

Four merchants from the held-out set, chosen because each one demonstrates a
claim that would otherwise just be a paragraph:

  1. a rented account -- the agent escalates instead of closing on a clean
     homepage, and the second channel is the script inventory
  2. a legitimate merchant whose telemetry looks like a laundering front, saved
     by the gate
  3. a licensing violation -- caught by vision, invisible to telemetry, and
     correctly stopped at review rather than a hold
  4. a storefront carrying text addressed to the classifier

Run after `python -m sentinel.pipeline`:

    python scripts/demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sentinel.config import ARTIFACTS

W = 78
TIER_LABEL = {0: "MONITOR", 1: "RECHECK", 2: "REVIEW (settlements run)",
              3: "HOLD (settlements frozen)"}


def rule(char: str = "-") -> None:
    print(char * W)


def head(title: str, subtitle: str = "") -> None:
    print()
    rule("=")
    print(title)
    if subtitle:
        print(subtitle)
    rule("=")


def wrap(text: str, indent: str = "  ") -> None:
    import textwrap

    for line in textwrap.wrap(text, W - len(indent)):
        print(indent + line)


def load_cards() -> dict[str, dict]:
    d = ARTIFACTS / "cards"
    if not d.exists() or not any(d.glob("*.json")):
        sys.exit("No evidence cards. Run `python -m sentinel.pipeline` first.")
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*.json")}


def show_case(card: dict, why: str) -> None:
    d = card["decision"]
    inv = card["investigation"]
    print()
    rule()
    print(f"  {card['merchant_id']}   declared: {card['declared_category']}"
          f"   onboarded {card['onboarded_days_ago']}d ago")
    rule()
    print(f"\n  WHY THIS CASE: {why}\n")

    print("  CHANNELS")
    for ch in d["channels"]:
        mark = {"DISSENT": "!!", "BLIND": "??", "AGREES": "ok"}[ch["verdict"]]
        print(f"    [{mark}] {ch['channel']:<15} {ch['verdict']}")
        wrap(ch["statement"], "         ")

    if inv.get("steps"):
        print(f"\n  INVESTIGATION  ({inv['steps_used']} of {inv['step_budget']} steps, "
              f"Rs {inv.get('total_cost_inr', 0):.2f})")
        for s in inv["steps"]:
            args = ", ".join(f"{k}={v!r}" for k, v in s["args"].items())
            cost = f"Rs {s['cost_inr']:.2f}" if s["cost_inr"] else "free"
            print(f"    {s['step']}. {s['tool']}({args})   [{cost}]")
            wrap(f"reasoning: {s['rationale']}", "       ")
            wrap(f"returned:  {s['returned']}", "       ")
            print(f"       belief now {s['confidence_after'] * 100:.1f}%")
        print(f"    stopped: {inv['stop_reason'].replace('_', ' ')}")

    for surface, v in (card.get("vision") or {}).items():
        if v.get("injection", {}).get("present"):
            print(f"\n  PAGE INTEGRITY  ({surface})")
            wrap(v["injection"]["headline"], "    ")
            for detail in v["injection"].get("detail", [])[:2]:
                wrap(f'quoted: "{detail["quoted"]}"', "    ")

    print(f"\n  DECISION: Tier {d['tier']} - {TIER_LABEL[d['tier']]}")
    if d.get("second_channel_source"):
        print(f"  Second corroborating channel: {d['second_channel_source']}")
    wrap(d["justification"], "    ")
    for n in d.get("notes", []):
        wrap(f"note: {n}", "    ")

    print("\n  ANALYST RATIONALE")
    wrap(card["narrative"]["text"], "    ")
    print(f"    [provenance: {card['narrative']['provenance']}]")

    gt = card["ground_truth"]
    print(f"\n  GROUND TRUTH (synthetic corpus only): label={gt['label']} "
          f"archetype={gt['archetype']} hard_negative={gt['hard_negative']}")


def pick(cards: dict, **criteria):
    for card in cards.values():
        gt, d = card["ground_truth"], card["decision"]
        if all(
            (gt.get(k) == v if k in gt else d.get(k) == v)
            for k, v in criteria.items()
        ):
            yield card


def main() -> None:
    cards = load_cards()
    decisions = pd.read_csv(ARTIFACTS / "decisions.csv")
    test = decisions[decisions.in_test_split]

    head("SENTINEL", "Post-onboarding merchant monitoring. Held-out cases only.")
    print(f"  Held-out set: {len(test)} merchants, {int(test.label.sum())} diverged "
          f"({test.label.mean():.1%} prevalence)")
    print(f"  Reaching an analyst: {int((test.tier >= 2).sum())}   "
          f"of which settlements held: {int((test.tier >= 3).sum())}")
    print(f"  Legitimate merchants wrongly held: "
          f"{int(((test.label == 0) & (test.tier >= 3)).sum())}")

    # 1. rented account, second channel = scripts
    head("CASE 1 - THE ACCOUNT THAT WAS RENTED OUT",
         "A clean homepage would close the case. Telemetry dissents, so it does not.")
    pool = [c for c in pick(cards, archetype="rented_account")
            if c["decision"]["second_channel_source"] == "scripts"] \
        or list(pick(cards, archetype="rented_account"))
    if pool:
        show_case(pool[0],
                  "The genuine storefront keeps trading while the same merchant ID "
                  "processes for a second business. Only the checkout surface "
                  "disagrees with the declaration.")

    # 2. legitimate merchant the gate protects
    head("CASE 2 - THE LEGITIMATE MERCHANT THE GATE PROTECTS",
         "Telemetry dissents. Settlements keep running anyway.")
    pool = list(pick(cards, label=0))
    if pool:
        show_case(pool[0],
                  "A real business whose payment behaviour is atypical for its "
                  "category. One channel dissents, so it never reaches a hold. "
                  "This is the case a single-signal rule freezes.")
    else:
        print("\n  No legitimate merchant reached the queue in this run -- which is\n"
              "  itself the result: zero false positives at review or above.")

    # 3. licensing violation
    head("CASE 3 - THE VIOLATION TELEMETRY CANNOT SEE",
         "Identical payment shape to a licensed pharmacy. Caught by the storefront.")
    pool = list(pick(cards, archetype="licensing_violation"))
    if pool:
        show_case(pool[0],
                  "An unregistered pharmacy has exactly the payment shape of a "
                  "registered one -- same basket, same prices, same hours. No "
                  "telemetry feature will ever separate them. Vision can, and "
                  "one channel means review, not a hold.")

    # 4. injection as a signal
    head("CASE 4 - THE STOREFRONT THAT TALKED TO THE CLASSIFIER",
         "Adversarial copy, treated as evidence rather than followed.")
    pool = [
        c for c in cards.values()
        if any(v.get("injection", {}).get("present")
               for v in (c.get("vision") or {}).values())
    ]
    if pool:
        show_case(pool[0],
                  "The page carries text written for an automated reviewer. It is "
                  "recorded as a suspicion signal, never acted on as an "
                  "instruction, and never gates a hold by itself. No legitimate "
                  "merchant has a reason to instruct a reviewing model.")

    head("THE RULE UNDERNEATH ALL FOUR")
    wrap("No single channel can trigger a settlement hold. Tier 3 requires the "
         "storefront channel to dissent from the declaration and one independent "
         "channel to agree with it. Independent evidence types agreeing is a far "
         "stronger signal than either being loud -- and it is what keeps a "
         "legitimate business whose telemetry looks alarming out of a settlement "
         "freeze.", "  ")
    print()
    print("  Full metrics:  python -m sentinel.eval.harness")
    print("  Console:       python -m uvicorn sentinel.api.app:app --port 8000")
    print()


if __name__ == "__main__":
    main()
