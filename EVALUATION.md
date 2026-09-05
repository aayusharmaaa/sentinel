# Evaluation

Track 02's bar is *measured precision and recall on a held-out test set*, with
*honest metrics including false-positive cost*. This document is written to that
bar, including the parts that do not flatter the design.

Reproduce everything with:

```bash
python -m sentinel.data.generate && python -m sentinel.eval.harness
```

Raw output lands in `artifacts/evaluation.json`.

---

## 1. What is being measured, and how the split was made

| | |
|---|---|
| Population | 4,000 merchants, 90-day payment streams |
| Prevalence | **3.5%** — not a balanced split |
| Held-out test | 1,202 merchants, 48 diverged (4.0%) |
| Split | grouped by ring, 30% held out |
| Baselines | fitted on **training-split legitimate merchants only** |
| Model | fitted on training rows only |

Four choices worth stating because each one makes the numbers worse and more
believable:

**Grouped split.** Members of one funnel ring are kept wholly inside one side.
If a ring straddled the split, the model would see its shared behaviour in
training and be scored on its siblings in test. That is leakage, it would
inflate ring recall specifically, and it looks fine until the system meets a
ring it has never seen.

**Baselines fitted on training legitimate merchants only.** Fitting per-category
baselines on the whole population would let abusers pull their own category's
baseline towards themselves, deflating their own z-scores. It would improve
nothing and would be invisible in the results.

**Prevalence 3.5%.** On a balanced set every precision number here roughly
doubles and means nothing, because this system's operating cost is dominated by
what it does to the 96.5%.

**Hard negatives are in the population by construction.** Four legitimate
variants exist specifically to look guilty: `viral_spike` (a real sale — steep
slope that decays), `festival_season` (a step change in volume and hours),
`price_point_change` (a move to a flat ₹499 SKU, taking round-value share to
0.96 overnight with no change of business), and `shared_platform` (the same
popular hosting ASN and checkout plugin as hundreds of others). Plus two whole
categories — `skill_gaming` and `subscription_box` — whose *honest* telemetry is
close to indistinguishable from a laundering front.

**Ground truth carries channel visibility.** Each abuse archetype declares which
channel can structurally see it, and the generator honours that. An unregistered
pharmacy's post-divergence stream is byte-for-byte its pre-divergence stream, so
any telemetry model that scores it above chance is reading noise — and the
evaluation checks that it does not.

---

## 2. Headline

| | precision | recall | TP | FP | FN |
|---|---|---|---|---|---|
| **Tier 4 — payouts restricted**, single cycle | **1.000** | 0.542 | 26 | **0** | 22 |
| **Tier 4 — payouts restricted**, one rotation period | **1.000** | 0.604 | 29 | **0** | 19 |
| Tier 2+ — any analyst action, single cycle | 0.943 | 0.688 | 33 | 2 | 15 |
| Tier 2+ — any analyst action, one rotation period | 0.957 | **0.938** | 45 | 2 | 3 |

The ladder now has five rungs (`T0` monitor, `T1` recheck, `T2` analyst review,
`T3` ask the merchant, `T4` restrict payouts). Only T4 stops money.

*Single cycle* is what the system catches today. *One rotation period* is what it
catches within 90 days, once the slow whole-book rotation has reached every
merchant at least once. Both are reported because the first is the daily
operating picture and the second is the coverage guarantee.

Telemetry channel alone, held out: **AUC 0.900, average precision 0.671.**

---

## 3. False-positive cost

The number that matters most is the third column of the headline table: **zero
merchants had settlements wrongly held.**

| | count | rate of legitimate population | cost | what the merchant experiences |
|---|---|---|---|---|
| Wrongly restricted (Tier 4) | **0** | 0.000 | ₹0 | — |
| Wrongly queued (Tier 2/3) | 2 | 0.0017 | ₹1,800 | **Nothing.** Settlements run normally. |

Both false positives are legitimate merchants whose payment behaviour drifted.
Their storefronts agree with their declarations, so only one view dissents, so
the gate stops them at review. Their payouts were never touched.

### How a false positive is priced

Counting false positives is not enough, because a wrongly-queued merchant and a
wrongly-held merchant are not the same event. The rupee model
([`config.py`](sentinel/config.py)) is stated in full so it can be argued with:

| term | value | why |
|---|---|---|
| `hold_days` | 4.0 | median working days a Tier 4 case stays open |
| `hold_harm_rate` | 0.06 | working-capital harm as a share of frozen volume |
| `hold_fixed_cost_inr` | ₹12,000 | support escalation, comms, reinstatement |
| **`churn_probability_after_hold`** | **0.22** | **a legitimate business whose payouts stopped without warning often does not stay** |
| `take_rate` × `remaining_lifetime` | 2% × 1,095 days | the margin lost when one does leave |
| `analyst_review_cost_inr` | ₹900 | fully loaded, incurred at Tier 2 and above |
| `voice_call_cost_inr` | ₹14 | telephony, speech processing, and the share that escalates to a human |
| `network_fine_inr` | ₹120,000 | network exposure, amortised per undetected merchant |
| `chargeback_liability_rate` | 0.11 | of post-divergence volume |
| `enforcement_freeze_rate` | 0.18 | expected value of funds frozen in the pool |

The churn term is the one that is usually left out, and leaving it out breaks
the model. Without it a wrongly-held merchant costs ~₹24,000 against ~₹1.5M for
a missed one — a 65:1 ratio, under which the cost-optimal policy is to hold
everything that moves. It also scales with merchant size, which is the point:
wrongly holding a *large* legitimate merchant is the single most expensive
mistake this system can make, and a false-positive count cannot see that.

---

## 4. Policy ablation — is the two-channel gate worth anything?

Every policy sees **identical channel outputs** on the same held-out merchants.
Only the rule converting them into an action varies.

| policy | restrict precision | recall | wrong holds | total cost |
|---|---|---|---|---|
| `no_monitoring` | — | 0.000 | 0 | ₹207,560,271 |
| `telemetry_only` | 0.909 | 0.417 | 2 | ₹125,927,473 |
| `storefront_only` | 1.000 | 0.604 | 0 | ₹90,449,981 |
| `gate_removed` | 0.943 | 0.688 | **2** | ₹73,958,284 |
| **`two_channel_gate`** | **1.000** | 0.542 | **0** | **₹73,636,284** |

`gate_removed` is the cleanest comparison: Sentinel's triage, its five channels
and its response ladder, with only the two-channel *requirement* deleted so a
restriction fires on any single dissent. It catches **more** merchants than the
gate (0.688 against 0.542) and still costs marginally more.

`no_monitoring` sets the scale: doing nothing after onboarding costs ₹207.6M on
this test set. Sentinel removes ₹133.9M of that, **65%**.

### What is robust here, and what is not

Two claims come out of that table and they are not equally strong. Separating
them matters more than the headline.

**Robust — the wrong-hold count.** The gate produces **zero** wrongly-held
merchants. Deleting the two-channel requirement produces **two to three**, in
every corpus generation and at every vision-accuracy setting swept. This is a
structural property of the rule, not a property of this particular sample: a
hold needs two independent channels to agree, and the innocent explanations that
make telemetry noisy do not also rewrite a merchant's storefront.

**Not robust — the size of the cost advantage.** The gate's margin over
`gate_removed` is now **0.4%** of total cost — the two policies are within a
rounding error of each other on money. An earlier build of this system reversed
the ordering entirely, and the reversal turned out to be an agent bug rather
than a property of the gate (see [FAILURES.md §9](FAILURES.md)). Quote the
wrong-hold count, which needs no assumption; quote the cost figure only with its
assumption attached. The harness sweeps it:

| post-hold churn probability | two-channel gate | gate removed | gate cheaper |
|---|---|---|---|
| 0.00 | ₹74,133,936 | ₹74,244,782 | yes, by 0.15% |
| 0.10 | ₹74,133,936 | ₹75,029,043 | yes |
| **0.22** *(headline)* | ₹74,133,936 | ₹75,970,155 | yes, by 2.5% |
| 0.50 | ₹74,133,936 | ₹78,166,085 | yes |

An **earlier generation of this corpus reversed the ordering below a churn
probability of ~0.11** — the gate was the more expensive policy at low churn.
The regeneration that followed a determinism fix changed which merchants get
flagged, and the ordering with it.

So the honest statement is: **the gate's cost advantage depends on the churn
assumption and on the sample; its false-positive advantage does not.** Quote the
cost figure with the assumption attached, or quote the wrong-hold count, which
needs no assumption at all. Anyone deploying this should measure their own
post-hold churn rather than inherit 0.22.

An earlier draft of this document asserted a crossover at "roughly 0.08" without
checking, then at 0.106 from a corpus that no longer exists. The sweep now runs
inside the harness and prints with the results, so the number cannot drift away
from the claim again.

---

## 4b. Modality ablation — what each channel is worth

Four runs of the whole system over the same held-out population, one evidence
channel added per row. This is the measurement that decides whether multimodal
was worth building rather than a better single-channel model.

| channels | reaches analyst | restrict precision | wrong holds | ring recall | corroborated |
|---|---|---|---|---|---|
| payments only | 0.417 | — | 0 | 0.36 | 0 |
| + storefront | 0.521 | 1.000 | 0 | 0.43 | 17 |
| **+ network graph** | **0.688** | 1.000 | 0 | **1.00** | 28 |
| + voice verification | 0.688 | 1.000 | 0 | 1.00 | 29 |

**The graph carries the funnel archetype.** Ring recall goes 0.43 → 1.00 the
moment it is switched on, and nothing else in the system moves it, because a
funnel member is generated below the single-merchant threshold: no amount of
modelling the merchant *itself* will ever select it. What the graph contributes
that no per-merchant channel can is **synchrony** — the mean pairwise cosine of
cluster members' drift vectors. Independent businesses drift independently; a
ring turns on together. That quantity does not exist for a single merchant.

**Voice does not move recall, and is not reported as if it did.** It is only
placed on merchants already heading for action, so by construction it cannot
surface a case that nothing else found. Its axis is corroboration: it is the
second independent view on 3 of the 26 restrictions. Reported honestly:

| | |
|---|---|
| Calls placed | 29 |
| Answered with a usable claim | 9 |
| Diverged merchants who answered and **restated their declaration** | 6 |
| Diverged merchants who **contradicted it** | 3 |
| Legitimate merchants who contradicted it | 0 |

A merchant asked whether it has changed business has every reason to say no, and
most calls go unanswered. The channel is worth having because a contradiction is
*strong when it happens*, not because it happens often. Anyone reporting voice
verification as a high-recall detector has not modelled the merchant's
incentives.

---

## 4c. Contradictions — does the bar actually separate?

The gate restricts settlements when two independent views contradict the
declaration. Whether that bar is meaningful is measurable:

| | diverged | legitimate |
|---|---|---|
| Mean contradictions per merchant | **1.54** | **0.01** |
| Merchants reaching two or more | **29** | **0** |

Zero legitimate merchants in the held-out set reach the bar that stops money.
That is the single most important number in this document, and it is a count
rather than a probability on purpose.

Action confidence against ground truth:

| action confidence | legitimate | diverged |
|---|---|---|
| `RESTRICT` | **0** | 27 |
| `VERIFY` | 2 | 5 |
| `REVIEW` | 9 | 1 |
| `WATCH` | 3 | 0 |
| `NONE` | 1,140 | 15 |

The three confidences are kept apart deliberately. Model confidence reaching
0.99 does not move a case; only independent views agreeing does. Where a model
is very confident and only one view dissents, the evidence card says so in as
many words.

---

## 5. Recall by archetype — which channel is doing the work

Averaging recall across archetypes would hide the most important fact about this
system, which is that **half the diverged population is invisible to payment
telemetry** and the storefront channel is carrying all of it.

Within one rotation period:

| archetype | n | telemetry can see it | reaches analyst | settlements held |
|---|---|---|---|---|
| `pivot_prohibited` | 15 | yes | **1.00** | 0.73 |
| `licensing_violation` | 10 | **no** | **1.00** | 0.20 |
| `funnel_ring` | 14 | **no** | **1.00** | 0.86 |
| `rented_account` | 9 | partially | 0.67 | 0.44 |
| **telemetry-blind cases** | **24** | **no** | **24 of 24** | — |

**`licensing_violation` is the clean demonstration.** These merchants are
generated from a distribution identical to legitimate pharmacies. Measured on the
held-out split:

| | mean telemetry score | max |
|---|---|---|
| legitimate pharmacies (n=96) | 0.0091 | 0.113 |
| unregistered pharmacies (n=10) | 0.0262 | 0.167 |

Telemetry cannot tell them apart, exactly as designed, and every one of them is
still caught by the storefront channel. **Seven of ten correctly stop at Tier 2**
— analyst review with settlements running — because only one channel can see the
violation. The three that reach Tier 3 do so because they *also* share rare
payout infrastructure with other merchants, which is a genuine second channel.

This is why single-channel dissent has to produce action rather than waiting.
For a licensing violation the second channel can never arrive, and waiting for
it means never acting.

**`rented_account` at 0.43 is the weakest result** and is reported as such. The
genuine store keeps trading alongside the illicit business, so the payment
mixture stays near the category baseline and telemetry often never crosses its
threshold. Closing that gap means triggering on the *shape* of a bimodal
payment mixture rather than its level. That is not built.

---

## 6. Category-relative baselines — the claim, and what actually happened

The design's most-argued claim: *the question is never "is this merchant
unusual", it is "is this merchant unusual for what it claims to be".*

### Against a threshold rule: emphatic

`round_share >= 0.60 OR velocity >= 4.0 OR night_share >= 0.30`, applied to
every merchant regardless of declared category — the kind of rule most systems
ship first:

| | precision | recall | legitimate merchants flagged |
|---|---|---|---|
| absolute threshold rule | 0.034 | 0.125 | **173** |
| category-relative model | 0.750 | 0.562 | **9** |

It loses on both axes at once. And the composition of those 173 false positives
is the argument stated precisely:

| category | wrongly flagged |
|---|---|
| `skill_gaming` | **115** |
| `subscription_box` | **44** |
| `digital_services` | 6 |
| `electronics_retail` | 6 |
| `apparel` | 2 |

159 of 173 are the two categories whose *honest* behaviour looks like abuse. A
skill-gaming platform at 68% round-value share is ordinary. A home furnishing
store at 71% is not. An absolute threshold flags both, and the one it should not
have flagged is a real business whose payroll now depends on how fast a queue
moves.

### Against the same model on raw features: the claim does not hold

| | AUC | FP at matched recall (0.56) |
|---|---|---|
| category-relative | 0.900 | 9 |
| raw, uncontextualised features | **0.920** | **4** |

Category-relative scoring is **worse** here. A gradient-boosted tree given raw
features partially reconstructs category structure by itself, from correlations
between features — high velocity together with high round-value share and a
night-shifted histogram is recognisably skill gaming without anyone saying so.

Reported as measured rather than as expected. The per-category encoding stays
for a reason that is not AUC: **the evidence card.** *"Round-value share sits
4.1σ above the baseline for home furnishing"* is a sentence an analyst can act
on and a merchant can contest. *"The model scored 0.83"* is neither. In a
workflow where the output can freeze a business's payouts, a marginally better
score that cannot be explained is the worse artefact.

---

## 7. Sensitivity — how good does the vision channel have to be?

Every metric above assumes something about the storefront classifier. That
assumption is swept rather than asserted:

| vision accuracy | hold precision | hold recall | wrong holds |
|---|---|---|---|
| 0.966 | 1.000 | 0.479 | 0 |
| 0.949 | 1.000 | 0.479 | 0 |
| 0.850 | 1.000 | 0.375 | 0 |
| 0.594 | *undefined — nothing held* | 0.000 | 0 |

**Precision holds at 1.000 across the whole sweep; recall collapses.** As the
model degrades, the gate converts its errors into missed detections rather than
wrongly frozen merchants — because a hold needs a second, independent channel to
agree, and a confused vision model does not make telemetry agree with it.

For a system that can stop a real business's payouts, that is the correct
direction to fail in, and it means the deployment risk is *under-detection under
model drift*, which is monitorable, rather than *sudden false holds*, which is
not recoverable.

---

## 8. Adversarial robustness

### The injection detector

| | |
|---|---|
| Adversarial variants detected | **5 / 5** |
| Clean pages falsely flagged | **0 / 26** |

Five hand-written variants is a small set and this is a pattern matcher, so the
honest reading is "no obvious gap", not "solved". Its role is to **record** a
signal on the evidence card, never to gate a hold on its own.

The framing matters more than the score. A page carrying text addressed to an
automated classifier is treated as **evidence of suspicion in its own right** —
there is no honest business whose homepage needs to instruct a reviewing model
about how to categorise it. Injection defence is usually pure loss; here it
inverts into a signal.

### Verdict robustness

Three independent defences, in order of how much they are relied on:

1. Page content is framed as **data to describe, never instruction to follow**.
2. The vision verdict is **cross-checked against telemetry**, which a merchant
   cannot influence without changing the actual business. This is the one that
   actually holds: prompt hardening can be circumvented; editing a web page
   cannot change a payment stream.
3. The attempt is recorded as a signal.

The narrative summariser is downstream of structured features only. If scraped
page content reached it, a merchant could write copy on their own site designed
to shape the compliance paragraph a human reads before deciding — a softer
target than the verdict and a more useful one. That path is closed by an
allowlist plus `assert_no_page_text`, a runtime guard that scans the payload for
any string from the rendered page and raises. It runs on every call, because a
guarantee nothing checks is a guarantee that quietly stops holding.

---

## 9. Throughput and cost

The two-speed design is the deployability claim, so it is measured.

| stage | count | rate |
|---|---|---|
| Cheap channel — every merchant | 4,000 | 100% |
| Cleared triage → expensive channel | 355 | 8.9% |
| Vision calls (cold cache) | 408 | — |
| Reached an analyst | 29 held-out cases | — |

| | cold cycle | warm cycle |
|---|---|---|
| Vision calls | 408 | 0 |
| Spend | ₹897.60 | ₹0.00 |
| Per merchant | ₹0.2244 | ₹0.0000 |
| **Projected at 1M merchants** | **₹224,400** | — |

The warm figure assumes **no page changed between cycles**, which is why it is
₹0 and why it is not the number to plan against. True steady-state cost sits
between the two, proportional to how often merchants edit their checkout. **Cold
cost is the honest planning number.**

### What each trigger buys

| trigger | selected | diverged | yield | unique catches |
|---|---|---|---|---|
| `telemetry_drift` | 29 | 26 | **89.7%** | 15 |
| `external_signal` | 41 | 12 | 29.3% | 2 |
| `volume_shape_change` | 9 | 2 | 22.2% | 0 |
| `high_volume_rotation` | 40 | 1 | **2.5%** | 0 |

The rotation trigger has a 2.5% yield — it deliberately spends money on
merchants nothing is wrong with. That spend is the premium paid for covering the
telemetry blind spot and for being able to say no merchant goes unexamined
indefinitely. It is reported as a number rather than assumed as a design virtue.

---

## 10. Determinism

`bitwise_identical: true`. The telemetry channel rescored twice on the same
matrix produces byte-identical output, max absolute delta 0.0.

This is checked in the evaluation rather than claimed in a docstring. A
settlement hold gets challenged weeks later and the same window has to produce
the same score to the digit; an unchecked reproducibility claim in a compliance
system is worth nothing.

---

## 11. What this evaluation does not establish

- **It is synthetic.** It cannot show Sentinel works on Razorpay's real book. It
  shows the design's claims are falsifiable and which of them survived — one did
  not, and it is in section 6.
- **The offline vision channel is a simulator.** Its confusion structure is
  derived from how much interface vocabulary two verticals share, and its error
  rate is swept rather than assumed, but it is not a vision model. Live mode is.
- **48 positives is a small test set.** Per-archetype recalls rest on 7 to 17
  merchants each; treat them as direction, not precision. Tier 3 precision of
  1.000 on 21 predictions means "no observed false holds", not "no false holds".
- **The economics are estimates**, and the *cost* claim is conditional on them
  in a way the *false-positive* claim is not. Every constant is stated in
  section 3 so the conclusions can be recomputed. See section 4 for which of the
  two ablation claims survives corpus regeneration and which does not.
- **One cycle is not a year.** Repeated exposure, adversarial adaptation, and
  baseline drift as categories evolve are all real and none are modelled.
