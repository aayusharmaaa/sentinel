# What broke, and how I got out

Ordered by how much each one changed the system, not by when it happened.

---

## 1. The detector was reading the generator

**Symptom.** The first honest-looking run: held-out AUC 0.986, average precision
0.93. Every legitimate merchant scored below 0.19, every diverged merchant above
0.99. Perfect separation at any threshold, zero false positives.

I had written this in my own docstring before the run: *"a model that scores 1.0
on this set has memorised the generator, not learned the task."* Then I nearly
shipped it.

**Cause.** Every legitimate merchant inside a category was drawn from
**identical** parameters. A furniture shop's round-value share was 0.14, full
stop — no variance. So the per-category baseline had almost no spread, the
robust scale in the denominator of every z-score was tiny, and a legitimate
merchant sat at 0σ while any diverged merchant sat at 40σ. The model was not
separating businesses. It was separating *"drawn from the category profile"*
from *"drawn from something else"*, which is a property of my generator and
exists nowhere in the world.

Two related things made it worse. Abusers swapped **every** feature family at
once, which no real operator does; and post-divergence volume was keyed to the
archetype's absolute scale, so a furniture shop that pivoted acquired a 7×
transaction-count step overnight that any rule would catch.

**Fix.** Three changes in [`data/generate.py`](sentinel/data/generate.py):

- `_personalise()` — every merchant draws its own profile around the category's,
  with real spread on price point, repeat rate, peak hours and risk rates. This
  is what makes a z-score against the category baseline mean anything, and it is
  what lets a legitimate merchant land in the tail, which is the case the whole
  two-channel gate exists to handle.
- `_partial_diverge()` — an abuser moves on two to five of the five feature
  families, chosen at random and blended by a share drawn from `Beta(2.2, 2.0)`.
  The ones that move on two are genuinely hard, and some of them are supposed to
  be missed.
- Post-divergence volume anchored to the merchant's **own** prior trading level.

**Result.** AUC fell from 0.986 to 0.900, average precision from 0.93 to 0.67,
and legitimate merchants started appearing at the triage gate. That is the
number I trust.

**What I took from it.** A suspiciously good result on synthetic data is a bug
report about the data. I had written the warning down and still had to be
persuaded by the metric, which is the part worth remembering.

---

## 2. A whole abuse class was unreachable, and no model work would have fixed it

**Symptom.** `licensing_violation` recall: 0.00 at Tier 3, and 2 of 10 reaching
an analyst. I spent a while assuming the telemetry model was weak.

**Cause.** It wasn't the model. I had built exactly one of the four triage
triggers the design calls for — telemetry drift — so the expensive channel only
ever ran on merchants whose telemetry had already moved. An unregistered
pharmacy sells the same basket at the same prices to the same people at the same
hours as a registered one. It will never drift, never change volume shape, and
never throw an external signal. It was **structurally unreachable**, and every
hour spent on the telemetry model would have changed nothing.

**Fix.** Built [`triage.py`](sentinel/triage.py) properly, with all four
triggers. The one that mattered is the rotation: a fast slice over high-volume
accounts, plus a slow sweep over the *entire* book on a 90-cycle rotation. The
whole-book sweep is the part I nearly left out — without it, the coverage
guarantee only applies to merchants large enough to be worth rotating, and a
small unregistered pharmacy doing ₹50,000 a day is never examined by anything,
ever.

**Result.** `licensing_violation` went to 1.00 reaching an analyst within one
rotation period. It costs money on merchants nothing is wrong with, and the
evaluation reports that premium per trigger rather than hiding it: the rotation
trigger has a 2.5% yield against telemetry drift's 90%.

**What I took from it.** When recall is zero for a whole class, check whether the
class can be *reached* before improving the thing that scores it.

---

## 3. The agent decided everything on one channel and never opened its eyes

**Symptom.** 130 investigations, 3 vision calls. 123 of them terminated after a
single step. The four-step trace the design is built around never happened once.

**Cause.** Two bugs compounding.

The belief was seeded with the merchant's telemetry score converted to log-odds,
and then the `query_transaction_features` tool **added telemetry dissent as
evidence on top of it**. The cheap channel was counted twice. A merchant at 0.62
started at 0.49 log-odds, gained 1.30, and crossed the 0.80 stop threshold at
step one.

And the stop condition was `confidence >= threshold`, with nothing requiring
that the storefront had actually been looked at. So the agent reached a
confident belief, stopped, and handed the gate a decision the gate cannot act on
— because a settlement hold **requires** the storefront channel, so a
telemetry-only belief of 0.97 produces exactly the same outcome as 0.50.

**Fix.** In [`agent/loop.py`](sentinel/agent/loop.py): the prior is now the
prevalence among merchants that cleared triage
(`TRIAGE_CONDITIONAL_PRIOR_LOG_ODDS`), each channel enters exactly once, and the
stop condition carries `and b.vision` — the threshold alone is not a licence to
stop. Raised the threshold to 0.90 so the agent corroborates before concluding.

**Result.** The canonical trace now runs as designed, on real held-out data:
homepage matches the declaration → escalate rather than close → checkout
classifies as lending → scripts corroborate → stop at step 4 of 8.

---

## 4. A silent cache bug that only showed up as an impossible statistic

**Symptom.** The throughput report said `hits: 0, misses: 0` — while the
classifier reported 3 calls made. A cache that has been consulted cannot have
zero of both. That impossibility was the only visible sign.

**Cause.** `VerdictCache` defines `__len__`. The classifier did
`self.cache = cache or VerdictCache()`. An empty cache has `len() == 0`, so it is
**falsy**, so on every cold start the classifier silently discarded the shared
cache the pipeline handed it and built its own. Nothing failed. The pipeline's
cache stayed empty and the reported cost was wrong.

**Fix.** `self.cache = VerdictCache() if cache is None else cache`.

**What I took from it.** `x or default` is wrong for any object with `__len__` or
`__bool__`. It failed silently and only surfaced because two counters
contradicted each other in a report I happened to print.

---

## 5. The vision cache was poisoning eleven merchants from one bad roll

**Symptom.** Eleven `pivot_prohibited` merchants — unrelated, different
categories — all classified `unknown` with confidence **exactly 0.1990**. An
identical float across unrelated merchants is not a coincidence.

**Cause.** The cache is content-addressed by rendered page, which is correct.
But every merchant in a vertical rendered the *same* fixture, so they shared one
cache key. The first merchant to be classified happened to hit the simulator's
1.5% abstain path, and its verdict was then served to all eleven.

The same mistake was inflating the headline cost number: the reported 99% cache
hit rate was measuring **how few fixtures I had written**, not how rarely
merchant pages change.

**Fix.** Each merchant renders its own page (`merchant_ref` in
[`storefronts.py`](sentinel/vision/storefronts.py)) — which is also just true,
since two betting fronts do not serve byte-identical HTML. Cache hits now come
from re-examining the same merchant across cycles, which is the real source.

**Result.** `pivot_prohibited` Tier 3 recall went from 5/17 to 15/17. The honest
cold-cycle cost is ₹893 rather than ₹2.20, and the evaluation now reports cold
and warm separately.

---

## 6. The system claimed reproducibility it did not have

**Symptom.** Two consecutive evaluation runs on unchanged data gave different
numbers. Tier 3 recall moved between 0.438 and 0.479; the reported spend moved
by a few rupees. Small enough to look like rounding, and I nearly wrote it off.

**Cause.** The storefront simulator's per-merchant seed was
`abs(hash((merchant_id, surface)))`. **Python randomises `hash()` for strings on
every interpreter start** unless `PYTHONHASHSEED` is pinned. Same merchant, same
page, different process, different verdict. The ring infrastructure assignment in
the generator had the same defect.

The part that matters is not the wobble. It is that the system's headline
compliance claim is *a decision can be re-derived months later and come out
identical*, and the evaluation printed `bitwise_identical: True` the whole time
— because it scored the same matrix twice **inside one process**, which is
exactly the condition under which this bug is invisible. The check was real and
it was testing the wrong thing.

**Fix.** A `stable_hash()` helper in `config.py` built on SHA-256, and every
behaviour-affecting seed routed through it. Then the determinism check was
extended to cover what it had been missing: it now asserts fixed reference seed
values, so a regression to a process-local hash fails loudly instead of quietly
moving the numbers.

**Result.** Two independent runs now produce byte-identical `decisions.csv`
(same MD5). The numbers in the README are from a run that can be reproduced.

**What I took from it.** A green check on a property you care about is worth
nothing until you know what would make it go red. This one could not go red.

---

## 7. The cost model recommended holding everyone

**Symptom.** The policy ablation showed almost no gap between the two-channel
gate and the naive union — the thing the entire product is built on had no
measurable value.

**Cause.** My false-positive cost was working-capital harm plus a flat
operational fee: about ₹24,000 for a wrongly-held merchant. A missed illicit
merchant cost ₹1.5M. At a 65:1 ratio the cost-optimal policy is to hold
everything that moves, and any ablation run against that model will say the
gate is a waste.

The ratio was wrong because I had left out the dominant real cost: **a
legitimate business whose payouts stopped for four days without warning often
does not come back.**

**Fix.** Added `churn_probability_after_hold` (0.22) against remaining lifetime
margin (`daily_gmv × take_rate × 1,095 days`) in
[`config.py`](sentinel/config.py). This term scales with merchant size, which is
the point — wrongly holding a *large* legitimate merchant is the single most
expensive mistake the system can make, and a false-positive count cannot see
that.

**Result.** The gate became measurably the cheapest policy, and for the right
reason: it catches fewer merchants than the union and still costs less.

**What I took from it.** A cost model that recommends the degenerate policy is
not a cost model. Sanity-check it against the extreme before trusting an
ablation run through it.

---

## 8. The result that contradicted the design, and stayed in

The design's most-argued claim is that per-category baselines beat absolute
thresholds. Against a threshold rule that is emphatic: 173 legitimate merchants
flagged against 9.

But against **the same gradient-boosted model handed raw features**,
category-relative scoring is *worse* — 9 false positives against 4 at matched
recall, AUC 0.900 against 0.920. A boosted tree partially reconstructs category
structure by itself from correlations between features.

My first instinct was that the ablation was broken. It wasn't; my first version
of it genuinely was (I compared the two scorers at one shared threshold, which
compares two different operating points and means nothing — fixed to matched
recall). Once it was correct, the result held.

It is in [EVALUATION.md](EVALUATION.md) and the README as a finding, with the
reason per-category baselines stay in anyway: the evidence card. *"Round-value
share sits 4.1σ above the baseline for home furnishing"* is something an analyst
can act on and a merchant can contest. *"The model scored 0.83"* is neither,
whatever its AUC.

---

## Smaller ones

- **Attribution collapsed to `0.000` on the confident cases.** Neutralisation
  measured in probability space is uninformative once the score saturates near
  1.0 — the explanation column went blank exactly where an analyst most needs
  it. Moved to log-odds.
- **The console was serving evidence cards for training-split merchants.** A
  GBDT memorises its training positives, so those cards showed telemetry scores
  the model could not have produced on a merchant it had never seen. Cards are
  now built for held-out merchants only.
- **Stale cards.** The pipeline wrote into `artifacts/cards/` without clearing
  it, so the console served decisions from runs that no longer existed. 135
  cards for 116 real cases.
- **I gave unregistered pharmacies a prescription-upload script** as their
  vertical tell. That is exactly backwards — `rx-upload` is what a *licensed*
  pharmacy loads. It handed the corroborating channel a signal that does not
  exist in reality and pushed licensing violations to Tier 3 on invented
  evidence. Removed; they now correctly stop at Tier 2, which is what the design
  says should happen when only one channel can see the violation.
- **Heredocs mangled backslashes** in regex patches during the build, silently
  producing no-op edits that reported success. Switched to writing patch scripts
  to a file. Not interesting, but it cost time twice.

---

## 9. The graph and the voice channel measured as worthless, and both were my fault

**Symptom.** The modality ablation — four runs of the whole system, one channel
added per row — came back flat:

```
payments only          analyst recall 0.417   ring recall 0.36
+ storefront                          0.521               0.43
+ network graph                       0.521               0.43
+ voice verification                  0.521               0.43
```

Two brand-new subsystems, no measurable effect. The tempting reading is that
multimodal was not worth building. The real reading was that I had wired both of
them so they could not possibly help.

**Cause 1 — the graph could not surface anyone.** Network dissent alone only
reached `RECHECK`, and I never connected clusters to triage. So a tight,
synchronised ring earned no storefront look, and the one signal that exists
*only* at the group level could never start an investigation.

**Cause 2 — the investigator did not know why the case arrived.** This was the
worse one. After wiring clusters into triage, ring members finally got looked
at, and still went nowhere: eight of them stopped after two steps with
`evidence_exhausted`, having classified only the **homepage**. A funnel member's
homepage is an ordinary shop — that is the entire design of a funnel. The tell is
on the checkout. The planner had no access to the network score, so an agent told
only "look at this merchant" checked the front page, found a shop, and closed.

**Cause 3 — voice could not corroborate.** A merchant whose storefront dissented
*and* who contradicted its own declaration on a verification call still could not
reach a settlement restriction, because I had left `statement` out of the
corroborator list. The strongest possible corroboration — the merchant telling
you itself — counted for nothing.

**Fix.** Clusters feed triage; `Belief` carries the network score and the planner
pursues it (confirm the link, which is free, then go to the checkout); the
merchant's own statement joins the corroborator list, first in the ordering
because it is the hardest to fake.

**Result.** Settlement restrictions went from 18 to **26** diverged merchants
with wrong holds still at **zero**, ring recall from 4/14 to 12/14, and all four
corroborating channels are now actually used: telemetry 14, infrastructure 7,
merchant statement 3, scripts 2.

**What I took from it.** A channel that measures as worthless is a claim about
the wiring before it is a claim about the channel. All three of these looked like
"the graph doesn't help" and all three were plumbing. The ablation was doing its
job — it just took reading the trace to see what it was actually saying.
