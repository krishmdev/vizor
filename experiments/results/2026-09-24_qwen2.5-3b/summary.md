# qwen2.5:3b-instruct via Ollama (small local model, reduced design)

- Results: `experiments/results/2026-09-24_qwen2.5-3b` (git b9f509a, 2026-09-25)
- Answer model: `qwen2.5:3b-instruct`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 20 queries x 2 samples, 22 pages
- This is a small local model. It needed an extra system message (recorded in the manifest) to cite in-line.
- Local calls: 83 new, 557 cached (no API spend)

## Baseline visibility

| Domain | Retrieved % | Cited % | Cited when retrieved % | C-SoV % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment |
|---|---|---|---|---|---|---|---|---|
| crema-lab.example | 75 | 57 | 77 | 42.1 | 39.3 | 2.04 | 23 | +0.26 |
| brewline.example (target) | 75 | 45 | 60 | 20.7 | 21.9 | 2.94 | 40 | +0.28 |
| shotreport.example | 55 | 30 | 55 | 18.2 | 15.9 | 2.42 | 45 | +0.44 |
| beanbudget.example | 65 | 18 | 27 | 8.3 | 9.8 | 3.00 | 73 | +0.66 |
| grindandtamp.example | 40 | 20 | 50 | 10.7 | 8.1 | 4.00 | 50 | +0.01 |

## Sandbox arms (target: brewline.example)

An arm counts as having an effect when its Holm-adjusted Wilcoxon p-value is below 0.05 (`*`). The 95% confidence intervals are descriptive. `noop` and `aa_resample` are controls and are excluded from the Holm family. Page arms change each query's focus page. They are tested across (fold, page) units because queries that share an edited page are not independent; `n` reports queries / units. Arms that use the tracked queries are cross-fitted: built on one half of the queries and scored on the other. "ΔPAWC if cited" counts only answers with citations. Citation-format failures instead appear in the uncited rate.

| Arm | ΔPAWC pp [95% CI] | Rel Δ % | ΔC-SoV pp [95% CI] | Δ cited pp | ΔPAWC if cited pp | Uncited answers % | Δ sentiment | p (Holm) | n |
|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0 | +0.0 [+0.0, +0.0] | +0.0 | +0.0 | 5 | +0.000 | control | 20 / 4 |
| `aa_resample` | +2.7 [-5.6, +9.7] | +12 | +1.8 [-6.5, +8.8] | +0.0 | +5.9 | 10 | +0.048 | control | 20 / 20 |
| `metadata` | -0.7 [-8.3, +8.4] | -3 | -2.5 [-10.6, +6.7] | -2.5 | +2.4 | 12 | -0.115 | 1.000 | 20 / 7 |
| `faq_rewrite` | +0.9 [-4.6, +8.4] | +4 | -1.0 [-5.9, +7.1] | +0.0 | +3.0 | 5 | -0.028 | 1.000 | 20 / 7 |
| `jsonld_insert` | -3.7 [-7.0, +3.7] | -17 | -3.6 [-7.3, +4.7] | -5.0 | +0.9 | 12 | -0.139 | 1.000 | 20 / 4 |
| `internal_links` | +1.4 [-2.6, +8.8] | +6 | +1.0 [-3.0, +8.5] | +2.5 | +5.4 | 10 | +0.012 | 1.000 | 20 / 4 |
| `stats_surface` | +0.3 [-1.7, +3.8] | +1 | +0.7 [-0.6, +3.9] | +7.5 | +5.3 | 10 | +0.025 | 1.000 | 20 / 4 |
| `keyword_stuffing` | +1.1 [-2.5, +7.4] | +5 | +0.7 [-3.6, +8.5] | +5.0 | +3.3 | 10 | +0.003 | 1.000 | 20 / 7 |
| `engine:reverse` | +5.4 [-12.7, +25.0] | +25 | +5.0 [-12.8, +24.3] | +2.5 | +3.8 | 0 | -0.143 | 1.000 | 20 / 20 |

## Position sweep

| Target slot | PAWC share % [95% CI] | Δ vs slot 1 pp [95% CI] | Cited % | p (Holm, sweeps) | n |
|---|---|---|---|---|---|
| 1 | 28.0 [18.4, 37.8] | ref | 58 |  | 19 |
| 3 | 30.1 [20.6, 40.1] | +2.1 [-12.4, +17.2] | 71 | 1.000 | 19 |
| 5 | 7.8 [2.6, 14.0] | -20.2 [-31.9, -7.6] | 21 | 0.042 * | 19 |

## Boost sweep

For each query, these classes compare the boosted source list with w=0: a different set of sources, the same sources in a different order, or no change. Each cell shows `n queries: mean ΔPAWC pp`.

| Boost w | Target retrieved % | PAWC share % [95% CI] | Δ vs w=0 pp [95% CI] | Cited % | p (Holm, sweeps) | Set changed | Order only | Unchanged |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 75 | 21.9 [10.7, 34.4] | ref | 45 |  |  |  |  |
| 0.02 | 75 | 18.1 [9.0, 28.0] | -3.8 [-14.0, +4.7] | 45 | 1.000 | 1: +0.0 | 6: -12.8 | 13: +0.0 |
| 0.10 | 80 | 20.3 [9.3, 33.6] | -1.7 [-10.7, +6.4] | 40 | 1.000 | 7: +14.0 | 6: -21.9 | 7: +0.0 |

## What the sweeps show

- Context order (same pages, target forced into each slot, n=19 queries): PAWC share slot 1 28.0%, slot 3 30.1%, slot 5 7.8%. Holm-significant differences: slot 5 vs 1: -20.2 [-31.9, -7.6] pp (Holm p 0.042).
- How fragile the slot 5 result is: its raw p is 0.011 over 19 queries, but those queries are served by only 4 target pages. Per page, slot 5 minus slot 1 is duo -18.0, dialing-in -17.7, mill-64 +15.7, solo -43.6 pp, and a Wilcoxon test on the page means gives p = 0.250. It also depends on the sweeps forming their own Holm family: in one family with the arms (11 tests) its Holm p would be 0.116.
- Retrieval weighting (final scores span 0.192 across the top 5 at the median; the 5th-to-6th gap is 0.036): w=0.02: target retrieval unchanged at 75%, ΔPAWC -3.8 [-14.0, +4.7] pp; w=0.10: target retrieval 75% → 80%, ΔPAWC -1.7 [-10.7, +6.4] pp. No boost has a Holm-significant effect.
- Noise floor: re-sampling the unchanged prompts (A/A) moved PAWC share by +2.7 [-5.6, +9.7] pp.
- Page and engine arms: 0 of 7 have a Holm-significant effect.
- Sensitivity (80% power, strictest Holm step; A/A per-query SD 18.3 pp; slot sweep n=19, boost sweep n=20 queries). Moving the target from slot 1 to slot 5: **detected**; small retrieval boosts (w ≤ 0.10): not detected (MDE ≈ 13.7 pp); page edits: **untestable at this design** (7 independent page units; the minimum achievable Holm p is 0.109, above 0.05 at any effect size). Only the boosts and page edits are small changes; moving a source to the last slot is a large one. An effect the size of the MDE would be detected with 80% power; smaller ones could be missed. The sweep MDE uses the A/A per-query SD as its noise scale, so it is approximate for slot and boost comparisons, which have their own variance.
- Bandit, held out: the frozen contextual policy had regret 1.778 (±1.194) vs 1.970 for random and 2.508 for the best fixed arm chosen on the training half. On the held-out queries it is not distinguishable from random.

## Bandit replay

Offline replay uses rewards from the same table the policies learn from. It runs 2000 rounds x 20 runs. Regret is summed over rounds in PAWC share (1.0 = 100 pp). The hindsight row knows the best single arm in advance and serves as a reference, not a policy.

| Policy | Cumulative regret at T (±95% CI) | Mean reward pp/round | Most chosen arm |
|---|---|---|---|
| linucb(a=0.1) | 226.89 ± 19.41 | +7.30 | `noop` (25%) |
| linucb(a=0.5) | 158.79 ± 16.45 | +10.71 | `faq_rewrite` (21%) |
| linucb(a=1) | 155.72 ± 9.23 | +10.83 | `faq_rewrite` (20%) |
| lints(v=0.1) | 182.30 ± 17.90 | +9.56 | `stats_surface` (18%) |
| eps-greedy(0.1) | 200.54 ± 19.19 | +8.60 | `noop` (19%) |
| linucb-bias-only(a=0.1) | 363.41 ± 6.75 | +0.33 | `noop` (53%) |
| random | 376.40 ± 4.47 | -0.17 | `metadata` (15%) |
| best fixed arm (hindsight) | 344.96 ± 4.13 | +1.32 | `internal_links` (100%) |
| oracle | 0.00 ± 0.00 | +18.69 |  |

Held-out check: the policy is fit on fold-1 queries, then frozen and scored on fold-2 queries. Regret is summed over held-out queries, with one decision per query. Caveat: for the cross-fitted arms (metadata, FAQ, keyword stuffing) the fold-1 rewards were measured on pages built from fold-2 query text, so this is a split of contexts rather than a fully independent test.

| Policy | Held-out regret (±95% CI) | Mean reward pp | Most chosen arm |
|---|---|---|---|
| linucb (frozen, contextual) | 1.778 ± 1.194 | +2.72 | `internal_links` (40%) |
| linucb-bias-only (frozen) | 2.508 ± 1.718 | -4.58 | `metadata` (100%) |
| best fixed arm (chosen on train) | 2.508 ± 1.718 | -4.58 | `metadata` (100%) |
| random (expected) | 1.970 ± n/a | +0.79 |  |
| oracle | 0.000 ± 0.000 | +20.50 |  |

## Greedy optimization loop

Held-out queries: 10. The loop keeps an arm only when the lower 95% confidence bound on held-out ΔPAWC is above zero.

| Step | Proposed | Predicted pp | Held-out ΔPAWC pp [95% CI] | Decision | Applied so far | Held-out PAWC % |
|---|---|---|---|---|---|---|
| 1 | `internal_links` | +3.20 | +0.68 [-5.35, +7.15] | rejected | `none` | 23.7 |
| 2 | `metadata` | +0.77 | -4.58 [-11.63, +20.34] | rejected | `none` | 23.7 |
