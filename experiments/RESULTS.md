# Results

`vizor report` builds this page from the run directories under `experiments/results/`. All numbers come from those files. See `experiments/README.md` for reproduction steps.

## qwen2.5:3b-instruct via Ollama (small local model, reduced design)

- Results: `experiments/results/2026-09-24_qwen2.5-3b` (git b9f509a, 2026-09-25)
- Answer model: `qwen2.5:3b-instruct`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 20 queries x 2 samples, 22 pages
- This is a small local model. It needed an extra system message (recorded in the manifest) to cite in-line.
- Local calls: 83 new, 557 cached (no API spend)

### Baseline visibility

| Domain | Retrieved % | Cited % | Cited when retrieved % | C-SoV % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment |
|---|---|---|---|---|---|---|---|---|
| crema-lab.example | 75 | 57 | 77 | 42.1 | 39.3 | 2.04 | 23 | +0.26 |
| brewline.example (target) | 75 | 45 | 60 | 20.7 | 21.9 | 2.94 | 40 | +0.28 |
| shotreport.example | 55 | 30 | 55 | 18.2 | 15.9 | 2.42 | 45 | +0.44 |
| beanbudget.example | 65 | 18 | 27 | 8.3 | 9.8 | 3.00 | 73 | +0.66 |
| grindandtamp.example | 40 | 20 | 50 | 10.7 | 8.1 | 4.00 | 50 | +0.01 |

### Sandbox arms (target: brewline.example)

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

### Position sweep

| Target slot | PAWC share % [95% CI] | Δ vs slot 1 pp [95% CI] | Cited % | p (Holm, sweeps) | n |
|---|---|---|---|---|---|
| 1 | 28.0 [18.4, 37.8] | ref | 58 |  | 19 |
| 3 | 30.1 [20.6, 40.1] | +2.1 [-12.4, +17.2] | 71 | 1.000 | 19 |
| 5 | 7.8 [2.6, 14.0] | -20.2 [-31.9, -7.6] | 21 | 0.042 * | 19 |

### Boost sweep

For each query, these classes compare the boosted source list with w=0: a different set of sources, the same sources in a different order, or no change. Each cell shows `n queries: mean ΔPAWC pp`.

| Boost w | Target retrieved % | PAWC share % [95% CI] | Δ vs w=0 pp [95% CI] | Cited % | p (Holm, sweeps) | Set changed | Order only | Unchanged |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 75 | 21.9 [10.7, 34.4] | ref | 45 |  |  |  |  |
| 0.02 | 75 | 18.1 [9.0, 28.0] | -3.8 [-14.0, +4.7] | 45 | 1.000 | 1: +0.0 | 6: -12.8 | 13: +0.0 |
| 0.10 | 80 | 20.3 [9.3, 33.6] | -1.7 [-10.7, +6.4] | 40 | 1.000 | 7: +14.0 | 6: -21.9 | 7: +0.0 |

### What the sweeps show

- Context order (same pages, target forced into each slot, n=19 queries): PAWC share slot 1 28.0%, slot 3 30.1%, slot 5 7.8%. Holm-significant differences: slot 5 vs 1: -20.2 [-31.9, -7.6] pp (Holm p 0.042).
- How fragile the slot 5 result is: its raw p is 0.011 over 19 queries, but those queries are served by only 4 target pages. Per page, slot 5 minus slot 1 is duo -18.0, dialing-in -17.7, mill-64 +15.7, solo -43.6 pp, and a Wilcoxon test on the page means gives p = 0.250. It also depends on the sweeps forming their own Holm family: in one family with the arms (11 tests) its Holm p would be 0.116.
- Retrieval weighting (final scores span 0.192 across the top 5 at the median; the 5th-to-6th gap is 0.036): w=0.02: target retrieval unchanged at 75%, ΔPAWC -3.8 [-14.0, +4.7] pp; w=0.10: target retrieval 75% → 80%, ΔPAWC -1.7 [-10.7, +6.4] pp. No boost has a Holm-significant effect.
- Noise floor: re-sampling the unchanged prompts (A/A) moved PAWC share by +2.7 [-5.6, +9.7] pp.
- Page and engine arms: 0 of 7 have a Holm-significant effect.
- Sensitivity (80% power, strictest Holm step; A/A per-query SD 18.3 pp; slot sweep n=19, boost sweep n=20 queries). Moving the target from slot 1 to slot 5: **detected**; small retrieval boosts (w ≤ 0.10): not detected (MDE ≈ 13.7 pp); page edits: **untestable at this design** (4–7 independent page units per arm; the minimum achievable Holm p is 0.109, above 0.05 at any effect size). Only the boosts and page edits are small changes; moving a source to the last slot is a large one. An effect the size of the MDE would be detected with 80% power; smaller ones could be missed. The sweep MDE uses the A/A per-query SD as its noise scale, so it is approximate for slot and boost comparisons, which have their own variance.
- Bandit, held out: the frozen contextual policy had regret 1.778 (±1.194) vs 1.970 for random and 2.508 for the best fixed arm chosen on the training half. On the held-out queries it is not distinguishable from random.

### Bandit replay

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

### Greedy optimization loop

Held-out queries: 10. The loop keeps an arm only when the lower 95% confidence bound on held-out ΔPAWC is above zero.

| Step | Proposed | Predicted pp | Held-out ΔPAWC pp [95% CI] | Decision | Applied so far | Held-out PAWC % |
|---|---|---|---|---|---|---|
| 1 | `internal_links` | +3.20 | +0.68 [-5.35, +7.15] | rejected | `none` | 23.7 |
| 2 | `metadata` | +0.77 | -4.58 [-11.63, +20.34] | rejected | `none` | 23.7 |


## FakeLLM (pipeline check)

- Results: `experiments/results/2026-09-24_fakellm` (git b9f509a, 2026-09-25)
- Answer model: `fake-extractive/v1/BAAI/bge-small-en-v1.5` (deterministic extractive stand-in; pipeline check, not model evidence)
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 40 queries x 5 samples, 22 pages

### Baseline visibility

| Domain | Retrieved % | Cited % | Cited when retrieved % | C-SoV % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment |
|---|---|---|---|---|---|---|---|---|
| crema-lab.example | 70 | 57 | 81 | 29.3 | 30.2 | 1.66 | 19 | +0.03 |
| beanbudget.example | 65 | 53 | 82 | 30.0 | 28.1 | 1.67 | 18 | +0.39 |
| brewline.example (target) | 80 | 52 | 64 | 18.2 | 19.5 | 2.25 | 36 | +0.06 |
| shotreport.example | 65 | 40 | 62 | 14.5 | 14.8 | 2.25 | 38 | +0.05 |
| grindandtamp.example | 35 | 24 | 70 | 8.0 | 7.4 | 2.57 | 30 | -0.12 |

### Sandbox arms (target: brewline.example)

An arm counts as having an effect when its Holm-adjusted Wilcoxon p-value is below 0.05 (`*`). The 95% confidence intervals are descriptive. `noop` and `aa_resample` are controls and are excluded from the Holm family. Page arms change each query's focus page. They are tested across (fold, page) units because queries that share an edited page are not independent; `n` reports queries / units. Arms that use the tracked queries are cross-fitted: built on one half of the queries and scored on the other. "ΔPAWC if cited" counts only answers with citations. Citation-format failures instead appear in the uncited rate.

| Arm | ΔPAWC pp [95% CI] | Rel Δ % | ΔC-SoV pp [95% CI] | Δ cited pp | ΔPAWC if cited pp | Uncited answers % | Δ sentiment | p (Holm) | n |
|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0 | +0.0 [+0.0, +0.0] | +0.0 | +0.0 | 0 | +0.000 | control | 40 / 5 |
| `aa_resample` | +2.6 [-0.0, +5.2] | +13 | +2.1 [+0.0, +4.3] | +2.0 | +2.6 | 0 | +0.011 | control | 40 / 40 |
| `metadata` | +0.6 [-1.4, +2.3] | +3 | +0.6 [-1.5, +2.4] | -2.5 | +0.6 | 0 | -0.007 | 1.000 | 40 / 10 |
| `faq_rewrite` | +1.3 [-2.2, +4.0] | +7 | +0.7 [-2.0, +2.8] | +2.5 | +1.3 | 0 | -0.010 | 1.000 | 40 / 10 |
| `jsonld_insert` | +0.0 [+0.0, +0.0] | +0 | +0.0 [+0.0, +0.0] | +0.0 | +0.0 | 0 | +0.000 | 1.000 | 40 / 5 |
| `internal_links` | -0.7 [-1.1, +0.0] | -3 | -0.8 [-1.6, +0.0] | -2.5 | -0.7 | 0 | -0.010 | 1.000 | 40 / 5 |
| `stats_surface` | -1.5 [-2.8, -0.6] | -8 | -1.1 [-1.8, +0.1] | -2.5 | -1.5 | 0 | -0.047 | 1.000 | 40 / 5 |
| `quote_surface` | -2.1 [-3.7, +0.0] | -11 | -1.7 [-3.1, +0.0] | -4.0 | -2.1 | 0 | -0.023 | 1.000 | 40 / 5 |
| `keyword_stuffing` | +1.7 [-0.6, +3.5] | +9 | +1.2 [-1.0, +2.9] | +2.0 | +1.7 | 0 | -0.006 | 1.000 | 40 / 10 |
| `engine:reverse` | +11.4 [+3.2, +19.7] | +59 | +9.7 [+2.9, +16.8] | +11.0 | +11.4 | 0 | +0.025 | 0.075 | 40 / 40 |

### Position sweep

| Target slot | PAWC share % [95% CI] | Δ vs slot 1 pp [95% CI] | Cited % | p (Holm, sweeps) | n |
|---|---|---|---|---|---|
| 1 | 40.2 [36.9, 43.4] | ref | 90 |  | 39 |
| 2 | 24.5 [22.0, 27.1] | -15.7 [-19.0, -12.4] | 73 | <0.001 * | 39 |
| 3 | 18.2 [14.9, 21.7] | -21.9 [-26.6, -17.2] | 59 | <0.001 * | 39 |
| 4 | 10.1 [7.7, 12.7] | -30.1 [-33.8, -26.2] | 39 | <0.001 * | 39 |
| 5 | 6.8 [5.4, 8.4] | -33.4 [-36.4, -30.2] | 29 | <0.001 * | 39 |

### Boost sweep

For each query, these classes compare the boosted source list with w=0: a different set of sources, the same sources in a different order, or no change. Each cell shows `n queries: mean ΔPAWC pp`.

| Boost w | Target retrieved % | PAWC share % [95% CI] | Δ vs w=0 pp [95% CI] | Cited % | p (Holm, sweeps) | Set changed | Order only | Unchanged |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 80 | 19.5 [14.0, 25.3] | ref | 52 |  |  |  |  |
| 0.02 | 80 | 24.9 [18.1, 31.7] | +5.4 [+2.5, +8.8] | 58 | 0.004 * | 4: +13.3 | 10: +16.2 | 26: +0.0 |
| 0.05 | 80 | 29.4 [22.1, 36.7] | +9.9 [+5.5, +14.7] | 61 | <0.001 * | 9: +22.5 | 11: +17.5 | 20: +0.0 |
| 0.10 | 88 | 35.7 [28.4, 42.9] | +16.2 [+10.9, +21.7] | 70 | <0.001 * | 14: +24.1 | 15: +20.8 | 11: +0.0 |

### What the sweeps show

- FakeLLM applies an explicit −0.05 per-slot position penalty; this sweep recovers that built-in prior and is not evidence about real models.

### Bandit replay

Offline replay uses rewards from the same table the policies learn from. It runs 2000 rounds x 20 runs. Regret is summed over rounds in PAWC share (1.0 = 100 pp). The hindsight row knows the best single arm in advance and serves as a reference, not a policy.

| Policy | Cumulative regret at T (±95% CI) | Mean reward pp/round | Most chosen arm |
|---|---|---|---|
| linucb(a=0.1) | 84.74 ± 4.28 | +2.31 | `keyword_stuffing` (22%) |
| linucb(a=0.5) | 77.08 ± 3.11 | +2.72 | `keyword_stuffing` (20%) |
| linucb(a=1) | 86.36 ± 2.39 | +2.27 | `keyword_stuffing` (18%) |
| lints(v=0.1) | 91.50 ± 4.24 | +1.95 | `keyword_stuffing` (21%) |
| eps-greedy(0.1) | 90.85 ± 3.72 | +1.99 | `faq_rewrite` (21%) |
| linucb-bias-only(a=0.1) | 112.52 ± 6.82 | +0.90 | `keyword_stuffing` (39%) |
| random | 132.50 ± 1.69 | -0.05 | `metadata` (13%) |
| best fixed arm (hindsight) | 95.20 ± 1.60 | +1.77 | `keyword_stuffing` (100%) |
| oracle | 0.00 ± 0.00 | +6.57 |  |

Held-out check: the policy is fit on fold-1 queries, then frozen and scored on fold-2 queries. Regret is summed over held-out queries, with one decision per query. Caveat: for the cross-fitted arms (metadata, FAQ, keyword stuffing) the fold-1 rewards were measured on pages built from fold-2 query text, so this is a split of contexts rather than a fully independent test.

| Policy | Held-out regret (±95% CI) | Mean reward pp | Most chosen arm |
|---|---|---|---|
| linucb (frozen, contextual) | 0.793 ± 0.473 | +3.23 | `faq_rewrite` (30%) |
| linucb-bias-only (frozen) | 0.852 ± 0.522 | +2.94 | `keyword_stuffing` (100%) |
| best fixed arm (chosen on train) | 0.852 ± 0.522 | +2.94 | `keyword_stuffing` (100%) |
| random (expected) | 1.420 ± n/a | +0.10 |  |
| oracle | 0.000 ± 0.000 | +7.20 |  |

### Greedy optimization loop

Held-out queries: 20. The loop keeps an arm only when the lower 95% confidence bound on held-out ΔPAWC is above zero.

| Step | Proposed | Predicted pp | Held-out ΔPAWC pp [95% CI] | Decision | Applied so far | Held-out PAWC % |
|---|---|---|---|---|---|---|
| 1 | `keyword_stuffing` | +0.40 | +2.94 [+0.02, +4.99] | kept | `keyword_stuffing` | 21.2 |
| 2 | `faq_rewrite` | +0.35 | -0.91 [-6.35, +1.75] | rejected | `keyword_stuffing` | 21.2 |
| 3 | `metadata` | +0.01 | +1.31 [-2.72, +3.57] | rejected | `keyword_stuffing` | 21.2 |
| 4 | `jsonld_insert` | +0.00 | +0.00 [+0.00, +0.00] | rejected | `keyword_stuffing` | 21.2 |
