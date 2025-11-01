# FakeLLM (pipeline check)

- Results: `experiments/results/2026-09-24_fakellm` (git b9f509a, 2026-09-25)
- Answer model: `fake-extractive/v1/BAAI/bge-small-en-v1.5` (deterministic extractive stand-in; pipeline check, not model evidence)
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 40 queries x 5 samples, 22 pages

## Baseline visibility

| Domain | Retrieved % | Cited % | Cited when retrieved % | C-SoV % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment |
|---|---|---|---|---|---|---|---|---|
| crema-lab.example | 70 | 57 | 81 | 29.3 | 30.2 | 1.66 | 19 | +0.03 |
| beanbudget.example | 65 | 53 | 82 | 30.0 | 28.1 | 1.67 | 18 | +0.39 |
| brewline.example (target) | 80 | 52 | 64 | 18.2 | 19.5 | 2.25 | 36 | +0.06 |
| shotreport.example | 65 | 40 | 62 | 14.5 | 14.8 | 2.25 | 38 | +0.05 |
| grindandtamp.example | 35 | 24 | 70 | 8.0 | 7.4 | 2.57 | 30 | -0.12 |

## Sandbox arms (target: brewline.example)

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

## Position sweep

| Target slot | PAWC share % [95% CI] | Δ vs slot 1 pp [95% CI] | Cited % | p (Holm, sweeps) | n |
|---|---|---|---|---|---|
| 1 | 40.2 [36.9, 43.4] | ref | 90 |  | 39 |
| 2 | 24.5 [22.0, 27.1] | -15.7 [-19.0, -12.4] | 73 | <0.001 * | 39 |
| 3 | 18.2 [14.9, 21.7] | -21.9 [-26.6, -17.2] | 59 | <0.001 * | 39 |
| 4 | 10.1 [7.7, 12.7] | -30.1 [-33.8, -26.2] | 39 | <0.001 * | 39 |
| 5 | 6.8 [5.4, 8.4] | -33.4 [-36.4, -30.2] | 29 | <0.001 * | 39 |

## Boost sweep

For each query, these classes compare the boosted source list with w=0: a different set of sources, the same sources in a different order, or no change. Each cell shows `n queries: mean ΔPAWC pp`.

| Boost w | Target retrieved % | PAWC share % [95% CI] | Δ vs w=0 pp [95% CI] | Cited % | p (Holm, sweeps) | Set changed | Order only | Unchanged |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 80 | 19.5 [14.0, 25.3] | ref | 52 |  |  |  |  |
| 0.02 | 80 | 24.9 [18.1, 31.7] | +5.4 [+2.5, +8.8] | 58 | 0.004 * | 4: +13.3 | 10: +16.2 | 26: +0.0 |
| 0.05 | 80 | 29.4 [22.1, 36.7] | +9.9 [+5.5, +14.7] | 61 | <0.001 * | 9: +22.5 | 11: +17.5 | 20: +0.0 |
| 0.10 | 88 | 35.7 [28.4, 42.9] | +16.2 [+10.9, +21.7] | 70 | <0.001 * | 14: +24.1 | 15: +20.8 | 11: +0.0 |

## What the sweeps show

- FakeLLM applies an explicit −0.05 per-slot position penalty; this sweep recovers that built-in prior and is not evidence about real models.

## Bandit replay

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

Held-out check: the policy is fit on fold-1 queries, then frozen and scored on fold-2 queries. Regret is summed over held-out queries, with one decision per query.

| Policy | Held-out regret | Mean reward pp | Most chosen arm |
|---|---|---|---|
| linucb (frozen, contextual) | 0.793 | +3.23 | `faq_rewrite` (30%) |
| linucb-bias-only (frozen) | 0.852 | +2.94 | `keyword_stuffing` (100%) |
| best fixed arm (chosen on train) | 0.852 | +2.94 | `keyword_stuffing` (100%) |
| random (expected) | 1.420 | +0.10 |  |
| oracle | 0.000 | +7.20 |  |

## Greedy optimization loop

Held-out queries: 20. The loop keeps an arm only when the lower 95% confidence bound on held-out ΔPAWC is above zero.

| Step | Proposed | Predicted pp | Held-out ΔPAWC pp [95% CI] | Decision | Applied so far | Held-out PAWC % |
|---|---|---|---|---|---|---|
| 1 | `keyword_stuffing` | +0.40 | +2.9 [+0.0, +5.0] | kept | `keyword_stuffing` | 21.2 |
| 2 | `faq_rewrite` | +0.35 | -0.9 [-6.3, +1.7] | rejected | `keyword_stuffing` | 21.2 |
| 3 | `metadata` | +0.01 | +1.3 [-2.7, +3.6] | rejected | `keyword_stuffing` | 21.2 |
| 4 | `jsonld_insert` | +0.00 | +0.0 [+0.0, +0.0] | rejected | `keyword_stuffing` | 21.2 |
