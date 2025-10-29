# FakeLLM (pipeline check)

- Results: `experiments/results/2026-09-24_fakellm` (git 7230b9d, 2026-09-24)
- Answer model: `fake-extractive/v1/BAAI/bge-small-en-v1.5` (deterministic extractive stand-in: a pipeline check, not model evidence)
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 40 queries x 5 samples, 22 pages

## Baseline visibility

| Domain | Retrieved % | Cited % | Cited when retrieved % | C-SoV % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment |
|---|---|---|---|---|---|---|---|---|
| crema-lab.example | 70 | 57 | 81 | 28.8 | 29.7 | 1.68 | 19 | +0.03 |
| beanbudget.example | 65 | 52 | 81 | 29.8 | 28.0 | 1.69 | 19 | +0.39 |
| brewline.example (target) | 80 | 52 | 66 | 18.6 | 19.7 | 2.29 | 34 | +0.06 |
| shotreport.example | 65 | 40 | 62 | 14.5 | 14.8 | 2.25 | 38 | +0.05 |
| grindandtamp.example | 35 | 25 | 71 | 8.3 | 7.8 | 2.62 | 29 | -0.14 |

## Sandbox arms (target: brewline.example)

Test: Wilcoxon signed-rank on per-query deltas, Holm-adjusted across the non-control arms; `*` marks p(Holm) < 0.05. The bootstrap CI is descriptive. `noop` and `aa_resample` are controls outside the Holm family. Page arms that read the tracked queries are cross-fitted: built from one half of the queries, scored on the other. A/A noise band (re-sampled unchanged prompts): +2.3 [-0.2, +4.9] pp; 'Beyond A/A' says whether an arm's mean falls outside that band.

| Arm | ΔPAWC pp [95% CI] | Rel Δ % | ΔC-SoV pp [95% CI] | Δ cited pp | Δ retrieved pp | Δ sentiment | Uncited answers % | p (Holm) | Beyond A/A | n |
|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0 | +0.0 [+0.0, +0.0] | +0.0 | +0.0 | +0.000 | 0 | n/a |  | 40 |
| `aa_resample` | +2.3 [-0.2, +4.9] | +12 | +1.9 [-0.2, +4.2] | +1.5 | +0.0 | +0.020 | 0 | n/a |  | 40 |
| `metadata` | +0.6 [-1.5, +2.7] | +3 | +0.7 [-1.2, +2.8] | -3.0 | +0.0 | +0.001 | 0 | 1.000 | no | 40 |
| `faq_rewrite` | +1.0 [-1.4, +3.5] | +5 | +0.5 [-1.6, +2.6] | +1.5 | -2.5 | -0.002 | 0 | 1.000 | no | 40 |
| `jsonld_insert` | +0.0 [+0.0, +0.0] | +0 | +0.0 [+0.0, +0.0] | +0.0 | +0.0 | +0.000 | 0 | 1.000 | no | 40 |
| `internal_links` | -0.7 [-1.6, +0.0] | -3 | -0.8 [-2.1, +0.0] | -2.5 | +0.0 | -0.010 | 0 | 1.000 | yes | 40 |
| `stats_surface` | -1.7 [-4.1, +0.9] | -8 | -1.2 [-3.5, +1.0] | -2.5 | -2.5 | -0.023 | 0 | 1.000 | yes | 40 |
| `quote_surface` | -1.9 [-5.3, +0.5] | -10 | -1.6 [-4.6, +0.6] | -3.5 | +0.0 | -0.004 | 0 | 1.000 | yes | 40 |
| `keyword_stuffing` | +1.5 [-0.4, +4.0] | +8 | +1.2 [-0.6, +3.6] | +1.5 | -2.5 | +0.000 | 0 | 1.000 | no | 40 |
| `engine:reverse` | +11.4 [+3.2, +19.5] | +58 | +9.5 [+2.6, +16.4] | +10.0 | +0.0 | +0.022 | 0 | 0.047 * | yes | 40 |

## Position sweep

| Target slot | PAWC share % [95% CI] | Δ vs slot 1 pp [95% CI] | Cited % | p | n |
|---|---|---|---|---|---|
| 1 | 40.5 [37.1, 43.9] | ref | 90 |  | 39 |
| 2 | 24.2 [21.7, 26.8] | -16.3 [-19.7, -13.0] | 72 | <0.001 | 39 |
| 3 | 18.5 [15.2, 22.0] | -22.1 [-26.9, -17.1] | 61 | <0.001 | 39 |
| 4 | 10.3 [7.9, 12.9] | -30.2 [-34.0, -26.4] | 40 | <0.001 | 39 |
| 5 | 6.6 [5.2, 8.2] | -33.9 [-37.1, -30.6] | 29 | <0.001 | 39 |

## Boost sweep

Per-query classes compare the boosted source list with w=0: a different set of sources, the same sources in a different order, or no change. Each class cell is `n queries: mean ΔPAWC pp`.

| Boost w | Target retrieved % | PAWC share % [95% CI] | Δ vs w=0 pp [95% CI] | Cited % | p | Set changed | Order only | Unchanged |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 80 | 19.7 [14.2, 25.4] | ref | 52 |  |  |  |  |
| 0.02 | 80 | 25.1 [18.4, 31.8] | +5.4 [+2.6, +8.7] | 58 | 0.004 | 4: +13.3 | 10: +16.3 | 26: +0.0 |
| 0.05 | 80 | 29.4 [22.1, 36.6] | +9.6 [+5.3, +14.4] | 61 | <0.001 | 9: +22.4 | 11: +16.8 | 20: +0.0 |
| 0.10 | 88 | 36.0 [28.6, 43.2] | +16.3 [+10.8, +21.9] | 70 | <0.001 | 14: +24.9 | 15: +20.2 | 11: +0.0 |

## What the sweeps show

- FakeLLM applies an explicit −0.05 per-slot position penalty; this sweep recovers that built-in prior and is not evidence about real models.

## Bandit replay

Offline replay, 2000 rounds x 20 runs; regret is in units of PAWC share (1.0 = 100 pp) summed over rounds. The hindsight row knows the best single arm in advance, so it is a reference line, not a policy.

| Policy | Cumulative regret at T (±95% CI) | Mean reward pp/round | Most chosen arm |
|---|---|---|---|
| linucb(a=0.1) | 80.45 ± 5.03 | +2.15 | `keyword_stuffing` (26%) |
| linucb(a=0.5) | 75.36 ± 3.56 | +2.52 | `keyword_stuffing` (20%) |
| linucb(a=1) | 81.36 ± 2.32 | +2.07 | `keyword_stuffing` (18%) |
| lints(v=0.1) | 84.25 ± 3.10 | +2.03 | `keyword_stuffing` (20%) |
| eps-greedy(0.1) | 84.93 ± 4.88 | +2.06 | `faq_rewrite` (22%) |
| linucb-bias-only(a=0.1) | 107.25 ± 4.41 | +0.85 | `keyword_stuffing` (39%) |
| random | 127.71 ± 1.74 | -0.12 | `metadata` (13%) |
| best fixed arm (hindsight) | 93.30 ± 1.64 | +1.52 | `keyword_stuffing` (100%) |
| oracle | 0.00 ± 0.00 | +6.24 |  |

Held-out check: fit on fold-1 queries, freeze, score the frozen choices on fold-2 queries (regret summed over held-out queries, one decision each).

| Policy | Held-out regret | Mean reward pp | Most chosen arm |
|---|---|---|---|
| linucb (frozen, contextual) | 1.002 | +1.78 | `metadata` (25%) |
| linucb-bias-only (frozen) | 1.270 | +0.44 | `metadata` (100%) |
| best fixed arm (chosen on train) | 1.270 | +0.44 | `metadata` (100%) |
| random (expected) | 1.377 | -0.09 |  |
| oracle | 0.000 | +6.79 |  |

## Greedy optimization loop

Held-out queries: 20. An arm is kept only if the lower 95% bound on its held-out ΔPAWC is above zero.

| Step | Proposed | Predicted pp | Held-out ΔPAWC pp [95% CI] | Decision | Applied so far | Held-out PAWC % |
|---|---|---|---|---|---|---|
| 1 | `keyword_stuffing` | +0.57 | +2.5 [-0.5, +7.0] | rejected | `none` | 18.9 |
| 2 | `metadata` | +0.47 | +0.4 [-2.1, +2.8] | rejected | `none` | 18.9 |
| 3 | `faq_rewrite` | +0.12 | +1.8 [-2.0, +6.1] | rejected | `none` | 18.9 |
| 4 | `jsonld_insert` | +0.00 | +0.0 [+0.0, +0.0] | rejected | `none` | 18.9 |
