# qwen2.5:3b-instruct via Ollama (small local model, reduced design)

- Results: `experiments/results/2026-09-23_qwen2.5-3b` (git 7511cb0, 2026-09-23)
- Answer model: `qwen2.5:3b-instruct`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 20 queries x 2 samples, 22 pages
- A small local model, not a GPT-class engine. It needed an extra system message (recorded in the manifest) before it would cite sentence by sentence.
- Local calls: 422 new, 218 cached (no API spend)

## Baseline visibility

| Domain | Retrieved % | Cited % | Cited when retrieved % | C-SoV % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment |
|---|---|---|---|---|---|---|---|---|
| crema-lab.example | 75 | 70 | 93 | 34.8 | 38.9 | 1.89 | 7 | +0.04 |
| brewline.example (target) | 75 | 60 | 80 | 18.4 | 26.9 | 2.25 | 20 | +0.16 |
| shotreport.example | 55 | 40 | 73 | 31.9 | 21.5 | 2.06 | 27 | +0.33 |
| beanbudget.example | 65 | 12 | 19 | 10.1 | 6.6 | 2.20 | 81 | +0.36 |
| grindandtamp.example | 40 | 12 | 31 | 4.8 | 3.6 | 2.00 | 69 | -0.19 |

## Sandbox arms (target: brewline.example)

Test: Wilcoxon signed-rank on per-query deltas, Holm-adjusted across the non-control arms; `*` marks p(Holm) < 0.05. The bootstrap CI is descriptive. `noop` and `aa_resample` are controls outside the Holm family. Page arms that read the tracked queries are cross-fitted: built from one half of the queries, scored on the other. A/A noise band (re-sampled unchanged prompts): -3.4 [-11.3, +5.2] pp; 'Beyond A/A' says whether an arm's mean falls outside that band.

| Arm | ΔPAWC pp [95% CI] | Rel Δ % | ΔC-SoV pp [95% CI] | Δ cited pp | Δ retrieved pp | Δ sentiment | Uncited answers % | p (Holm) | Beyond A/A | n |
|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0 | +0.0 [+0.0, +0.0] | +0.0 | +0.0 | +0.000 | 2 | n/a |  | 20 |
| `aa_resample` | -3.4 [-11.3, +5.2] | -12 | -3.7 [-11.8, +5.1] | -7.5 | +0.0 | +0.102 | 0 | n/a |  | 20 |
| `metadata` | +0.9 [-6.3, +8.6] | +3 | +0.8 [-6.0, +7.7] | +2.5 | +0.0 | +0.070 | 2 | 1.000 | no | 20 |
| `faq_rewrite` | -7.7 [-17.8, +4.2] | -29 | -9.5 [-19.2, +0.8] | -20.0 | +0.0 | +0.019 | 8 | 0.318 | no | 20 |
| `jsonld_insert` | +1.9 [-6.6, +10.1] | +7 | +0.4 [-8.4, +8.6] | +2.5 | +0.0 | +0.112 | 0 | 1.000 | no | 20 |
| `internal_links` | -7.0 [-14.8, +0.7] | -26 | -8.9 [-15.7, -1.7] | -10.0 | +0.0 | +0.022 | 2 | 0.574 | no | 20 |
| `stats_surface` | -6.6 [-16.6, +3.3] | -25 | -7.3 [-16.8, +2.0] | -10.0 | +0.0 | +0.021 | 0 | 1.000 | no | 20 |
| `keyword_stuffing` | -0.7 [-5.5, +4.3] | -3 | -3.2 [-8.9, +2.3] | -2.5 | +0.0 | +0.068 | 2 | 1.000 | no | 20 |
| `engine:reverse` | +1.4 [-10.3, +12.5] | +5 | -1.0 [-12.9, +10.2] | -12.5 | +0.0 | +0.033 | 0 | 1.000 | no | 20 |

## Position sweep

| Target slot | PAWC share % [95% CI] | Δ vs slot 1 pp [95% CI] | Cited % | p | n |
|---|---|---|---|---|---|
| 1 | 21.9 [13.0, 31.6] | ref | 58 |  | 19 |
| 3 | 33.6 [23.1, 44.8] | +11.7 [-4.1, +27.8] | 66 | 0.220 | 19 |
| 5 | 12.1 [6.1, 18.3] | -9.8 [-21.9, +1.8] | 37 | 0.116 | 19 |

## Boost sweep

Per-query classes compare the boosted source list with w=0: a different set of sources, the same sources in a different order, or no change. Each class cell is `n queries: mean ΔPAWC pp`.

| Boost w | Target retrieved % | PAWC share % [95% CI] | Δ vs w=0 pp [95% CI] | Cited % | p | Set changed | Order only | Unchanged |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 75 | 26.9 [17.6, 35.8] | ref | 60 |  |  |  |  |
| 0.02 | 75 | 23.1 [13.4, 33.2] | -3.8 [-9.9, +2.1] | 50 | 0.289 | 1: -31.6 | 6: -7.4 | 13: +0.0 |
| 0.10 | 80 | 28.6 [17.1, 41.3] | +1.7 [-7.8, +11.7] | 58 | 0.821 | 7: +11.6 | 6: -7.9 | 7: +0.0 |

## What the sweeps show

- Context order: with identical page content, moving the target from slot 1 to slot 5 changed its PAWC share from 21.9% to 12.1% (Δ -9.8 [-21.9, +1.8] pp, 1.8x, n=19 queries).
- Retrieval weighting: the median spread of final scores across the top 5 was 0.192 and the median gap between the 5th and 6th candidate 0.036. A boost of 0.02 moved target retrieval from 75% to 75% and PAWC share by -3.8 [-9.9, +2.1] pp. No boost in the sweep moved PAWC share with a CI excluding zero.
- Noise floor: re-sampling the unchanged prompts (A/A) moved PAWC share by -3.4 [-11.3, +5.2] pp.
- Metadata only (title + description): +0.9 [-6.3, +8.6] pp, Holm p 1.000.

## Bandit replay

Offline replay, 2000 rounds x 20 runs; regret is in units of PAWC share (1.0 = 100 pp) summed over rounds. The hindsight row knows the best single arm in advance, so it is a reference line, not a policy.

| Policy | Cumulative regret at T (±95% CI) | Mean reward pp/round | Most chosen arm |
|---|---|---|---|
| linucb(a=0.1) | 221.84 ± 14.82 | +5.10 | `noop` (38%) |
| linucb(a=0.5) | 196.12 ± 13.86 | +6.28 | `noop` (20%) |
| linucb(a=1) | 199.39 ± 12.90 | +6.15 | `jsonld_insert` (20%) |
| lints(v=0.1) | 193.22 ± 15.37 | +6.41 | `noop` (23%) |
| eps-greedy(0.1) | 220.29 ± 13.62 | +5.13 | `noop` (26%) |
| linucb-bias-only(a=0.1) | 319.87 ± 5.84 | +0.16 | `noop` (71%) |
| random | 380.45 ± 5.14 | -2.87 | `metadata` (15%) |
| best fixed arm (hindsight) | 286.61 ± 2.84 | +1.87 | `jsonld_insert` (100%) |
| oracle | 0.00 ± 0.00 | +16.24 |  |

Held-out check: fit on fold-1 queries, freeze, score the frozen choices on fold-2 queries (regret summed over held-out queries, one decision each).

| Policy | Held-out regret | Mean reward pp | Most chosen arm |
|---|---|---|---|
| linucb (frozen, contextual) | 2.161 | -1.79 | `keyword_stuffing` (40%) |
| linucb-bias-only (frozen) | 2.339 | -3.57 | `keyword_stuffing` (100%) |
| best fixed arm (chosen on train) | 2.339 | -3.57 | `keyword_stuffing` (100%) |
| random (expected) | 1.919 | +0.63 |  |
| oracle | 0.000 | +19.82 |  |

## Greedy optimization loop

Held-out queries: 10. An arm is kept only if the lower 95% bound on its held-out ΔPAWC is above zero.

| Step | Proposed | Predicted pp | Held-out ΔPAWC pp [95% CI] | Decision | Applied so far | Held-out PAWC % |
|---|---|---|---|---|---|---|
| 1 | `keyword_stuffing` | +3.77 | -3.6 [-8.0, +1.1] | rejected | `none` | 22.8 |
| 2 | `jsonld_insert` | -0.60 | +6.0 [-4.5, +17.8] | rejected | `none` | 22.8 |
