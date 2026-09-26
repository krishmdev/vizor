# qwen2.5:3b-instruct via Ollama, bench corpus (pre-registered main run)

- Results: `experiments/results/2026-09-05_bench-qwen3b` (git 0444d1e, 2026-09-05)
- Answer model: `qwen2.5:3b-instruct`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 72 queries x 5 samples, 72 pages
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 2530 new, 3230 cached (no API spend)

## Baseline visibility

Pooled marker share is the domain's share of all valid markers across every answer, so long, heavily cited answers count for more. It is not the per-answer C-SoV used as the primary metric in the arm tables, which averages each answer's own share.

| Domain | Retrieved % | Cited % | Cited when retrieved % | Pooled marker share % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment | Named % | Named, not cited % | Cited, not named % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| larkspur.example (target) | 93 | 69 | 75 | 50.8 | 42.5 | 3.58 | 25 | +0.15 | 59 | 16 | 27 |
| cogandchain.example | 56 | 29 | 52 | 15.8 | 14.1 | 3.37 | 48 | +0.14 | 24 | 11 | 16 |
| spokewise.example | 53 | 21 | 39 | 12.4 | 9.6 | 3.64 | 61 | +0.19 | 2 | 1 | 19 |
| pedalcheap.example | 56 | 16 | 28 | 11.1 | 7.4 | 4.07 | 72 | +0.39 | 0 | 0 | 16 |
| saddlesore.example | 54 | 19 | 34 | 10.0 | 6.2 | 4.93 | 66 | +0.11 | 0 | 0 | 19 |

## Sandbox arms (target: larkspur.example)

The verdict is the Holm-adjusted Wilcoxon p on the primary metric (citation share (C-SoV, share of the answer's valid markers)), below 0.05 (`*`). The brand-mention test (“named”: the answer names the target brand, with or without a citation) gets its own Holm adjustment. Content-only arms (`content:`) keep the baseline's sources and their order and change only the edited page's text; they form a separate Holm family. `noop` and `aa_resample` are controls outside every family. Page arms are tested on edited-page units (queries sharing an edited page are not independent); `n` is queries / units. The 95% CIs resample those units and are descriptive. The last four columns describe the arm's answers: no valid citation at all, citation-like text the parser could not map to a source, and every citation collected on the final sentence (for those answers PAWC's position weighting is meaningless). “Queries whose sources changed” counts queries where the arm changed the list of sources the model saw.

| Arm | ΔC-SoV pp [95% CI] (primary) | ΔPAWC pp [95% CI] | Δnamed pp [95% CI] | p (Holm, primary) | p (Holm, named) | Queries whose sources changed | Uncited % | Unparsed markers % | Cites only on last sentence % | n |
|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | control | control | 0 | 20 | 0.0 | 41 | 72 / 24 |
| `aa_resample` | +0.2 [-3.7, +4.4] | +0.7 [-3.4, +5.1] | -1.4 [-5.3, +2.5] | control | control |  | 18 | 0.3 | 40 | 72 / 72 |
| `metadata` | -2.3 [-7.0, +2.6] | -2.7 [-7.7, +2.7] | +1.1 [-2.3, +4.1] | 1.000 | 1.000 | 5 | 23 | 0.0 | 39 | 72 / 24 |
| `faq_rewrite` | -15.4 [-21.3, -9.2] | -15.7 [-21.8, -9.5] | -8.3 [-15.1, -2.1] | 0.003 * | 0.036 * | 10 | 27 | 0.0 | 40 | 72 / 24 |
| `jsonld_insert` | -3.2 [-6.4, +0.2] | -2.6 [-6.3, +1.3] | +4.4 [+0.0, +9.0] | 0.380 | 1.000 | 1 | 22 | 0.0 | 39 | 72 / 24 |
| `internal_links` | -5.2 [-9.5, -0.7] | -5.3 [-9.9, -0.5] | -2.5 [-4.8, +0.0] | 0.097 | 0.142 | 0 | 21 | 0.0 | 43 | 72 / 24 |
| `stats_surface` | -0.5 [-5.4, +4.6] | +0.1 [-5.0, +5.4] | +1.4 [-2.5, +5.4] | 1.000 | 1.000 | 25 | 17 | 0.0 | 42 | 72 / 24 |
| `keyword_stuffing` | +1.7 [-2.7, +6.4] | +1.9 [-2.6, +6.8] | -1.1 [-4.3, +1.9] | 1.000 | 1.000 | 2 | 18 | 0.3 | 42 | 72 / 24 |
| `content:metadata` | -2.4 [-6.5, +1.8] | -2.7 [-7.2, +1.8] | +0.0 [-3.3, +3.1] | 0.992 | 1.000 | 0 | 23 | 0.0 | 39 | 72 / 24 |
| `content:faq_rewrite` | -15.3 [-20.3, -9.8] | -15.7 [-20.7, -10.2] | -7.8 [-14.5, -1.8] | 0.003 * | 0.043 * | 0 | 27 | 0.0 | 42 | 72 / 24 |
| `content:jsonld_insert` | -3.5 [-6.7, +0.0] | -2.9 [-6.6, +1.1] | +3.1 [-1.4, +7.7] | 0.315 | 1.000 | 0 | 22 | 0.0 | 39 | 72 / 24 |
| `content:internal_links` | -5.2 [-9.5, -0.7] | -5.3 [-9.9, -0.5] | -2.5 [-4.8, +0.0] | 0.097 | 0.142 | 0 | 21 | 0.0 | 43 | 72 / 24 |
| `content:stats_surface` | +0.8 [-3.1, +4.9] | +1.4 [-2.5, +5.6] | +0.8 [-4.0, +5.5] | 0.992 | 1.000 | 0 | 18 | 0.3 | 42 | 72 / 24 |
| `content:keyword_stuffing` | +0.6 [-3.4, +4.9] | +0.8 [-3.3, +5.2] | -0.6 [-3.9, +2.4] | 0.992 | 1.000 | 0 | 18 | 0.3 | 41 | 72 / 24 |

## Content vs rank

Each page edit split into what the new text did with the same sources in the same order (content-only arm) and what it did by changing retrieval (full arm minus content-only arm, paired per query). Total = content + rank-mediated. The p next to the rank-mediated effect is an unadjusted Wilcoxon on page units.

| Page edit | Total ΔC-SoV pp | Content-only ΔC-SoV pp | Rank-mediated ΔC-SoV pp | Total Δnamed pp | Content-only Δnamed pp | Rank-mediated Δnamed pp | Queries whose sources changed |
|---|---|---|---|---|---|---|---|
| `metadata` | -2.3 [-7.0, +2.6] | -2.4 [-6.5, +1.8] | +0.1 [-1.7, +2.4] (p 0.738) | +1.1 [-2.3, +4.1] | +0.0 [-3.3, +3.1] | +1.1 [+0.0, +2.4] | 5 |
| `faq_rewrite` | -15.4 [-21.3, -9.2] | -15.3 [-20.3, -9.8] | -0.1 [-2.2, +1.9] (p 0.964) | -8.3 [-15.1, -2.1] | -7.8 [-14.5, -1.8] | -0.6 [-1.7, +0.0] | 10 |
| `jsonld_insert` | -3.2 [-6.4, +0.2] | -3.5 [-6.7, +0.0] | +0.3 [+0.0, +0.9] (p 0.700) | +4.4 [+0.0, +9.0] | +3.1 [-1.4, +7.7] | +1.4 [+0.0, +4.3] | 1 |
| `internal_links` | -5.2 [-9.5, -0.7] | -5.2 [-9.5, -0.7] | +0.0 [+0.0, +0.0] (p 1.000) | -2.5 [-4.8, +0.0] | -2.5 [-4.8, +0.0] | +0.0 [+0.0, +0.0] | 0 |
| `stats_surface` | -0.5 [-5.4, +4.6] | +0.8 [-3.1, +4.9] | -1.3 [-5.1, +1.9] (p 0.920) | +1.4 [-2.5, +5.4] | +0.8 [-4.0, +5.5] | +0.6 [-1.7, +2.8] | 25 |
| `keyword_stuffing` | +1.7 [-2.7, +6.4] | +0.6 [-3.4, +4.9] | +1.1 [+0.0, +2.9] (p 0.458) | -1.1 [-4.3, +1.9] | -0.6 [-3.9, +2.4] | -0.6 [-1.6, +0.0] | 2 |

## Position sweep

| Target slot | C-SoV % | PAWC share % [95% CI] | ΔC-SoV vs slot 1 pp [95% CI] | Cited % | p (Holm, sweeps) | n |
|---|---|---|---|---|---|---|
| 1 | 43.1 | 43.8 [35.0, 52.6] | ref | 72 |  | 36 |
| 5 | 13.0 | 12.6 [5.7, 20.9] | -30.1 [-39.9, -20.5] | 22 | <0.001 * | 36 |

## What the sweeps show

- Context order (same pages, target forced into each slot, n=36 queries): C-SoV slot 1 43.1%, slot 5 13.0%. Holm-significant differences: slot 5 vs 1: -30.1 [-39.9, -20.5] pp (Holm p <0.001).
- How fragile the slot 5 result is: its raw p is <0.001 over 36 queries, but those queries are served by only 17 target pages. Per page, slot 5 minus slot 1 on C-SoV is beam-800 -51.6, floor-pump -90.0, fold-20 -33.4, gravel-gx -18.3, chain-care -46.4, first-commute -10.0, flat-tire -7.5, tire-pressure +1.3, haul -61.1, lock-d9 -41.4, metro-7 -54.3, pannier-20 -40.0, shell +8.7, spin-t2 +16.9, sprout-16 -30.8, gear-indexing -2.0, volt-e1 -13.1 pp, and a Wilcoxon test on the page means gives p = 0.001. It also depends on the sweeps forming their own Holm family: in one family with the arms (13 tests) its Holm p would be <0.001.
- Noise floor: re-sampling the unchanged prompts (A/A) moved C-SoV by +0.2 [-3.7, +4.4] pp on 72 query units and the named rate by -1.4 [-5.3, +2.5] pp.
- Page and engine arms: 1 of 6 have a Holm-significant effect on C-SoV: `faq_rewrite`; 1 of 6 on the named rate: `faq_rewrite`.
- Content-only arms: 1 of 6 have a Holm-significant effect on C-SoV: `content:faq_rewrite`; 1 of 6 on the named rate: `content:faq_rewrite`.
- Content vs rank: for 2 of 6 page edits the rank-mediated part of the C-SoV change is larger in size than the content-only part (point estimates; see the decomposition table for intervals).
- Weighting: the C-SoV estimates and intervals above weight every query equally, while the Wilcoxon test ranks unweighted page means. The page means are `metadata` -1.8, `faq_rewrite` -14.1, `jsonld_insert` -2.5, `internal_links` -4.6, `stats_surface` 0.6, `keyword_stuffing` 2.5, `content:metadata` -2.1, `content:faq_rewrite` -13.7, `content:jsonld_insert` -2.7, `content:internal_links` -4.6, `content:stats_surface` 1.6, `content:keyword_stuffing` 1.5 pp.
- Minimum detectable effect on C-SoV (80% power, strictest Holm step, normal approximation, from the A/A re-sample; per-query A/A SD 17.7 pp): page edits ≈ 7.9 pp (8.5 pp with t quantiles) on 24 underlying page units (both cross-fit folds clustered by page); content-only arms ≈ 7.9 pp; named rate for page edits ≈ 7.6 pp; slot sweep ≈ 8.3 pp. These are approximate: the test is a Wilcoxon signed-rank test, not a t test. Effects smaller than these could be missed.
- Positive control (target moved from slot 1 to slot 5, same pages): C-SoV -30.1 [-39.9, -20.5] pp, Holm p <0.001 *.
- Exploratory, not pre-registered: on the held-out queries the frozen contextual bandit policy had regret 6.268 (±2.233) vs 7.705 for random and 5.772 for the best fixed arm chosen on the training half. It is not distinguishable from random.

## Bandit replay

Offline replay uses rewards from the same table the policies learn from. It runs 2000 rounds x 20 runs. Regret is summed over rounds in citation share (1.0 = 100 pp). The hindsight row knows the best single arm in advance and serves as a reference, not a policy.

| Policy | Cumulative regret at T (±95% CI) | Mean reward pp/round | Most chosen arm |
|---|---|---|---|
| linucb(a=0.1) | 308.01 ± 7.46 | +1.32 | `noop` (57%) |
| linucb(a=0.5) | 307.28 ± 8.29 | +1.51 | `noop` (31%) |
| linucb(a=1) | 321.87 ± 5.22 | +0.54 | `noop` (24%) |
| lints(v=0.1) | 312.00 ± 5.98 | +1.18 | `noop` (43%) |
| eps-greedy(0.1) | 313.11 ± 4.95 | +1.18 | `noop` (38%) |
| linucb-bias-only(a=0.1) | 335.61 ± 6.37 | +0.01 | `noop` (83%) |
| random | 408.53 ± 3.28 | -3.73 | `metadata` (15%) |
| best fixed arm (hindsight) | 305.85 ± 2.90 | +1.31 | `keyword_stuffing` (100%) |
| oracle | 0.00 ± 0.00 | +16.93 |  |

Held-out check: the policy is fit on fold-1 queries, then frozen and scored on fold-2 queries. Regret is summed over held-out queries, with one decision per query. Caveat: for the cross-fitted arms (metadata, FAQ, keyword stuffing) the fold-1 rewards were measured on pages built from fold-2 query text, so this is a split of contexts rather than a fully independent test.

| Policy | Held-out regret (±95% CI) | Mean reward pp | Most chosen arm |
|---|---|---|---|
| linucb (frozen, contextual) | 6.268 ± 2.233 | +1.56 | `keyword_stuffing` (42%) |
| linucb-bias-only (frozen) | 5.772 ± 1.985 | +2.94 | `keyword_stuffing` (100%) |
| best fixed arm (chosen on train) | 5.772 ± 1.985 | +2.94 | `keyword_stuffing` (100%) |
| random (expected) | 7.705 ± n/a | -2.43 |  |
| oracle | 0.000 ± 0.000 | +18.97 |  |
