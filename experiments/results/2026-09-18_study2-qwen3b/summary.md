# Study 2: qwen2.5-3b-mlx4 via Localhost AI, bench corpus, Study 2 queries (pre-registered)

- Results: `experiments/results/2026-09-18_study2-qwen3b` (git c07c57b, 2026-09-18)
- Answer model: `qwen2.5-3b-mlx4`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `vader/vaderSentiment-3.3.2/compound`; PAWC decay: paper
- 72 queries x 3 samples, 72 pages
- Served by server `localhost-ai`, preset `qwen2.5-3b-mlx4`, commit `c61e05d4992d6cb68c7c16fc986eeb84dda74c5f` at `http://127.0.0.1:8431/v1`
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 1179 new, 333 cached (no API spend)

## Baseline visibility

Pooled marker share is the domain's share of all valid markers across every answer, so long, heavily cited answers count for more. It is not the per-answer C-SoV used as the primary metric in the arm tables, which averages each answer's own share.

| Domain | Retrieved % | Cited % | Cited when retrieved % | Pooled marker share % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment | Named % | Named, not cited % | Cited, not named % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| larkspur.example (target) | 99 | 93 | 94 | 49.6 | 60.6 | 1.54 | 6 | +0.10 | 56 | 3 | 40 |
| cogandchain.example | 69 | 41 | 59 | 18.1 | 17.2 | 2.68 | 41 | +0.14 | 23 | 3 | 21 |
| spokewise.example | 53 | 22 | 42 | 9.3 | 8.5 | 2.60 | 58 | +0.13 | 1 | 0 | 21 |
| saddlesore.example | 51 | 26 | 50 | 18.6 | 7.4 | 3.05 | 50 | +0.06 | 0 | 0 | 26 |
| pedalcheap.example | 46 | 14 | 31 | 4.4 | 5.1 | 2.32 | 69 | +0.22 | 0 | 0 | 14 |

## Sandbox arms (target: larkspur.example)

The verdict follows `sampled_primary.json`: AP failed its validation gate on criteria 3 and 4, so the primary is the sampled citation share (C-SoV) on 24 page units: an arm has an effect (`*`) when the exact Wilcoxon and the sign-flip p are both below 0.05 after Holm over the 3 arms. The ΔC-SoV column, its intervals and the `n` units come from that file (page units, A/A included; the A/A's ΔPAWC and Δnamed stay on its query units). The Study 1 column is the Holm-adjusted Wilcoxon (zero_method zsplit) that decided Study 1; it is shown for continuity and decides nothing here. The brand-mention test (“named”: the answer names the target brand, with or without a citation) gets its own Holm adjustment. `noop` and `aa_resample` are controls outside every family. Page arms are tested on edited-page units (queries sharing an edited page are not independent); `n` is queries / units. The 95% CIs resample those units and are descriptive. The last four columns describe the arm's answers: no valid citation at all, citation-like text the parser could not map to a source, and every citation collected on the final sentence (for those answers PAWC's position weighting is meaningless). “Queries whose sources changed” counts queries where the arm changed the list of sources the model saw.

| Arm | ΔC-SoV pp [95% CI] (primary) | ΔPAWC pp [95% CI] | Δnamed pp [95% CI] | p (Holm, primary: Wilcoxon / sign-flip) | Study 1 test (continuity, not this study's rule) | p (Holm, named) | Queries whose sources changed | Uncited % | Unparsed markers % | Cites only on last sentence % | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | control | control | control | 0 | 1 | 0.0 | 5 | 72 / 24 |
| `aa_resample` | -3.4 [-6.6, -0.5] | -3.3 [-6.9, +0.3] | +4.6 [-0.0, +9.7] | control | control | control |  | 1 | 1.4 | 6 | 72 / 24 |
| `answer_first` | -9.5 [-15.8, -3.8] | -10.3 [-16.5, -4.4] | +2.1 [-3.1, +7.4] | 0.008 / 0.005 * | 0.011 | 0.514 | 29 | 0 | 0.9 | 4 | 72 / 24 |
| `evidence_surface_llm` | -4.3 [-9.1, -0.4] | -4.5 [-9.4, -0.7] | +4.6 [+0.8, +8.7] | 0.046 / 0.044 * | 0.052 | 0.133 | 4 | 1 | 0.0 | 5 | 72 / 24 |
| `faq_rewrite_v2` | -10.0 [-15.8, -4.8] | -10.1 [-16.2, -4.6] | -2.8 [-6.9, +1.0] | 0.004 / 0.003 * | 0.011 | 0.299 | 2 | 2 | 0.5 | 8 | 72 / 24 |

## Position sweep

| Target slot | C-SoV % | PAWC share % [95% CI] | ΔC-SoV vs slot 1 pp [95% CI] | Cited % | p (Holm, sweeps) | n |
|---|---|---|---|---|---|---|
| 1 | 55.4 | 58.7 [49.2, 67.9] | ref | 90 |  | 36 |
| 5 | 28.9 | 28.2 [19.3, 37.6] | -26.6 [-36.6, -15.7] | 48 | <0.001 * | 36 |

## What the sweeps show

- Context order (same pages, target forced into each slot, n=36 queries): C-SoV slot 1 55.4%, slot 5 28.9%. Holm-significant differences: slot 5 vs 1: -26.6 [-36.6, -15.7] pp (Holm p <0.001).
- How fragile the slot 5 result is: its raw p is <0.001 over 36 queries, but those queries are served by only 16 target pages. Per page, slot 5 minus slot 1 on C-SoV is beam-800 -51.9, floor-pump -44.7, fold-20 -15.1, gravel-gx +11.8, bike-theft -28.9, chain-care -16.7, tire-pressure -16.7, haul -67.5, lock-d9 -27.8, metro-7 -0.7, pannier-20 -38.9, shell -23.5, spin-t2 -12.0, sprout-16 -40.3, wheel-truing -6.7, volt-e1 -32.5 pp, and a Wilcoxon test on the page means gives p = <0.001. It also depends on the sweeps forming their own Holm family: in one family with the arms (4 tests) its Holm p would be <0.001.
- Noise floor: re-sampling the unchanged prompts (A/A) moved C-SoV by -3.4 [-6.6, -0.5] pp on 24 page units (raw sign-flip p 0.045) and the named rate by +4.6 [-0.0, +9.7] pp.
- Page and engine arms: 3 of 3 have an effect under the sampled primary rule on C-SoV: `answer_first`, `evidence_surface_llm`, `faq_rewrite_v2`; 0 of 3 on the named rate.
- Exploratory, not pre-registered (the reference choice, `sampled_primary.md`): against the A/A re-sample `answer_first` -6.1 (Holm p 0.137 / 0.140), `evidence_surface_llm` -0.9 (Holm p 0.812 / 0.742), `faq_rewrite_v2` -6.6 (Holm p 0.137 / 0.140); against the mean of baseline and A/A `answer_first` -7.8 (Holm p 0.029 / 0.020), `evidence_surface_llm` -2.6 (Holm p 0.406 / 0.287), `faq_rewrite_v2` -8.3 (Holm p 0.029 / 0.020). A lucky baseline draw shifts every arm the same way; an arm that holds up only against the baseline alone is not robust to that choice.
- Weighting: the C-SoV estimates, intervals and tests above all use unweighted page means.
- Minimum detectable effect on C-SoV (80% power, strictest Holm step, normal approximation, from the A/A re-sample; per-query A/A SD 16.7 pp): page edits ≈ 5.3 pp (5.6 pp with t quantiles) on 24 underlying page units (both cross-fit folds clustered by page); named rate for page edits ≈ 10.0 pp; slot sweep ≈ 7.8 pp. These are approximate: the test is a Wilcoxon signed-rank test, not a t test. Effects smaller than these could be missed.
- Positive control (target moved from slot 1 to slot 5, same pages): C-SoV -26.6 [-36.6, -15.7] pp, Holm p <0.001 *.
- Bandit, held out: the frozen contextual policy had regret 4.553 (±1.964) vs 4.762 for random and 2.502 for the best fixed arm chosen on the training half. On the held-out queries it is not distinguishable from random.

## Bandit replay

Offline replay uses rewards from the same table the policies learn from. It runs 2000 rounds x 20 runs. Regret is summed over rounds in citation share (1.0 = 100 pp). The hindsight row knows the best single arm in advance and serves as a reference, not a policy.

| Policy | Cumulative regret at T (±95% CI) | Mean reward pp/round | Most chosen arm |
|---|---|---|---|
| linucb(a=0.1) | 163.18 ± 3.06 | -0.10 | `noop` (89%) |
| linucb(a=0.5) | 173.86 ± 3.99 | -0.58 | `noop` (67%) |
| linucb(a=1) | 190.49 ± 4.92 | -1.48 | `noop` (53%) |
| lints(v=0.1) | 168.46 ± 3.18 | -0.40 | `noop` (82%) |
| eps-greedy(0.1) | 175.72 ± 2.63 | -0.62 | `noop` (75%) |
| linucb-bias-only(a=0.1) | 163.21 ± 2.59 | -0.07 | `noop` (99%) |
| random | 275.26 ± 3.35 | -5.57 | `noop` (25%) |
| best fixed arm (hindsight) | 161.74 ± 2.42 | +0.00 | `noop` (100%) |
| oracle | 0.00 ± 0.00 | +8.05 |  |

Held-out check: the policy is fit on fold-1 queries, then frozen and scored on fold-2 queries. Regret is summed over held-out queries, with one decision per query. Caveat: for the cross-fitted arms (metadata, FAQ, keyword stuffing) the fold-1 rewards were measured on pages built from fold-2 query text, so this is a split of contexts rather than a fully independent test.

| Policy | Held-out regret (±95% CI) | Mean reward pp | Most chosen arm |
|---|---|---|---|
| linucb (frozen, contextual) | 4.553 ± 1.964 | -5.70 | `noop` (56%) |
| linucb-bias-only (frozen) | 2.502 ± 1.265 | +0.00 | `noop` (100%) |
| best fixed arm (chosen on train) | 2.502 ± 1.265 | +0.00 | `noop` (100%) |
| random (expected) | 4.762 ± n/a | -6.28 |  |
| oracle | 0.000 ± 0.000 | +6.95 |  |
