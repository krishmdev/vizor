# Results

`vizor report` builds this page from the run directories under `experiments/results/`. All numbers come from those files. See `experiments/README.md` for reproduction steps.

## qwen2.5:3b-instruct via Ollama, bench corpus (pre-registered main run)

- Results: `experiments/results/2026-09-05_bench-qwen3b` (git 0444d1e, 2026-09-05)
- Answer model: `qwen2.5:3b-instruct`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 72 queries x 5 samples, 72 pages
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 2530 new, 3230 cached (no API spend)

### Baseline visibility

Pooled marker share is the domain's share of all valid markers across every answer, so long, heavily cited answers count for more. It is not the per-answer C-SoV used as the primary metric in the arm tables, which averages each answer's own share.

| Domain | Retrieved % | Cited % | Cited when retrieved % | Pooled marker share % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment | Named % | Named, not cited % | Cited, not named % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| larkspur.example (target) | 93 | 69 | 75 | 50.8 | 42.5 | 3.58 | 25 | +0.15 | 59 | 16 | 27 |
| cogandchain.example | 56 | 29 | 52 | 15.8 | 14.1 | 3.37 | 48 | +0.14 | 24 | 11 | 16 |
| spokewise.example | 53 | 21 | 39 | 12.4 | 9.6 | 3.64 | 61 | +0.19 | 2 | 1 | 19 |
| pedalcheap.example | 56 | 16 | 28 | 11.1 | 7.4 | 4.07 | 72 | +0.39 | 0 | 0 | 16 |
| saddlesore.example | 54 | 19 | 34 | 10.0 | 6.2 | 4.93 | 66 | +0.11 | 0 | 0 | 19 |

### Sandbox arms (target: larkspur.example)

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

### Content vs rank

Each page edit split into what the new text did with the same sources in the same order (content-only arm) and what it did by changing retrieval (full arm minus content-only arm, paired per query). Total = content + rank-mediated. The p next to the rank-mediated effect is an unadjusted Wilcoxon on page units.

| Page edit | Total ΔC-SoV pp | Content-only ΔC-SoV pp | Rank-mediated ΔC-SoV pp | Total Δnamed pp | Content-only Δnamed pp | Rank-mediated Δnamed pp | Queries whose sources changed |
|---|---|---|---|---|---|---|---|
| `metadata` | -2.3 [-7.0, +2.6] | -2.4 [-6.5, +1.8] | +0.1 [-1.7, +2.4] (p 0.738) | +1.1 [-2.3, +4.1] | +0.0 [-3.3, +3.1] | +1.1 [+0.0, +2.4] | 5 |
| `faq_rewrite` | -15.4 [-21.3, -9.2] | -15.3 [-20.3, -9.8] | -0.1 [-2.2, +1.9] (p 0.964) | -8.3 [-15.1, -2.1] | -7.8 [-14.5, -1.8] | -0.6 [-1.7, +0.0] | 10 |
| `jsonld_insert` | -3.2 [-6.4, +0.2] | -3.5 [-6.7, +0.0] | +0.3 [+0.0, +0.9] (p 0.700) | +4.4 [+0.0, +9.0] | +3.1 [-1.4, +7.7] | +1.4 [+0.0, +4.3] | 1 |
| `internal_links` | -5.2 [-9.5, -0.7] | -5.2 [-9.5, -0.7] | +0.0 [+0.0, +0.0] (p 1.000) | -2.5 [-4.8, +0.0] | -2.5 [-4.8, +0.0] | +0.0 [+0.0, +0.0] | 0 |
| `stats_surface` | -0.5 [-5.4, +4.6] | +0.8 [-3.1, +4.9] | -1.3 [-5.1, +1.9] (p 0.920) | +1.4 [-2.5, +5.4] | +0.8 [-4.0, +5.5] | +0.6 [-1.7, +2.8] | 25 |
| `keyword_stuffing` | +1.7 [-2.7, +6.4] | +0.6 [-3.4, +4.9] | +1.1 [+0.0, +2.9] (p 0.458) | -1.1 [-4.3, +1.9] | -0.6 [-3.9, +2.4] | -0.6 [-1.6, +0.0] | 2 |

### Position sweep

| Target slot | C-SoV % | PAWC share % [95% CI] | ΔC-SoV vs slot 1 pp [95% CI] | Cited % | p (Holm, sweeps) | n |
|---|---|---|---|---|---|---|
| 1 | 43.1 | 43.8 [35.0, 52.6] | ref | 72 |  | 36 |
| 5 | 13.0 | 12.6 [5.7, 20.9] | -30.1 [-39.9, -20.5] | 22 | <0.001 * | 36 |

### What the sweeps show

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

### Bandit replay

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


## qwen2.5:3b-instruct via Ollama, bench corpus (pre-registered main run), pilot: baseline and A/A only

- Results: `experiments/results/2026-09-05_bench-qwen3b-pilot` (git 0a441a4, 2026-09-05)
- Answer model: `qwen2.5:3b-instruct`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 72 queries x 5 samples, 72 pages
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 720 new, 360 cached (no API spend)

### Baseline visibility

Pooled marker share is the domain's share of all valid markers across every answer, so long, heavily cited answers count for more. It is not the per-answer C-SoV used as the primary metric in the arm tables, which averages each answer's own share.

| Domain | Retrieved % | Cited % | Cited when retrieved % | Pooled marker share % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment | Named % | Named, not cited % | Cited, not named % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| larkspur.example (target) | 93 | 69 | 75 | 50.8 | 42.5 | 3.58 | 25 | +0.15 | 59 | 16 | 27 |
| cogandchain.example | 56 | 29 | 52 | 15.8 | 14.1 | 3.37 | 48 | +0.14 | 24 | 11 | 16 |
| spokewise.example | 53 | 21 | 39 | 12.4 | 9.6 | 3.64 | 61 | +0.19 | 2 | 1 | 19 |
| pedalcheap.example | 56 | 16 | 28 | 11.1 | 7.4 | 4.07 | 72 | +0.39 | 0 | 0 | 16 |
| saddlesore.example | 54 | 19 | 34 | 10.0 | 6.2 | 4.93 | 66 | +0.11 | 0 | 0 | 19 |

### Sandbox arms (target: larkspur.example)

The verdict is the Holm-adjusted Wilcoxon p on the primary metric (citation share (C-SoV, share of the answer's valid markers)), below 0.05 (`*`). The brand-mention test (“named”: the answer names the target brand, with or without a citation) gets its own Holm adjustment. `noop` and `aa_resample` are controls outside every family. Page arms are tested on edited-page units (queries sharing an edited page are not independent); `n` is queries / units. The 95% CIs resample those units and are descriptive. The last four columns describe the arm's answers: no valid citation at all, citation-like text the parser could not map to a source, and every citation collected on the final sentence (for those answers PAWC's position weighting is meaningless). “Queries whose sources changed” counts queries where the arm changed the list of sources the model saw.

| Arm | ΔC-SoV pp [95% CI] (primary) | ΔPAWC pp [95% CI] | Δnamed pp [95% CI] | p (Holm, primary) | p (Holm, named) | Queries whose sources changed | Uncited % | Unparsed markers % | Cites only on last sentence % | n |
|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | control | control | 0 | 20 | 0.0 | 41 | 72 / 24 |
| `aa_resample` | +0.2 [-3.7, +4.4] | +0.7 [-3.4, +5.1] | -1.4 [-5.3, +2.5] | control | control |  | 18 | 0.3 | 40 | 72 / 72 |

### What the sweeps show

- Noise floor: re-sampling the unchanged prompts (A/A) moved C-SoV by +0.2 [-3.7, +4.4] pp on 72 query units and the named rate by -1.4 [-5.3, +2.5] pp.
- Minimum detectable effect on C-SoV (for the planned design of 6 page edits and 6 content-only twins, each its own Holm family; 80% power, strictest Holm step, normal approximation, from the A/A re-sample; per-query A/A SD 17.7 pp): page edits ≈ 7.9 pp (8.5 pp with t quantiles) on 24 underlying page units (both cross-fit folds clustered by page); content-only arms ≈ 7.9 pp; named rate for page edits ≈ 7.6 pp. These are approximate: the test is a Wilcoxon signed-rank test, not a t test. Effects smaller than these could be missed.


## Study 2: qwen2.5-3b-mlx4 via Localhost AI, bench corpus, Study 2 queries (pre-registered)

- Results: `experiments/results/2026-09-18_study2-qwen3b` (git c07c57b, 2026-09-18)
- Answer model: `qwen2.5-3b-mlx4`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `vader/vaderSentiment-3.3.2/compound`; PAWC decay: paper
- 72 queries x 3 samples, 72 pages
- Served by server `localhost-ai`, preset `qwen2.5-3b-mlx4`, commit `c61e05d4992d6cb68c7c16fc986eeb84dda74c5f` at `http://127.0.0.1:8431/v1`
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 1179 new, 333 cached (no API spend)

### Baseline visibility

Pooled marker share is the domain's share of all valid markers across every answer, so long, heavily cited answers count for more. It is not the per-answer C-SoV used as the primary metric in the arm tables, which averages each answer's own share.

| Domain | Retrieved % | Cited % | Cited when retrieved % | Pooled marker share % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment | Named % | Named, not cited % | Cited, not named % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| larkspur.example (target) | 99 | 93 | 94 | 49.6 | 60.6 | 1.54 | 6 | +0.10 | 56 | 3 | 40 |
| cogandchain.example | 69 | 41 | 59 | 18.1 | 17.2 | 2.68 | 41 | +0.14 | 23 | 3 | 21 |
| spokewise.example | 53 | 22 | 42 | 9.3 | 8.5 | 2.60 | 58 | +0.13 | 1 | 0 | 21 |
| saddlesore.example | 51 | 26 | 50 | 18.6 | 7.4 | 3.05 | 50 | +0.06 | 0 | 0 | 26 |
| pedalcheap.example | 46 | 14 | 31 | 4.4 | 5.1 | 2.32 | 69 | +0.22 | 0 | 0 | 14 |

### Sandbox arms (target: larkspur.example)

The verdict follows `sampled_primary.json`: AP failed its validation gate on criteria 3 and 4, so the primary is the sampled citation share (C-SoV) on 24 page units: an arm has an effect (`*`) when the exact Wilcoxon and the sign-flip p are both below 0.05 after Holm over the 3 arms. The ΔC-SoV column, its intervals and the `n` units come from that file (page units, A/A included; the A/A's ΔPAWC and Δnamed stay on its query units). The Study 1 column is the Holm-adjusted Wilcoxon (zero_method zsplit) that decided Study 1; it is shown for continuity and decides nothing here. The brand-mention test (“named”: the answer names the target brand, with or without a citation) gets its own Holm adjustment. `noop` and `aa_resample` are controls outside every family. Page arms are tested on edited-page units (queries sharing an edited page are not independent); `n` is queries / units. The 95% CIs resample those units and are descriptive. The last four columns describe the arm's answers: no valid citation at all, citation-like text the parser could not map to a source, and every citation collected on the final sentence (for those answers PAWC's position weighting is meaningless). “Queries whose sources changed” counts queries where the arm changed the list of sources the model saw.

| Arm | ΔC-SoV pp [95% CI] (primary) | ΔPAWC pp [95% CI] | Δnamed pp [95% CI] | p (Holm, primary: Wilcoxon / sign-flip) | Study 1 test (continuity, not this study's rule) | p (Holm, named) | Queries whose sources changed | Uncited % | Unparsed markers % | Cites only on last sentence % | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | control | control | control | 0 | 1 | 0.0 | 5 | 72 / 24 |
| `aa_resample` | -3.4 [-6.6, -0.5] | -3.3 [-6.9, +0.3] | +4.6 [-0.0, +9.7] | control | control | control |  | 1 | 1.4 | 6 | 72 / 24 |
| `answer_first` | -9.5 [-15.8, -3.8] | -10.3 [-16.5, -4.4] | +2.1 [-3.1, +7.4] | 0.008 / 0.005 * | 0.011 | 0.514 | 29 | 0 | 0.9 | 4 | 72 / 24 |
| `evidence_surface_llm` | -4.3 [-9.1, -0.4] | -4.5 [-9.4, -0.7] | +4.6 [+0.8, +8.7] | 0.046 / 0.044 * | 0.052 | 0.133 | 4 | 1 | 0.0 | 5 | 72 / 24 |
| `faq_rewrite_v2` | -10.0 [-15.8, -4.8] | -10.1 [-16.2, -4.6] | -2.8 [-6.9, +1.0] | 0.004 / 0.003 * | 0.011 | 0.299 | 2 | 2 | 0.5 | 8 | 72 / 24 |

### Position sweep

| Target slot | C-SoV % | PAWC share % [95% CI] | ΔC-SoV vs slot 1 pp [95% CI] | Cited % | p (Holm, sweeps) | n |
|---|---|---|---|---|---|---|
| 1 | 55.4 | 58.7 [49.2, 67.9] | ref | 90 |  | 36 |
| 5 | 28.9 | 28.2 [19.3, 37.6] | -26.6 [-36.6, -15.7] | 48 | <0.001 * | 36 |

### What the sweeps show

- Context order (same pages, target forced into each slot, n=36 queries): C-SoV slot 1 55.4%, slot 5 28.9%. Holm-significant differences: slot 5 vs 1: -26.6 [-36.6, -15.7] pp (Holm p <0.001).
- How fragile the slot 5 result is: its raw p is <0.001 over 36 queries, but those queries are served by only 16 target pages. Per page, slot 5 minus slot 1 on C-SoV is beam-800 -51.9, floor-pump -44.7, fold-20 -15.1, gravel-gx +11.8, bike-theft -28.9, chain-care -16.7, tire-pressure -16.7, haul -67.5, lock-d9 -27.8, metro-7 -0.7, pannier-20 -38.9, shell -23.5, spin-t2 -12.0, sprout-16 -40.3, wheel-truing -6.7, volt-e1 -32.5 pp, and a Wilcoxon test on the page means gives p = <0.001. It also depends on the sweeps forming their own Holm family: in one family with the arms (4 tests) its Holm p would be <0.001.
- Noise floor: re-sampling the unchanged prompts (A/A) moved C-SoV by -3.4 [-6.6, -0.5] pp on 24 page units (raw sign-flip p 0.045) and the named rate by +4.6 [-0.0, +9.7] pp.
- Page and engine arms: 3 of 3 have an effect under the sampled primary rule on C-SoV: `answer_first`, `evidence_surface_llm`, `faq_rewrite_v2`; 0 of 3 on the named rate.
- Exploratory, not pre-registered (the reference choice, `sampled_primary.md`): against the A/A re-sample `answer_first` -6.1 (Holm p 0.137 / 0.140), `evidence_surface_llm` -0.9 (Holm p 0.812 / 0.742), `faq_rewrite_v2` -6.6 (Holm p 0.137 / 0.140); against the mean of baseline and A/A `answer_first` -7.8 (Holm p 0.029 / 0.020), `evidence_surface_llm` -2.6 (Holm p 0.406 / 0.287), `faq_rewrite_v2` -8.3 (Holm p 0.029 / 0.020). A lucky baseline draw shifts every arm the same way; an arm that holds up only against the baseline alone is not robust to that choice.
- Weighting: the C-SoV estimates, intervals and tests above all use unweighted page means.
- Minimum detectable effect on C-SoV (80% power, strictest Holm step, normal approximation, from the A/A re-sample; per-query A/A SD 16.7 pp): page edits ≈ 5.3 pp (5.6 pp with t quantiles) on 24 underlying page units; named rate for page edits ≈ 10.0 pp; slot sweep ≈ 7.8 pp. These are approximate: the test is a Wilcoxon signed-rank test, not a t test. Effects smaller than these could be missed.
- Positive control (target moved from slot 1 to slot 5, same pages): C-SoV -26.6 [-36.6, -15.7] pp, Holm p <0.001 *.
- Exploratory, not pre-registered: on the held-out queries the frozen contextual bandit policy had regret 4.553 (±1.964) vs 4.762 for random and 2.502 for the best fixed arm chosen on the training half. It is not distinguishable from random.

### Bandit replay

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


## Study 3 pilot: qwen3.5-9b-mlx4 via Localhost AI, baseline and A/A at 2 samples

- Results: `experiments/results/2026-09-18_study3-pilot` (git 4803da9, 2026-09-18)
- Answer model: `qwen3.5-9b-mlx4`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `vader/vaderSentiment-3.3.2/compound`; PAWC decay: paper
- 72 queries x 2 samples, 72 pages
- Served by server `localhost-ai`, preset `qwen3.5-9b-mlx4`, commit `fdb4fbeb361e6244300b999ed1fdaeaef1fb1599` at `http://127.0.0.1:8431/v1`
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 288 new, 144 cached (no API spend)

### Baseline visibility

Pooled marker share is the domain's share of all valid markers across every answer, so long, heavily cited answers count for more. It is not the per-answer C-SoV used as the primary metric in the arm tables, which averages each answer's own share.

| Domain | Retrieved % | Cited % | Cited when retrieved % | Pooled marker share % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment | Named % | Named, not cited % | Cited, not named % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| larkspur.example (target) | 100 | 99 | 99 | 55.0 | 55.8 | 2.00 | 1 | +0.07 | 49 | 1 | 51 |
| cogandchain.example | 69 | 53 | 77 | 18.8 | 18.4 | 3.71 | 23 | +0.10 | 19 | 0 | 34 |
| spokewise.example | 53 | 43 | 82 | 12.0 | 12.2 | 4.05 | 18 | +0.02 | 1 | 0 | 42 |
| saddlesore.example | 57 | 42 | 74 | 9.4 | 8.5 | 5.00 | 26 | +0.01 | 0 | 0 | 42 |
| pedalcheap.example | 38 | 19 | 52 | 4.9 | 4.4 | 3.68 | 48 | +0.23 | 0 | 0 | 19 |

### Sandbox arms (target: larkspur.example)

The verdict is the Holm-adjusted Wilcoxon p on the primary metric (citation share (C-SoV, share of the answer's valid markers)), below 0.05 (`*`). The brand-mention test (“named”: the answer names the target brand, with or without a citation) gets its own Holm adjustment. `noop` and `aa_resample` are controls outside every family. Page arms are tested on edited-page units (queries sharing an edited page are not independent); `n` is queries / units. The 95% CIs resample those units and are descriptive. The last four columns describe the arm's answers: no valid citation at all, citation-like text the parser could not map to a source, and every citation collected on the final sentence (for those answers PAWC's position weighting is meaningless). “Queries whose sources changed” counts queries where the arm changed the list of sources the model saw.

| Arm | ΔC-SoV pp [95% CI] (primary) | ΔPAWC pp [95% CI] | Δnamed pp [95% CI] | p (Holm, primary) | p (Holm, named) | Queries whose sources changed | Uncited % | Unparsed markers % | Cites only on last sentence % | n |
|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | control | control | 0 | 1 | 0.0 | 0 | 72 / 24 |
| `aa_resample` | +0.5 [-1.9, +2.8] | +0.4 [-1.9, +2.8] | +4.9 [-2.1, +11.8] | control | control |  | 1 | 0.0 | 0 | 72 / 72 |

### What the sweeps show

- Noise floor: re-sampling the unchanged prompts (A/A) moved C-SoV by +0.5 [-1.9, +2.8] pp on 72 query units and the named rate by +4.9 [-2.1, +11.8] pp.
- Weighting: the C-SoV estimates, intervals and tests above use unweighted page means, except the A/A row, which uses its 72 query units (on page units the A/A is +0.11 pp).
- No page arms in this run (a pilot), so it states no MDE of its own; `vizor sensitivity <dir> --family N` gives the MDE for a planned design.


## Study 3: qwen3.5-9b-mlx4 via Localhost AI, bench corpus, Study 3 queries (pre-registered)

- Results: `experiments/results/2026-09-28_study3-qwen9b` (git 7e5d5b1, 2026-09-28)
- Answer model: `qwen3.5-9b-mlx4`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `vader/vaderSentiment-3.3.2/compound`; PAWC decay: paper
- 72 queries x 2 samples, 72 pages
- Served by server `localhost-ai`, preset `qwen3.5-9b-mlx4`, commit `8fa3d561868856735aa6196773b30bd73e92b0cb` at `http://127.0.0.1:8431/v1`
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 677 new, 403 cached (no API spend)

### Baseline visibility

Pooled marker share is the domain's share of all valid markers across every answer, so long, heavily cited answers count for more. It is not the per-answer C-SoV used as the primary metric in the arm tables, which averages each answer's own share.

| Domain | Retrieved % | Cited % | Cited when retrieved % | Pooled marker share % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment | Named % | Named, not cited % | Cited, not named % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| larkspur.example (target) | 100 | 99 | 99 | 55.0 | 55.8 | 2.00 | 1 | +0.07 | 49 | 1 | 51 |
| cogandchain.example | 69 | 53 | 77 | 18.8 | 18.4 | 3.71 | 23 | +0.10 | 19 | 0 | 34 |
| spokewise.example | 53 | 43 | 82 | 12.0 | 12.2 | 4.05 | 18 | +0.02 | 1 | 0 | 42 |
| saddlesore.example | 57 | 42 | 74 | 9.4 | 8.5 | 5.00 | 26 | +0.01 | 0 | 0 | 42 |
| pedalcheap.example | 38 | 19 | 52 | 4.9 | 4.4 | 3.68 | 48 | +0.23 | 0 | 0 | 19 |

### Sandbox arms (target: larkspur.example)

The verdict follows `sampled_primary.json`: the primary is the sampled citation share (C-SoV) on 24 page units: an arm has an effect (`*`) when the exact Wilcoxon and the sign-flip p are both below 0.05 after Holm over the 3 arms, and its delta against the A/A re-sample has the same sign as against the baseline. The ΔC-SoV column, its intervals and the `n` units come from that file (page units, A/A included; the A/A's ΔPAWC and Δnamed stay on its query units). The Study 1 column is the Holm-adjusted Wilcoxon (zero_method zsplit) that decided Study 1; it is shown for continuity and decides nothing here. The brand-mention test (“named”: the answer names the target brand, with or without a citation) gets its own Holm adjustment. Content-only arms (`content:`) keep the baseline's sources and their order and change only the edited page's text; they form a separate Holm family. `noop` and `aa_resample` are controls outside every family. Page arms are tested on edited-page units (queries sharing an edited page are not independent); `n` is queries / units. The 95% CIs resample those units and are descriptive. The last four columns describe the arm's answers: no valid citation at all, citation-like text the parser could not map to a source, and every citation collected on the final sentence (for those answers PAWC's position weighting is meaningless). “Queries whose sources changed” counts queries where the arm changed the list of sources the model saw.

| Arm | ΔC-SoV pp [95% CI] (primary) | ΔPAWC pp [95% CI] | Δnamed pp [95% CI] | p (Holm, primary: Wilcoxon / sign-flip) | Study 1 test (continuity, not this study's rule) | p (Holm, named) | Queries whose sources changed | Uncited % | Unparsed markers % | Cites only on last sentence % | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | control | control | control | 0 | 1 | 0.0 | 0 | 72 / 24 |
| `aa_resample` | +0.1 [-2.9, +3.1] | +0.4 [-1.9, +2.8] | +4.9 [-2.1, +11.8] | control | control | control |  | 1 | 0.0 | 0 | 72 / 24 |
| `fact_passage` | -5.5 [-10.3, -0.6] | -5.2 [-9.9, -0.6] | -5.4 [-16.1, +3.0] | 0.081 / 0.117 | 0.081 | 1.000 | 22 | 3 | 0.0 | 0 | 72 / 24 |
| `entity_anchor` | -1.7 [-3.4, -0.2] | -1.3 [-3.1, +0.4] | +1.2 [-3.1, +5.6] | 0.259 / 0.117 | 0.404 | 1.000 | 1 | 1 | 0.0 | 0 | 72 / 24 |
| `retrieval_meta` | -0.7 [-3.3, +2.0] | +0.6 [-2.0, +3.3] | +0.2 [-5.9, +5.0] | 0.565 / 0.626 | 0.565 | 1.000 | 10 | 1 | 0.0 | 0 | 72 / 24 |
| `content:fact_passage` | -1.5 [-5.8, +2.9] | -0.2 [-4.2, +3.9] | -4.9 [-15.6, +3.8] | secondary | 0.623 | 1.000 | 0 | 0 | 0.0 | 0 | 72 / 24 |
| `content:entity_anchor` | +0.2 [-2.1, +2.2] | +0.4 [-2.0, +2.7] | -0.3 [-5.9, +5.2] | secondary | 0.812 | 1.000 | 0 | 0 | 0.0 | 0 | 72 / 24 |
| `content:retrieval_meta` | -0.7 [-3.9, +2.7] | +0.6 [-2.8, +4.0] | -5.2 [-16.7, +3.6] | secondary | 0.812 | 1.000 | 0 | 0 | 0.0 | 0 | 72 / 24 |

### Content vs rank

Each page edit split into what the new text did with the same sources in the same order (content-only arm) and what it did by changing retrieval (full arm minus content-only arm, paired per query). Total = content + rank-mediated. The p next to the rank-mediated effect is an unadjusted Wilcoxon on page units.

| Page edit | Total ΔC-SoV pp | Content-only ΔC-SoV pp | Rank-mediated ΔC-SoV pp | Total Δnamed pp | Content-only Δnamed pp | Rank-mediated Δnamed pp | Queries whose sources changed |
|---|---|---|---|---|---|---|---|
| `fact_passage` | -5.5 [-10.3, -0.6] | -1.5 [-5.8, +2.9] | -3.9 [-8.3, -0.1] (p 0.152) | -5.4 [-16.1, +3.0] | -4.9 [-15.6, +3.8] | -0.5 [-4.5, +3.5] | 22 |
| `entity_anchor` | -1.7 [-3.4, -0.2] | +0.2 [-2.1, +2.2] | -1.8 [-3.5, -0.1] (p 0.065) | +1.2 [-3.1, +5.6] | -0.3 [-5.9, +5.2] | +1.6 [-3.0, +6.2] | 1 |
| `retrieval_meta` | -0.7 [-3.3, +2.0] | -0.7 [-3.9, +2.7] | -0.0 [-2.0, +1.8] (p 0.898) | +0.2 [-5.9, +5.0] | -5.2 [-16.7, +3.6] | +5.4 [-0.9, +12.3] | 10 |

### What the sweeps show

- Noise floor: re-sampling the unchanged prompts (A/A) moved C-SoV by +0.1 [-2.9, +3.1] pp on 24 page units (raw sign-flip p 0.946) and the named rate by +4.9 [-2.1, +11.8] pp.
- Page and engine arms: 0 of 3 have an effect under the sampled primary rule on C-SoV; 0 of 3 on the named rate.
- Content-only arms: 0 of 3 have a Holm-significant effect on C-SoV; 0 of 3 on the named rate.
- Exploratory, not pre-registered (the reference choice, `sampled_primary.md`): against the A/A re-sample `fact_passage` -5.6 (Holm p 0.054 / 0.242), `entity_anchor` -1.8 (Holm p 0.481 / 0.613), `retrieval_meta` -0.8 (Holm p 0.481 / 0.625); against the mean of baseline and A/A `fact_passage` -5.5 (Holm p 0.035 / 0.151), `entity_anchor` -1.7 (Holm p 0.242 / 0.253), `retrieval_meta` -0.8 (Holm p 0.360 / 0.573). A lucky baseline draw shifts every arm the same way; an arm that holds up only against the baseline alone is not robust to that choice.
- Content vs rank, exploratory and not pre-registered (the split on sample 0 alone, where each twin and its full arm share a seed; `study3_secondary.md`), in pp: `fact_passage` -2.3 content / -4.3 rank-mediated, `entity_anchor` -0.7 content / -0.2 rank-mediated, `retrieval_meta` -1.5 content / -0.7 rank-mediated.
- The pre-registered content-vs-rank split (the decomposition table) is confounded by the 1-sample twins (see the design doc): where a full arm's prompt equals its twin's, full minus twin is sampling noise, not a rank effect. Taken at face value, it has the rank-mediated part larger in size than the content-only part for 2 of 3 page edits. `entity_anchor` changed the sources of only 1 query, so its rank-mediated part is noise.
- Weighting: the C-SoV estimates, intervals and tests above all use unweighted page means.
- Minimum detectable effect on C-SoV (80% power, strictest Holm step, normal approximation, from the A/A re-sample; per-query A/A SD 10.4 pp): page edits ≈ 5.1 pp (5.4 pp with t quantiles) on 24 underlying page units; content-only arms ≈ 5.1 pp (this assumes 2 samples per query, as for the page edits; the twins ran at 1, so their MDE is larger); named rate for page edits ≈ 11.0 pp. These are approximate: the test is a Wilcoxon signed-rank test, not a t test. Effects smaller than these could be missed.
- Exploratory, not pre-registered: on the held-out queries the frozen contextual bandit policy had regret 1.480 (±0.590) vs 2.968 for random and 2.782 for the best fixed arm chosen on the training half. It beat random by more than its 95% interval.

### Bandit replay

Offline replay uses rewards from the same table the policies learn from. It runs 2000 rounds x 20 runs. Regret is summed over rounds in citation share (1.0 = 100 pp). The hindsight row knows the best single arm in advance and serves as a reference, not a policy.

| Policy | Cumulative regret at T (±95% CI) | Mean reward pp/round | Most chosen arm |
|---|---|---|---|
| linucb(a=0.1) | 124.03 ± 2.71 | +0.34 | `noop` (68%) |
| linucb(a=0.5) | 130.69 ± 2.74 | +0.04 | `noop` (41%) |
| linucb(a=1) | 140.93 ± 2.65 | -0.50 | `noop` (34%) |
| lints(v=0.1) | 129.45 ± 2.44 | +0.10 | `noop` (56%) |
| eps-greedy(0.1) | 130.86 ± 3.25 | +0.01 | `noop` (64%) |
| linucb-bias-only(a=0.1) | 132.33 ± 1.90 | -0.08 | `noop` (93%) |
| random | 179.74 ± 2.08 | -2.45 | `noop` (25%) |
| best fixed arm (hindsight) | 130.94 ± 1.78 | +0.00 | `noop` (100%) |
| oracle | 0.00 ± 0.00 | +6.61 |  |

Held-out check: the policy is fit on fold-1 queries, then frozen and scored on fold-2 queries. Regret is summed over held-out queries, with one decision per query. Caveat: for the cross-fitted arms (metadata, FAQ, keyword stuffing) the fold-1 rewards were measured on pages built from fold-2 query text, so this is a split of contexts rather than a fully independent test.

| Policy | Held-out regret (±95% CI) | Mean reward pp | Most chosen arm |
|---|---|---|---|
| linucb (frozen, contextual) | 1.480 ± 0.590 | +0.25 | `retrieval_meta` (61%) |
| linucb-bias-only (frozen) | 2.782 ± 1.081 | -3.37 | `retrieval_meta` (100%) |
| best fixed arm (chosen on train) | 2.782 ± 1.081 | -3.37 | `retrieval_meta` (100%) |
| random (expected) | 2.968 ± n/a | -3.89 |  |
| oracle | 0.000 ± 0.000 | +4.36 |  |


## FakeLLM (pipeline check)

- Results: `experiments/results/2026-08-05_fakellm` (git b9f509a, 2026-08-20)
- Answer model: `fake-extractive/v1/BAAI/bge-small-en-v1.5` (deterministic extractive stand-in; pipeline check, not model evidence)
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 40 queries x 5 samples, 22 pages

### Baseline visibility

Pooled marker share is the domain's share of all valid markers across every answer, so long, heavily cited answers count for more. It is not the per-answer C-SoV used as the primary metric in the arm tables, which averages each answer's own share.

| Domain | Retrieved % | Cited % | Cited when retrieved % | Pooled marker share % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment |
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

| Target slot | PAWC share % [95% CI] | ΔPAWC vs slot 1 pp [95% CI] | Cited % | p (Holm, sweeps) | n |
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
