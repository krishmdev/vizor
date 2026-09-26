# Study 3: qwen3.5-9b-mlx4 via Localhost AI, bench corpus, Study 3 queries (pre-registered)

- Results: `experiments/results/2026-09-28_study3-qwen9b` (git 7e5d5b1, 2026-09-28)
- Answer model: `qwen3.5-9b-mlx4`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `vader/vaderSentiment-3.3.2/compound`; PAWC decay: paper
- 72 queries x 2 samples, 72 pages
- Served by server `localhost-ai`, preset `qwen3.5-9b-mlx4`, commit `8fa3d561868856735aa6196773b30bd73e92b0cb` at `http://127.0.0.1:8431/v1`
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 677 new, 403 cached (no API spend)

## Baseline visibility

Pooled marker share is the domain's share of all valid markers across every answer, so long, heavily cited answers count for more. It is not the per-answer C-SoV used as the primary metric in the arm tables, which averages each answer's own share.

| Domain | Retrieved % | Cited % | Cited when retrieved % | Pooled marker share % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment | Named % | Named, not cited % | Cited, not named % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| larkspur.example (target) | 100 | 99 | 99 | 55.0 | 55.8 | 2.00 | 1 | +0.07 | 49 | 1 | 51 |
| cogandchain.example | 69 | 53 | 77 | 18.8 | 18.4 | 3.71 | 23 | +0.10 | 19 | 0 | 34 |
| spokewise.example | 53 | 43 | 82 | 12.0 | 12.2 | 4.05 | 18 | +0.02 | 1 | 0 | 42 |
| saddlesore.example | 57 | 42 | 74 | 9.4 | 8.5 | 5.00 | 26 | +0.01 | 0 | 0 | 42 |
| pedalcheap.example | 38 | 19 | 52 | 4.9 | 4.4 | 3.68 | 48 | +0.23 | 0 | 0 | 19 |

## Sandbox arms (target: larkspur.example)

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

## Content vs rank

Each page edit split into what the new text did with the same sources in the same order (content-only arm) and what it did by changing retrieval (full arm minus content-only arm, paired per query). Total = content + rank-mediated. The p next to the rank-mediated effect is an unadjusted Wilcoxon on page units.

| Page edit | Total ΔC-SoV pp | Content-only ΔC-SoV pp | Rank-mediated ΔC-SoV pp | Total Δnamed pp | Content-only Δnamed pp | Rank-mediated Δnamed pp | Queries whose sources changed |
|---|---|---|---|---|---|---|---|
| `fact_passage` | -5.5 [-10.3, -0.6] | -1.5 [-5.8, +2.9] | -3.9 [-8.3, -0.1] (p 0.152) | -5.4 [-16.1, +3.0] | -4.9 [-15.6, +3.8] | -0.5 [-4.5, +3.5] | 22 |
| `entity_anchor` | -1.7 [-3.4, -0.2] | +0.2 [-2.1, +2.2] | -1.8 [-3.5, -0.1] (p 0.065) | +1.2 [-3.1, +5.6] | -0.3 [-5.9, +5.2] | +1.6 [-3.0, +6.2] | 1 |
| `retrieval_meta` | -0.7 [-3.3, +2.0] | -0.7 [-3.9, +2.7] | -0.0 [-2.0, +1.8] (p 0.898) | +0.2 [-5.9, +5.0] | -5.2 [-16.7, +3.6] | +5.4 [-0.9, +12.3] | 10 |

## What the sweeps show

- Noise floor: re-sampling the unchanged prompts (A/A) moved C-SoV by +0.1 [-2.9, +3.1] pp on 24 page units (raw sign-flip p 0.946) and the named rate by +4.9 [-2.1, +11.8] pp.
- Page and engine arms: 0 of 3 have an effect under the sampled primary rule on C-SoV; 0 of 3 on the named rate.
- Content-only arms: 0 of 3 have a Holm-significant effect on C-SoV; 0 of 3 on the named rate.
- Exploratory, not pre-registered (the reference choice, `sampled_primary.md`): against the A/A re-sample `fact_passage` -5.6 (Holm p 0.054 / 0.242), `entity_anchor` -1.8 (Holm p 0.481 / 0.613), `retrieval_meta` -0.8 (Holm p 0.481 / 0.625); against the mean of baseline and A/A `fact_passage` -5.5 (Holm p 0.035 / 0.151), `entity_anchor` -1.7 (Holm p 0.242 / 0.253), `retrieval_meta` -0.8 (Holm p 0.360 / 0.573). A lucky baseline draw shifts every arm the same way; an arm that holds up only against the baseline alone is not robust to that choice.
- Content vs rank, exploratory and not pre-registered (the split on sample 0 alone, where each twin and its full arm share a seed; `study3_secondary.md`), in pp: `fact_passage` -2.3 content / -4.3 rank-mediated, `entity_anchor` -0.7 content / -0.2 rank-mediated, `retrieval_meta` -1.5 content / -0.7 rank-mediated.
- The pre-registered content-vs-rank split (the decomposition table) is confounded by the 1-sample twins (see the design doc): where a full arm's prompt equals its twin's, full minus twin is sampling noise, not a rank effect. Taken at face value, it has the rank-mediated part larger in size than the content-only part for 2 of 3 page edits. `entity_anchor` changed the sources of only 1 query, so its rank-mediated part is noise.
- Weighting: the C-SoV estimates, intervals and tests above all use unweighted page means.
- Minimum detectable effect on C-SoV (80% power, strictest Holm step, normal approximation, from the A/A re-sample; per-query A/A SD 10.4 pp): page edits ≈ 5.1 pp (5.4 pp with t quantiles) on 24 underlying page units; content-only arms ≈ 5.1 pp (this assumes 2 samples per query, as for the page edits; the twins ran at 1, so their MDE is larger); named rate for page edits ≈ 11.0 pp. These are approximate: the test is a Wilcoxon signed-rank test, not a t test. Effects smaller than these could be missed.
- Exploratory, not pre-registered: on the held-out queries the frozen contextual bandit policy had regret 1.480 (±0.590) vs 2.968 for random and 2.782 for the best fixed arm chosen on the training half. It beat random by more than its 95% interval.

## Bandit replay

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
