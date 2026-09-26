# Study 3 pilot: qwen3.5-9b-mlx4 via Localhost AI, baseline and A/A at 2 samples

- Results: `experiments/results/2026-09-18_study3-pilot` (git 4803da9, 2026-09-18)
- Answer model: `qwen3.5-9b-mlx4`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `vader/vaderSentiment-3.3.2/compound`; PAWC decay: paper
- 72 queries x 2 samples, 72 pages
- Served by server `localhost-ai`, preset `qwen3.5-9b-mlx4`, commit `fdb4fbeb361e6244300b999ed1fdaeaef1fb1599` at `http://127.0.0.1:8431/v1`
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 288 new, 144 cached (no API spend)

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

The verdict is the Holm-adjusted Wilcoxon p on the primary metric (citation share (C-SoV, share of the answer's valid markers)), below 0.05 (`*`). The brand-mention test (“named”: the answer names the target brand, with or without a citation) gets its own Holm adjustment. `noop` and `aa_resample` are controls outside every family. Page arms are tested on edited-page units (queries sharing an edited page are not independent); `n` is queries / units. The 95% CIs resample those units and are descriptive. The last four columns describe the arm's answers: no valid citation at all, citation-like text the parser could not map to a source, and every citation collected on the final sentence (for those answers PAWC's position weighting is meaningless). “Queries whose sources changed” counts queries where the arm changed the list of sources the model saw.

| Arm | ΔC-SoV pp [95% CI] (primary) | ΔPAWC pp [95% CI] | Δnamed pp [95% CI] | p (Holm, primary) | p (Holm, named) | Queries whose sources changed | Uncited % | Unparsed markers % | Cites only on last sentence % | n |
|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | control | control | 0 | 1 | 0.0 | 0 | 72 / 24 |
| `aa_resample` | +0.5 [-1.9, +2.8] | +0.4 [-1.9, +2.8] | +4.9 [-2.1, +11.8] | control | control |  | 1 | 0.0 | 0 | 72 / 72 |

## What the sweeps show

- Noise floor: re-sampling the unchanged prompts (A/A) moved C-SoV by +0.5 [-1.9, +2.8] pp on 72 query units and the named rate by +4.9 [-2.1, +11.8] pp.
- Weighting: the C-SoV estimates, intervals and tests above use unweighted page means, except the A/A row, which uses its 72 query units (on page units the A/A is +0.11 pp).
- No page arms in this run (a pilot), so it states no MDE of its own; `vizor sensitivity <dir> --family N` gives the MDE for a planned design.
