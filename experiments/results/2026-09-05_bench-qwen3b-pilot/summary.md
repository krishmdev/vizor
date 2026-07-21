# qwen2.5:3b-instruct via Ollama, bench corpus (pre-registered main run), pilot: baseline and A/A only

- Results: `experiments/results/2026-09-05_bench-qwen3b-pilot` (git 0a441a4, 2026-09-05)
- Answer model: `qwen2.5:3b-instruct`
- Retrieval: `sentence-transformers/BAAI/bge-small-en-v1.5/5c38ec7c405e/384/21fa4cb7` + `cross-encoder/cross-encoder/ms-marco-MiniLM-L-6-v2/233902d25c44`, index `numpy`
- Sentiment: `hf/cardiffnlp/twitter-roberta-base-sentiment-latest/3216a57f2a0d/p_pos-p_neg`; PAWC decay: paper
- 72 queries x 5 samples, 72 pages
- System message (sha256 `aec4c3c34309`, full text in the manifest): “You answer questions using numbered search results. Write short sentences. Put a citation…”
- Primary metric for the verdict: citation share (C-SoV, share of the answer's valid markers)
- Local calls: 720 new, 360 cached (no API spend)

## Baseline visibility

| Domain | Retrieved % | Cited % | Cited when retrieved % | C-SoV % | PAWC share % | First cite (sentence) | Ignored when retrieved % | Answer sentiment | Named % | Named, not cited % | Cited, not named % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| larkspur.example (target) | 93 | 69 | 75 | 50.8 | 42.5 | 3.58 | 25 | +0.15 | 59 | 16 | 27 |
| cogandchain.example | 56 | 29 | 52 | 15.8 | 14.1 | 3.37 | 48 | +0.14 | 24 | 11 | 16 |
| spokewise.example | 53 | 21 | 39 | 12.4 | 9.6 | 3.64 | 61 | +0.19 | 2 | 1 | 19 |
| pedalcheap.example | 56 | 16 | 28 | 11.1 | 7.4 | 4.07 | 72 | +0.39 | 0 | 0 | 16 |
| saddlesore.example | 54 | 19 | 34 | 10.0 | 6.2 | 4.93 | 66 | +0.11 | 0 | 0 | 19 |

## Sandbox arms (target: larkspur.example)

The verdict is the Holm-adjusted Wilcoxon p on the primary metric (citation share (C-SoV, share of the answer's valid markers)), below 0.05 (`*`). The brand-mention test (“named”: the answer names the target brand, with or without a citation) gets its own Holm adjustment. `noop` and `aa_resample` are controls outside every family. Page arms are tested on edited-page units (queries sharing an edited page are not independent); `n` is queries / units. The 95% CIs resample those units and are descriptive. The last four columns describe the arm's answers: no valid citation at all, citation-like text the parser could not map to a source, and every citation collected on the final sentence (for those answers PAWC's position weighting is meaningless). “Queries whose sources changed” counts queries where the arm changed the list of sources the model saw.

| Arm | ΔC-SoV pp [95% CI] (primary) | ΔPAWC pp [95% CI] | Δnamed pp [95% CI] | p (Holm, primary) | p (Holm, named) | Queries whose sources changed | Uncited % | Unparsed markers % | Cites only on last sentence % | n |
|---|---|---|---|---|---|---|---|---|---|---|
| `noop` | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | control | control | 0 | 20 | 0.0 | 41 | 72 / 24 |
| `aa_resample` | +0.2 [-3.7, +4.4] | +0.7 [-3.4, +5.1] | -1.4 [-5.3, +2.5] | control | control |  | 18 | 0.3 | 40 | 72 / 72 |

## What the sweeps show

- Noise floor: re-sampling the unchanged prompts (A/A) moved C-SoV by +0.2 [-3.7, +4.4] pp and the named rate by -1.4 [-5.3, +2.5] pp.
- Minimum detectable effect on C-SoV (80% power, strictest Holm step, from the A/A re-sample; per-query A/A SD 17.7 pp): named rate for page edits ≈ 6.1 pp. Effects smaller than these could be missed.
