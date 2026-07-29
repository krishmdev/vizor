# Bench run design (written before the run)

This file fixes the design of the local-model run on the bench corpus before any answer for it
exists. The earlier local run (20 queries x 2 samples on the demo corpus, git 434953b) could not
test page edits at all: each arm had only 4 to 7 independent edited pages, so the smallest
p-value an exact Wilcoxon test could return was above the Holm threshold. That run also changed
its system prompt once, and its results changed sign between the two versions. This design is
meant to avoid both problems: enough independent pages to test, and one prompt chosen in advance.

## Question

In the Vizor sandbox engine, answering with qwen2.5:3b-instruct, do edits to a target site's
pages change how often that site shows up in the answers? "Shows up" is measured two ways: the
share of the answer's citations that point at the target's pages, and whether the answer names
the target brand at all.

This is a question about this sandbox engine and this model. Nothing here says anything about
ChatGPT, Perplexity, Google or any other production answer engine.

## Engine and corpus

- Corpus: `data/bench` (synthetic, 72 pages on 5 fictional `.example` sites). The target is
  `larkspur.example` with 24 pages; each has two competitor pages on the same topic.
- Queries: the 72 in `data/bench/queries.jsonl`, 3 per target page and 18 per intent. None names
  the target brand.
- Retrieval: bge-small-en-v1.5 passages, top 50, MaxP to 15 documents, ms-marco-MiniLM-L-6-v2
  rerank, top 5 sources in the prompt (unchanged from the demo engine).
- User prompt: the GEO-derived instruction in `src/vizor/generate/prompt.py` (sha256 prefix
  `29ee6ad65d4659bd`), then the question and the five sources.
- Model: `qwen2.5:3b-instruct` through Ollama 0.34.4, temperature 0.7, max 450 new tokens,
  `num_ctx` 4096, one request at a time.
- Samples: 5 per query. The seed of sample k of query q is the same in every arm (common random
  numbers); the A/A arm uses a different seed salt.

System prompt A, the one pre-registered prompt for the main run (`configs/bench_qwen3b.yaml`,
sha256 prefix `aec4c3c34309`):

> You answer questions using numbered search results. Write short sentences. Put a citation in
> square brackets at the end of EVERY sentence, right before the period, naming the search result
> that supports that sentence. Format example, with made-up facts unrelated to any question:
> "The Harrow kettle holds 1.7 litres [2]. It ships in three colours [2]. The Linden toaster has
> four slots [4]." Never collect citations at the end of the answer.

It is the prompt of the 2026-08-05 run with the example moved from bicycles (which would now be
the corpus's own subject) to kitchen appliances. It will not be changed after the pilot starts.

## Arms

- Controls: `noop` (must give exactly zero) and `aa_resample` (same prompts, fresh seeds; the
  noise floor).
- Page edits, applied to each query's focus page (the target page retrieval ranks highest for
  it): `metadata`, `faq_rewrite`, `jsonld_insert`, `internal_links`, `stats_surface`,
  `keyword_stuffing`. `metadata`, `faq_rewrite` and `keyword_stuffing` read the tracked queries
  and are cross-fitted over two folds.
- Content-only twins of the six edits (`content:<edit>`): the same edited page, but the sources
  and their order are taken from the unedited corpus, so only the text shown for that page
  changes. The full arm minus its twin is the rank-mediated part of the edit's effect.
- Positive control: the position sweep with the target forced into slot 1 and slot 5 (same
  pages), on a balanced subset of 36 queries.

No LLM rewrite arms, no engine re-ordering arm, no boost sweep, no greedy loop. The bandit replay
is still computed from the page-arm rewards (it costs no calls) but is not part of any test.

## Metrics

- Primary: the target's citation share per answer (valid `[n]` markers pointing at a target
  page divided by all valid markers; 0 when the answer cites nothing), averaged over the 5
  samples of a query.
- Secondary, tested: the named rate, the share of a query's answers that name the target brand
  ("Larkspur", whole word, case-insensitive), with or without a citation.
- Reported, not tested: PAWC share (the GEO position-adjusted word count), the share of an
  answer's sentences that name the brand, cited rate, and per arm the rates of answers with no
  valid citation, answers with citation-like text the parser could not map
  ("unparsed markers"), and answers whose citations all sit on the last sentence.

Citation parsing is `src/vizor/attribution/citations.py` as of this commit, checked against 31
hand-read qwen answers (`tests/fixtures/qwen_citations.jsonl`).

## Analysis and decision rule

- Per query: delta = mean over samples (arm) minus mean over samples (baseline).
- Unit of analysis: the edited page. Queries that share an edited page are averaged into one
  unit: the page for edits that don't read queries (24 units), the page within its fold for the
  cross-fitted edits (up to 48 units; superseded before the pilot by the amendment under
  Deviations, which makes it 24 units for every page arm). Engine-side comparisons use queries
  as units.
- Test: two-sided Wilcoxon signed-rank on unit means, Holm-adjusted within a family, alpha 0.05.
  95% intervals resample units and are descriptive.
- Families: (1) the six full page edits on the primary metric; (2) the same six on the named
  rate; (3) the six content-only twins on the primary metric; (4) the twins on the named rate;
  (5) the slot 5 vs slot 1 positive control alone.
- A page edit "changes how often the site shows up" only if its full arm passes in family 1 or 2.
  It has "a content effect" only if its content-only twin passes in family 3 or 4. The
  rank-mediated part (full minus twin) is reported with an interval and an unadjusted p, and is
  not a claim on its own.
- Because the headline claim passes if family 1 or family 2 passes, and each family is held to
  0.05 on its own, the chance of a false headline claim is up to about 0.10, not 0.05. This was
  not corrected before the run and is stated here so the results are read with it in mind.
- If nothing passes, the result is reported as null together with the MDE below.

## Power

Before any page arm runs, a pilot runs only the baseline and the A/A re-sample with exactly this
design (`vizor experiment --pilot`). Its answers are the same cached answers the main run then
reuses. From the A/A per-query deltas, `vizor sensitivity <pilot> --metric c_share --family 6
--content-family 6` gives the minimum detectable effect at 80% power and the strictest Holm
step (alpha / 6) for page-level units. That number is written into the "Pilot and MDE" section
below and committed before the main run starts. The pilot cannot change the design; it only
states what the design can detect.

## Budget

Prompts don't depend on the answer model here (no LLM rewrites), so a FakeLLM dry run of this
exact config counts the calls: 650 distinct prompt and seed-scheme pairs, or 3,250 model calls
at 5 samples. Most content-only twins reuse their full arm's prompts, because an edit rarely
changes the source list. A timing probe on four bench prompts measured 3.4 s per answer with one
request at a time (67 s for 20 answers) and 2.8 s with two concurrent requests. The run uses one
request at a time: about 3 hours.

## Robustness checks

Both are run after the main run and reported next to it with the same tables. Neither can
change the main result.

1. System prompt B, same model and page arms (no twins, no sweep), 3 samples per query
   (`configs/bench_qwen3b_prompt_b.yaml`, sha256 prefix `c8ff533300a3`):

   > Answer the user's question from the numbered search results only. After every sentence,
   > write the number of each search result that supports it in square brackets, like [3] or
   > [1][4]. If the results do not answer the question, say so.

2. A second model: Qwen3.5-9B (MLX 4-bit, preset `qwen3.5-9b-mlx4`) served by Localhost AI, the
   author's own inference server, with system prompt A, the six page edits and the controls, 3
   samples per query (`configs/bench_qwen9b_localhost_ai.yaml`). Before any 9B answer is
   collected, the same prompt and seed are sent alone and inside a concurrent batch of about 10
   prompts; if the outputs are not byte-identical, the mismatch rate is recorded and the run uses
   one request at a time (or is skipped, with the reason). `scripts/determinism_check.py` does
   this check and writes `determinism.json`. The server's commit hash goes into the run's
   manifest (`vizor experiment --server-commit`).

## Pilot and MDE

The pilot ran on 2026-09-05 from 20:34 to 21:18 (git 0a441a4, after the amendment below), with
`configs/bench_qwen3b.yaml` and `--pilot`: baseline and A/A re-sample only, 72 queries x 5
samples, 720 new qwen2.5:3b calls. Results: `experiments/results/2026-09-05_bench-qwen3b-pilot`.
No page arm had been run when these numbers were written down.

From `vizor sensitivity <pilot> --metric c_share --family 6 --content-family 6` (normal
approximation, 80% power, two-sided, alpha 0.05 / 6 for the strictest Holm step, 24 page units):

| Metric | A/A mean delta | A/A SD per query | A/A SD per page unit | MDE, page edit | MDE, content-only twin |
|---|---|---|---|---|---|
| Citation share (primary) | +0.2 pp | 17.7 pp | 11.1 pp | 7.9 pp | 7.9 pp |
| Named rate | -1.4 pp | 16.9 pp | 10.7 pp | 7.6 pp | 7.6 pp |

At baseline the target's citation share averages 41.9% per answer and 59% of answers name the
brand. So the main run can detect a page edit that moves the citation share by about 8-9
percentage points and will likely miss smaller effects. The 7.9 pp above is the normal
approximation; with t(23) quantiles it is 8.5 pp, and a simulated Wilcoxon test at alpha / 6
has about 76% power at 7.9 pp. With 24 units an exact Wilcoxon test can reach p = 1.2e-7, so every page
arm is testable. The main run reuses the pilot's baseline and A/A answers from the cache.

## Deviations

(Any change from this plan, with the reason, is listed here.)

### Analysis amendment before the pilot (2026-09-05)

The original plan above counted the two cross-fit versions of one page as separate units. They
share the same underlying page, topic and source facts. The full and content-only page arms now
cluster query deltas by the **underlying page across both folds** before the Wilcoxon test and
cluster bootstrap. This makes 24 planned page units, including for cross-fitted edits. The
`page x fold` A/A spread remains a diagnostic only; it cannot raise the inferential sample size.
The planned page MDE uses 24 page means. No answer from the bench pilot or main run existed when
this amendment was made. The corpus, prompts, arms, seeds, primary metric, Holm families and
decision threshold remain as specified above.

The positive-control slot comparison is tested on the run's primary metric, citation share.
The generated verdict and per-page robustness calculation now name and compute that same metric.
PAWC for the slot sweep remains descriptive. The reason: a positive control is only useful if it
checks the measurement the edits are judged on. If the slot sweep were tested on PAWC while the
page edits are tested on citation share, a passing control would say nothing about whether the
citation-share measurement can see a real effect.

### Prompt B is a reduced robustness check (2026-09-05, before the main run finished)

The prompt-B run keeps only the controls (`noop`, `aa_resample`) and the six full page edits,
with 3 samples per query instead of 5, and has no content-only twins and no position sweep
(`configs/bench_qwen3b_prompt_b.yaml`, unchanged since it was committed with this design). The
reason is GPU time on a machine shared with two other projects: the reduced design needs about
a third of the main run's calls. It can say whether the direction and rough size of the page-edit
effects hold under a different system prompt; it has less power than the main run and is not
used for the content vs rank question. No main-run result had been read when this was written.

### Corpus and query disclosure (written after the main run)

Two properties of the bench limit what the results can say, and should have been stated up front:

- The corpus was written knowing which edits would be tested. The target site's pages lack
  exactly the features the edits add: the 24 Larkspur pages have no FAQ block and no JSON-LD,
  while all 12 cogandchain pages have both. An edit that adds one of these features therefore
  always has something to add, which a real site might not.
- The queries are hand-written, 3 per topic, and the target page is retrieved for 93% of
  answers. The cross-fitted edits read the other fold's queries, and those include same-topic
  siblings of the held-out query. "Held out" here means a held-out query, not a held-out topic.

### Prompt B dropped (2026-09-18, superseded by Study 2)

The prompt-B robustness run was not run. After the main run, the page-edit MDE (about 8-9 pp)
was found to come from sampling noise at temperature 0.7 rather than from the number of pages,
so a second sampled run with 3 samples per query would have had even less power than the main
run. The GPU time is spent instead on Study 2 (`docs/bench-design-study2.md`), which uses a
teacher-forced citation metric with no sampling noise and new, pre-registered queries. The
prompt-B config stays in `configs/` for reference. The 9B robustness run is still planned as an
optional run after Study 2.
