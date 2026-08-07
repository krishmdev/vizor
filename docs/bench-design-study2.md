# Study 2 design (written before the run)

Written on 2026-09-18, after Study 1 (`docs/bench-design.md`, results in
`experiments/results/2026-09-05_bench-qwen3b`) and before any Study 2 answer, rewrite or score
exists. The validation gate below runs first, on Study 1's stored data. The MDE section is filled
in from the gate before Study 2 starts. Nothing else in this file changes after the gate starts;
any later change goes under Deviations with its reason.

## Why a second study

Study 1 found one clear effect: the FAQ rewrite lowered the target's citation share by 15.4
points, and its content-only twin gave the same -15.3, so the loss comes from the page text, not
from retrieval rank. The other five edits were null, with an MDE of about 8-9 points. Moving the
target from the first to the fifth source slot lowered its citation share by 30 points.

Two things limit what Study 1 can say next:

- The MDE is set by sampling noise at temperature 0.7 (the A/A re-sample has a per-query SD of
  about 18 points), not by the number of pages. More samples of the same metric buy little.
- The FAQ loss has a likely mechanism. The prompt shows each source's three passages most similar
  to the question, FAQ passages included. `faq_rewrite` appends up to four FAQ pairs whose
  answers are page sentences, and those passages can push the spec and fact passages out of the
  three slots. For the floor pump page and query comp-11, the specs passage ($49, 160 psi)
  disappears from the prompt and a FAQ pair about truing a wheel takes its place.

Study 2 therefore (1) replaces the primary metric with one that has no sampling noise, (2) checks
that metric against Study 1 before relying on it, and (3) tests three edits chosen from the
diagnosis on queries that did not exist when Study 1 ran.

## Engine and corpus

- Corpus: the same `data/bench` corpus as Study 1 (72 pages, 24 on the target site
  `larkspur.example`). The disclosure in `docs/bench-design.md` applies unchanged: the corpus was
  written knowing which edits would be tested, and the target pages have no FAQ block and no
  JSON-LD.
- Queries: 72 new queries in `data/bench/queries_study2.jsonl`, 3 per target page and 18 per
  intent, loaded through `data/bench/project_study2.yaml`. They were written after Study 1's
  results were known, by reading the target pages; no Study 2 output existed. None names the
  target brand and none repeats a Study 1 query. The same caveat as Study 1 holds: they are
  hand-written, a few per topic, and aimed at pages the target has.
- Retrieval: unchanged from Study 1 (bge-small-en-v1.5, top 50 passages, MaxP to 15 documents,
  ms-marco-MiniLM-L-6-v2 rerank, top 5 sources).
- Passage policy: `query-top3`, Study 1's rendering, for every primary comparison.
- Model: `qwen2.5-3b-mlx4` (Qwen2.5-3B-Instruct, MLX 4-bit) served by Localhost AI, the author's
  inference server. The same server and weights produce the sampled answers, the LLM page
  rewrites (temperature 0) and the teacher-forced scores. These are not Study 1's weights
  (Ollama's GGUF Q4_K_M), so no Study 1 number is compared with a Study 2 number as if they came
  from one model.
- System prompt A from Study 1, word for word. Sampled answers use temperature 0.7, 450 new
  tokens, 3 samples per query and the same common-random-number seeds.
- Before any sampled answer, `scripts/determinism_check.py` checks that an answer sent alone and
  inside a concurrent batch is byte-identical. The run uses one request at a time either way.
- Config: `configs/study2_qwen3b_localhost_ai.yaml`. The server's commit goes into the manifest
  and into both caches (`--server-commit`).

## Primary metric: attribution propensity (AP), content-pinned

Take a reference answer R, which is baseline sample 0 or 1 of this study's own run. Each citation
site of R is the first index digit of a marker whose first index is a valid source (the same
marker parser as Study 1). Under a prompt P, the scorer returns the log-probability of each
index "1".."5" as the next text at that site, given P and R's text before the site, and the
probabilities are renormalized over the prompt's valid indices. AP(P, R) is the mean over R's
sites of the probability on the target's slots.

- Content-pinned (primary): the arm's prompt is its `content:` twin, which keeps the baseline's
  sources in the baseline's order and changes only the edited page's text. The target's slot is
  the same in both prompts.
- For each page arm a, query q and reference R: delta = AP(content:a, R) - AP(baseline, R). A
  query's delta is the mean over its reference answers with at least one citation site; a page's
  delta is the mean over its queries. Queries whose references have no citation site drop out.
- Because R is fixed and nothing is sampled, identical prompts give a delta of exactly 0. The
  A/A difference is 0 by construction; the variance left is between reference answers and pages.
- One scorer call per (prompt, query, reference) scores every site of that reference. The rows
  keep each site's candidate log-probabilities, and `vizor recompute` rebuilds AP from them.

## Arms

All three are query-blind: they read only the page, so there is no cross-fitting and no query
can leak into its own edit.

1. `answer_first`: moves the page's key-fact sentences into a new first paragraph, so they fall
   in the first body passage. A key fact is a sentence with a price, a number with a unit (kg,
   psi, mm, W, Wh, lumens, percent and the like) or a spec pattern ("2 x 10", "7-speed"). At most
   4 sentences and 90 words, in page order. Nothing is reworded.
2. `evidence_surface_llm`: GEO's statistics addition, restricted to numbers already on the page.
   The rewriter (a separate `rewriter` config, the same model at temperature 0) is asked to state
   the page's own numbers early and explicitly and to add nothing. A guard rejects the output if
   it contains any number or capitalized name that is not on the page, or keeps fewer than 90% of
   the page's distinct numbers. A rejected page keeps its original text, which gives that page a
   delta of exactly 0. The rejection rate is reported in the manifest (`rewrite_guard`).
3. `faq_rewrite_v2`: a direct test of the diagnosis. Questions come from the page's own section
   headings, not from queries. Each answer is the single page sentence that best matches its
   question, no sentence is used twice, and at most 2 pairs are added (the two best matched).

Controls: `noop`, `aa_resample` and the slot control (target forced into slot 1 or slot 5, on a
36-query subset balanced across intents, chosen the same way as in Study 1). All are reported on AP and on the sampled metrics.

Prompt sets scored: the baseline, every sampled arm, the three content-only twins (prompt only,
not sampled), and the baseline, the arms and the twins re-rendered under `body-top3` (prompt
only). The full list of comparisons is in the config.

## Analysis and decision rule

- Unit: the target page, 24 units. The estimate, the 95% percentile bootstrap interval (5,000
  resamples of pages) and both tests use the same unweighted mean of page deltas.
- Tests on the 24 page deltas: an exact two-sided Wilcoxon signed-rank test (zero deltas dropped;
  the normal approximation if nonzero |delta| values tie) and a two-sided sign-flip permutation
  test on the mean with 20,000 draws.
- Family: the three content-pinned AP comparisons. Holm is applied over these three, separately
  for each test.
- An arm "changes how often the target is cited" only if both its Holm-adjusted p values are
  below 0.05. The direction is reported whichever it is. This replaces Study 1's "family 1 or
  family 2" rule, which allowed a familywise error of up to about 0.10: there is one primary
  metric and one family, and requiring both tests can only lower the error rate.
- Everything else is secondary, reported with intervals and unadjusted or within-family Holm p
  values, and not used for claims:
  - full mode (the arm's own retrieval) and the rank-mediated part (full minus content-pinned);
  - the `body-top3` re-render, which tests the displacement mechanism directly;
  - the sampled citation share and named rate at 3 samples, page-weighted, from `vizor report`,
    to show whether AP moves with real citations;
  - the slot control on AP;
  - leave-one-passage-out tables (`vizor attribute`) for chosen queries.

Predictions, stated so they can be checked, not tested as hypotheses: `faq_rewrite_v2` harms
less than Study 1's `faq_rewrite` under `query-top3`, and has a delta of exactly 0 under
`body-top3`, where FAQ passages are never shown. `answer_first` is not negative. No prediction
for `evidence_surface_llm`.

## Validation gate (runs first, on Study 1)

Before Study 2 runs, AP is computed on Study 1's stored prompts with `qwen2.5-3b-mlx4`, using
Study 1's baseline samples 0 and 1 as reference text (they were generated by Ollama and are used
only as text). `scripts/validation_gate.py` with `configs/study2_gate.yaml` does this and writes
`gate.json`. The gate passes only if all four hold:

1. AP detects the slot effect: slot 5 minus slot 1 is negative, with both Holm p values below
   0.01 (the slot comparison is its own family, on page units).
2. AP gives the FAQ rewrite (content-pinned, Study 1's `content:faq_rewrite` prompts) a
   negative sign.
3. That AP delta's page-level 95% CI half-width is at most 60% of the half-width of Study 1's
   sampled citation-share delta for the same prompts, both from unweighted page means with the
   same bootstrap.
4. Across Study 1's twelve page arms (six full, six content-only), the Spearman correlation
   between per-page AP deltas and per-page citation-share deltas of the same prompts is at least
   0.5, pooled over (arm, page) points.

If the gate fails, the failure is reported, and Study 2's primary metric becomes the sampled
citation share with the same units, weighting, tests and Holm over the three arms. AP is then
reported as a secondary metric. Criterion 4 is demanding: most Study 1 arms were null and the
sampled citation share has a page-level SD of about 11 points, which attenuates any correlation,
so a failure there does not by itself mean AP is wrong. It still counts as a failure.

## MDE

(Filled in from `gate.json` after the gate and before Study 2 starts: the page-level SD of the
content-pinned AP deltas of Study 1's page arms, t(23) quantiles, 80% power, two-sided alpha
0.05 / 3, reported as a range over arms.)

## Budget

A FakeLLM and FakeScorer dry run of this exact design (`configs/study2_dryrun.yaml`) makes 1,512
sampled answers, of which 1,188 are distinct prompt and seed pairs (`noop` reuses the
baseline's), 24 rewrite calls, and 2,016 scored (prompt set, query, reference) rows, of which 776
are distinct scorer calls in that dry run. The real count of scorer calls depends on how many
distinct prompts the edits produce. At Study 1's Ollama rate of 3.4 s per answer (Localhost AI's
rate on this model has not been measured here), the sampled answers take about 1 hour and 10
minutes. The gate scores 2,160 rows (1,156 distinct calls) on Study 1's prompts.

## What this study cannot say

It uses one small local model, a synthetic corpus written knowing the edits, and hand-written
queries. It says nothing about ChatGPT, Perplexity or any production answer engine. AP measures
where a fixed answer's citations would point, not whether a model would write a different answer
after an edit; the sampled metrics are there to check that the two move together.

## Deviations

(Any change from this plan, with the reason, is listed here.)
