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
   delta of exactly 0. The rejection rate is reported in the manifest (`rewrite_guard`). The guard
   was amended before the gate, and the arm needs at least 12 accepted pages; see Deviations.
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
   same bootstrap. (Replaced by relative precision; see the amendment under Deviations.)
4. Across Study 1's twelve page arms (six full, six content-only), the Spearman correlation
   between per-page AP deltas and per-page citation-share deltas of the same prompts is at least
   0.5, pooled over (arm, page) points. (Replaced by a reliability-corrected test on the six
   content-pinned arms; see the amendment under Deviations.)

If the gate fails, the failure is reported, and Study 2's primary metric becomes the sampled
citation share with the same units, weighting, tests and Holm over the three arms. AP is then
reported as a secondary metric. Criterion 4 is demanding: most Study 1 arms were null and the
sampled citation share has a page-level SD of about 11 points, which attenuates any correlation,
so a failure there does not by itself mean AP is wrong. It still counts as a failure.

## MDE

(Filled in from `gate.json` after the gate and before Study 2 starts: the page-level SD of the
content-pinned AP deltas of Study 1's page arms, t(23) quantiles, 80% power, two-sided alpha
0.05 / 3, reported as a range over arms.)

Filled in on 2026-09-18 from `experiments/results/2026-09-18_study2-gate/gate.json`, before any
Study 2 answer: the page-level SDs of the six content-pinned AP deltas on Study 1 range from
2.97 to 7.22 pp, which gives an MDE on AP of 2.1 to 5.1 pp (24 pages, t quantiles, 80% power,
two-sided alpha 0.05 / 3). The gate failed (see Deviations), so AP is a secondary metric and
these numbers describe it only. The primary is now the sampled citation share, whose MDE at 3
samples is computed from Study 2's own A/A re-sample and reported with the results; Study 1's
was about 8 pp at 5 samples, so something near or above that is expected.

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

### Amendment before the gate (2026-09-18, before any GPU run)

A review of the code and design before the gate found problems in two gate criteria, in the
rewrite guard and in how reference answers are scored under reordered prompts. No gate score,
Study 2 answer, rewrite or score existed when this was written. The changes:

1. Gate criterion 4 is replaced. The sampled citation-share deltas are noisy at the page
   level: in Study 1 the A/A re-sample's page-delta variance is 122.71, and the pooled page-delta
   variance of the six content-only arms is 171.38, so their reliability is
   rel = 1 - 122.71 / 171.38 = 0.284. Even a perfect metric could only correlate with them at
   about sqrt(0.284) = 0.53, which made the old threshold of 0.5 close to unpassable. The new
   criterion uses the six content-pinned arms only (the full arms repeat the same pages with rank
   noise added). It passes if rho / sqrt(0.284) >= 0.5 (a raw Spearman rho of at least about
   0.27), rho > 0, and a one-sided page-permutation p < 0.05. The permutation relabels pages the
   same way in every arm, so arm-level agreement (every page moving with the FAQ arm) is kept
   under the null and the test asks whether AP agrees with the citation share page by page.
   rel = 0.284 is fixed in `configs/study2_gate.yaml`; the gate also reports the value it
   measures from the run, for information.
2. Gate criterion 3 is replaced. AP deltas are in points of citation probability and citation
   share deltas in points of marker share, so a ratio of CI half-widths depends on the scale of
   each metric. The criterion now compares relative precision, |delta| / CI half-width: it
   passes if AP's is at least 1 / 0.6 times the citation share's, for the same content-pinned
   FAQ prompts and the same bootstrap.
3. Reported for every AP comparison (secondary, no claims): the share of queries whose focus
   page is among the prompt's sources, and the number of pages whose delta is exactly 0 (for
   `evidence_surface_llm`, these include the rejected pages).
4. In full mode and in the slot control the arm's prompt numbers the sources differently from
   the baseline the reference answer was written against. The reference's citation indices are
   now renumbered through the sources' page ids before scoring, so earlier markers in the prefix
   still point at the same pages; an index whose page is not in the arm's prompt keeps its
   number and is counted. AP from each reference's first citation site alone, whose prefix holds
   no earlier marker, is reported next to every comparison as a check that criterion 1 and the
   full-mode results do not come from the prefix's markers. Content-pinned prompts keep the
   baseline order, so the primary analysis is unchanged.
5. The rewrite guard for `evidence_surface_llm` is tightened and loosened where it was wrong.
   It flagged ordinary sentence starters ("This", "Additionally") and words from the page's own
   headings, which would have rejected most rewrites. It now checks capitalized words after a
   sentence's start against the page's body, title, description, headings and FAQ, and a
   sentence-initial word only if it looks like a name (followed by another capitalized word, or
   capitalized elsewhere mid-sentence) and is not a common starter. It could also be bypassed,
   so it now also compares number and unit pairs ("160 psi" cannot become "160 bar"), rejects
   spelled-out number words the page does not use, and requires that at least 70% of the page's
   distinct content words survive, next to the existing 90% of its numbers.
6. The 24 rewrites are generated right after the gate, before any Study 2 answer, with
   `scripts/pregenerate_rewrites.py` and the run's rewriter config, and the run reuses them from
   the cache. The pre-registered minimum is 12 accepted pages of 24
   (`sandbox.rewrite_min_accepted`). With fewer, most of the arm's pages would be unedited and
   its delta would mostly measure zeros, so the arm is reported as inconclusive: its comparisons
   stay in the table and in the Holm family of three, and it cannot count as an effect.
7. Key-fact detection for `answer_first` no longer treats "in", "m", "l" or "x" after a number
   as a unit (spec patterns such as "2 x 10" are still matched on their own).

Implementation fixes with no effect on the design: the scorer renormalizes each site's
probabilities itself from the stored log-probabilities (the server's renormalized values are
only checked against them, to 1e-5), so `vizor recompute` reproduces every AP value exactly;
and it refuses to score unless the server reports its commit, checkpoint revision and tokenizer
hash, with the chat-template options also part of the cache key.


### Second amendment, before the gate and the rewrites (2026-09-18)

Written after the first amendment and before any gate score, rewrite, Study 2 answer or score
exists. Each item was committed on its own, before the GPU step it affects.

1. The rewriter changes (`evidence_surface_llm` only). It was `qwen2.5-3b-mlx4` at temperature
   0; it is now `qwen3.5-9b-mlx4` (Qwen3.5-9B, MLX 4-bit, pinned revision in Localhost AI's
   `models.yaml`) with thinking on, served by Localhost AI in its own session. Reasons: the
   rewriter is no longer the answer model, so the arm is less exposed to a model preferring its
   own phrasing; a larger model that reasons first should pass the grounding guard more often;
   and rewriting with a stronger model than the one answering is closer to how such edits are
   made in practice. Settings, in the `rewriter` block of `configs/study2_qwen3b_localhost_ai.yaml`:
   temperature 0.6 and top_p 0.95 (Qwen's advice for thinking mode), seed 0, at most 4,096 new
   tokens of which at most 2,048 are thinking (`max_thinking_tokens`; the server closes the
   block at the budget and the model then writes the page), and
   `chat_template_kwargs: {enable_thinking: true}` sent per request. The server's preset is not
   changed, so every answering run still has thinking off. The server returns the thinking block
   separately; only the final page text goes through the guard and into the arm. The server
   runs with `LHAI_MAX_CONTEXT=8192` so a page plus 4,096 new tokens fits.
   The 24 rewrites are generated once by `scripts/pregenerate_rewrites.py`, and the output is a
   frozen artifact committed with the results: `rewrites.json` (every page's raw output,
   verdict, violations and thinking-token count, plus the rewriter's model, revision, server
   commit, seed and settings) and `reasoning.jsonl` (the thinking blocks, kept for provenance
   and never shown to an answer model). The run reads the pages from that file
   (`sandbox.frozen_rewrites`) and refuses it if its rewriter settings or prompt differ from the
   config's; it makes no rewriter call. This replaces "the run reuses them from the cache" in
   item 6 above. How many of the 24 rewrites reached the thinking budget is reported; a
   truncated rewrite that fails the guard counts as rejected like any other. Unchanged: the
   guard, the minimum of 12 accepted pages, the other two arms, and the answer model.
2. Everything sampled or scored runs at batch 1 and concurrency 1. Localhost AI's own check
   (its commit 2aeada3) found that its MLX models give byte-identical output from run to run
   when a request runs alone, but not when it shares a batch with other requests, at any batch
   width from 2 up (the padded multi-row prefill appears to be the cause). The earlier plan
   above ("an answer sent alone and inside a concurrent batch is byte-identical") therefore
   cannot be met by batching, and is replaced: every server session in this study (the
   sampled answers, the reference answers, the rewrites and the `/v1/score` scoring) starts
   with `LHAI_CONTROLLER=fixed LHAI_FIXED_BATCH=1`, and the client keeps `workers: 1`.
   `scripts/determinism_check.py` still runs on the answer model before its first sampled
   answer, now at that setting: ten prompts sent alone twice, then all at once (the server
   queues them and runs them one at a time), and it passes only if all three sets are
   byte-identical. The result is saved with the run.
3. A replication on a second model family, run last, after Study 2: Gemma 4 E4B
   (`gemma-4-e4b-mlx4`, Gemma 4 E4B instruct, MLX 4-bit) served by Localhost AI, with thinking
   off (`chat_template_kwargs: {enable_thinking: false}` on every answer and every scoring
   request). Only the answer and scoring model changes. The pages are the same, including the
   frozen Qwen3.5-9B rewrites, so this is a cross-family check of the same edits. Two parts:
   - Study 1's key arms on Study 1's 72 queries (`configs/gemma_e4b_study1.yaml`), sampled at
     3 samples per query (Study 1 used 5): `noop`, `aa_resample`, `faq_rewrite`,
     `content:faq_rewrite`, and the slot control (target forced to slot 1 or slot 5 on 36
     queries). The primary metric is the sampled citation share with Study 1's weighting and
     tests (Holm-adjusted exact Wilcoxon on page units). `faq_rewrite` is the only page arm, so
     Holm over one arm changes nothing; `content:faq_rewrite` is its own family of one.
   - Study 2's arms on Study 2's 72 queries (`configs/gemma_e4b_study2.yaml`): the same arms,
     controls, prompt-only sets, 3 samples, and AP scored by Gemma through `/v1/score` against
     reference answers that are Gemma's own baseline samples 0 and 1. The same tests as Study 2,
     with Holm over the three content-pinned arms and the same both-tests rule.
   Claim rule, fixed now: a Study 1 or Study 2 result "replicates" on Gemma if the Gemma
   estimate has the same sign and its Holm-adjusted p is below 0.05 (for Study 2's AP arms, both
   Holm-adjusted p values). A result that does not meet this is reported as not replicated,
   whatever the reason. With 3 samples the replication has less power than the originals; its
   MDE is computed afterwards from its own page-level SDs (the Study 1 part from the A/A
   re-sample, the Study 2 part the same way as Study 2's) and reported with the results. Before
   the first Gemma answer the determinism check runs on Gemma at batch 1 (item 2), and
   `/v1/score` on Gemma is checked against in-process mlx-lm scoring (`MLXScorer`) on a sample
   of rows, since Gemma's sliding-window layers are a different code path from Qwen2.5's.
4. `MLXScorer`, the in-process mlx-lm scorer used to check `/v1/score`, now passes
   `enable_thinking: false` to the chat template unless told otherwise. mlx-lm's tokenizer
   wrapper turns thinking on by default for Gemma 4 and Qwen3.5, while Localhost AI scores with
   it off, so without this the two would build different prompts for those models and the
   equivalence check in item 3 would compare different things. Qwen2.5's template ignores the
   option, so nothing scored with Qwen2.5 changes.

### Gate result (2026-09-18, before any Study 2 answer)

The gate ran on `qwen2.5-3b-mlx4` through Localhost AI at commit c61e05d (a docs and tests
commit on top of 154b4cd; the server code is the same), at batch 1, and failed two of four
criteria (`experiments/results/2026-09-18_study2-gate/gate.md`):

1. Slot 5 vs slot 1 on AP: -44.4 pp [-58.5, -29.7], Holm p 0.0002 (sign-flip) and 0.0002
   (Wilcoxon) on 17 pages. Pass.
2. FAQ rewrite, content-pinned, on AP: -8.1 pp [-11.1, -5.4]. Pass.
3. Relative precision: AP's |delta| / CI half-width is 2.83, the citation share's 2.24. AP's
   is 1.27 times the citation share's, short of the 1 / 0.6 = 1.67 required. Fail.
4. Spearman between AP and citation-share page deltas over the six content-pinned arms: rho
   0.25 over 138 points, 0.47 after the reliability correction (rel 0.284; the run itself
   measures 0.284), one-sided page-permutation p 0.33. Fail.

As pre-registered, Study 2's primary metric is therefore the sampled citation share, with page
units and weighting, the same two tests and Holm over the three arms; AP is reported as a
secondary metric with no claims. Criterion 4's failure means AP does not track the sampled
citation share page by page on Study 1's mostly null arms; it does not show that AP is wrong,
and the design said beforehand that it would count as a failure anyway.

Operational note: during the gate some `/v1/score` requests got no reply (the server went on
serving later requests). The first attempt stopped on a 600 s timeout; the scorer now resends a
request after 60 s on a fresh connection (b6c47a0, c017b31), and the gate that counts resent 9
requests. Scoring is deterministic at batch 1 and cached, so a resent request returns the same
values; `vizor recompute` matches all 2,160 rows.
