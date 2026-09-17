# Study 3 design (written before the run)

Written on 2026-09-18, after Study 2 (`docs/bench-design-study2.md`, results in
`experiments/results/2026-09-18_study2-qwen3b`) and before any Study 3 answer exists. No Study 3
query has been sent to any language model. The pilot below runs first; its section is filled in
from the pilot and committed before the main run starts. Nothing else in this file changes after
the pilot starts; any later change goes under Deviations with its reason.

## Why a third study

Studies 1 and 2 tested content edits (a FAQ block, moving key facts to the lead, an LLM rewrite
that surfaces the page's numbers). None raised the target's citation share, and most lowered it:
in Study 2 all three arms lost 4 to 10 points. The likely mechanism is the engine's passage
policy. Each source shows its three body or FAQ passages most similar to the question, and the
edits put new text into those slots in place of the passages that answered the question.

The one large lever in both studies was position: moving the target from the first source slot
to the fifth cost 30 to 44 points. Relevance and position dominate in the published work on
generative engines too.

Study 3 therefore tests edits built to win passage selection and retrieval rank, rather than
edits that restate the page. It also moves to a stronger answer model, Qwen3.5-9B.

## Engine and corpus

- Corpus: the same `data/bench` corpus as Studies 1 and 2 (72 pages, 24 on the target site
  `larkspur.example`). It is synthetic, and it was written knowing which kinds of edit would be
  tested; see `data/bench/README.md` and `docs/bench-design.md`.
- Retrieval: unchanged (bge-small-en-v1.5, top 50 passages, MaxP to 15 documents,
  ms-marco-MiniLM-L-6-v2 rerank, top 5 sources). Passage policy `query-top3`.
- Answer model: `qwen3.5-9b-mlx4` (Qwen3.5-9B, MLX 4-bit) served by Localhost AI, the author's
  inference server, at commit ce3d03d or later (the commit is recorded in every manifest and in
  the cache key). Thinking is off (the preset's default, and the request also sends
  `chat_template_kwargs: {enable_thinking: false}`). The server runs with
  `LHAI_CONTROLLER=fixed LHAI_FIXED_BATCH=1` and the prefix cache off (`LHAI_PREFIX_CACHE=0`),
  and Vizor sends one request at a time (`workers: 1`). Localhost AI found this model
  byte-identical run to run at batch 1 but not batch-invariant from batch 2 up.
- Before any Study 3 answer, `scripts/determinism_check.py` runs on this model at batch 1 with
  thinking off, on 10 Study 3 prompts, and its result is committed with the pilot.
- System prompt A from Studies 1 and 2, word for word. Temperature 0.7, 450 new tokens, common
  random-number seeds (sample k of a query has the same seed under every arm).
- Configs: `configs/study3_qwen9b.yaml` (main), `configs/study3_qwen9b_pilot.yaml` (pilot; the
  same model block, so the main run reuses the pilot's answers from the cache) and
  `configs/study3_dryrun.yaml` (FakeLLM, same arms and retrieval).

## Queries

72 new queries in `data/bench/queries_study3.jsonl`, loaded through
`data/bench/project_study3.yaml`: 3 per target page (the `topic` field) and 18 per intent. They
were written by hand after Study 2, by reading the target pages. None names the target brand, and
none repeats a Study 1 or Study 2 query; `tests/test_study3.py` checks the counts and the
overlap.

Six queries were reworded after checking the baseline retrieval only (the FakeLLM dry run; the
cascade does not depend on the answer model), because with the first wording the frame-size
guide and then the Metro 7 page were the focus page of no query at all, which would have left 23
page units instead of 24:

| Query | First wording | Final wording |
|---|---|---|
| s3-comp-07 | medium or large frame if i am 182 cm tall | size chart vs test ride for choosing a bike frame size |
| s3-tran-07 | which size commuter bike should i buy if i am 170 cm | what frame size gravel bike should i buy at 175 cm tall |
| s3-trou-07 | i cannot stand over my bike frame without it touching | my bike frame has almost no standover clearance |
| s3-info-11 | how much weight can the rear rack on a city bike carry | does a city bike with a hub gear need a chainguard |
| s3-comp-11 | chainguard vs no chainguard for riding to work in trousers | city bike with a chainguard vs a gravel bike for riding in work clothes |
| s3-trou-12 | my derailleur gets dirty and skips on my city bike | my city bike chain keeps getting my trousers dirty |

The Metro 7 queries went through one more wording each in between. With the final queries,
every target page is the focus of 1 to 4 queries (4 queries: 8 pages, 3: 10, 2: 4, 1: 2), and 60
of the 72 queries have their own topic's page as focus. As in Studies 1 and 2, the focus page is
the target page the baseline retrieval ranks highest for the query, and each page arm edits only
that page.

## Arms

All three arms read only the page (and, for `retrieval_meta`, how common each term is across the
corpus). None reads a query, so there is no cross-fitting. None invents anything: every sentence
comes from the page, and the only added words are the product and brand name, "For the", "On
the", "In the ... guide" and the page's own key terms. Their output still goes through the
grounding guard from Study 2 (no number, quantity or capitalized name that is not on the page),
and the guard's verdict and whether the page changed at all are logged per page in the
manifest's `rewrite_guard` (the no-op flags `unchanged_pages` and `reordered_pages`). In the dry
run the guard accepted all 72 edits.

1. `fact_passage`: one self-contained key-facts paragraph, placed after the intro. It collects
   the page's sentences that carry a price, a number with a unit or a spec pattern (the same
   `key_fact` test as Study 2's `answer_first`), in page order, up to 110 words so that it falls
   inside one 120-word body window. Each sentence is made to name the product and brand: a
   leading bare reference is replaced ("The pump reaches 160 psi" becomes "The Larkspur Floor
   Pump reaches 160 psi"), and otherwise the sentence gets "For the Larkspur Floor Pump, ..." (on
   a guide, "In the Larkspur Tire Pressure guide, ..."). The original sentences stay where they
   were. Every one of the 24 pages gets a passage (34 to 108 words).
2. `entity_anchor`: nothing is rewritten except the first sentence of each ~120-word body window
   (the windows the retriever and the passage policy use). In that sentence a leading pronoun
   ("It", "Its") or bare reference to the product ("The Beam 800", "The pump", "This bike")
   becomes the product and brand name; a leading bare reference to a part ("The motor is ...")
   becomes "On the Larkspur Haul, the motor is ..."; and an unbranded name of one of the site's
   products gets the brand. At most 16 words are added per page. This changes the 12 product
   pages (+1 to +8 words each). The 12 guides have no product of their own and no unbranded
   product name in a window's first sentence, so they are unchanged, and their queries get a
   delta of exactly 0 in full mode.
3. `retrieval_meta`: the title, the meta description, the H1 and the H2s are rebuilt from the
   page's own key terms. Key terms are alphabetic 1-3-grams of the page with no function word in
   them (a multi-word term must occur twice), scored by TF-IDF of the page against the 72-page
   corpus, excluding the brand and the words of the H1, no two overlapping, in the page's own
   spelling. Title: H1 plus the top three terms, at most 70 characters. H1: H1 plus the top term.
   Each H2: the H2 plus the unused term closest to it (bi-encoder cosine). Description: the page
   sentences carrying the most key terms, at most 155 characters. The body is not touched. The
   head passage (title, description, headings) is retrievable, and the title and description
   are shown in the prompt, so this edit works through rank and through those two fields; it
   can never change which body passages are shown. Study 1's `metadata` arm used query
   keyphrases and was null; this one reads no query and aims at relevance.

Controls: `noop` (must give exactly 0) and `aa_resample` (the same prompts with fresh seeds, the
noise floor). Content-only twins `content:<arm>` pin the sources and their order to the
baseline's and change only the edited page's rendered text; they run at 1 sample
(`sandbox.arm_samples`) and are secondary.

## What retrieval does with the edits (known before any answer)

Retrieval and prompt building do not depend on the answer model, so the FakeLLM dry run
(`vizor experiment --config configs/study3_dryrun.yaml`) logs exactly what the real run will
log in `retrieval.csv` (one row per arm and query: the focus page's candidate rank and source
slot, whether its MaxP passage is edited or new, and how many of its shown passages are edited
or new). Summary over the 72 queries:

| Arm | Focus page shown | Mean candidate rank | Edited/new passage shown (queries where shown) | MaxP passage edited/new |
|---|---|---|---|---|
| baseline | 100% | 1.74 | 0% | 0% |
| `fact_passage` | 95.8% | 2.27 | 100% | 61% |
| `entity_anchor` | 98.6% | 1.72 | 45% | 6% |
| `retrieval_meta` | 100% | 1.56 | 0% (by design) | 22% (the head passage) |
| `content:fact_passage` | 100% | 1.74 (pinned) | 100% | 0% |
| `content:entity_anchor` | 100% | 1.74 (pinned) | 46% | 0% |

So the key-facts passage wins a display slot every time its page is shown, but it moves the page
down (it becomes the page's best-matching passage for 61% of queries, and the cross-encoder
scores it below the passage it replaced). `retrieval_meta` moves the page up, by 0.18 ranks on
average. `entity_anchor` changes almost nothing about rank.

## Primary metric, tests and decision rule

- Primary: the sampled citation share (C-SoV: the target's share of the answer's valid citation
  markers), in full mode (the edited page is re-indexed, so retrieval can rank it differently;
  rank is part of the mechanism being tested), at 3 samples per query.
- Unit: the focus page, 24 units. A query's delta is the mean over samples of arm minus
  baseline; a page's delta is the mean over its queries; the estimate is the unweighted mean of
  the 24 page deltas, with a 95% percentile bootstrap interval (5,000 resamples of pages).
- Tests on the 24 page deltas: the exact two-sided Wilcoxon signed-rank test (zero deltas
  dropped; the normal approximation if nonzero |delta| values tie) and the two-sided sign-flip
  permutation test on the mean (20,000 draws, as in Study 2; the exact 2^24 version is reported
  next to it). Holm over the three arms, separately for each test.
- Decision rule. An arm changes how often the target is cited only if:
  1. both Holm-adjusted p values are below 0.05, and
  2. its delta against the A/A arm (arm minus `aa_resample`, same page units) has the same sign
     as its delta against the baseline.
  Condition 2 is the robustness check the Study 2 review asked for: the baseline is one draw of
  3 samples per query, and a lucky or unlucky baseline moves every arm's delta the same way. The
  direction is reported whichever it is.
- Computed by
  `scripts/sampled_primary.py <run> --config configs/study3_qwen9b.yaml --arms fact_passage entity_anchor retrieval_meta --require-aa-sign --study "Study 3"`.
- AP (teacher-forced attribution propensity) is not used: it failed the Study 2 validation gate.

## Secondary metrics (no claims)

Reported with intervals and raw p values by `scripts/study3_secondary.py`:

- the named rate (the answer names the target brand), arm minus baseline;
- the content-vs-rank split for each arm: the full-mode delta, the content-only twin's delta and
  the rank-mediated rest (full minus twin); the twins have 1 sample, so their deltas are noisier;
- the focus page's retrieval rate (shown among the 5 sources) and mean candidate rank, arm minus
  baseline, per page;
- the passage-selection share: how often an edited or new passage of the page is among the
  passages shown.

## Predictions

These are stated so they can be checked afterwards; they are not tested as hypotheses.

- `fact_passage`: negative. The passage is always shown when the page is, so it takes one of the
  three slots from a passage chosen for the question, which is the displacement that hurt in
  Studies 1 and 2; and the dry run shows the page losing rank (mean 1.74 to 2.27, and shown for
  95.8% instead of 100% of queries), and position is the strongest lever measured. Every sentence
  naming the product might help attribution, but not enough to outweigh both.
- `entity_anchor`: positive but small, below the MDE, so not expected to meet the rule. It adds
  1 to 8 words to 12 pages, leaves the other 12 unchanged, and barely moves rank; a named
  passage should be slightly easier to attribute, and nothing is displaced.
- `retrieval_meta`: positive, most likely below the MDE. The page moves up by 0.18 ranks on
  average, and the slot effect was 30 to 44 points over four slots, which suggests about +1 to +2
  points from rank; the new title and description are shown to the model but change no passage.

## Pilot and MDE

Before the main run, the pilot (`configs/study3_qwen9b_pilot.yaml`) runs the baseline, `noop`
and `aa_resample` at 2 samples on the Study 3 queries: 72 x 2 x 2 = 288 generated answers
(`noop` reuses the baseline's). `scripts/study3_pilot.py` then computes:

- the A/A page SD at 2 samples and the MDE (t quantiles on 23 df, 80% power, two-sided alpha
  0.05 / 3), and the same projected to the main run's samples by multiplying the SD by
  sqrt(2 / k), which assumes the A/A spread is sampling noise;
- the mean time per generated answer (cache hits excluded).

Budget contingency, pre-registered: if the pilot's mean time per answer is above 15 s, the main
run uses 2 samples per query instead of 3 (the twins stay at 1), and the MDE is projected to 2
samples. The pilot's numbers, and the samples the main run will use, are written into this
section and committed before the main run.

(Filled in from the pilot.)

Filled in on 2026-09-28 from `experiments/results/2026-09-18_study3-pilot` (`pilot.json`,
`pilot.md`), before any page arm was run:

- Localhost AI was at commit fdb4fbe (later than ce3d03d, as allowed above). The determinism
  check on `qwen3.5-9b-mlx4` at batch 1 with thinking off passed: 10 of 10 prompts
  byte-identical sent alone twice, and 10 of 10 identical when sent together (queued one at a
  time by the server) (`determinism.json`).
- A/A (`aa_resample` minus baseline) on the 24 pages at 2 samples: mean +0.11 pp, page SD 7.73
  pp, MDE 5.4 pp (t quantiles, 80% power, two-sided alpha 0.05 / 3). The per-query A/A SD is 10.4
  pp (Study 2, qwen2.5-3b at 3 samples: 16.7 pp).
- Time per generated answer: mean 31.9 s, median 29.2 s over the 288 pilot answers (2.55 hours
  of generation). This is well above the spec's estimate of 13 s.
- The contingency applies (31.9 s is over 15 s): the main run uses 2 samples per query, run as
  `vizor experiment --config configs/study3_qwen9b.yaml --samples 2`; the content-only twins stay
  at 1 sample. At 2 samples the main run is the pilot's own design, so the MDE for the primary
  is the pilot's 5.4 pp with no projection. The main run reuses the pilot's baseline and A/A
  answers from the cache, which needs the same server commit (fdb4fbe).
- Remaining budget at 2 samples, from the dry run's distinct prompts: about 389 new answers
  (`fact_passage` 144, `entity_anchor` 68, `retrieval_meta` 144, twins 33), about 3.5 hours at
  31.9 s.

## Budget

The spec's estimate, at about 13 s per answer (34.7 ms per decoded token plus about 2.5k tokens
of prefill, measured on this model at batch 1): pilot 288 answers (about 1 hour); main run 72 x
5 x 3 + 72 x 3 x 1 = 1,296 answers (about 4.7 hours). The dry run counts fewer distinct (prompt,
seed) pairs, because an unchanged page or an unchanged selection gives the same prompt and the
cache answers it: 999 in all (baseline 216, `aa_resample` 216, `fact_passage` 216,
`entity_anchor` 102, `retrieval_meta` 216, twins 22, 1 and 10). 288 of them come from the
pilot, which leaves 711 new answers for the main run (about 2.6 hours at 13 s).

## What this study cannot say

One answer model, one synthetic corpus written knowing the edits, hand-written queries, a
simulated engine whose passage policy and prompt layout are modeling choices. Nothing here is a
claim about ChatGPT, Perplexity or any production answer engine. A positive result is not
expected by design; nulls and harms are reported the same way.

## Deviations

(Any change from this plan, with the reason, is listed here.)
