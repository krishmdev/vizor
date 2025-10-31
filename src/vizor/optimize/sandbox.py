"""The sandbox: apply an arm, re-run the same queries through the same engine, measure the shift.

Design:
- A fixed query set Q and K samples per query at the engine's temperature.
- Common random numbers: sample k of query q uses the same seed under every arm. With FakeLLM
  or a local model this pairs the randomness; OpenAI's seed is best effort, so there the pairing
  buys little and `aa_resample` is what tells you how big chance differences are.
- Page arms are applied to the query's *focus page*: the target page the baseline retrieval ranked
  highest for that query (or, if none made the candidate list, the target page closest to the
  query). Each query sees exactly one edited page, which is what makes per-query contexts in the
  bandit meaningful.
- Cross-fitting: arms that read the tracked queries (metadata keyphrases, FAQ questions, keyword
  stuffing) are built from one half of the queries and scored on the other half, so no page is
  ever edited with the very query it is then scored on.
- Engine arms change retrieval weighting or source order and leave pages alone.
- `noop` must give exactly zero difference (a pipeline and cache check). `aa_resample` re-samples
  the unchanged prompts with fresh seeds: the noise floor every other arm is read against.
- The unit of analysis is the query: delta_q = mean_k(variant) - mean_k(baseline). The test is a
  Wilcoxon signed-rank on delta_q, Holm-adjusted across the real arms (noop and aa_resample are
  controls, outside the family); the paired-bootstrap 95% CI is descriptive.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vizor.generate.engine import Engine, EngineResult
from vizor.generate.prompt import parse_prompt
from vizor.metrics.impression import relative_improvement
from vizor.metrics.visibility import answer_rows
from vizor.optimize.retrieval_policy import RELEVANCE, RetrievalPolicy
from vizor.optimize.stats import holm, paired_clustered
from vizor.optimize.transforms import TRANSFORMS, TransformContext, apply_chain
from vizor.types import Query, SourceDoc, stable_seed

TARGET_METRICS = [
    "imp_pwc",
    "imp_word",
    "c_share",
    "cited",
    "retrieved",
    "answer_sentiment",
    "uncited",
    "n_hallucinated",
    "imp_pwc_cited",
]
CONTROLS = ("noop", "aa_resample")


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # "doc" | "engine" | "aa"
    transforms: tuple[str, ...] = ()
    policy: RetrievalPolicy = RELEVANCE
    salt: str = ""

    @classmethod
    def parse(cls, spec: str) -> Arm:
        if spec == "aa_resample":
            return cls(spec, "aa", salt="aa")
        if spec.startswith("engine:"):
            return cls(spec, "engine", policy=RetrievalPolicy.parse(spec.removeprefix("engine:")))
        names = tuple(spec.split("+"))
        for n in names:
            if n not in TRANSFORMS:
                raise ValueError(f"unknown arm {n!r}")
        return cls(spec, "doc", transforms=names)

    @property
    def uses_queries(self) -> bool:
        return any(TRANSFORMS[t].uses_queries for t in self.transforms)


@dataclass
class ArmRun:
    arm: Arm
    results: list[EngineResult]
    rows: pd.DataFrame
    # fold -> doc_id -> diff text / edited doc; "all" when the arm doesn't read queries
    diffs: dict[str, dict[str, str]] = field(default_factory=dict)
    changed: dict[str, dict[str, SourceDoc]] = field(default_factory=dict)
    # query_id -> (fold, doc_id) of the page edited for that query
    scored_against: dict[str, tuple[str, str]] = field(default_factory=dict)

    def per_query(self) -> pd.DataFrame:
        """Mean over samples of the target's metrics, one row per query."""
        t = target_frame(self.rows)
        return t.groupby("query_id")[TARGET_METRICS].mean()


def target_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse target-domain rows to one row per answer (sums shares if a project has several
    target domains)."""
    t = rows[rows["role"] == "target"]
    agg = t.groupby(["query_id", "sample"]).agg(
        imp_pwc=("imp_pwc", "sum"),
        imp_word=("imp_word", "sum"),
        c_share=("c_share", "sum"),
        cited=("cited", "max"),
        retrieved=("retrieved", "max"),
        answer_sentiment=("answer_sentiment", "mean"),
        source_sentiment=("source_sentiment", "mean"),
        uncited=("answer_uncited", "max"),
        n_hallucinated=("n_hallucinated", "max"),
    )
    agg[["cited", "retrieved", "uncited"]] = agg[["cited", "retrieved", "uncited"]].astype(float)
    # PAWC share only over answers that cited anything, to separate "cited the target less" from
    # "stopped citing altogether" (a formatting failure).
    agg["imp_pwc_cited"] = agg["imp_pwc"].where(agg["uncited"] == 0)
    return agg.reset_index()


def two_folds(queries: Sequence[Query]) -> tuple[list[Query], list[Query]]:
    """Deterministic 50/50 split by hash, stratified by intent."""
    a: list[Query] = []
    b: list[Query] = []
    by: dict[str, list[Query]] = defaultdict(list)
    for q in queries:
        by[q.intent].append(q)
    # Hash-order within each intent, then deal alternately across the concatenation, so both
    # folds stay balanced by intent and neither is empty even with one query per intent.
    ordered = [
        q for g in by.values() for q in sorted(g, key=lambda q: stable_seed("split", q.query_id))
    ]
    for i, q in enumerate(ordered):
        (a if i % 2 == 0 else b).append(q)
    return a, b


class Sandbox:
    def __init__(
        self,
        engine: Engine,
        queries: Sequence[Query],
        samples: int,
        domains: dict[str, str],
        ctx_factory: Callable[[Sequence[Query]], TransformContext],
        sentiment=None,
        decay: str = "paper",
        bootstrap: int = 5000,
    ) -> None:
        self.engine = engine
        self.queries = list(queries)
        self.samples = samples
        self.domains = domains
        self.ctx_factory = ctx_factory
        self.ctx = ctx_factory(self.queries)
        self.sentiment = sentiment
        self.decay = decay
        self.bootstrap = bootstrap
        self.baseline: ArmRun | None = None
        self.folds = two_folds(self.queries)
        self.focus: dict[str, str] = {}

    def rows_for(self, results: list[EngineResult], arm: str) -> pd.DataFrame:
        out = []
        for r in results:
            source_text = None
            if self.sentiment is not None:
                _, srcs = parse_prompt(r.prompt)
                source_text = {s.index: s.content for s in srcs}
            for row in answer_rows(r.answer, self.domains, self.decay, self.sentiment, source_text):
                row["arm"] = arm
                out.append(row)
        return pd.DataFrame(out)

    def _set_focus(self, base: ArmRun) -> None:
        targets = [d for d in self.engine.cascade.docs.values() if d.role == "target"]
        if not targets:
            return
        tv = np.stack([self.ctx.doc_vector(d) for d in targets])
        for r in base.results:
            qid = r.answer.query_id
            if qid in self.focus:
                continue
            hit = next((c.doc for c in r.selection.candidates if c.doc.role == "target"), None)
            if hit is None:
                q = next(q for q in self.queries if q.query_id == qid)
                qv = self.engine.cascade.query_vector(q.text)
                hit = targets[int(np.argmax(tv @ qv))]
            self.focus[qid] = hit.doc_id

    def run_baseline(self) -> ArmRun:
        base = self._run_engine(Arm("baseline", "doc"), self.queries)
        self._set_focus(base)
        self.baseline = base
        return base

    def _run_engine(
        self, arm: Arm, queries: Sequence[Query], engine: Engine | None = None
    ) -> ArmRun:
        results = (engine or self.engine).run(queries, self.samples, arm.policy, arm.salt)
        return ArmRun(arm, results, self.rows_for(results, arm.name))

    def _run_doc_arm(
        self, arm: Arm, plan: list[tuple[str, Sequence[Query], TransformContext]]
    ) -> ArmRun:
        results: list[EngineResult] = []
        diffs: dict[str, dict[str, str]] = {}
        changed: dict[str, dict[str, SourceDoc]] = {}
        scored: dict[str, tuple[str, str]] = {}
        docs = self.engine.cascade.docs
        for fold, queries, ctx in plan:
            by_page: dict[str, list[Query]] = defaultdict(list)
            for q in queries:
                by_page[self.focus[q.query_id]].append(q)
            for doc_id, group in by_page.items():
                new, diff = apply_chain(docs[doc_id], arm.transforms, ctx)
                engine = self.engine
                if new is not docs[doc_id]:
                    engine = engine.with_cascade(engine.cascade.with_docs({doc_id: new}))
                    changed.setdefault(fold, {})[doc_id] = new
                    diffs.setdefault(fold, {})[doc_id] = diff
                results += engine.run(group, self.samples, arm.policy, arm.salt)
                for q in group:
                    scored[q.query_id] = (fold, doc_id)
        order = {q.query_id: i for i, q in enumerate(self.queries)}
        results.sort(key=lambda r: (order.get(r.answer.query_id, 0), r.answer.sample))
        return ArmRun(arm, results, self.rows_for(results, arm.name), diffs, changed, scored)

    def run(
        self,
        arm: Arm,
        queries: Sequence[Query] | None = None,
        ctx_queries: Sequence[Query] | None = None,
    ) -> ArmRun:
        """Run an arm. Page arms are cross-fitted over the two folds unless `ctx_queries` pins
        the queries the transforms may read (used by the held-out greedy loop)."""
        queries = list(queries or self.queries)
        if arm.kind != "doc":
            return self._run_engine(arm, queries)
        if self.baseline is None:
            raise RuntimeError("run_baseline() first: page arms need each query's focus page")
        if ctx_queries is not None:
            plan = [("pinned", queries, self.ctx_factory(ctx_queries))]
        elif arm.uses_queries:
            ids = {q.query_id for q in queries}
            a, b = self.folds
            plan = [
                ("fold1", [q for q in a if q.query_id in ids], self.ctx_factory(b)),
                ("fold2", [q for q in b if q.query_id in ids], self.ctx_factory(a)),
            ]
        else:
            plan = [("all", queries, self.ctx)]
        return self._run_doc_arm(arm, plan)

    def clusters(self, run: ArmRun, ids: pd.Index) -> np.ndarray:
        """Unit of treatment for each query: its edited page within its fold for page arms
        (queries sharing a page are not independent), the query itself otherwise."""
        if run.scored_against:
            return np.array(["|".join(run.scored_against.get(q, ("", q))) for q in ids])
        return np.asarray(ids)

    def compare(self, base: ArmRun, var: ArmRun, subset: Sequence[str] | None = None) -> dict:
        b, v = base.per_query(), var.per_query()
        idx = b.index.intersection(v.index)
        if subset is not None:
            idx = idx.intersection(pd.Index(subset))
        b, v = b.loc[idx], v.loc[idx]
        d = v - b
        cl = self.clusters(var, idx)
        pwc = paired_clustered(d["imp_pwc"].to_numpy() * 100, cl, b=self.bootstrap)
        csov = paired_clustered(d["c_share"].to_numpy() * 100, cl, b=self.bootstrap)
        retrieved_b = b["retrieved"] > 0
        cond = d.loc[retrieved_b, "imp_pwc"] * 100
        fresh = [r for r in var.results if not r.answer.usage.get("cached", False)]
        return {
            "arm": var.arm.name,
            "kind": var.arm.kind,
            "n_queries": len(idx),
            "n_units": pwc.n,
            "unit": "page x fold" if var.scored_against else "query",
            "base_pwc_pct": b["imp_pwc"].mean() * 100,
            "d_pwc_pp": pwc.mean,
            "d_pwc_lo": pwc.lo,
            "d_pwc_hi": pwc.hi,
            "p": pwc.p,
            "rel_pct": relative_improvement(b["imp_pwc"].mean(), v["imp_pwc"].mean()),
            "d_csov_pp": csov.mean,
            "d_csov_lo": csov.lo,
            "d_csov_hi": csov.hi,
            "d_cite_rate_pp": d["cited"].mean() * 100,
            "d_retrieval_pp": d["retrieved"].mean() * 100,
            "d_pwc_given_retrieved_pp": cond.mean() if len(cond) else np.nan,
            "d_pwc_given_cited_pp": d["imp_pwc_cited"].mean() * 100,
            "d_sentiment": d["answer_sentiment"].mean(),
            "uncited_rate_pct": v["uncited"].mean() * 100,
            "base_uncited_rate_pct": b["uncited"].mean() * 100,
            "d_uncited_pp": d["uncited"].mean() * 100,
            "hallucinated_per_answer": v["n_hallucinated"].mean(),
            "new_calls": len(fresh),
            "new_cost_usd": sum(r.answer.usage.get("cost_usd", 0.0) for r in fresh),
            "transform_scope": "crossfit-2" if var.arm.uses_queries else "none",
        }

    @staticmethod
    def with_holm(rows: list[dict], controls: Sequence[str] = CONTROLS) -> pd.DataFrame:
        """The one verdict rule: an effect counts if its Holm-adjusted Wilcoxon p is below 0.05.
        Controls are reported but sit outside the family."""
        df = pd.DataFrame(rows)
        tested = ~df["arm"].isin(controls)
        df["p_holm"] = np.nan
        if tested.any():
            df.loc[tested, "p_holm"] = holm(df.loc[tested, "p"].tolist())
        df["significant"] = tested & (df["p_holm"] < 0.05)
        return df


def level_ci(values: np.ndarray, b: int = 5000) -> tuple[float, float, float]:
    from vizor.optimize.stats import bootstrap_ci

    return bootstrap_ci(np.asarray(values, dtype=float), b=b)


def position_sweep(sb: Sandbox, positions: Sequence[int]) -> tuple[pd.DataFrame, dict[int, ArmRun]]:
    """Force exactly one target page into slot p (1..5), everything else unchanged."""
    runs = {
        p: sb.run(
            Arm(f"engine:target_at:{p}", "engine", policy=RetrievalPolicy.parse(f"target_at:{p}"))
        )
        for p in positions
    }
    forced = {r.answer.query_id for r in runs[positions[0]].results if r.answer.usage.get("forced")}
    ref = runs[positions[0]]
    out = []
    for p, run in runs.items():
        pq = run.per_query().loc[sorted(forced)]
        mean, lo, hi = level_ci(pq["imp_pwc"].to_numpy() * 100, sb.bootstrap)
        cmp = sb.compare(ref, run, subset=sorted(forced)) if p != positions[0] else None
        out.append(
            {
                "position": p,
                "n_queries": len(pq),
                "pwc_pct": mean,
                "pwc_lo": lo,
                "pwc_hi": hi,
                "cite_rate_pct": pq["cited"].mean() * 100,
                "c_share_pct": pq["c_share"].mean() * 100,
                "d_pwc_vs_first_pp": cmp["d_pwc_pp"] if cmp else 0.0,
                "d_lo": cmp["d_pwc_lo"] if cmp else 0.0,
                "d_hi": cmp["d_pwc_hi"] if cmp else 0.0,
                "p": cmp["p"] if cmp else 1.0,
            }
        )
    return pd.DataFrame(out), runs


def boost_sweep(sb: Sandbox, boosts: Sequence[float]) -> tuple[pd.DataFrame, dict[float, ArmRun]]:
    """Add w to the target pages' final retrieval score (sigmoid of the cross-encoder logit)."""
    runs: dict[float, ArmRun] = {}
    for w in boosts:
        if w == 0 and sb.baseline is not None:
            runs[w] = sb.baseline
        else:
            runs[w] = sb.run(
                Arm(f"engine:boost:{w:g}", "engine", policy=RetrievalPolicy(target_boost=w))
            )
    ref = runs[boosts[0]]

    def source_lists(run: ArmRun) -> dict[str, list[str]]:
        return {
            r.answer.query_id: [s.doc_id for s in r.answer.sources]
            for r in run.results
            if r.answer.sample == 0
        }

    ref_src = source_lists(ref)
    out = []
    for w, run in runs.items():
        pq = run.per_query()
        src = source_lists(run)
        cls = {}
        for qid, docs in src.items():
            base = ref_src.get(qid, [])
            cls[qid] = (
                "unchanged"
                if docs == base
                else ("order_only" if set(docs) == set(base) else "set_changed")
            )
        d_q = (pq["imp_pwc"] - ref.per_query()["imp_pwc"]) * 100
        by_cls = {}
        for c in ("set_changed", "order_only", "unchanged"):
            ids = [q for q, k in cls.items() if k == c and q in d_q.index]
            by_cls[f"n_{c}"] = len(ids)
            by_cls[f"d_pwc_{c}_pp"] = float(d_q.loc[ids].mean()) if ids else np.nan
        mean, lo, hi = level_ci(pq["imp_pwc"].to_numpy() * 100, sb.bootstrap)
        cmp = sb.compare(ref, run) if w != boosts[0] else None
        out.append(
            {
                "boost": w,
                "n_queries": len(pq),
                "retrieval_pct": pq["retrieved"].mean() * 100,
                "pwc_pct": mean,
                "pwc_lo": lo,
                "pwc_hi": hi,
                "c_share_pct": pq["c_share"].mean() * 100,
                "cite_rate_pct": pq["cited"].mean() * 100,
                "d_pwc_pp": cmp["d_pwc_pp"] if cmp else 0.0,
                "d_lo": cmp["d_pwc_lo"] if cmp else 0.0,
                "d_hi": cmp["d_pwc_hi"] if cmp else 0.0,
                "p": cmp["p"] if cmp else 1.0,
                **by_cls,
            }
        )
    return pd.DataFrame(out), runs


def score_scale(base: ArmRun, n: int = 5) -> dict:
    """How big a boost is relative to the scores it shifts: the spread of final scores in the
    top-n and the gap between the n-th and (n+1)-th candidate, per query."""
    spread, gap = [], []
    seen = set()
    for r in base.results:
        if r.answer.query_id in seen:
            continue
        seen.add(r.answer.query_id)
        s = [c.final_score for c in r.selection.candidates]
        if len(s) > n:
            spread.append(s[0] - s[n - 1])
            gap.append(s[n - 1] - s[n])
    return {
        "median_top5_spread": float(np.median(spread)) if spread else np.nan,
        "median_gap_5_6": float(np.median(gap)) if gap else np.nan,
        "score_range": [0.0, 1.0],
    }
