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
- Each query has delta_q = mean_k(variant) - mean_k(baseline). For page arms, queries are
  clustered by the underlying edited page, including across cross-fit folds. Engine arms use
  queries as units. The Wilcoxon test runs on unit means; the clustered-bootstrap 95% CI is
  descriptive.
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
from vizor.metrics.mentions import brand_patterns
from vizor.metrics.visibility import answer_rows
from vizor.optimize.retrieval_policy import RELEVANCE, RetrievalPolicy
from vizor.optimize.stats import PairedResult, holm, paired_clustered
from vizor.optimize.transforms import TRANSFORMS, TransformContext, apply_chain
from vizor.retrieve.cascade import Selection
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
    "mentioned",
    "mention_share",
    "mentioned_not_cited",
    "unparsed",
    "cites_last_only",
]
CONTROLS = ("noop", "aa_resample")


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # "doc" | "engine" | "aa"
    transforms: tuple[str, ...] = ()
    policy: RetrievalPolicy = RELEVANCE
    salt: str = ""
    # "full": the edited page is re-indexed, so retrieval may rank it (and its neighbours)
    # differently. "content": sources and their order are pinned to the baseline selection and
    # only the edited page's rendered text changes, which isolates the content effect.
    mode: str = "full"

    @classmethod
    def parse(cls, spec: str) -> Arm:
        if spec == "aa_resample":
            return cls(spec, "aa", salt="aa")
        if spec.startswith("engine:"):
            return cls(spec, "engine", policy=RetrievalPolicy.parse(spec.removeprefix("engine:")))
        mode = "full"
        body = spec
        if spec.startswith("content:"):
            mode, body = "content", spec.removeprefix("content:")
        names = tuple(body.split("+"))
        for n in names:
            if n not in TRANSFORMS:
                raise ValueError(f"unknown arm {n!r}")
        return cls(spec, "doc", transforms=names, mode=mode)

    @property
    def base_name(self) -> str:
        """The full-mode arm this one pairs with ('content:faq_rewrite' -> 'faq_rewrite')."""
        return "+".join(self.transforms) if self.kind == "doc" else self.name

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
        mentioned=("mentioned", "max"),
        mention_share=("mention_share", "sum"),
        unparsed=("n_unparsed", "max"),
        cites_last_only=("cites_last_only", "max"),
    )
    agg["mention_share"] = agg["mention_share"].clip(upper=1.0)
    agg["unparsed"] = agg["unparsed"] > 0
    agg["mentioned_not_cited"] = agg["mentioned"].astype(bool) & ~agg["cited"].astype(bool)
    flags = ["cited", "retrieved", "uncited", "mentioned", "unparsed", "cites_last_only"]
    agg[[*flags, "mentioned_not_cited"]] = agg[[*flags, "mentioned_not_cited"]].astype(float)
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
        brands: dict | None = None,
        primary: str = "imp_pwc",
        weighting: str = "query",
        arm_samples: dict[str, int] | None = None,
    ) -> None:
        if primary not in ("imp_pwc", "c_share", "mentioned"):
            raise ValueError(f"unknown primary metric {primary!r}")
        self.brands = brands if brands is not None else brand_patterns(domains)
        self.primary = primary
        self.weighting = weighting
        self.engine = engine
        self.queries = list(queries)
        self.samples = samples
        # arm name -> samples per query for that arm (e.g. content-only twins at 1 sample)
        self.arm_samples = dict(arm_samples or {})
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
            for row in answer_rows(
                r.answer, self.domains, self.decay, self.sentiment, source_text, self.brands
            ):
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
        k = self.samples_for(arm)
        results = (engine or self.engine).run(queries, k, arm.policy, arm.salt)
        return ArmRun(arm, results, self.rows_for(results, arm.name))

    def samples_for(self, arm: Arm) -> int:
        return self.arm_samples.get(arm.name, self.samples)

    def _doc_engines(self, arm: Arm, plan: list[tuple[str, Sequence[Query], TransformContext]]):
        """(fold, doc_id, queries, engine, edited doc or None, diff) for each edited page."""
        docs = self.engine.cascade.docs
        for fold, queries, ctx in plan:
            by_page: dict[str, list[Query]] = defaultdict(list)
            for q in queries:
                by_page[self.focus[q.query_id]].append(q)
            for doc_id, group in by_page.items():
                new, diff = apply_chain(docs[doc_id], arm.transforms, ctx)
                engine = self.engine
                if new is docs[doc_id]:
                    yield fold, doc_id, group, engine, None, diff
                    continue
                edited = engine.cascade.with_docs({doc_id: new})
                engine = (
                    engine.content_only(edited)
                    if arm.mode == "content"
                    else engine.with_cascade(edited)
                )
                yield fold, doc_id, group, engine, new, diff

    def _run_doc_arm(
        self, arm: Arm, plan: list[tuple[str, Sequence[Query], TransformContext]]
    ) -> ArmRun:
        results: list[EngineResult] = []
        diffs: dict[str, dict[str, str]] = {}
        changed: dict[str, dict[str, SourceDoc]] = {}
        scored: dict[str, tuple[str, str]] = {}
        for fold, doc_id, group, engine, new, diff in self._doc_engines(arm, plan):
            if new is not None:
                changed.setdefault(fold, {})[doc_id] = new
                diffs.setdefault(fold, {})[doc_id] = diff
            results += engine.run(group, self.samples_for(arm), arm.policy, arm.salt)
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
        return self._run_doc_arm(arm, self._plan(arm, queries, ctx_queries))

    def _plan(
        self, arm: Arm, queries: list[Query], ctx_queries: Sequence[Query] | None = None
    ) -> list[tuple[str, Sequence[Query], TransformContext]]:
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
        return plan

    def build_prompts(
        self, arm: Arm, passage_policy: str | None = None, queries: Sequence[Query] | None = None
    ) -> dict[str, tuple[str, Selection]]:
        """Each query's prompt under this arm, built exactly as `run` would build it (same
        edits, folds and source pinning) but with no answer sampled. `passage_policy` re-renders
        the same sources under another passage policy."""
        queries = list(queries or self.queries)

        def render(engine: Engine) -> Engine:
            return engine.with_passage_policy(passage_policy) if passage_policy else engine

        if arm.kind != "doc":
            eng = render(self.engine)
            return {q.query_id: eng.build_prompt(q, arm.policy) for q in queries}
        out: dict[str, tuple[str, Selection]] = {}
        for _fold, _doc, group, engine, _new, _diff in self._doc_engines(
            arm, self._plan(arm, queries)
        ):
            eng = render(engine)
            for q in group:
                out[q.query_id] = eng.build_prompt(q, arm.policy)
        return out

    def clusters(self, run: ArmRun, ids: pd.Index) -> np.ndarray:
        """Cluster page-arm queries by the underlying page, across both cross-fit folds."""
        if run.scored_against:
            return np.array([run.scored_against[q][1] for q in ids])
        return np.asarray(ids)

    def compare(self, base: ArmRun, var: ArmRun, subset: Sequence[str] | None = None) -> dict:
        b, v = base.per_query(), var.per_query()
        idx = b.index.intersection(v.index)
        if subset is not None:
            idx = idx.intersection(pd.Index(subset))
        b, v = b.loc[idx], v.loc[idx]
        d = v - b
        cl = self.clusters(var, idx)

        def test(col: str) -> PairedResult:
            return paired_clustered(
                d[col].to_numpy() * 100, cl, b=self.bootstrap, weighting=self.weighting
            )

        pwc, csov = test("imp_pwc"), test("c_share")
        ment, ment_s = test("mentioned"), test("mention_share")
        prim = {"imp_pwc": pwc, "c_share": csov, "mentioned": ment}[self.primary]
        retrieved_b = b["retrieved"] > 0
        cond = d.loc[retrieved_b, "imp_pwc"] * 100
        fresh = [r for r in var.results if not r.answer.usage.get("cached", False)]
        # Did the arm change what the model was shown? Compare each query's source list (doc ids
        # in prompt order) with the baseline's.
        moved = self._sources_changed(base, var, idx)
        dp = d[self.primary] * 100
        return {
            "arm": var.arm.name,
            "kind": var.arm.kind,
            "mode": var.arm.mode,
            "family": self.family(var.arm),
            "primary": self.primary,
            "n_queries": len(idx),
            "n_units": prim.n,
            "unit": "page" if var.scored_against else "query",
            "base_pwc_pct": b["imp_pwc"].mean() * 100,
            "d_pwc_pp": pwc.mean,
            "d_pwc_lo": pwc.lo,
            "d_pwc_hi": pwc.hi,
            "p_pwc": pwc.p,
            "p": prim.p,
            "rel_pct": relative_improvement(b["imp_pwc"].mean(), v["imp_pwc"].mean()),
            "base_csov_pct": b["c_share"].mean() * 100,
            "d_csov_pp": csov.mean,
            "d_csov_lo": csov.lo,
            "d_csov_hi": csov.hi,
            "p_csov": csov.p,
            "base_mention_pct": b["mentioned"].mean() * 100,
            "d_mention_pp": ment.mean,
            "d_mention_lo": ment.lo,
            "d_mention_hi": ment.hi,
            "p_mention": ment.p,
            "base_mention_share_pct": b["mention_share"].mean() * 100,
            "d_mention_share_pp": ment_s.mean,
            "d_mention_share_lo": ment_s.lo,
            "d_mention_share_hi": ment_s.hi,
            "d_cite_rate_pp": d["cited"].mean() * 100,
            "d_retrieval_pp": d["retrieved"].mean() * 100,
            "d_pwc_given_retrieved_pp": cond.mean() if len(cond) else np.nan,
            "d_pwc_given_cited_pp": d["imp_pwc_cited"].mean() * 100,
            "d_sentiment": d["answer_sentiment"].mean(),
            "uncited_rate_pct": v["uncited"].mean() * 100,
            "base_uncited_rate_pct": b["uncited"].mean() * 100,
            "d_uncited_pp": d["uncited"].mean() * 100,
            "unparsed_rate_pct": v["unparsed"].mean() * 100,
            "last_only_rate_pct": v["cites_last_only"].mean() * 100,
            "mentioned_not_cited_pct": v["mentioned_not_cited"].mean() * 100,
            "hallucinated_per_answer": v["n_hallucinated"].mean(),
            "n_sources_changed": int(moved.sum()),
            "d_primary_sources_changed_pp": dp[moved].mean() if moved.any() else np.nan,
            "d_primary_sources_same_pp": dp[~moved].mean() if (~moved).any() else np.nan,
            "new_calls": len(fresh),
            "new_cost_usd": sum(r.answer.usage.get("cost_usd", 0.0) for r in fresh),
            "transform_scope": "crossfit-2" if var.arm.uses_queries else "none",
        }

    @staticmethod
    def _sources_changed(base: ArmRun, var: ArmRun, idx: pd.Index) -> pd.Series:
        def lists(run: ArmRun) -> dict[str, tuple[str, ...]]:
            return {
                r.answer.query_id: tuple(s.doc_id for s in r.answer.sources)
                for r in run.results
                if r.answer.sample == 0
            }

        lb, lv = lists(base), lists(var)
        return pd.Series([lb.get(q) != lv.get(q) for q in idx], index=idx, dtype=bool)

    @staticmethod
    def family(arm: Arm) -> str:
        """Holm family: content-only arms are tested apart from the arms a site owner would ship."""
        return "content" if arm.mode == "content" else "arms"

    @staticmethod
    def with_holm(rows: list[dict], controls: Sequence[str] = CONTROLS) -> pd.DataFrame:
        """The one verdict rule: an effect counts if its Holm-adjusted Wilcoxon p (on the primary
        metric) is below 0.05, within its family. Controls are reported but sit outside every
        family. The mention test gets its own Holm adjustment within the same families."""
        df = pd.DataFrame(rows)
        if "family" not in df:
            df["family"] = "arms"
        tested = ~df["arm"].isin(controls)
        df["p_holm"] = np.nan
        df["p_mention_holm"] = np.nan
        for _, g in df[tested].groupby("family"):
            df.loc[g.index, "p_holm"] = holm(g["p"].tolist())
            if "p_mention" in g:
                df.loc[g.index, "p_mention_holm"] = holm(g["p_mention"].tolist())
        df["significant"] = tested & (df["p_holm"] < 0.05)
        df["mention_significant"] = tested & (df["p_mention_holm"] < 0.05)
        return df

    def decompose(self, base: ArmRun, full: ArmRun, content: ArmRun) -> dict:
        """Total = content + rank-mediated. The content arm keeps the baseline's sources and
        order, so full - content is what the edit did through retrieval (re-ranking the edited
        page or displacing others). Paired per query, clustered by edited page."""
        tot, con, rank = (
            self.compare(base, full),
            self.compare(base, content),
            self.compare(content, full),
        )
        out = {"arm": full.arm.name, "primary": self.primary}
        for label, c in (("total", tot), ("content", con), ("rank", rank)):
            for m in ("csov", "pwc", "mention"):
                out[f"{label}_{m}_pp"] = c[f"d_{m}_pp"]
                out[f"{label}_{m}_lo"] = c[f"d_{m}_lo"]
                out[f"{label}_{m}_hi"] = c[f"d_{m}_hi"]
            out[f"{label}_p"] = c["p"]
            out[f"{label}_n_units"] = c["n_units"]
        out["n_sources_changed"] = tot["n_sources_changed"]
        return out


def level_ci(values: np.ndarray, b: int = 5000) -> tuple[float, float, float]:
    from vizor.optimize.stats import bootstrap_ci

    return bootstrap_ci(np.asarray(values, dtype=float), b=b)


def position_sweep(
    sb: Sandbox, positions: Sequence[int], queries: Sequence[Query] | None = None
) -> tuple[pd.DataFrame, dict[int, ArmRun]]:
    """Force exactly one target page into slot p (1..5), everything else unchanged."""
    runs = {
        p: sb.run(
            Arm(f"engine:target_at:{p}", "engine", policy=RetrievalPolicy.parse(f"target_at:{p}")),
            queries=queries,
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
                "d_csov_vs_first_pp": cmp["d_csov_pp"] if cmp else 0.0,
                "d_csov_lo": cmp["d_csov_lo"] if cmp else 0.0,
                "d_csov_hi": cmp["d_csov_hi"] if cmp else 0.0,
                "mention_pct": pq["mentioned"].mean() * 100,
                "p_pwc": cmp["p_pwc"] if cmp else 1.0,
                "p": cmp["p"] if cmp else 1.0,
                "primary": sb.primary,
            }
        )
    return pd.DataFrame(out), runs


def boost_sweep(
    sb: Sandbox, boosts: Sequence[float], queries: Sequence[Query] | None = None
) -> tuple[pd.DataFrame, dict[float, ArmRun]]:
    """Add w to the target pages' final retrieval score (sigmoid of the cross-encoder logit)."""
    runs: dict[float, ArmRun] = {}
    query_ids = {q.query_id for q in (queries or sb.queries)}
    for w in boosts:
        if w == 0 and sb.baseline is not None:
            runs[w] = sb.baseline
        else:
            runs[w] = sb.run(
                Arm(f"engine:boost:{w:g}", "engine", policy=RetrievalPolicy(target_boost=w)),
                queries=queries,
            )
    ref = runs[boosts[0]]

    def source_lists(run: ArmRun) -> dict[str, list[str]]:
        return {
            r.answer.query_id: [s.doc_id for s in r.answer.sources]
            for r in run.results
            if r.answer.sample == 0 and r.answer.query_id in query_ids
        }

    ref_src = source_lists(ref)
    out = []
    for w, run in runs.items():
        pq = run.per_query()
        pq = pq.loc[pq.index.isin(query_ids)]
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
            matching_ids = [q for q, k in cls.items() if k == c and q in d_q.index]
            by_cls[f"n_{c}"] = len(matching_ids)
            by_cls[f"d_pwc_{c}_pp"] = (
                float(d_q.loc[matching_ids].mean()) if matching_ids else np.nan
            )
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
