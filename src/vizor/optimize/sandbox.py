"""The sandbox: apply an arm, re-run the same queries through the same engine, measure the shift.

Design:
- A fixed query set Q and K samples per query at the engine's temperature.
- Common random numbers: sample k of query q uses the same seed under every arm, so the only
  thing that differs between baseline and variant is the arm.
- Doc arms transform the target site's pages; only changed passages are re-embedded.
- Engine arms change retrieval weighting or source order and leave pages alone.
- `noop` must give exactly zero difference (checks the pipeline). `aa_resample` re-samples the
  unchanged prompt with fresh seeds, which measures pure sampling noise: the noise floor that
  every other arm should be read against.
- The unit of analysis is the query: delta_q = mean_k(variant) - mean_k(baseline). Report the mean
  with a paired-bootstrap 95% CI over queries, a Wilcoxon signed-rank p, and Holm across arms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vizor.generate.engine import Engine, EngineResult
from vizor.generate.prompt import parse_prompt
from vizor.metrics.impression import relative_improvement
from vizor.metrics.visibility import answer_rows
from vizor.optimize.retrieval_policy import RELEVANCE, RetrievalPolicy
from vizor.optimize.stats import holm, paired
from vizor.optimize.transforms import TRANSFORMS, TransformContext, apply_to_targets
from vizor.types import Query, SourceDoc

TARGET_METRICS = ["imp_pwc", "imp_word", "c_share", "cited", "retrieved", "answer_sentiment"]


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


@dataclass
class ArmRun:
    arm: Arm
    results: list[EngineResult]
    rows: pd.DataFrame
    diffs: dict[str, str] = field(default_factory=dict)
    changed: dict[str, SourceDoc] = field(default_factory=dict)

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
    )
    agg[["cited", "retrieved"]] = agg[["cited", "retrieved"]].astype(float)
    return agg.reset_index()


class Sandbox:
    def __init__(
        self,
        engine: Engine,
        queries: Sequence[Query],
        samples: int,
        domains: dict[str, str],
        ctx: TransformContext,
        sentiment=None,
        decay: str = "paper",
        bootstrap: int = 5000,
    ) -> None:
        self.engine = engine
        self.queries = list(queries)
        self.samples = samples
        self.domains = domains
        self.ctx = ctx
        self.sentiment = sentiment
        self.decay = decay
        self.bootstrap = bootstrap
        self.baseline: ArmRun | None = None

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

    def run(self, arm: Arm, queries: Sequence[Query] | None = None, docs_ctx=None) -> ArmRun:
        queries = list(queries or self.queries)
        engine = self.engine
        diffs: dict[str, str] = {}
        changed: dict[str, SourceDoc] = {}
        if arm.kind == "doc":
            changed, diffs = apply_to_targets(arm.transforms, docs_ctx or self.ctx)
            if changed:
                engine = engine.with_cascade(engine.cascade.with_docs(changed))
        results = engine.run(queries, self.samples, arm.policy, arm.salt)
        return ArmRun(arm, results, self.rows_for(results, arm.name), diffs, changed)

    def run_baseline(self) -> ArmRun:
        self.baseline = self.run(Arm("baseline", "doc"))
        return self.baseline

    def compare(self, base: ArmRun, var: ArmRun, subset: Sequence[str] | None = None) -> dict:
        b, v = base.per_query(), var.per_query()
        idx = b.index.intersection(v.index)
        if subset is not None:
            idx = idx.intersection(pd.Index(subset))
        b, v = b.loc[idx], v.loc[idx]
        d = v - b
        pwc = paired(d["imp_pwc"].to_numpy() * 100, b=self.bootstrap)
        csov = paired(d["c_share"].to_numpy() * 100, b=self.bootstrap)
        retrieved_b = b["retrieved"] > 0
        cond = d.loc[retrieved_b, "imp_pwc"] * 100
        calls = sum(1 for r in var.results if not r.answer.usage.get("cached", False))
        cost = sum(r.answer.usage.get("cost_usd", 0.0) for r in var.results)
        return {
            "arm": var.arm.name,
            "kind": var.arm.kind,
            "n_queries": pwc.n,
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
            "d_sentiment": d["answer_sentiment"].mean(),
            "new_calls": calls,
            "cost_usd": cost,
        }

    @staticmethod
    def with_holm(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        tested = df["arm"] != "noop"
        df["p_holm"] = np.nan
        if tested.any():
            df.loc[tested, "p_holm"] = holm(df.loc[tested, "p"].tolist())
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
    out = []
    for w, run in runs.items():
        pq = run.per_query()
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
