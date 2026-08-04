"""Attribution-propensity scoring over a stored run, and its paired analysis.

Input is a finished run directory: reference answers are baseline samples (by default 0 and 1),
and the prompts come from the run itself. A *prompt set* is one prompt per query:
- every arm in `responses.jsonl.gz` (its sample-0 prompt), named after the arm;
- for runs that wrote `arm_prompts.jsonl.gz`, also the prompt-only sets built without sampling
  (content-only twins, other passage policies), named `<arm>` or `<arm>@<policy>`.

One scorer call per (prompt set, query, reference answer) scores every citation site of the
reference. Rows keep the per-site candidate log-probabilities, so `vizor recompute` can rebuild
AP without the model.

The analysis unit is the page: a query's delta is the mean over its reference answers, a page's
delta the mean over its queries, and the estimate, bootstrap interval and both tests use the
unweighted mean of page deltas.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from vizor.generate.prompt import prompt_hash
from vizor.generate.scorer import CachedScorer, FakeScorer, LocalhostScorer, Scorer
from vizor.metrics.propensity import (
    ap_from_logprobs,
    ap_from_result,
    citation_sites,
    is_close,
    target_slots,
)
from vizor.optimize.stats import bootstrap_ci, holm, sign_flip_p, wilcoxon_exact_p
from vizor.runstore import read_jsonl_gz, write_csv, write_jsonl_gz

Log = Callable[[str], None]
PINNED_ARMS = ("baseline", "noop", "aa_resample")


def make_scorer(sc, cache: Path | None) -> Scorer:
    """A cached scorer from a `ScoreConfig`."""
    if sc.backend == "fake":
        inner: Scorer = FakeScorer()
    elif sc.backend == "localhost":
        inner = LocalhostScorer(
            sc.model,
            base_url=sc.base_url,
            chat_template_kwargs=sc.chat_template_kwargs,
            server_meta=sc.server_meta,
        )
    elif sc.backend == "mlx":
        from vizor.generate.scorer import MLXScorer

        inner = MLXScorer(sc.model, sc.revision, sc.chat_template_kwargs)
    else:
        raise ValueError(f"unknown scorer backend {sc.backend!r}")
    # Like FakeLLM, the fake scorer is cheap and deterministic, so it is never cached on disk.
    return CachedScorer(inner, None if sc.backend == "fake" else cache)


def _mode(arm: str) -> str:
    return "pinned" if arm.startswith("content:") or arm in PINNED_ARMS else "full"


def load_prompt_sets(run_dir: Path) -> dict[str, dict[str, dict]]:
    """set name -> query_id -> {arm, mode, policy, prompt, sources}."""
    prompts = {r["prompt_hash"]: r["prompt"] for r in read_jsonl_gz(run_dir / "prompts.jsonl.gz")}
    manifest = json.loads((run_dir / "manifest.json").read_text())
    policy = manifest.get("passage_policy", "query-top3")
    sets: dict[str, dict[str, dict]] = {}
    for r in read_jsonl_gz(run_dir / "responses.jsonl.gz"):
        if r["sample"] != 0:
            continue
        sets.setdefault(r["arm"], {})[r["query_id"]] = {
            "arm": r["arm"],
            "mode": _mode(r["arm"]),
            "policy": policy,
            "prompt": prompts[r["prompt_hash"]],
            "sources": r["sources"],
        }
    extra = run_dir / "arm_prompts.jsonl.gz"
    if extra.exists():
        for r in read_jsonl_gz(extra):
            sets.setdefault(r["set"], {}).setdefault(
                r["query_id"],
                {
                    "arm": r["arm"],
                    "mode": r["mode"],
                    "policy": r["policy"],
                    "prompt": prompts[r["prompt_hash"]],
                    "sources": r["sources"],
                },
            )
    return sets


def load_references(run_dir: Path, samples, arm: str = "baseline") -> dict[tuple[str, int], dict]:
    out = {}
    for r in read_jsonl_gz(run_dir / "responses.jsonl.gz"):
        if r["arm"] == arm and r["sample"] in samples:
            out[(r["query_id"], r["sample"])] = {"text": r["text"], "n_sources": len(r["sources"])}
    return out


def load_units(run_dir: Path) -> dict[str, str]:
    """query_id -> the underlying target page it is scored against."""
    q = pd.read_csv(run_dir / "queries.csv")
    return dict(zip(q["query_id"], q["focus_doc"], strict=True))


def build_jobs(
    sets: dict[str, dict[str, dict]],
    refs: dict[tuple[str, int], dict],
    targets: list[str],
    system_prompt: str,
    which: list[str] | None = None,
) -> list[dict]:
    jobs = []
    for name in which or sorted(sets):
        if name not in sets:
            raise KeyError(f"prompt set {name!r} is not in this run")
        for qid, e in sets[name].items():
            for (rq, k), ref in sorted(refs.items()):
                if rq != qid:
                    continue
                n = len(e["sources"])
                sites = citation_sites(ref["text"], ref["n_sources"], n)
                messages = [{"role": "user", "content": e["prompt"]}]
                if system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})
                jobs.append(
                    {
                        "set": name,
                        "arm": e["arm"],
                        "mode": e["mode"],
                        "policy": e["policy"],
                        "query_id": qid,
                        "ref_sample": k,
                        "prompt_hash": prompt_hash(e["prompt"]),
                        "n_sources": n,
                        "target_slots": list(target_slots(e["sources"], targets)),
                        "messages": messages,
                        "continuation": ref["text"],
                        "sites": sites,
                    }
                )
    return jobs


def run_jobs(scorer: Scorer, jobs: list[dict], workers: int = 1, log: Log = print) -> list[dict]:
    t0 = time.time()

    def one(j: dict) -> dict:
        res = scorer.score(j["messages"], j["continuation"], j["sites"])
        return {
            **{k: j[k] for k in ("set", "arm", "mode", "policy", "query_id", "ref_sample")},
            "prompt_hash": j["prompt_hash"],
            "n_sources": j["n_sources"],
            "target_slots": j["target_slots"],
            "n_sites": len(res.sites),
            "ap": ap_from_result(res, j["target_slots"]),
            "sites": [
                {"char_offset": s.char_offset, "token_index": s.token_index, "logprobs": s.logprobs}
                for s in res.sites
            ],
            "cached": res.cached,
        }

    out: list[dict] = []
    if workers <= 1:
        for i, j in enumerate(jobs):
            out.append(one(j))
            if (i + 1) % 200 == 0:
                log(f"scored {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)")
    else:
        with ThreadPoolExecutor(workers) as pool:
            out = list(pool.map(one, jobs))
    return out


def rows_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "sites"} for r in rows])


def paired_deltas(
    df: pd.DataFrame, name: str, ref: str, units: dict[str, str], subset=None
) -> pd.DataFrame:
    """Per-query AP deltas (arm minus reference set, mean over reference answers) with units."""
    a = df[df["set"] == name].set_index(["query_id", "ref_sample"])["ap"]
    b = df[df["set"] == ref].set_index(["query_id", "ref_sample"])["ap"]
    j = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    j["d"] = j["a"] - j["b"]
    q = j.groupby(level="query_id")[["a", "b", "d"]].mean()
    if subset is not None:
        q = q.loc[q.index.intersection(pd.Index(subset))]
    q["unit"] = [units.get(x, x) for x in q.index]
    return q


def compare_ap(
    df: pd.DataFrame,
    name: str,
    ref: str,
    units: dict[str, str],
    bootstrap: int = 5000,
    draws: int = 20000,
    subset=None,
) -> dict:
    q = paired_deltas(df, name, ref, units, subset)
    page = q.groupby("unit")[["a", "b", "d"]].mean() * 100
    d = page["d"].to_numpy()
    est, lo, hi = bootstrap_ci(d, b=bootstrap) if len(d) else (np.nan, np.nan, np.nan)
    return {
        "set": name,
        "ref": ref,
        "n_queries": len(q),
        "n_units": len(page),
        "ap_ref_pct": float(page["b"].mean()) if len(page) else np.nan,
        "ap_pct": float(page["a"].mean()) if len(page) else np.nan,
        "d_ap_pp": est,
        "d_ap_lo": lo,
        "d_ap_hi": hi,
        "sd_unit_pp": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
        "p_wilcoxon": wilcoxon_exact_p(d),
        "p_perm": sign_flip_p(d, draws=draws),
        "n_zero_units": int(np.sum(np.abs(d) < 1e-9)),
    }


def analyze(df: pd.DataFrame, comparisons, units: dict[str, str], sc) -> pd.DataFrame:
    """One row per comparison; Holm within each family on both tests. An effect counts only in
    the `primary` family, and only if both Holm-adjusted p values are below 0.05."""
    subsets: dict[str, list[str]] = {}
    for name in df["set"].unique():
        subsets[name] = sorted(df.loc[df["set"] == name, "query_id"].unique())
    out = []
    for c in comparisons:
        common = sorted(set(subsets.get(c.set, [])) & set(subsets.get(c.ref, [])))
        r = compare_ap(df, c.set, c.ref, units, sc.bootstrap, sc.draws, subset=common)
        out.append({"name": c.name, "family": c.family, **r})
    res = pd.DataFrame(out)
    if not len(res):
        return res
    res["p_wilcoxon_holm"] = np.nan
    res["p_perm_holm"] = np.nan
    for _, g in res.groupby("family"):
        res.loc[g.index, "p_wilcoxon_holm"] = holm(g["p_wilcoxon"].tolist())
        res.loc[g.index, "p_perm_holm"] = holm(g["p_perm"].tolist())
    res["significant"] = (
        (res["family"] == "primary") & (res["p_wilcoxon_holm"] < 0.05) & (res["p_perm_holm"] < 0.05)
    )
    return res


def default_comparisons(sets: list[str]):
    """Every set against the baseline in its own passage policy, as secondary comparisons."""
    from vizor.config import Comparison

    out = []
    for s in sets:
        arm, _, pol = s.partition("@")
        ref = f"baseline@{pol}" if pol else "baseline"
        if s != ref and ref in sets:
            out.append(Comparison(name=s, set=s, ref=ref, family="secondary"))
    return out


def _summary(res: pd.DataFrame, meta: dict) -> str:
    lines = [
        f"# Attribution propensity: {meta['run']}",
        "",
        f"Scorer `{meta['scorer_id']}`, pin `{json.dumps(meta['pin'], sort_keys=True)}`. "
        f"Reference answers: baseline samples {meta['refs']}. {meta['n_rows']} scored "
        f"(prompt set, query, reference) rows, {meta['n_rows_with_sites']} with at least one "
        "citation site. Units are pages; the estimate, interval and tests use unweighted page "
        "means. AP deltas are in percentage points of citation probability.",
        "",
        "| Comparison | Family | ΔAP pp [95% CI] | AP ref % | n queries / pages | p Wilcoxon "
        "(Holm) | p sign-flip (Holm) | Significant |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res.itertuples():
        lines.append(
            f"| `{r.name}` | {r.family} | {r.d_ap_pp:+.2f} [{r.d_ap_lo:+.2f}, {r.d_ap_hi:+.2f}] "
            f"| {r.ap_ref_pct:.1f} | {r.n_queries} / {r.n_units} | {r.p_wilcoxon:.3g} "
            f"({r.p_wilcoxon_holm:.3g}) | {r.p_perm:.3g} ({r.p_perm_holm:.3g}) "
            f"| {'yes' if r.significant else 'no'} |"
        )
    lines += [
        "",
        "Significant: primary family only, both Holm-adjusted p values below 0.05.",
    ]
    return "\n".join(lines) + "\n"


def score_run(run_dir: Path, cfg, out: Path | None = None, log: Log = print) -> Path:
    from vizor.config import cache_dir
    from vizor.ingest.corpus import load_project

    sc = cfg.score
    out = out or run_dir / "ap"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    project = load_project(cfg.project_path())
    sets = load_prompt_sets(run_dir)
    refs = load_references(run_dir, set(sc.refs))
    system_prompt = manifest.get("system_prompt") or ""
    which = sc.sets or None
    if which is None and sc.comparisons:
        which = sorted({x for c in sc.comparisons for x in (c.set, c.ref)})
    jobs = build_jobs(sets, refs, project.target_domains, system_prompt, which)
    scorer = make_scorer(sc, cache_dir() / "score")
    pin = scorer.pin()
    log(
        f"{len(jobs)} scoring jobs over {len(which or sets)} prompt sets, scorer {scorer.scorer_id}"
    )
    t0 = time.time()
    rows = run_jobs(scorer, jobs, sc.workers, log)
    df = rows_frame(rows)
    comparisons = sc.comparisons or default_comparisons(sorted(df["set"].unique()))
    res = analyze(df, comparisons, load_units(run_dir), sc)
    write_jsonl_gz(out / "ap_rows.jsonl.gz", rows)
    write_csv(out / "ap_deltas.csv", res)
    meta = {
        "run": run_dir.name,
        "run_git_commit": manifest.get("git_commit"),
        "scorer_id": scorer.scorer_id,
        "pin": pin,
        "score_schema": "score-v1",
        "refs": list(sc.refs),
        "config": cfg.model_dump(mode="json"),
        "n_rows": len(rows),
        "n_rows_with_sites": int(sum(r["n_sites"] > 0 for r in rows)),
        "scorer_stats": scorer.stats() if hasattr(scorer, "stats") else {},
        "elapsed_s": round(time.time() - t0, 1),
        "rows_sha256": hashlib.sha256((out / "ap_rows.jsonl.gz").read_bytes()).hexdigest(),
    }
    (out / "ap_manifest.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (out / "ap_summary.md").write_text(_summary(res, meta))
    log(f"wrote {out} ({meta['scorer_stats']})")
    return out


def recompute_ap(ap_dir: Path) -> tuple[bool, int]:
    """Rebuild every AP value from its stored per-site log-probabilities."""
    n = 0
    for r in read_jsonl_gz(ap_dir / "ap_rows.jsonl.gz"):
        again = ap_from_logprobs([s["logprobs"] for s in r["sites"]], r["target_slots"])
        if not is_close(again, float("nan") if r["ap"] is None else r["ap"], 1e-9):
            return False, n
        n += 1
    return True, n
