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
from vizor.generate.scorer import CachedScorer, FakeScorer, LocalhostScorer, Scorer, renormalize
from vizor.metrics.propensity import (
    ap_from_logprobs,
    ap_from_probs,
    ap_from_result,
    citation_sites,
    is_close,
    remap_citations,
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
            out[(r["query_id"], r["sample"])] = {
                "text": r["text"],
                "n_sources": len(r["sources"]),
                "doc_ids": [s["doc_id"] for s in sorted(r["sources"], key=lambda s: s["position"])],
            }
    return out


def load_units(run_dir: Path) -> dict[str, str]:
    """query_id -> the underlying target page it is scored against."""
    q = pd.read_csv(run_dir / "queries.csv")
    return dict(zip(q["query_id"], q["focus_doc"], strict=True))


def reference_text(ref: dict, doc_ids: list[str]) -> tuple[str, bool, int]:
    """The reference answer renumbered to a prompt whose sources are `doc_ids` (in order):
    (text, whether anything was renumbered, markers whose page is not in that prompt)."""
    base_ids = ref.get("doc_ids") or doc_ids
    pos = {d: i + 1 for i, d in enumerate(doc_ids)}
    mapping = {i + 1: pos[d] for i, d in enumerate(base_ids) if d in pos}
    if all(a == b for a, b in mapping.items()) and len(mapping) == len(base_ids):
        return ref["text"], False, 0
    text, unmapped = remap_citations(ref["text"], mapping)
    return text, True, unmapped


def build_jobs(
    sets: dict[str, dict[str, dict]],
    refs: dict[tuple[str, int], dict],
    targets: list[str],
    system_prompt: str,
    which: list[str] | None = None,
    units: dict[str, str] | None = None,
) -> list[dict]:
    """One job per (prompt set, query, reference answer). When a prompt orders the sources
    differently from the baseline the reference was written against (full mode, the slot
    control), the reference's citation indices are renumbered to that prompt's order first, so
    the earlier markers in the prefix point at the same pages they did in the baseline."""
    jobs = []
    units = units or {}
    for name in which or sorted(sets):
        if name not in sets:
            raise KeyError(f"prompt set {name!r} is not in this run")
        for qid, e in sets[name].items():
            doc_ids = [s["doc_id"] for s in sorted(e["sources"], key=lambda s: s["position"])]
            for (rq, k), ref in sorted(refs.items()):
                if rq != qid:
                    continue
                n = len(e["sources"])
                sites = citation_sites(ref["text"], ref["n_sources"], n)
                text, remapped, unmapped = reference_text(ref, doc_ids)
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
                        "focus_in_sources": units.get(qid) in doc_ids if units else None,
                        "remapped": remapped,
                        "n_unmapped": unmapped,
                        "messages": messages,
                        "continuation": text,
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
            "focus_in_sources": j.get("focus_in_sources"),
            "remapped": j.get("remapped", False),
            "n_unmapped": j.get("n_unmapped", 0),
            "n_sites": len(res.sites),
            "ap": ap_from_result(res, j["target_slots"]),
            # the first site's prefix holds no earlier marker, so it cannot be steered by them
            "ap_first": ap_from_probs([s.renorm for s in res.sites[:1]], j["target_slots"]),
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
    df: pd.DataFrame, name: str, ref: str, units: dict[str, str], subset=None, col: str = "ap"
) -> pd.DataFrame:
    """Per-query AP deltas (arm minus reference set, mean over reference answers) with units.
    `col="ap_first"` uses only each reference answer's first citation site."""
    a = df[df["set"] == name].set_index(["query_id", "ref_sample"])[col]
    b = df[df["set"] == ref].set_index(["query_id", "ref_sample"])[col]
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
    qf = paired_deltas(df, name, ref, units, subset, col="ap_first")
    df_ = (qf.groupby("unit")["d"].mean() * 100).to_numpy()
    f_est, f_lo, f_hi = bootstrap_ci(df_, b=bootstrap) if len(df_) else (np.nan,) * 3
    arm = df[df["set"] == name]
    if subset is not None:
        arm = arm[arm["query_id"].isin(subset)]
    focus = (
        arm.groupby("query_id")["focus_in_sources"].first().astype(float).mean() * 100
        if "focus_in_sources" in arm and arm["focus_in_sources"].notna().any()
        else np.nan
    )
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
        "focus_in_sources_pct": focus,
        "remapped_pct": float(arm["remapped"].mean() * 100) if "remapped" in arm else np.nan,
        "d_ap_first_pp": f_est,
        "d_ap_first_lo": f_lo,
        "d_ap_first_hi": f_hi,
        "p_perm_first": sign_flip_p(df_, draws=draws),
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
        "(Holm) | p sign-flip (Holm) | Significant | ΔAP first site pp | Focus page shown % "
        "| Pages with Δ = 0 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res.itertuples():
        lines.append(
            f"| `{r.name}` | {r.family} | {r.d_ap_pp:+.2f} [{r.d_ap_lo:+.2f}, {r.d_ap_hi:+.2f}] "
            f"| {r.ap_ref_pct:.1f} | {r.n_queries} / {r.n_units} | {r.p_wilcoxon:.3g} "
            f"({r.p_wilcoxon_holm:.3g}) | {r.p_perm:.3g} ({r.p_perm_holm:.3g}) "
            f"| {'yes' if r.significant else 'no'} | {r.d_ap_first_pp:+.2f} "
            f"| {r.focus_in_sources_pct:.0f} | {r.n_zero_units} |"
        )
    lines += [
        "",
        "Significant: primary family only, both Holm-adjusted p values below 0.05. In full mode "
        "and the slot control the reference answer's indices are renumbered to each prompt's "
        "source order; the first-site column uses only each reference's first citation site, "
        "whose prefix holds no earlier marker.",
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
    units = load_units(run_dir)
    jobs = build_jobs(sets, refs, project.target_domains, system_prompt, which, units)
    scorer = make_scorer(sc, cache_dir() / "score")
    pin = scorer.pin()
    log(
        f"{len(jobs)} scoring jobs over {len(which or sets)} prompt sets, scorer {scorer.scorer_id}"
    )
    t0 = time.time()
    rows = run_jobs(scorer, jobs, sc.workers, log)
    df = rows_frame(rows)
    comparisons = sc.comparisons or default_comparisons(sorted(df["set"].unique()))
    res = analyze(df, comparisons, units, sc)
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
        if "ap_first" in r:
            first = ap_from_logprobs([s["logprobs"] for s in r["sites"][:1]], r["target_slots"])
            if not is_close(first, float("nan") if r["ap_first"] is None else r["ap_first"], 1e-9):
                return False, n
        n += 1
    return True, n


# --------------------------------------------------------------------- leave-one-passage-out
def _passage_lines(content: str) -> list[int]:
    """Indices of the rendered passage lines in a source's content (not the "Related:" line)."""
    return [i for i, line in enumerate(content.split("\n")) if not line.startswith("Related: ")]


def drop_passage(prompt: str, source: int, line: int) -> str:
    """The prompt with one rendered passage line of one source removed, all else unchanged."""
    from dataclasses import replace

    from vizor.generate.prompt import format_prompt, parse_prompt

    question, sources = parse_prompt(prompt)
    if format_prompt(question, sources) != prompt:
        raise ValueError("prompt does not round-trip through parse_prompt/format_prompt")
    out = []
    for s in sources:
        if s.index == source:
            lines = s.content.split("\n")
            s = replace(s, content="\n".join(lines[:line] + lines[line + 1 :]))
        out.append(s)
    return format_prompt(question, out)


def attribute(
    run_dir: Path,
    cfg,
    set_name: str,
    query_ids: list[str],
    log: Log = print,
) -> pd.DataFrame:
    """Leave-one-passage-out AP for (prompt set, query) pairs: for each rendered passage of each
    source, AP of the target with that passage removed. d_ap_pp = AP(full) - AP(without), so a
    positive value means the passage pushes citations toward the target. A lighter ContextCite."""
    from vizor.config import cache_dir
    from vizor.generate.prompt import parse_prompt
    from vizor.ingest.corpus import load_project

    sc = cfg.score
    manifest = json.loads((run_dir / "manifest.json").read_text())
    project = load_project(cfg.project_path())
    sets = load_prompt_sets(run_dir)
    refs = load_references(run_dir, set(sc.refs))
    system_prompt = manifest.get("system_prompt") or ""
    scorer = make_scorer(sc, cache_dir() / "score")
    rows = []
    for qid in query_ids:
        e = sets[set_name][qid]
        slots = target_slots(e["sources"], project.target_domains)
        domains = {int(s["position"]): s["domain"] for s in e["sources"]}
        _, rendered = parse_prompt(e["prompt"])

        def msgs(prompt: str) -> list[dict]:
            m = [{"role": "user", "content": prompt}]
            return [{"role": "system", "content": system_prompt}, *m] if system_prompt else m

        for (rq, k), ref in sorted(refs.items()):
            if rq != qid:
                continue
            sites = citation_sites(ref["text"], ref["n_sources"], len(e["sources"]))
            if not sites:
                continue
            ids = [x["doc_id"] for x in sorted(e["sources"], key=lambda x: x["position"])]
            text = reference_text(ref, ids)[0]
            full = ap_from_result(scorer.score(msgs(e["prompt"]), text, sites), slots)
            for s in rendered:
                lines = s.content.split("\n")
                for i in _passage_lines(s.content):
                    p = drop_passage(e["prompt"], s.index, i)
                    ap = ap_from_result(scorer.score(msgs(p), text, sites), slots)
                    rows.append(
                        {
                            "set": set_name,
                            "query_id": qid,
                            "ref_sample": k,
                            "source": s.index,
                            "domain": domains.get(s.index, ""),
                            "is_target": s.index in slots,
                            "passage_no": i + 1,
                            "passage": lines[i][:160],
                            "ap_full_pct": full * 100,
                            "ap_without_pct": ap * 100,
                            "d_ap_pp": (full - ap) * 100,
                        }
                    )
        log(f"attributed {set_name} {qid}")
    return pd.DataFrame(rows)


def attribution_md(df: pd.DataFrame, top: int = 8) -> str:
    """Per (set, query): the passages whose removal moves the target's AP most, averaged over
    reference answers."""
    if not len(df):
        return "No citation sites in the reference answers.\n"
    lines = []
    keys = ["set", "query_id", "source", "domain", "is_target", "passage_no", "passage"]
    g = df.groupby(keys, as_index=False)[["ap_full_pct", "d_ap_pp"]].mean()
    for (s, q), part in g.groupby(["set", "query_id"]):
        part = part.reindex(part["d_ap_pp"].abs().sort_values(ascending=False).index).head(top)
        lines += [
            f"#### `{s}`, query `{q}` (target AP {part['ap_full_pct'].iloc[0]:.1f}%)",
            "",
            "| Source | Target | Passage | ΔAP pp if removed |",
            "|---|---|---|---|",
        ]
        for r in part.itertuples():
            text = r.passage.replace("|", "/")
            lines.append(
                f"| [{r.source}] {r.domain} | {'yes' if r.is_target else ''} | "
                f"{r.passage_no}: {text} | {-r.d_ap_pp:+.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------ side-by-side view
def ap_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "ap_rows.jsonl.gz").exists():
        return [run_dir]
    return sorted(p.parent for p in run_dir.glob("*/ap_rows.jsonl.gz"))


def side_by_side(
    run_dir: Path, query_id: str, sample: int, arm: str, ref: str = "baseline"
) -> dict:
    """Two prompt sets for one (query, sample): each side's rendered passages per source, and,
    when AP rows exist for that reference answer, the target's probability at every citation
    site of the reference under each side's prompt."""
    from vizor.generate.prompt import parse_prompt

    sets = load_prompt_sets(run_dir)
    sampled = {
        r["arm"] for r in read_jsonl_gz(run_dir / "responses.jsonl.gz") if r["sample"] == sample
    }
    sides = []
    for name in (ref, arm):
        e = sets.get(name, {}).get(query_id)
        if e is None:
            sides.append({"set": name, "sampled": False, "passages": []})
            continue
        _, rendered = parse_prompt(e["prompt"])
        dom = {int(s["position"]): s["domain"] for s in e["sources"]}
        sides.append(
            {
                "set": name,
                "mode": e["mode"],
                "policy": e["policy"],
                "sampled": name in sampled,
                "passages": [
                    {
                        "position": s.index,
                        "domain": dom.get(s.index, ""),
                        "title": s.title,
                        "lines": s.content.split("\n"),
                    }
                    for s in rendered
                ],
            }
        )
    sites: list[dict] = []
    ap: dict[str, float] = {}
    ref_text = ""
    for d in ap_dirs(run_dir):
        found = {
            r["set"]: r
            for r in read_jsonl_gz(d / "ap_rows.jsonl.gz")
            if r["query_id"] == query_id and r["ref_sample"] == sample and r["set"] in (ref, arm)
        }
        if len(found) < 2:
            continue
        refs = load_references(run_dir, {sample})
        ref_text = refs.get((query_id, sample), {}).get("text", "")
        for name, r in found.items():
            ap[name] = r["ap"]
        a, b = found[ref], found[arm]
        for sa, sb in zip(a["sites"], b["sites"], strict=True):
            off = sa["char_offset"]
            pa = sum(renormalize(sa["logprobs"]).get(str(t), 0.0) for t in a["target_slots"])
            pb = sum(renormalize(sb["logprobs"]).get(str(t), 0.0) for t in b["target_slots"])
            sites.append(
                {
                    "char_offset": off,
                    "context": ref_text[max(0, off - 60) : off + 2],
                    "cited": ref_text[off : off + 1],
                    "p_target_ref": pa,
                    "p_target_arm": pb,
                }
            )
        break
    return {
        "query_id": query_id,
        "sample": sample,
        "ref": ref,
        "arm": arm,
        "sides": sides,
        "reference_text": ref_text,
        "ap": ap,
        "sites": sites,
    }
