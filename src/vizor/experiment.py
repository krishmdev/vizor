"""One full experiment: baseline -> sandbox arms -> position/boost sweeps -> bandit replay ->
greedy optimization trajectory, written to a results directory."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

import vizor
from vizor.config import Config, build
from vizor.generate.llm import BudgetExceeded
from vizor.generate.prompt import INSTRUCTION
from vizor.metrics.sentiment import make_sentiment
from vizor.metrics.visibility import domain_summary
from vizor.models import PINS
from vizor.optimize import bandit as bandit_mod
from vizor.optimize.reward import N_FEATURES, context_features
from vizor.optimize.sandbox import (
    CONTROLS,
    Arm,
    ArmRun,
    Sandbox,
    boost_sweep,
    position_sweep,
    score_scale,
    target_frame,
    two_folds,
)
from vizor.optimize.stats import holm, mde
from vizor.optimize.transforms import FABRICATING, LLM_REWRITES, TransformContext
from vizor.retrieve.chunk import flatten_jsonld
from vizor.runstore import git_commit, host_manifest, src_tree, write_csv, write_jsonl_gz

Log = Callable[[str], None]


def _render_doc(d) -> str:
    parts = [f"Title: {d.title}", f"Description: {d.meta_description}", "", d.body]
    if d.faq:
        parts += ["", "FAQ:"] + [f"Q: {q}\nA: {a}" for q, a in d.faq]
    if d.jsonld:
        parts += ["", "JSON-LD: " + flatten_jsonld(d.jsonld)]
    if d.links:
        parts += ["", "Related: " + "; ".join(f"{a} ({h})" for a, h in d.links)]
    return "\n".join(parts)


def contexts_for(sb: Sandbox, ref: ArmRun, queries) -> np.ndarray:
    """13 features per query from its focus page. The baseline rate features come from `ref`,
    the A/A re-sample when there is one, so they don't share sampling noise with the rewards
    (which are variant minus baseline)."""
    pq = ref.per_query()
    docs = sb.engine.cascade.docs
    xs = []
    for q in queries:
        row = pq.loc[q.query_id]
        doc = docs[sb.focus[q.query_id]]
        xs.append(context_features(doc, q.intent, row["retrieved"], row["cited"], row["imp_pwc"]))
    return np.array(xs)


def reward_table(base: ArmRun, runs: dict[str, ArmRun], queries, samples: int) -> np.ndarray:
    b = target_frame(base.rows).set_index(["query_id", "sample"])["imp_pwc"]
    arms = list(runs)
    r = np.zeros((len(queries), len(arms), samples))
    for j, a in enumerate(arms):
        v = target_frame(runs[a].rows).set_index(["query_id", "sample"])["imp_pwc"]
        for i, q in enumerate(queries):
            for k in range(samples):
                d = v.get((q.query_id, k), 0.0) - b.get((q.query_id, k), 0.0)
                r[i, j, k] = np.clip(d, -1, 1)
    return r


def greedy_trajectory(
    sb: Sandbox,
    base: ArmRun,
    arms: list[str],
    rewards: np.ndarray,
    contexts: np.ndarray,
    queries,
    steps: int,
    log: Log,
) -> tuple[pd.DataFrame, list[ArmRun]]:
    """Fit a ridge model (LinUCB, alpha=0) on fold-1 queries' rewards, let it rank arms for the
    fold-2 queries, then apply them one at a time with transforms built from fold 1 only. An arm
    is kept only if the fold-2 paired 95% CI lower bound on delta PAWC is above zero."""
    train, test = sb.folds
    idx = {q.query_id: i for i, q in enumerate(queries)}
    pol = bandit_mod.LinUCB(arms, contexts.shape[1], alpha=0.0)
    for q in train:
        i = idx[q.query_id]
        for j, a in enumerate(arms):
            for k in range(rewards.shape[2]):
                pol.update(contexts[i], a, rewards[i, j, k])
    test_x = contexts[[idx[q.query_id] for q in test]]
    pred = {a: float(np.mean([pol.predict(x)[a] for x in test_x])) for a in arms}
    ranked = sorted(pred, key=lambda a: -pred[a])
    applied: list[str] = []
    current = base
    rows, extra_runs = [], []
    test_ids = [q.query_id for q in test]
    for step, arm in enumerate(ranked[:steps], start=1):
        trial = [*applied, arm]
        run = sb.run(
            Arm("greedy:" + "+".join(trial), "doc", tuple(trial)), queries=test, ctx_queries=train
        )
        extra_runs.append(run)
        cmp = sb.compare(current, run, subset=test_ids)
        kept = bool(cmp["d_pwc_lo"] > 0)
        log(
            f"greedy step {step}: +{arm} d_pwc={cmp['d_pwc_pp']:+.2f}pp "
            f"[{cmp['d_pwc_lo']:+.2f}, {cmp['d_pwc_hi']:+.2f}] {'kept' if kept else 'rejected'}"
        )
        if kept:
            applied, current = trial, run
        pq = current.per_query().loc[test_ids]
        rows.append(
            {
                "step": step,
                "proposed": arm,
                "predicted_reward_pp": pred[arm] * 100,
                "d_pwc_pp": cmp["d_pwc_pp"],
                "d_lo": cmp["d_pwc_lo"],
                "d_hi": cmp["d_pwc_hi"],
                "kept": kept,
                "applied": "+".join(applied) or "none",
                "heldout_pwc_pct": pq["imp_pwc"].mean() * 100,
                "n_heldout": len(test),
            }
        )
    return pd.DataFrame(rows), extra_runs


def minimum_detectable(
    sb: Sandbox, base: ArmRun, arm_runs: dict, sweep_family: int, n_sweep: int | None = None
) -> dict:
    """MDE at 80% power from the A/A re-sample's per-query noise: for engine arms and sweeps at
    the query level, for page arms at the (fold, focus page) cluster level they are tested at."""
    aa = arm_runs.get("aa_resample")
    if aa is None:
        return {}
    d = (aa.per_query()["imp_pwc"] - base.per_query()["imp_pwc"]).dropna() * 100
    page_runs = [r for r in arm_runs.values() if r.arm.kind == "doc" and r.scored_against]
    engine_arms = [a for a, r in arm_runs.items() if r.arm.kind == "engine"]
    out = {
        "aa_sd_per_query_pp": float(d.std(ddof=1)),
        "n_queries": int(len(d)),
        "engine_arm_pp": mde(
            float(d.std(ddof=1)), len(d), max(1, len(engine_arms) + len(page_runs))
        ),
        "sweep_pp": mde(float(d.std(ddof=1)), n_sweep or len(d), max(1, sweep_family)),
        "n_sweep_queries": n_sweep or int(len(d)),
        "power": 0.8,
        "alpha": 0.05,
    }
    if page_runs:
        units = pd.Series({q: "|".join(page_runs[0].scored_against[q]) for q in d.index})
        cm = d.groupby(units).mean()
        out["n_page_units"] = int(len(cm))
        out["page_arm_pp"] = mde(float(cm.std(ddof=1)), len(cm), len(page_runs) + len(engine_arms))
    return out


def preflight(cfg: Config, log: Log = print) -> dict:
    """Paid backends only start with a priced model, a cap, and an estimate that fits under the
    cap together with what the shared ledger has already spent."""
    from vizor.config import cache_dir
    from vizor.estimate import estimate_cost
    from vizor.generate.llm import SpendLedger, price_for

    if price_for(cfg.llm.model) is None:
        raise BudgetExceeded(f"no price for {cfg.llm.model!r} in PRICES; refusing a paid run")
    if cfg.llm.max_cost_usd is None:
        raise BudgetExceeded("llm.max_cost_usd is not set; refusing a paid run")
    est = estimate_cost(cfg)
    spent = SpendLedger(cache_dir() / "llm" / "ledger.json").total()
    log(
        f"pre-flight: estimate ${est['max_cost_usd_upper_bound']:.2f}, ledger ${spent:.2f}, "
        f"cap ${cfg.llm.max_cost_usd:.2f}"
    )
    if est["max_cost_usd_upper_bound"] + spent > cfg.llm.max_cost_usd:
        raise BudgetExceeded(
            f"estimate ${est['max_cost_usd_upper_bound']:.2f} + spent ${spent:.2f} exceeds "
            f"max_cost_usd ${cfg.llm.max_cost_usd:.2f}; not starting"
        )
    return est


def run_experiment(cfg: Config, out: Path, log: Log = print) -> Path:
    t0 = time.time()
    commit, tree = git_commit(), src_tree()
    if cfg.llm.backend == "openai":
        preflight(cfg, log)
    out.mkdir(parents=True, exist_ok=True)
    project, docs, queries, engine = build(cfg)
    sentiment = make_sentiment(cfg.sentiment)
    embedder = engine.cascade.embedder
    rewrite_llm = engine.llm if cfg.llm.backend != "fake" else None
    doc_map = {d.doc_id: d for d in docs}

    def ctx_for(qs) -> TransformContext:
        return TransformContext(doc_map, list(qs), embedder, llm=rewrite_llm)

    sb = Sandbox(
        engine,
        queries,
        cfg.samples,
        project.domains,
        ctx_for,
        sentiment,
        cfg.decay,
        cfg.sandbox.bootstrap,
    )
    log(
        f"{len(docs)} docs, {len(queries)} queries x {cfg.samples} samples, "
        f"llm={engine.llm.model_id}"
    )

    all_runs: list[ArmRun] = []
    skipped: list[str] = []
    budget_hit = False

    def attempt(label: str, fn):
        nonlocal budget_hit
        if budget_hit:
            skipped.append(label)
            return None
        try:
            return fn()
        except BudgetExceeded as e:
            log(f"budget reached during {label}: {e}")
            budget_hit = True
            skipped.append(label)
            return None

    base = sb.run_baseline()
    all_runs.append(base)
    log(f"baseline done ({time.time() - t0:.0f}s)")

    arm_specs = list(cfg.sandbox.arms)
    if cfg.sandbox.llm_rewrites and rewrite_llm is not None:
        arm_specs += LLM_REWRITES
        if cfg.sandbox.allow_fabrication:
            arm_specs += sorted(FABRICATING)
    parsed = {a: Arm.parse(a) for a in arm_specs}
    # Controls first, then page arms (what a site owner controls, and what the bandit needs),
    # then engine arms and sweeps, so a budget stop cuts the least important work.
    order = (
        [a for a in arm_specs if a in CONTROLS]
        + [a for a in arm_specs if a not in CONTROLS and parsed[a].kind == "doc"]
        + [a for a in arm_specs if a not in CONTROLS and parsed[a].kind != "doc"]
    )
    arm_runs: dict[str, ArmRun] = {}
    for spec in order:
        r = attempt(spec, lambda s=spec: sb.run(parsed[s]))
        if r:
            arm_runs[spec] = r
            log(f"arm {spec} done ({time.time() - t0:.0f}s)")
    all_runs += list(arm_runs.values())

    from vizor.config import _balanced

    sweep_qs = _balanced(queries, cfg.sandbox.sweep_queries) if cfg.sandbox.sweep_queries else None
    pos_df = attempt(
        "position_sweep", lambda: position_sweep(sb, cfg.sandbox.position_sweep, sweep_qs)
    )
    if pos_df:
        pos_df, pos_runs = pos_df
        all_runs += list(pos_runs.values())
        log(
            "position sweep:\n"
            + pos_df[["position", "pwc_pct", "d_pwc_vs_first_pp", "d_lo", "d_hi"]]
            .round(2)
            .to_string(index=False)
        )
    boost_df = attempt("boost_sweep", lambda: boost_sweep(sb, cfg.sandbox.boost_sweep, sweep_qs))
    if boost_df:
        boost_df, boost_runs = boost_df
        all_runs += [r for w, r in boost_runs.items() if w != 0]
        log(
            "boost sweep:\n"
            + boost_df[["boost", "retrieval_pct", "pwc_pct", "d_pwc_pp", "d_lo", "d_hi"]]
            .round(2)
            .to_string(index=False)
        )

    deltas = Sandbox.with_holm([sb.compare(base, r) for r in arm_runs.values()])

    # Sweeps are their own Holm family: every non-reference slot and boost comparison together.
    sweep_p = []
    if isinstance(pos_df, pd.DataFrame):
        sweep_p += [("pos", i, p) for i, p in zip(pos_df.index[1:], pos_df.p.iloc[1:], strict=True)]
    if isinstance(boost_df, pd.DataFrame):
        sweep_p += [
            ("boost", i, p) for i, p in zip(boost_df.index[1:], boost_df.p.iloc[1:], strict=True)
        ]
    if sweep_p:
        adj = holm([p for _, _, p in sweep_p])
        for df_name, df in (("pos", pos_df), ("boost", boost_df)):
            if isinstance(df, pd.DataFrame):
                df["p_holm"] = np.nan
                for (name, i, _), q in zip(sweep_p, adj, strict=True):
                    if name == df_name:
                        df.loc[i, "p_holm"] = q
                df["significant"] = df["p_holm"] < 0.05

    mde_info = minimum_detectable(
        sb, base, arm_runs, len(sweep_p), n_sweep=len(sweep_qs) if sweep_qs else None
    )
    log(f"minimum detectable effects: {mde_info}")
    log(
        "arms:\n"
        + deltas[["arm", "d_pwc_pp", "d_pwc_lo", "d_pwc_hi", "p_holm"]]
        .round(3)
        .to_string(index=False)
    )

    # Bandit over page arms only; engine interventions aren't something a site owner controls.
    b_arms = [a for a, r in arm_runs.items() if r.arm.kind == "doc"]
    traj = pd.DataFrame()
    curve = summary = pd.DataFrame()
    contexts = np.zeros((0, N_FEATURES))
    rewards = np.zeros((0, 0, cfg.samples))
    if len(b_arms) >= 2:
        rewards = reward_table(base, {a: arm_runs[a] for a in b_arms}, queries, cfg.samples)
        contexts = contexts_for(sb, arm_runs.get("aa_resample", base), queries)
        curve, summary = bandit_mod.replay(
            rewards, contexts, b_arms, cfg.bandit.rounds, cfg.bandit.runs, seed=cfg.seed
        )
        idx = {q.query_id: i for i, q in enumerate(queries)}
        train, test = two_folds(queries)
        held = bandit_mod.heldout(
            rewards,
            contexts,
            b_arms,
            [idx[q.query_id] for q in train],
            [idx[q.query_id] for q in test],
        )
        summary = pd.concat([summary, held], ignore_index=True)
        log("bandit:\n" + summary.round(3).to_string(index=False))
        cand = [a for a in b_arms if a != "noop"]
        res = attempt(
            "greedy",
            lambda: greedy_trajectory(
                sb,
                base,
                cand,
                rewards[:, [b_arms.index(a) for a in cand], :],
                contexts,
                queries,
                cfg.bandit.greedy_steps,
                log,
            ),
        )
        if res:
            traj, g_runs = res
            all_runs += g_runs

    # ------------------------------------------------------------------ write
    prompts: dict[str, str] = {}
    responses = []
    for run in all_runs:
        for r in run.results:
            prompts.setdefault(r.answer.prompt_hash, r.prompt)
            responses.append({"arm": run.arm.name, **r.answer.to_dict()})
    write_jsonl_gz(out / "responses.jsonl.gz", responses)
    write_jsonl_gz(
        out / "prompts.jsonl.gz",
        ({"prompt_hash": h, "prompt": p} for h, p in sorted(prompts.items())),
    )
    rows = pd.concat([r.rows for r in all_runs], ignore_index=True)
    write_csv(out / "rows.csv.gz", rows)
    write_csv(out / "baseline_domains.csv", domain_summary(base.rows))
    write_csv(out / "deltas.csv", deltas)
    per_query = pd.concat(
        [r.per_query().assign(arm=r.arm.name).reset_index() for r in [base, *arm_runs.values()]]
    )
    write_csv(out / "per_query.csv", per_query)
    focus = pd.DataFrame(
        [
            {
                "query_id": q.query_id,
                "query": q.text,
                "intent": q.intent,
                "fold": 1 if q in sb.folds[0] else 2,
                "focus_doc": sb.focus[q.query_id],
                "focus_url": doc_map[sb.focus[q.query_id]].url,
            }
            for q in queries
        ]
    )
    write_csv(out / "queries.csv", focus)
    if isinstance(pos_df, pd.DataFrame):
        write_csv(out / "position_sweep.csv", pos_df)
    if isinstance(boost_df, pd.DataFrame):
        write_csv(out / "boost_sweep.csv", boost_df)
    if len(curve):
        write_csv(out / "bandit_curve.csv", curve)
        write_csv(out / "bandit_summary.csv", summary)
        long = [
            {"query_id": q.query_id, "arm": a, "sample": k, "reward": rewards[i, j, k]}
            for i, q in enumerate(queries)
            for j, a in enumerate(b_arms)
            for k in range(rewards.shape[2])
        ]
        write_csv(out / "bandit_rewards.csv.gz", pd.DataFrame(long))
        ctx_df = pd.DataFrame(
            contexts,
            columns=[
                "bias",
                "has_faq",
                "has_jsonld",
                "meta_ok",
                "log_words",
                "links",
                "base_retrieval",
                "base_cite",
                "base_pwc",
                "i_informational",
                "i_comparison",
                "i_transactional",
                "i_troubleshooting",
            ],
        )
        ctx_df.insert(0, "query_id", [q.query_id for q in queries])
        write_csv(out / "bandit_contexts.csv", ctx_df)
    if len(traj):
        write_csv(out / "trajectory.csv", traj)

    # diffs[arm][fold][doc_id] = {url, notes, before, after}
    diffs = {
        name: {
            fold: {
                doc_id: {
                    "url": doc_map[doc_id].url,
                    "notes": run.diffs.get(fold, {}).get(doc_id, ""),
                    "before": _render_doc(doc_map[doc_id]),
                    "after": _render_doc(new),
                }
                for doc_id, new in pages.items()
            }
            for fold, pages in run.changed.items()
        }
        for name, run in arm_runs.items()
        if run.changed
    }
    (out / "diffs.json").write_text(json.dumps(diffs, indent=1, ensure_ascii=False))

    llm_stats = engine.llm.stats() if hasattr(engine.llm, "stats") else {}
    models_seen = sorted({r.answer.model for run in all_runs for r in run.results})
    fingerprints: dict[str, int] = {}
    for run in all_runs:
        for r in run.results:
            fp = r.answer.usage.get("system_fingerprint")
            if fp:
                fingerprints[fp] = fingerprints.get(fp, 0) + 1
    if len(models_seen) > 1:
        log(f"WARNING: answers came from more than one model snapshot: {models_seen}")
    manifest = {
        "vizor_version": vizor.__version__,
        "git_commit": commit,
        "src_tree": tree,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "elapsed_s": round(time.time() - t0, 1),
        "config": cfg.model_dump(),
        "project": project.name,
        "n_docs": len(docs),
        "n_queries": len(queries),
        "samples": cfg.samples,
        "embedder_id": embedder.embedder_id,
        "reranker_id": engine.cascade.reranker.reranker_id,
        "index_backend": engine.cascade.index.backend,
        "llm_model": engine.llm.model_id,
        "llm_is_fake": cfg.llm.backend == "fake",
        "fake_llm_position_prior": 0.05 if cfg.llm.backend == "fake" else None,
        "models_seen": models_seen,
        "single_model_snapshot": len(models_seen) == 1,
        "system_fingerprints": fingerprints,
        "prompt_instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest()[:16],
        "system_prompt": cfg.llm.system_prompt or None,
        "seed_scheme": "sha256(base_seed, query_id, sample, salt)[:8] & 0x7fffffff",
        "base_seed": cfg.seed,
        "sentiment_backend": sentiment.backend_id,
        "decay": cfg.decay,
        "transform_query_scope": "crossfit-2",
        "page_arm_scope": "focus page per query",
        "model_pins": {k: f"{v.repo}@{v.revision}" for k, v in PINS.items()},
        "score_scale": score_scale(base),
        "mde": mde_info,
        "llm_usage": llm_stats,
        "skipped_due_to_budget": skipped,
        "host": host_manifest(holder=Path(out).name, llm=engine.llm.model_id, device="cpu"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    log(f"wrote {out} in {time.time() - t0:.0f}s {llm_stats}")
    return out
