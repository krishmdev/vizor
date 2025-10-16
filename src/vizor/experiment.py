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
    Arm,
    ArmRun,
    Sandbox,
    boost_sweep,
    position_sweep,
    score_scale,
    target_frame,
)
from vizor.optimize.transforms import FABRICATING, LLM_REWRITES, TransformContext
from vizor.retrieve.chunk import flatten_jsonld
from vizor.runstore import git_commit, host_manifest, write_csv, write_jsonl_gz
from vizor.types import stable_seed

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


def split_queries(queries, frac: float = 0.5):
    """Deterministic train/held-out split by hash, stratified by intent."""
    train, test = [], []
    by: dict[str, list] = {}
    for q in queries:
        by.setdefault(q.intent, []).append(q)
    for group in by.values():
        group = sorted(group, key=lambda q: stable_seed("split", q.query_id))
        k = int(round(len(group) * frac))
        train += group[:k]
        test += group[k:]
    return train, test


def contexts_for(base: ArmRun, queries) -> np.ndarray:
    pq = base.per_query()
    focus = {}
    for r in base.results:
        if r.answer.query_id not in focus:
            focus[r.answer.query_id] = next(
                (c.doc for c in r.selection.candidates if c.doc.role == "target"), None
            )
    xs = []
    for q in queries:
        row = pq.loc[q.query_id]
        doc = focus.get(q.query_id)
        if doc is None:
            x = np.zeros(N_FEATURES)
            x[0] = 1.0
        else:
            x = context_features(doc, q.intent, row["retrieved"], row["cited"], row["imp_pwc"])
        xs.append(x)
    return np.array(xs)


def reward_table(base: ArmRun, runs: dict[str, ArmRun], queries, samples: int) -> np.ndarray:
    b = target_frame(base.rows).set_index(["query_id", "sample"])["imp_pwc"]
    arms = list(runs)
    r = np.zeros((len(queries), len(arms), samples))
    for j, a in enumerate(arms):
        v = target_frame(runs[a].rows).set_index(["query_id", "sample"])["imp_pwc"]
        for i, q in enumerate(queries):
            for k in range(samples):
                r[i, j, k] = np.clip(
                    v.get((q.query_id, k), 0.0) - b.get((q.query_id, k), 0.0), -1, 1
                )
    return r


def greedy_trajectory(
    sb: Sandbox,
    base: ArmRun,
    arms: list[str],
    rewards: np.ndarray,
    contexts: np.ndarray,
    queries,
    steps: int,
    ctx_factory: Callable[[list], TransformContext],
    log: Log,
) -> tuple[pd.DataFrame, list[ArmRun]]:
    """Fit LinUCB on the training queries' rewards, let it propose arms for the held-out queries,
    and keep an arm only if the held-out paired 95% CI lower bound on delta PAWC is above zero."""
    train, test = split_queries(queries)
    idx = {q.query_id: i for i, q in enumerate(queries)}
    pol = bandit_mod.LinUCB(arms, contexts.shape[1], alpha=0.5)
    for q in train:
        i = idx[q.query_id]
        for j, a in enumerate(arms):
            for k in range(rewards.shape[2]):
                pol.update(contexts[i], a, rewards[i, j, k])
    test_x = contexts[[idx[q.query_id] for q in test]]
    pred = {a: float(np.mean([pol.predict(x)[a] for x in test_x])) for a in arms}
    ranked = sorted(pred, key=lambda a: -pred[a])
    ctx = ctx_factory(train)
    applied: list[str] = []
    current = base
    rows, extra_runs = [], []
    for step, arm in enumerate(ranked[:steps], start=1):
        trial = [*applied, arm]
        run = sb.run(
            Arm("greedy:" + "+".join(trial), "doc", tuple(trial)), queries=test, docs_ctx=ctx
        )
        extra_runs.append(run)
        cmp = sb.compare(current, run, subset=[q.query_id for q in test])
        kept = bool(cmp["d_pwc_lo"] > 0)
        log(
            f"greedy step {step}: +{arm} d_pwc={cmp['d_pwc_pp']:+.2f}pp "
            f"[{cmp['d_pwc_lo']:+.2f}, {cmp['d_pwc_hi']:+.2f}] {'kept' if kept else 'rejected'}"
        )
        if kept:
            applied, current = trial, run
        pq = current.per_query().loc[[q.query_id for q in test]]
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


def run_experiment(cfg: Config, out: Path, log: Log = print) -> Path:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    project, docs, queries, engine = build(cfg)
    sentiment = make_sentiment(cfg.sentiment)
    embedder = engine.cascade.embedder
    rewrite_llm = engine.llm if cfg.llm.backend != "fake" else None
    doc_map = {d.doc_id: d for d in docs}

    def ctx_for(qs) -> TransformContext:
        return TransformContext(doc_map, qs, embedder, llm=rewrite_llm)

    sb = Sandbox(
        engine,
        queries,
        cfg.samples,
        project.domains,
        ctx_for(queries),
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
    controls = [a for a in arm_specs if a in ("noop", "aa_resample")]
    others = [a for a in arm_specs if a not in controls]

    arm_runs: dict[str, ArmRun] = {}
    for spec in controls:
        r = attempt(spec, lambda s=spec: sb.run(Arm.parse(s)))
        if r:
            arm_runs[spec] = r

    pos_df = attempt("position_sweep", lambda: position_sweep(sb, cfg.sandbox.position_sweep))
    boost_df = attempt("boost_sweep", lambda: boost_sweep(sb, cfg.sandbox.boost_sweep))
    if pos_df:
        pos_df, pos_runs = pos_df
        all_runs += list(pos_runs.values())
        log(
            "position sweep:\n"
            + pos_df[["position", "pwc_pct", "d_pwc_vs_first_pp", "d_lo", "d_hi"]]
            .round(2)
            .to_string(index=False)
        )
    if boost_df:
        boost_df, boost_runs = boost_df
        all_runs += [r for w, r in boost_runs.items() if w != 0]
        log(
            "boost sweep:\n"
            + boost_df[["boost", "retrieval_pct", "pwc_pct", "d_pwc_pp", "d_lo", "d_hi"]]
            .round(2)
            .to_string(index=False)
        )

    for spec in others:
        r = attempt(spec, lambda s=spec: sb.run(Arm.parse(s)))
        if r:
            arm_runs[spec] = r
            log(f"arm {spec} done ({time.time() - t0:.0f}s)")
    all_runs += list(arm_runs.values())

    deltas = Sandbox.with_holm([sb.compare(base, r) for r in arm_runs.values()])
    log(
        "arms:\n"
        + deltas[["arm", "d_pwc_pp", "d_pwc_lo", "d_pwc_hi", "p_holm"]]
        .round(3)
        .to_string(index=False)
    )

    # Bandit: page-side arms only; engine interventions aren't actions a site owner controls.
    b_arms = [a for a, r in arm_runs.items() if r.arm.kind == "doc"]
    traj = pd.DataFrame()
    curve = summary = pd.DataFrame()
    contexts = np.zeros((0, N_FEATURES))
    rewards = np.zeros((0, 0, cfg.samples))
    if len(b_arms) >= 2:
        b_runs = {a: arm_runs[a] for a in b_arms}
        rewards = reward_table(base, b_runs, queries, cfg.samples)
        contexts = contexts_for(base, queries)
        curve, summary = bandit_mod.replay(
            rewards, contexts, b_arms, cfg.bandit.rounds, cfg.bandit.runs, seed=cfg.seed
        )
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
                ctx_for,
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

    diffs = {
        name: {
            doc_id: {
                "url": doc_map[doc_id].url,
                "notes": run.diffs.get(doc_id, ""),
                "before": _render_doc(doc_map[doc_id]),
                "after": _render_doc(new),
            }
            for doc_id, new in run.changed.items()
        }
        for name, run in arm_runs.items()
        if run.changed
    }
    (out / "diffs.json").write_text(json.dumps(diffs, indent=1, ensure_ascii=False))

    llm_stats = engine.llm.stats() if hasattr(engine.llm, "stats") else {}
    manifest = {
        "vizor_version": vizor.__version__,
        "git_commit": git_commit(),
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
        "models_seen": sorted({r.answer.model for run in all_runs for r in run.results}),
        "prompt_instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest()[:16],
        "seed_scheme": "sha256(base_seed, query_id, sample, salt)[:8] & 0x7fffffff",
        "base_seed": cfg.seed,
        "sentiment_backend": sentiment.backend_id,
        "decay": cfg.decay,
        "model_pins": {k: f"{v.repo}@{v.revision}" for k, v in PINS.items()},
        "score_scale": score_scale(base),
        "llm_usage": llm_stats,
        "skipped_due_to_budget": skipped,
        "host": host_manifest(holder=Path(out).name, llm=engine.llm.model_id, device="cpu"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    log(f"wrote {out} in {time.time() - t0:.0f}s {llm_stats}")
    return out
