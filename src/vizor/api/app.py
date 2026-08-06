"""HTTP API over the engine and the results directories.

Results live on disk (experiments/results/<id> for committed runs, runs/<id> for new ones);
the API reads them and starts background jobs that write new ones.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from vizor.api.schemas import AnswerOut, AnswerRequest, JobOut, JobRequest, SentenceOut, SourceOut
from vizor.metrics.impression import decay_weights, impressions
from vizor.metrics.visibility import source_labels
from vizor.models import project_root

STATE: dict = {"engine": None, "egress": None, "jobs": {}}
_lock = threading.Lock()
_jobs_lock = threading.Lock()


def config_path() -> Path | None:
    p = Path(os.environ.get("VIZOR_CONFIG", project_root() / "configs" / "demo.yaml"))
    return p if p.exists() else None


def results_roots() -> list[Path]:
    root = project_root()
    return [root / "experiments" / "results", root / "runs"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("VIZOR_EGRESS_CANARY") == "1":
        from vizor.egress import check_egress

        leaks = check_egress()
        STATE["egress"] = "blocked" if not leaks else f"open: {', '.join(leaks)}"
    yield


app = FastAPI(title="Vizor", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("VIZOR_CORS", "http://localhost:8501").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _built():
    with _lock:
        if STATE["engine"] is None:
            from vizor.config import Config, build

            cfg = Config.load(config_path())
            STATE["cfg"] = cfg
            STATE["project"], STATE["docs"], STATE["queries"], STATE["engine"] = build(cfg)
        return STATE["cfg"], STATE["project"], STATE["docs"], STATE["queries"], STATE["engine"]


def _find(run_id: str) -> Path:
    if "/" in run_id or run_id.startswith("."):
        raise HTTPException(400, "bad id")
    for root in results_roots():
        p = root / run_id
        if (p / "manifest.json").exists():
            return p
    raise HTTPException(404, f"no results named {run_id}")


def _csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return json.loads(pd.read_csv(p).to_json(orient="records"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "egress": STATE["egress"], "engine_loaded": STATE["engine"] is not None}


@app.get("/project")
def project() -> dict:
    cfg, proj, docs, queries, engine = _built()
    return {
        "name": proj.name,
        "domains": proj.domains,
        "n_docs": len(docs),
        "queries": [{"query_id": q.query_id, "query": q.text, "intent": q.intent} for q in queries],
        "embedder_id": engine.cascade.embedder_id,
        "reranker_id": engine.cascade.reranker.reranker_id,
        "llm": engine.llm.model_id,
        "is_fake_llm": cfg.llm.backend == "fake",
    }


@app.get("/corpus/docs")
def corpus_docs() -> list[dict]:
    _, _, docs, _, _ = _built()
    return [
        {
            "doc_id": d.doc_id,
            "url": d.url,
            "domain": d.domain,
            "role": d.role,
            "title": d.title,
            "words": d.n_words,
            "has_faq": bool(d.faq),
            "has_jsonld": bool(d.jsonld),
            "links": len(d.links),
        }
        for d in docs
    ]


@app.get("/corpus/docs/{doc_id}")
def corpus_doc(doc_id: str) -> dict:
    _, _, docs, _, _ = _built()
    for d in docs:
        if d.doc_id == doc_id:
            return d.to_dict()
    raise HTTPException(404, "no such doc")


def answer_payload(ans, prompt: str, roles: dict[str, str], is_fake: bool) -> AnswerOut:
    from vizor.generate.prompt import parse_prompt

    question, srcs = parse_prompt(prompt) if prompt else ("", [])
    titles = {s.index: s.title for s in srcs}
    n = len(ans.sources)
    imp = impressions(ans.sentences, n)
    labels = source_labels(ans)
    counts = [sum(s.citations.count(i + 1) for s in ans.sentences) for i in range(n)]
    d = decay_weights(len(ans.sentences))
    return AnswerOut(
        query=question or ans.query_id,
        model=ans.model,
        is_fake_llm=is_fake,
        seed=ans.seed,
        text=ans.text,
        sources=[
            SourceOut(
                position=s.position,
                domain=s.domain,
                role=roles.get(s.domain, "competitor"),
                url=s.url,
                title=titles.get(s.position, ""),
                retrieval_score=s.retrieval_score,
                rerank_score=s.rerank_score,
                final_score=s.final_score,
                label=labels[i],
                pwc_share=imp.pwc[i],
                citations=counts[i],
            )
            for i, s in enumerate(ans.sources)
        ],
        sentences=[
            SentenceOut(
                pos=s.pos, text=s.text, n_words=s.n_words, decay=w, citations=list(s.citations)
            )
            for s, w in zip(ans.sentences, d, strict=True)
        ],
        hallucinated_citations=ans.hallucinated_citations,
    )


@app.post("/answer", response_model=AnswerOut)
def answer(req: AnswerRequest) -> AnswerOut:
    from vizor.optimize.retrieval_policy import RetrievalPolicy
    from vizor.optimize.transforms import TransformContext, apply_to_targets
    from vizor.types import Query, stable_id

    cfg, proj, docs, queries, engine = _built()
    try:
        policy = RetrievalPolicy.parse(req.policy)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    q = Query("live-" + stable_id(req.query, 8), req.query, "live")
    # Serializes live answers so concurrent requests don't rebuild the cascade in parallel; the
    # embedder/reranker caches themselves are locked.
    with _lock:
        if req.arms:
            ctx = TransformContext({d.doc_id: d for d in docs}, queries, engine.cascade.embedder)
            try:
                changed, _ = apply_to_targets(req.arms, ctx)
            except KeyError as e:
                raise HTTPException(422, f"unknown arm {e}") from e
            engine = engine.with_cascade(engine.cascade.with_docs(changed))
        r = engine.answer(q, req.sample, policy)
    return answer_payload(r.answer, r.prompt, proj.domains, cfg.llm.backend == "fake")


@app.get("/experiments")
def experiments() -> list[dict]:
    out = []
    for root in results_roots():
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            mf = p / "manifest.json"
            if mf.exists():
                m = json.loads(mf.read_text())
                out.append(
                    {
                        "id": p.name,
                        "location": root.name,
                        "llm_model": m.get("llm_model"),
                        "is_fake_llm": m.get("llm_is_fake"),
                        "n_queries": m.get("n_queries"),
                        "samples": m.get("samples"),
                        "started": m.get("started"),
                        "kind": "experiment" if (p / "deltas.csv").exists() else "baseline",
                    }
                )
    return out


@app.get("/experiments/{run_id}")
def experiment(run_id: str) -> dict:
    p = _find(run_id)
    return {
        "id": run_id,
        "manifest": json.loads((p / "manifest.json").read_text()),
        "domains": _csv(p / "baseline_domains.csv") or _csv(p / "domains.csv"),
        "deltas": _csv(p / "deltas.csv"),
        "position_sweep": _csv(p / "position_sweep.csv"),
        "boost_sweep": _csv(p / "boost_sweep.csv"),
        "bandit_summary": _csv(p / "bandit_summary.csv"),
        "bandit_curve": _csv(p / "bandit_curve.csv"),
        "trajectory": _csv(p / "trajectory.csv"),
    }


@app.get("/experiments/{run_id}/answers")
def experiment_answers(run_id: str, arm: str = "baseline") -> list[dict]:
    from vizor.runstore import read_jsonl_gz

    p = _find(run_id)
    return [
        {"query_id": r["query_id"], "sample": r["sample"], "prompt_hash": r["prompt_hash"]}
        for r in read_jsonl_gz(p / "responses.jsonl.gz")
        if r["arm"] == arm
    ]


@app.get("/experiments/{run_id}/answers/{query_id}/{sample}", response_model=AnswerOut)
def experiment_answer(run_id: str, query_id: str, sample: int, arm: str = "baseline") -> AnswerOut:
    from vizor.inspect import load_answer

    p = _find(run_id)
    m = json.loads((p / "manifest.json").read_text())
    try:
        ans, prompt = load_answer(p, arm=arm, query_id=query_id, sample=sample)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    from vizor.config import Config
    from vizor.ingest.corpus import load_project

    roles = load_project(Config.model_validate(m["config"]).project_path()).domains
    return answer_payload(ans, prompt, roles, bool(m.get("llm_is_fake")))


@app.get("/experiments/{run_id}/sets")
def experiment_sets(run_id: str) -> list[str]:
    """Prompt sets of a run: sampled arms plus prompt-only sets written for scoring."""
    from vizor.scoring import load_prompt_sets

    return sorted(load_prompt_sets(_find(run_id)))


@app.get("/experiments/{run_id}/compare/{query_id}/{sample}")
def experiment_compare(
    run_id: str, query_id: str, sample: int, arm: str, ref: str = "baseline"
) -> dict:
    """Side by side: two prompt sets for one (query, sample), with their rendered passages,
    their sampled answers where they exist, and per-site AP of the reference answer."""
    from vizor.scoring import side_by_side

    p = _find(run_id)
    out = side_by_side(p, query_id, sample, arm, ref)
    for side in out["sides"]:
        side["answer"] = (
            experiment_answer(run_id, query_id, sample, arm=side["set"]).model_dump()
            if side["sampled"]
            else None
        )
    return out


@app.get("/experiments/{run_id}/diffs")
def experiment_diffs(run_id: str) -> dict:
    p = _find(run_id) / "diffs.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _start(kind: str, req: JobRequest) -> JobOut:
    from vizor.config import Config

    cfg = Config.load(config_path())
    cfg.queries, cfg.samples = req.queries, req.samples or cfg.samples
    if req.arms is not None:
        cfg.sandbox.arms = req.arms
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{uuid.uuid4().hex[:6]}"
    out = project_root() / "runs" / job_id
    job = {"job_id": job_id, "kind": kind, "status": "running", "out_dir": str(out), "error": None}
    with _jobs_lock:
        if any(j["status"] == "running" for j in STATE["jobs"].values()):
            raise HTTPException(409, "another job is running; one at a time")
        STATE["jobs"][job_id] = job

    def work() -> None:
        try:
            if kind == "baseline":
                from vizor.baseline import run_baseline

                run_baseline(cfg, out, log=lambda _: None)
            else:
                from vizor.experiment import run_experiment
                from vizor.metrics.report import write_summary

                run_experiment(cfg, out, log=lambda _: None)
                write_summary(out)
            job["status"] = "done"
        except Exception as e:  # surfaced through GET /jobs/{id}
            job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"

    threading.Thread(target=work, daemon=True).start()
    return JobOut(**job)


@app.post("/runs", response_model=JobOut)
def start_run(req: JobRequest) -> JobOut:
    """Baseline visibility run in the background."""
    return _start("baseline", req)


@app.post("/sandbox", response_model=JobOut)
def start_sandbox(req: JobRequest) -> JobOut:
    """Sandbox + sweeps + bandit replay + greedy loop in the background."""
    return _start("experiment", req)


@app.get("/jobs/{job_id}", response_model=JobOut)
def job(job_id: str) -> JobOut:
    if job_id not in STATE["jobs"]:
        raise HTTPException(404, "no such job")
    return JobOut(**STATE["jobs"][job_id])
