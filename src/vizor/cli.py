from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Vizor: measure and test LLM citation visibility.",
)
models_app = typer.Typer(no_args_is_help=True, help="Pinned model artifacts.")
selfcheck_app = typer.Typer(no_args_is_help=True, help="Runtime checks.")
ingest_app = typer.Typer(no_args_is_help=True, help="Bring real pages into a corpus.")
queries_app = typer.Typer(no_args_is_help=True, help="Query sets.")
app.add_typer(models_app, name="models")
app.add_typer(selfcheck_app, name="selfcheck")
app.add_typer(ingest_app, name="ingest")
app.add_typer(queries_app, name="queries")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML config")]


def _log(msg: str) -> None:
    typer.echo(msg, err=True)


def _config(
    config: Path | None,
    embedder: str | None = None,
    reranker: str | None = None,
    llm: str | None = None,
    model: str | None = None,
    sentiment: str | None = None,
    queries: int | None = None,
    samples: int | None = None,
    max_cost_usd: float | None = None,
    workers: int | None = None,
):
    from vizor.config import Config

    cfg = Config.load(config)
    for key, val in [
        ("embedder", embedder),
        ("reranker", reranker),
        ("sentiment", sentiment),
        ("queries", queries),
        ("samples", samples),
    ]:
        if val is not None:
            setattr(cfg, key, val)
    if llm is not None:
        cfg.llm.backend = llm  # type: ignore[assignment]
    if model is not None:
        cfg.llm.model = model
    if max_cost_usd is not None:
        cfg.llm.max_cost_usd = max_cost_usd
    if workers is not None:
        cfg.llm.workers = workers
    return type(cfg).model_validate(cfg.model_dump())


@app.command()
def demo(
    config: ConfigOpt = Path("configs/demo.yaml"),
    out: Annotated[Path | None, typer.Option(help="results directory")] = None,
    embedder: Annotated[str | None, typer.Option(help="bge | hashing")] = None,
    reranker: Annotated[str | None, typer.Option(help="cross-encoder | none")] = None,
    llm: Annotated[str | None, typer.Option(help="fake | openai | ollama")] = None,
    sentiment: Annotated[str | None, typer.Option(help="auto | roberta | vader")] = None,
    queries: Annotated[int | None, typer.Option(help="limit number of queries")] = None,
    samples: Annotated[int | None, typer.Option(help="samples per query")] = None,
    report: Annotated[bool, typer.Option(help="write summary.md next to the results")] = True,
) -> None:
    """Keyless end-to-end run: baseline, sandbox arms, sweeps, bandit replay, greedy loop."""
    from vizor.experiment import run_experiment

    cfg = _config(
        config if config and config.exists() else None,
        embedder,
        reranker,
        llm,
        None,
        sentiment,
        queries,
        samples,
    )
    if cfg.llm.backend == "fake":
        _log(
            "LLM backend: FakeLLM (deterministic extractive stand-in). Results are a pipeline "
            "check, not evidence about real model behavior."
        )
    out = out or Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_demo"
    run_experiment(cfg, out, log=_log)
    if report:
        from vizor.metrics.report import write_summary

        write_summary(out)
    typer.echo(str(out))


@app.command()
def experiment(
    config: ConfigOpt = None,
    out: Annotated[Path, typer.Option(help="results directory")] = Path("runs/experiment"),
    max_cost_usd: Annotated[float | None, typer.Option(help="hard stop for API spend")] = None,
    workers: Annotated[int | None, typer.Option(help="parallel LLM calls")] = None,
    queries: Annotated[int | None, typer.Option()] = None,
    samples: Annotated[int | None, typer.Option()] = None,
    pilot: Annotated[
        bool,
        typer.Option(
            help="baseline and A/A re-sample only (no page arms, sweeps or bandit), to measure "
            "the noise and the MDE before the full run; its answers are reused from the cache"
        ),
    ] = False,
) -> None:
    """Full experiment from a config (see configs/openai.yaml and configs/ollama.yaml)."""
    from vizor.experiment import run_experiment
    from vizor.metrics.report import write_summary

    cfg = _config(
        config, queries=queries, samples=samples, max_cost_usd=max_cost_usd, workers=workers
    )
    planned = None
    if pilot:
        planned = list(cfg.sandbox.arms)
        cfg.sandbox.arms = ["noop", "aa_resample"]
        cfg.sandbox.position_sweep = []
        cfg.sandbox.boost_sweep = []
        cfg.sandbox.llm_rewrites = False
        if cfg.label:
            cfg.label += ", pilot: baseline and A/A only"
    run_experiment(cfg, out, log=_log, planned_arms=planned)
    write_summary(out)
    typer.echo(str(out))


@app.command()
def run(
    config: ConfigOpt = Path("configs/demo.yaml"),
    out: Annotated[Path | None, typer.Option()] = None,
    embedder: Annotated[str | None, typer.Option()] = None,
    reranker: Annotated[str | None, typer.Option()] = None,
    llm: Annotated[str | None, typer.Option()] = None,
    sentiment: Annotated[str | None, typer.Option()] = None,
    queries: Annotated[int | None, typer.Option()] = None,
    samples: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Baseline visibility only: who gets retrieved, cited and emphasized for each query."""
    from vizor.baseline import run_baseline

    cfg = _config(
        config if config and config.exists() else None,
        embedder,
        reranker,
        llm,
        None,
        sentiment,
        queries,
        samples,
    )
    out = out or Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_baseline"
    summary = run_baseline(cfg, out, log=_log)
    typer.echo(summary.round(3).to_string(index=False))
    typer.echo(str(out))


@app.command()
def inspect(
    run_dir: Path,
    query: Annotated[str | None, typer.Option(help="query_id")] = None,
    arm: str = "baseline",
    sample: int = 0,
) -> None:
    """Print one answer with per-sentence attribution."""
    from vizor.inspect import format_answer, load_answer

    ans, prompt = load_answer(run_dir, arm=arm, query_id=query, sample=sample)
    typer.echo(format_answer(ans, prompt))


@app.command()
def recompute(run_dir: Path) -> None:
    """Re-derive every metric from the committed raw responses (no LLM, no network)."""
    from vizor.inspect import recompute_rows

    ok, n = recompute_rows(run_dir)
    typer.echo(f"{n} rows recomputed; {'match' if ok else 'MISMATCH'}")
    raise typer.Exit(0 if ok else 1)


@app.command()
def report(
    results: Annotated[list[Path] | None, typer.Argument(help="results dirs")] = None,
    out: Path = Path("experiments/RESULTS.md"),
    readme: Path = Path("README.md"),
) -> None:
    """Regenerate RESULTS.md (and the README results block) from results directories."""
    from vizor.metrics.report import write_results

    dirs = results or sorted(p for p in Path("experiments/results").iterdir() if p.is_dir())
    write_results(dirs, out, readme if readme.exists() else None)
    typer.echo(str(out))


@app.command("sensitivity")
def sensitivity_cmd(
    run_dir: Path,
    metric: Annotated[str | None, typer.Option(help="imp_pwc, c_share or mentioned")] = None,
    family: Annotated[int | None, typer.Option(help="planned Holm family of page arms")] = None,
    content_family: Annotated[int | None, typer.Option(help="planned content-only family")] = None,
) -> None:
    """Minimum detectable effects from a run's A/A re-sample (JSON). With --family, for a
    planned design rather than the arms the run actually has."""
    from vizor.optimize.sensitivity import sensitivity

    typer.echo(json.dumps(sensitivity(run_dir, metric, family, content_family), indent=2))


@app.command()
def estimate(config: ConfigOpt = None) -> None:
    """Rough token and cost estimate for an experiment config, without calling any model."""
    from vizor.estimate import estimate_cost

    typer.echo(json.dumps(estimate_cost(_config(config)), indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the API."""
    import uvicorn

    uvicorn.run("vizor.api.app:app", host=host, port=port, log_level="info")


@models_app.command("fetch")
def models_fetch(write_lock: bool = False) -> None:
    """Download pinned models into .models/ and verify them against models.lock."""
    from vizor.models import fetch

    for k, v in fetch(write_lock=write_lock).items():
        typer.echo(f"{k}: {v}")


@models_app.command("verify")
def models_verify() -> None:
    from vizor.models import verify

    for k, v in verify().items():
        typer.echo(f"{k}: {v}")


@selfcheck_app.command("egress")
def selfcheck_egress(expect: str = "blocked") -> None:
    """Try external connections from this process. --expect blocked|open."""
    from vizor.egress import check_egress

    leaks = check_egress()
    if expect == "blocked":
        typer.echo("egress blocked for all targets" if not leaks else f"EGRESS OPEN: {leaks}")
        raise typer.Exit(1 if leaks else 0)
    typer.echo(f"reachable: {leaks}")
    raise typer.Exit(0 if leaks else 1)


@ingest_app.command("cc")
def ingest_cc(
    pattern: Annotated[str, typer.Option(help="URL pattern, e.g. '*.example.com/*'")],
    out: Annotated[Path, typer.Option(help="corpus JSONL to append to")],
    limit: int = 10,
    role: str = "competitor",
    index: str = "latest",
) -> None:
    """Common Crawl: CDX lookup -> WARC range fetch -> extract -> append to a corpus."""
    from vizor.ingest.commoncrawl import CommonCrawl
    from vizor.ingest.corpus import append_corpus

    cc = CommonCrawl(index=index)
    docs = cc.ingest(pattern, limit=limit, role=role)
    n = append_corpus(out, docs)
    typer.echo(f"{n} new documents from {cc.index_name} -> {out}")


@ingest_app.command("url")
def ingest_url(
    url: str,
    out: Annotated[Path, typer.Option()],
    role: str = "competitor",
) -> None:
    """Live fetch of one page (respects robots.txt)."""
    from vizor.ingest.corpus import append_corpus
    from vizor.ingest.live import fetch_page

    n = append_corpus(out, [fetch_page(url, role=role)])
    typer.echo(f"{n} new documents -> {out}")


@queries_app.command("generate")
def queries_generate(
    config: ConfigOpt = None,
    out: Annotated[Path, typer.Option()] = Path("queries.jsonl"),
    per_intent: int = 10,
) -> None:
    """Generate, dedupe (cos < 0.88), filter and cluster queries from the project's corpus."""
    from vizor.queries import generate_queries

    qs = generate_queries(_config(config), per_intent=per_intent)
    with open(out, "w") as fh:
        for q in qs:
            fh.write(
                json.dumps(
                    {
                        "query_id": q.query_id,
                        "query": q.text,
                        "intent": q.intent,
                        "cluster_id": q.cluster_id,
                    }
                )
                + "\n"
            )
    typer.echo(f"{len(qs)} queries -> {out}")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
