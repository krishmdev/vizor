"""Per-query retrieval log: where the page an arm edited ended up, and what the model saw of it.

One row per (arm, query), from the arm's source selection (the same for every sample of a
query) and its rendered prompt:
- `focus_doc`: the target page scored for this query (the page the arm edited);
- `focus_rank`: its rank among the reranked candidates (1-15; empty if it is not a candidate);
- `focus_slot`: its source slot in the prompt (1-5; empty if not shown) and `focus_retrieved`;
- `target_retrieved`: whether any target-site page is among the sources;
- `best_passage_kind` and `best_passage_new`: the passage that ranked the page (MaxP) and
  whether that passage is not on the original page (edited or new);
- `n_shown`, `n_shown_new` and `shown_new`: how many of the page's rendered passages there are,
  how many are not on the original page, and whether at least one is;
- `title_changed`, `description_changed`: whether the rendered title or description differ from
  the original page's (the fields `retrieval_meta` edits).

A passage counts as original if its text equals one of the original page's passages (the last
rendered passage may be cut by the prompt's word cap, so a cut line counts as original if an
original passage starts with it).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from vizor.generate.prompt import parse_prompt
from vizor.retrieve.chunk import passages
from vizor.types import SourceDoc


def _original_texts(doc: SourceDoc) -> set[str]:
    out = {p.text for p in passages(doc)}
    if doc.links:
        out.add("Related: " + "; ".join(f"{a} ({h})" for a, h in doc.links))
    return out


def _is_original(line: str, texts: set[str]) -> bool:
    if line in texts:
        return True
    if line.endswith(" …"):
        head = line[: -len(" …")].rstrip()
        return any(t.startswith(head) for t in texts)
    return False


def retrieval_rows(
    arm: str,
    results: Iterable,
    focus: Mapping[str, str],
    original: Mapping[str, SourceDoc],
) -> list[dict]:
    """`results`: the arm's EngineResults; `focus`: query_id -> the page edited (or scored)
    for it; `original`: doc_id -> the unedited page."""
    rows: list[dict] = []
    seen: set[str] = set()
    cache: dict[str, set[str]] = {}
    for r in results:
        qid = r.answer.query_id
        if qid in seen:
            continue
        seen.add(qid)
        doc_id = focus[qid]
        base = original[doc_id]
        texts = cache.setdefault(doc_id, _original_texts(base))
        sel = r.selection
        cand = next((i for i, c in enumerate(sel.candidates) if c.doc.doc_id == doc_id), None)
        slot = next((i for i, c in enumerate(sel.sources) if c.doc.doc_id == doc_id), None)
        row = {
            "arm": arm,
            "query_id": qid,
            "focus_doc": doc_id,
            "focus_rank": None if cand is None else cand + 1,
            "focus_slot": None if slot is None else slot + 1,
            "focus_retrieved": slot is not None,
            "target_retrieved": any(c.doc.role == "target" for c in sel.sources),
            "best_passage_kind": None,
            "best_passage_new": None,
            "n_shown": 0,
            "n_shown_new": 0,
            "shown_new": False,
            "title_changed": None,
            "description_changed": None,
        }
        if cand is not None:
            best = sel.candidates[cand].best_passage
            row["best_passage_kind"] = best.kind
            row["best_passage_new"] = not _is_original(best.text, texts)
        if slot is not None:
            _, rendered = parse_prompt(r.prompt)
            src = next(s for s in rendered if s.index == slot + 1)
            lines = [x for x in src.content.split("\n") if x.strip()]
            new = [x for x in lines if not _is_original(x, texts)]
            row.update(
                n_shown=len(lines),
                n_shown_new=len(new),
                shown_new=bool(new),
                title_changed=src.title != base.title,
                description_changed=src.description != base.meta_description,
            )
        rows.append(row)
    return rows


def retrieval_frame(runs: Iterable, focus_of, original: Mapping[str, SourceDoc]) -> pd.DataFrame:
    """Rows for several ArmRuns; `focus_of(run)` gives that run's query_id -> page map."""
    rows: list[dict] = []
    for run in runs:
        rows += retrieval_rows(run.arm.name, run.results, focus_of(run), original)
    return pd.DataFrame(rows)


def retrieval_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm: the focus page's retrieval rate, mean candidate rank (candidates only) and mean
    slot (shown only), and the passage-selection share: how often an edited or new passage is
    among the passages shown, over all queries and over queries where the page is shown."""
    g = df.groupby("arm", sort=False)
    out = pd.DataFrame(
        {
            "n_queries": g.size(),
            "focus_retrieved": g["focus_retrieved"].mean(),
            "focus_rank_mean": g["focus_rank"].mean(),
            "focus_slot_mean": g["focus_slot"].mean(),
            "target_retrieved": g["target_retrieved"].mean(),
            "shown_new_all": g["shown_new"].mean(),
            "shown_new_if_shown": g.apply(
                lambda x: x.loc[x["focus_retrieved"], "shown_new"].mean(), include_groups=False
            ),
            "best_passage_new": g["best_passage_new"].apply(
                lambda s: s.dropna().astype(float).mean()
            ),
        }
    )
    return out.reset_index()
