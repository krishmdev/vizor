"""The parser against real qwen2.5:3b answers, hand-checked.

`tests/fixtures/qwen_citations.jsonl` holds 31 distinct answers from the committed 2026-08-05
qwen run, stratified by how they cite: [Source [n]] headers, [Source n] as a sentence subject,
[n, m] lists, answers that collect every marker on the last sentence, uncited answers, and plain
[n] answers. `expected` is the citation list per sentence as a reader would assign the literal
markers (no inference about uncited sentences), checked by reading each answer.
"""

import json
from pathlib import Path

import pytest

from vizor.attribution.citations import parse_answer

FIXTURE = Path(__file__).parent / "fixtures" / "qwen_citations.jsonl"
CASES = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("case", CASES, ids=[f"{c['category']}-{i}" for i, c in enumerate(CASES)])
def test_real_qwen_answer(case):
    p = parse_answer(case["text"], case["n_sources"])
    assert [list(s.citations) for s in p.sentences] == case["expected"]
    assert not p.unparsed and not p.hallucinated


def test_fixture_covers_every_category():
    cats = {c["category"] for c in CASES}
    assert cats == {"nested", "source", "list", "uncited", "last", "plain"}
