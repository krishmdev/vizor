from vizor.generate.prompt import MAX_SOURCE_WORDS, RenderedSource, format_prompt, parse_prompt


def test_prompt_round_trip():
    srcs = [
        RenderedSource(
            1,
            "Brewline Duo",
            "https://brewline.example/duo",
            "Dual boiler.",
            "",
            "Line one.\nLine two.",
        ),
        RenderedSource(
            2,
            "Aria",
            "https://crema-lab.example/aria",
            "",
            "name: Aria; price: 499",
            "Content [3] here.",
        ),
    ]
    q, parsed = parse_prompt(format_prompt("is the duo quiet?", srcs))
    assert q == "is the duo quiet?"
    assert parsed == srcs


def test_engine_prompt_is_source_tagged_and_capped(engine, queries):
    prompt, sel = engine.build_prompt(queries[0])
    q, srcs = parse_prompt(prompt)
    assert q == queries[0].text
    assert [s.index for s in srcs] == [1, 2, 3, 4, 5]
    assert [s.url for s in srcs] == [c.doc.url for c in sel.sources]
    assert all(len(s.content.split()) <= MAX_SOURCE_WORDS + 1 for s in srcs)
    assert "[1][2][3]" in prompt


class _P:
    def __init__(self, kind, order):
        self.kind, self.order = kind, order


def _scored():
    # FAQ passages score highest, the way an appended FAQ can crowd out body passages
    return [
        (_P("head", 0), 0.99),
        (_P("body", 1), 0.5),
        (_P("body", 2), 0.7),
        (_P("body", 3), 0.4),
        (_P("body", 4), 0.3),
        (_P("faq", 5), 0.9),
        (_P("faq", 6), 0.8),
    ]


def test_passage_policies():
    from vizor.generate.prompt import choose_passages

    def orders(policy):
        return [p.order for p, _ in choose_passages(_scored(), policy)]

    assert orders("query-top3") == [2, 5, 6]
    assert orders("body-top3") == [1, 2, 3]
    assert orders("top2+faq1") == [1, 2, 5]
    no_faq = [x for x in _scored() if x[0].kind != "faq"]
    assert [p.order for p, _ in choose_passages(no_faq, "top2+faq1")] == [1, 2, 3]


def test_engine_passage_policy_keeps_sources(engine, queries):
    from dataclasses import replace

    q = queries[0]
    _, sel = engine.build_prompt(q)
    doc = sel.sources[0].doc
    faq = tuple((f"Question {i} about {q.text}?", q.text) for i in range(3))
    edited = engine.cascade.with_docs({doc.doc_id: replace(doc, faq=faq)})
    eng = engine.content_only(edited)
    p_query, s1 = eng.build_prompt(q)
    p_body, s2 = eng.with_passage_policy("body-top3").build_prompt(q)
    assert [c.doc.doc_id for c in s1.sources] == [c.doc.doc_id for c in s2.sources]
    assert "Q: Question" in p_query.split("### Source [2]")[0]
    assert "Q: Question" not in p_body
