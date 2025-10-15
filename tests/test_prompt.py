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
