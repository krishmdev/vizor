from vizor.attribution.citations import parse_answer


def test_fake_llm_is_deterministic_and_cites_valid_sources(engine, queries):
    for q in queries[:10]:
        a = engine.answer(q, 0).answer
        b = engine.answer(q, 0).answer
        assert a.text == b.text
        assert a.sentences and not a.hallucinated_citations
        assert all(s.citations for s in a.sentences)
        assert all(1 <= c <= len(a.sources) for s in a.sentences for c in s.citations)


def test_samples_differ(engine, queries):
    texts = {engine.answer(queries[3], k).answer.text for k in range(5)}
    assert len(texts) > 1


def test_answer_parses_back_to_same_sentences(engine, queries):
    a = engine.answer(queries[5], 1).answer
    assert parse_answer(a.text, 5).sentences == a.sentences
