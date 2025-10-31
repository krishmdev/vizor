from vizor.attribution.citations import (
    count_words,
    normalize_markers,
    parse_answer,
    split_sentences,
    strip_markers,
)


def test_marker_after_period_moves_inside():
    assert normalize_markers("Good crema.[1] Next.") == "Good crema[1]. Next."
    assert normalize_markers("Good crema. [1][2] Next.") == "Good crema[1][2]. Next."


def test_comma_and_range_markers_expand():
    p = parse_answer("Both boilers heat fast [1, 2]. All three agree [1-3].", n_sources=5)
    assert [s.citations for s in p.sentences] == [(1, 2), (1, 2, 3)]


def test_hallucinated_index_is_dropped_and_recorded():
    p = parse_answer("It pulls a shot in 25 seconds [2][9].", n_sources=5)
    assert p.sentences[0].citations == (2,)
    assert p.hallucinated == [9]


def test_decimals_and_abbreviations_do_not_split():
    s = split_sentences("It draws 1.5 kW, e.g. more than the Aria. It costs $499.99 [1].")
    assert s == ["It draws 1.5 kW, e.g. more than the Aria.", "It costs $499.99 [1]."]


def test_bullets_are_separate_sentences():
    text = "Options:\n- The Duo has two boilers [1]\n- The Aria heats in 30s [2]\n\nIn short [3]."
    p = parse_answer(text, n_sources=3)
    assert [s.citations for s in p.sentences] == [(), (1,), (2,), (3,)]


def test_uncited_heading_counts_as_a_sentence():
    p = parse_answer("## Best picks\nThe Duo is quiet [1].", n_sources=2)
    assert len(p.sentences) == 2
    assert p.sentences[0].citations == ()
    assert p.sentences[1].pos == 1


def test_word_count_excludes_markers():
    assert count_words("The Duo is quiet [1][2].") == 4
    assert count_words("The Duo is quiet [1].", mode="geo_reference") == 3
    assert strip_markers("The Duo is quiet [1][2].") == "The Duo is quiet."


def test_urls_do_not_split():
    s = split_sentences("See brewline.example/duo for specs [1]. Done.")
    assert len(s) == 2


def test_no_and_min_split_unless_a_number_follows():
    assert len(split_sentences("The answer is no. The Duo is louder [1].")) == 2
    assert len(split_sentences("Model No. 5 is quiet [1].")) == 1
    assert len(split_sentences("Wait a min. Then pull the shot [2].")) == 2


def test_huge_ranges_do_not_expand():
    p = parse_answer("Everything agrees [1-5000000].", n_sources=5)
    assert p.sentences[0].citations == (1,)
    assert p.hallucinated == [5000000]
