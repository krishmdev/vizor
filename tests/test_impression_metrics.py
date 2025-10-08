import math

import pytest

from vizor.metrics.impression import (
    citation_share,
    decay_weights,
    impressions,
    relative_improvement,
)
from vizor.types import CitedSentence


def _answer(words, cites):
    return [
        CitedSentence(i, "x", w, tuple(c))
        for i, (w, c) in enumerate(zip(words, cites, strict=True))
    ]


# Worked by hand: 4 sentences with 4, 6, 2, 4 words citing [1], [1][2], [2], [3].
# Paper decay d(i) = e^(-i/4): 1, 0.778801, 0.606531, 0.472367.
# raw_pwc(1) = 4*1 + 6*0.778801/2 = 6.336402
# raw_pwc(2) = 6*0.778801/2 + 2*0.606531 = 3.549464
# raw_pwc(3) = 4*0.472367 = 1.889466
VECTOR = _answer([4, 6, 2, 4], [[1], [1, 2], [2], [3]])


def test_word_count_share():
    imp = impressions(VECTOR, 3)
    assert imp.word == pytest.approx([0.4375, 0.3125, 0.25])


def test_pawc_paper_decay():
    imp = impressions(VECTOR, 3, decay="paper")
    assert imp.raw_pwc == pytest.approx([6.336402, 3.549464, 1.889466], abs=1e-6)
    assert imp.pwc == pytest.approx([0.538108, 0.301432, 0.160460], abs=1e-6)
    assert sum(imp.pwc) == pytest.approx(1.0)


def test_pawc_reference_decay():
    imp = impressions(VECTOR, 3, decay="reference")
    assert imp.pwc == pytest.approx([0.569537, 0.294181, 0.136283], abs=1e-6)


def test_position_share():
    imp = impressions(VECTOR, 3, decay="paper")
    assert imp.pos == pytest.approx([0.486196, 0.348508, 0.165296], abs=1e-6)


def test_citation_share_of_voice():
    assert citation_share(VECTOR, 3) == pytest.approx([0.4, 0.4, 0.2])


def test_relative_improvement_matches_paper_definition():
    assert relative_improvement(0.160460, 0.25) == pytest.approx(55.80, abs=0.01)
    assert relative_improvement(0.0, 0.3) is None


def test_uncited_response():
    ans = _answer([5, 7], [[], []])
    imp = impressions(ans, 3)
    assert imp.uncited
    assert imp.pwc == [0.0, 0.0, 0.0]
    assert impressions(ans, 3, no_citation="uniform").pwc == pytest.approx([1 / 3] * 3)


def test_uncited_sentences_still_count_toward_positions():
    with_filler = _answer([3, 4], [[], [1]])
    assert impressions(with_filler, 2).raw_pwc[0] == pytest.approx(4 * math.exp(-1 / 2))


def test_single_sentence_reference_decay_is_one():
    assert decay_weights(1, "reference") == [1.0]
    assert decay_weights(1, "paper") == [1.0]
