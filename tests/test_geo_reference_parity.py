"""Parity with the GEO authors' released impression functions (vendored under tests/reference)."""

import random

import pytest

from tests.reference import geo_reference as ref
from vizor.metrics.impression import impressions
from vizor.types import CitedSentence


def _random_answer(rng: random.Random, n_sources: int):
    paras = []
    for _ in range(rng.randint(1, 4)):
        para = []
        for _ in range(rng.randint(1, 5)):
            tokens = [
                rng.choice(["a", "of", "the", "machine", "boiler", "grinder", "x9"])
                for _ in range(rng.randint(0, 25))
            ]
            k = rng.choice([0, 0, 1, 1, 1, 2, 3])
            cites = [rng.randint(1, n_sources) for _ in range(k)]
            para.append((tokens, " ".join(tokens), cites))
        paras.append(para)
    return paras


def _ours(paras):
    flat = [s for p in paras for s in p]
    return [
        CitedSentence(i, text, ref.get_num_words(tokens), tuple(cites))
        for i, (tokens, text, cites) in enumerate(flat)
    ]


@pytest.mark.parametrize("seed", range(200))
def test_matches_reference_code(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 6)
    paras = _random_answer(rng, n)
    ours = impressions(_ours(paras), n, decay="reference", no_citation="uniform")
    assert ours.pwc == pytest.approx(ref.impression_wordpos_count_simple(paras, n=n))
    assert ours.word == pytest.approx(ref.impression_word_count_simple(paras, n=n))
    assert ours.pos == pytest.approx(ref.impression_pos_count_simple(paras, n=n))


def test_unnormalized_scores_match():
    paras = _random_answer(random.Random(7), 5)
    ours = impressions(_ours(paras), 5, decay="reference")
    assert ours.raw_pwc == pytest.approx(
        ref.impression_wordpos_count_simple(paras, n=5, normalize=False)
    )
