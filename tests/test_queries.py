import numpy as np

from vizor.queries import _parse_json_list, cluster, dedup


def test_dedup_threshold(hashing):
    texts = [
        "best espresso machine for beginners",
        "best espresso machine for beginners",
        "how to descale a boiler",
    ]
    keep = dedup(texts, hashing.encode(texts, kind="query"))
    assert keep == [0, 2]


def test_parse_llm_json():
    txt = 'Sure:\n[{"query": "is the duo loud", "intent": "comparison"}, {"x": 1}]'
    assert _parse_json_list(txt) == [{"query": "is the duo loud", "intent": "comparison"}]
    assert _parse_json_list("no json") == []


def test_cluster_finds_two_groups():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (10, 8)) + np.eye(8)[0]
    b = rng.normal(0, 0.05, (10, 8)) + np.eye(8)[1]
    lab = cluster(np.vstack([a, b]))
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1 and lab[0] != lab[10]
