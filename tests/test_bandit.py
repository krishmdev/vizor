import numpy as np

from vizor.optimize.bandit import LinTS, LinUCB, replay, sherman_morrison


def test_sherman_morrison_matches_inverse():
    rng = np.random.default_rng(0)
    a = np.eye(6) * 2.0
    a_inv = np.linalg.inv(a)
    for _ in range(50):
        x = rng.normal(size=6)
        a += np.outer(x, x)
        a_inv = sherman_morrison(a_inv, x)
    assert np.allclose(a_inv, np.linalg.inv(a), atol=1e-8)


def test_lints_posterior_mean_is_ridge_solution():
    rng = np.random.default_rng(1)
    pol = LinTS(["a"], 4, v=0.1, lam=1.0)
    xs, rs = rng.normal(size=(30, 4)), rng.normal(size=30)
    for x, r in zip(xs, rs, strict=True):
        pol.update(x, "a", r)
    ridge = np.linalg.solve(np.eye(4) + xs.T @ xs, xs.T @ rs)
    assert np.allclose(pol.mean("a"), ridge)


def _synthetic(n_ctx=40, n_arms=4, d=5, samples=3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.c_[np.ones(n_ctx), rng.random((n_ctx, d - 1))]
    theta = rng.normal(size=(n_arms, d))
    mu = x @ theta.T
    r = mu[:, :, None] + rng.normal(scale=0.1, size=(n_ctx, n_arms, samples))
    return r, x


def test_linucb_beats_random_on_linear_rewards():
    r, x = _synthetic()
    _, summary = replay(r, x, ["a", "b", "c", "d"], rounds=400, runs=5)
    s = summary.set_index("policy")["final_regret"]
    assert s["linucb(a=0.5)"] < 0.5 * s["random"]
    assert s["lints(v=0.1)"] < 0.5 * s["random"]


def test_linucb_save_load(tmp_path):
    pol = LinUCB(["a", "b"], 3, alpha=0.5)
    pol.update(np.array([1.0, 0.5, 0.2]), "a", 0.3)
    pol.save(tmp_path / "p.json")
    back = LinUCB.load(tmp_path / "p.json")
    x = np.array([1.0, 0.1, 0.9])
    assert back.scores(x) == pol.scores(x)
