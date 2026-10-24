"""Monte Carlo check of interference-to-signal ratios for each memory store.

Writes data/curves.json, which the plot scenes read.
"""
import json
import math
from pathlib import Path

import numpy as np

D = 64
rng = np.random.default_rng(0)


def unit(n, d=D):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ratio_score(M, score, n_queries=16, chunk=1 << 18):
    """RMS of ||sum_{i!=j} v_i s(k_i.k_j)|| / ||v_j s(1)|| over a few queries."""
    q = unit(n_queries)  # queries are the stored keys j
    acc = np.zeros((n_queries, D), dtype=np.float64)
    left = M - 1
    while left > 0:
        n = min(chunk, left)
        k, v = unit(n), unit(n)
        acc += (score(q @ k.T).astype(np.float64)) @ v
        left -= n
    return float(np.sqrt(np.mean(np.sum(acc**2, 1))) / score(np.float32(1.0)))


def ratio_optimal(M, trials=4):
    out = []
    for _ in range(trials):
        K = unit(M)
        if M <= D:
            return 0.0
        P = K @ np.linalg.solve(K.T @ K, K.T)
        d = np.diag(P)
        off = np.sum(P**2, 1) - d**2
        out.append(np.mean(off / d**2))
    return float(np.sqrt(np.mean(out)))


relu = lambda s: np.maximum(s, 0)
SCORES = {
    "linear": lambda s: s,
    "relu": relu,
    "relu2": lambda s: relu(s) ** 2,
    "swiglu": lambda s: s * s / (1 + np.exp(-s)),
    "relu3": lambda s: relu(s) ** 3,
    "softmax": lambda s: np.exp(8.0 * (s - 1)),  # beta = 8, shifted for stability
}

# analytic predictions (random unit keys, s ~ N(0, 1/d))
def dfact(n):
    return math.prod(range(n, 0, -2)) if n > 0 else 1

ANALYTIC = {
    "linear": lambda M: math.sqrt(M / D),
    "optimal": lambda M: math.sqrt(max(M / D - 1, 0)),
    "relu": lambda M: math.sqrt(M / (2 * D)),
    "relu2": lambda M: math.sqrt(M * dfact(3) / (2 * D**2)),
    "relu3": lambda M: math.sqrt(M * dfact(5) / (2 * D**3)),
    "softmax": lambda M: math.sqrt(M) * math.exp(64 / D - 8),
}

def softmax_beta(M, beta):
    return math.sqrt(M) * math.exp(beta**2 / D - beta)

L = 1024
BETA_RULES = {
    "fixed": lambda M: 8.0,
    "logn": lambda M: 8.0 * max(1.0, math.log(M) / math.log(L)),
    "yarn": lambda M: 8.0 * (0.1 * math.log(max(M / L, 1)) + 1) ** 2,
}

if __name__ == "__main__":
    grid = np.unique(np.round(np.logspace(0.5, 6.4, 40)).astype(int))
    out = {"d": D, "grid": grid.tolist(), "analytic": {}, "mc": {}}
    for name, f in ANALYTIC.items():
        out["analytic"][name] = [f(int(M)) for M in grid]
    out["long_context"] = {
        r: [softmax_beta(int(M), b(int(M))) for M in grid] for r, b in BETA_RULES.items()
    }
    # truncated exp series for scene 6
    def trunc(n):
        return lambda s: sum((8 * s) ** p / math.factorial(p) for p in range(n))
    out["mc_trunc"] = {}

    checks = [16, 64, 128, 512, 2731, 8192, 34953, 131072, 1_200_000]
    for name, f in SCORES.items():
        out["mc"][name] = {}
        for M in checks:
            if name in ("linear", "relu") and M > 8192:
                continue
            if name in ("relu2", "swiglu") and M > 131072:
                continue
            r = ratio_score(M, f)
            out["mc"][name][M] = r
            print(f"{name:8s} M={M:>8d}  ratio={r:.3f}")
    out["mc"]["optimal"] = {}
    for M in [32, 64, 96, 128, 256, 512]:
        r = ratio_optimal(M)
        out["mc"]["optimal"][M] = r
        print(f"optimal  M={M:>8d}  ratio={r:.3f}")

    # capacity of truncated exp series with n terms: find M where MC ratio crosses 1
    for n in (2, 3, 4):
        f = trunc(n)
        lo, hi = 2, 200000
        while hi / lo > 1.05:
            mid = int(math.sqrt(lo * hi))
            if ratio_score(mid, f, n_queries=32) < 1:
                lo = mid
            else:
                hi = mid
        out["mc_trunc"][n] = lo
        print(f"exp series {n} terms: capacity ~ {lo}")

    Path(__file__).with_name("data").joinpath("curves.json").write_text(json.dumps(out))
