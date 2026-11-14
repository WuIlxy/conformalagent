"""Conformal layer.  Two ways to set a call's threshold:

1. ACI (online, README section 2): per-tool calibrated quantiles + one shared level.
2. Selector-aware batch thresholds (README section 4b): BatchGCP / BatchMVP of
   Jung, Noarov, Ramalingam & Roth, "Batch Multivalid Conformal Prediction" (2022),
   with the groups being tool families and selectors (selector.py).

Gibbs & Candes, "Adaptive Conformal Inference Under Distribution Shift" (2021):
    C_t(a_t)   = { y : S_t(X_t, y) <= Q_t(1 - a_t) }
    a_{t+1}    = a_t + gamma * (alpha* - err_t),   err_t = 1[Y_t not in C_t(a_t)]
a_t is NOT clamped (it stays in [-gamma, 1+gamma], Lemma 4.1).  a_t <= 0 gives
the full label space, a_t >= 1 the empty set.  The paper lets S_t and Q_t vary
with t; we use that to share one level a_t across tools while each tool family
k keeps its own calibration scores D_cal,k (README section 2).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# tool -> calibration family.  The three yes/no tools share one VLM prompt and
# score shape, so they pool scores (each alone is called too rarely to calibrate).
CAL_KEY = {"find": "find", "query": "query",
           "verify": "yesno", "exists": "yesno", "relate": "yesno"}


def quantile(scores: Sequence[float], p: float) -> float:
    """Split-conformal quantile with the (n+1) correction: Q(p) is the
    ceil((n+1)p)-th smallest score.  p <= 0 -> -inf (empty set),
    ceil((n+1)p) > n -> +inf (full set), matching ACI's conventions."""
    if p <= 0:
        return -math.inf
    s = sorted(scores)
    k = math.ceil((len(s) + 1) * p)
    return math.inf if k > len(s) else s[k - 1]


@dataclass
class ACI:
    cal: Dict[str, List[float]]     # cal family -> true-label scores
    target: float = 0.1             # alpha*
    gamma: float = 0.05
    alpha: float = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.alpha = self.target    # alpha_1 = alpha*, as in the paper

    def threshold(self, tool: str, groups=None) -> float:
        return quantile(self.cal.get(CAL_KEY[tool], []), 1 - self.alpha)

    def update(self, covered: bool):
        self.alpha += self.gamma * (self.target - (0.0 if covered else 1.0))


# ---------------------------------------------------------------- batch, group-aware
# rows: [(true-label score, groups of the call)], one row per labelled calibration call.
# A selector picks at most one call per trace, so its rows are one per question (iid).

@dataclass
class Fitted:
    """Static threshold  f(call) = sum of lam[g] over the call's groups, then (MVP)
    snapped to the grid k/m and patched in order.  Same interface as ACI, so
    tools.Env doesn't care which one it holds."""
    lam: Dict[str, float]
    target: float = 0.1
    m: int = 0                                         # 0: GCP (no grid, no patches)
    patches: List[Tuple[str, int, int]] = field(default_factory=list)   # (group, level k, shift dk)
    method: str = "gcp"

    @property
    def alpha(self) -> float:
        return self.target

    def reset(self):
        pass

    def update(self, covered: bool):
        pass

    def level(self, groups) -> int:
        if groups[0] not in self.lam:                  # no calibration data for this family
            return self.m
        k = _snap(sum(self.lam.get(g, 0.0) for g in groups), self.m)
        for g, v, dk in self.patches:
            if k == v and g in groups:
                k = min(max(k + dk, 0), self.m)
        return k

    def threshold(self, tool: str, groups=()) -> float:
        if not self.m:
            return sum(self.lam.get(g, 0.0) for g in groups) if groups[0] in self.lam else 1.0
        return self.level(groups) / self.m


def _snap(x: float, m: int) -> int:
    return min(max(round(x * m), 0), m)


def _groups(rows, min_count):
    from collections import Counter
    n = Counter(g for _, gs in rows for g in gs)
    return sorted(g for g, c in n.items() if c >= min_count or g.startswith("fam:"))


def fit_gcp(rows, target: float = 0.1, min_count: int = 20) -> Fitted:
    """BatchGCP (Alg. 1): minimise pinball loss over f = sum_g lam_g 1[g], an LP.
    Selectors with fewer than min_count rows get lam = 0; family intercepts are
    always fitted (groups[0] of every call is its family)."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, eye, hstack
    G = _groups(rows, min_count)
    col = {g: j for j, g in enumerate(G)}
    n, q = len(rows), 1 - target
    ij = [(i, col[g]) for i, (_, gs) in enumerate(rows) for g in gs if g in col]
    A = coo_matrix((np.ones(len(ij)), tuple(zip(*ij))), shape=(n, len(G)))
    # f_i + u_i - v_i = s_i,  u = (s - f)+,  v = (f - s)+
    res = linprog(np.r_[np.zeros(len(G)), np.full(n, q), np.full(n, 1 - q)],
                  A_eq=hstack([A, eye(n), -eye(n)]).tocsr(), b_eq=[s for s, _ in rows],
                  bounds=[(None, None)] * len(G) + [(0, None)] * (2 * n), method="highs")
    assert res.success, res.message
    return Fitted({g: float(res.x[col[g]]) for g in G}, target)


def fit_mvp(rows, target: float = 0.1, m: int = 20, tol: float = 0.002,
            min_count: int = 20, max_rounds: int = 1000) -> Fitted:
    """BatchMVP (Alg. 2), started from BatchGCP.  While some group's quantile
    calibration error  Q(f,g) = sum_v P(f=v|g) (q - P(s<=f | f=v, g))^2  exceeds
    tol * P(g), patch the worst (group, level) cell by the grid shift that brings
    its coverage closest to q.  Two deliberate changes from the paper, both for
    finite samples: the shift is the closest one with coverage >= q when there is
    one (not just closest), and a cell no shift improves is skipped, not a stop."""
    import numpy as np
    start = fit_gcp(rows, target, min_count)
    s = np.array([r[0] for r in rows])
    q = 1 - target
    mem = {g: np.array([g in gs for _, gs in rows]) for g in _groups(rows, min_count)}
    k = np.array([_snap(sum(start.lam.get(g, 0.0) for g in gs), m) for _, gs in rows])
    patches, stuck = [], set()
    for _ in range(max_rounds):
        worst, done = None, True
        for g, in_g in mem.items():
            Q = 0.0
            for v in np.unique(k[in_g]):
                B = in_g & (k == v)
                e = B.mean() * (q - (s[B] <= v / m).mean()) ** 2
                Q += e
                if (g, v) not in stuck and (worst is None or e > worst[0]):
                    worst = (e, g, int(v), B)
            done &= Q <= tol * in_g.mean()
        if done or worst is None:
            break
        _, g, v, B = worst
        cov = lambda dk: (s[B] <= min(max(v + dk, 0), m) / m).mean()
        dk = min(range(-m, m + 1), key=lambda d: (cov(d) < q, abs(cov(d) - q), abs(d)))
        if dk == 0:
            stuck.add((g, v))
            continue
        k[B] = min(max(v + dk, 0), m)
        patches.append((g, v, dk))
        stuck.clear()                    # the patch moved rows; re-check every cell
    return Fitted(start.lam, target, m, patches, "mvp")


if __name__ == "__main__":
    s = [i / 100 for i in range(100)]
    assert quantile(s, 0.9) <= quantile(s, 0.95)            # smaller alpha -> bigger set
    assert quantile(s, 0.0) == -math.inf and quantile(s, 1.0) == math.inf
    a = ACI({"find": s}, target=0.1, gamma=0.1)
    a.update(False); assert abs(a.alpha - 0.01) < 1e-9       # miss -> alpha down
    a.update(False); assert a.alpha < 0 and a.threshold("find") == math.inf  # no clamp -> full set
    a.update(True); assert a.alpha > -0.08                   # hit -> back up
    # batch: two selector groups whose true scores differ in scale; one shared
    # intercept would under-cover the hard group and over-cover the easy one
    import random
    rng = random.Random(0)
    rows = [(rng.random() * 0.5, ["fam:find", "easy"]) for _ in range(400)] + \
           [(rng.random(), ["fam:find", "hard"]) for _ in range(400)]
    cover = lambda f, g: sum(s <= f.threshold("find", gs) for s, gs in rows if g in gs) / 400
    for f in (fit_gcp(rows), fit_mvp(rows)):
        assert all(abs(cover(f, g) - 0.9) < 0.03 for g in ("easy", "hard")), (f.method, f)
    assert fit_gcp(rows, min_count=500).lam.keys() == {"fam:find"}        # small groups dropped
    assert fit_gcp(rows).threshold("query", ["fam:query"]) == 1.0         # unseen family: full set
    print("conformal ok")
