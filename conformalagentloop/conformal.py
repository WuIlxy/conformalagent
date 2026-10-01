"""The online conformal layer that sits between a raw model and the agent.

Formalism (from the project note).  A trace is a sequence of calls
    {(x_t, f_t, C_hat_t, y_t)}_{t=1..T}
where x_t is the input, f_t the conformal predictor called, C_hat_t=f_t(x_t)
the predicted *set*, and y_t the ground truth.  The hard part: x_t is produced
by the agent's own prior actions, so it is NOT exchangeable with the offline
calibration data for f_t -- the agent induces covariate shift on its own tools.

Two guarantees, two mechanisms:

  Trace-level  -- coverage along a single trace.  Adaptive Conformal Inference
                  (Gibbs & Candes 2021): after each call, nudge alpha by the
                  realised miscoverage, alpha_{t+1} = alpha_t + gamma*(target - err_t).
                  Robust to the shift because it reacts to observed error
                  instead of assuming exchangeability.

  Task-level   -- 95% of traces achieve 95% coverage.  Run ACI *within* each
                  trace, but choose its initial alpha_1 (equivalently the
                  starting quantile) by ordinary split conformal over a held-out
                  set of whole traces.  The outer split-conformal picks the init;
                  the inner ACI adapts from there.

This module is deliberately thin and honest about what is stubbed -- see the
`# OPEN:` notes and README section 5.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple
import math


# --------------------------------------------------------------------------- #
# split conformal: turn nonconformity scores into a threshold (the "init")
# --------------------------------------------------------------------------- #
def split_conformal_quantile(cal_scores: Sequence[float], alpha: float) -> float:
    """The conformal quantile q_hat: the smallest threshold s.t. a fresh score
    <= q_hat with prob >= 1-alpha, with the finite-sample (n+1) correction.

    A prediction set is then {label : nonconformity(label) <= q_hat}.  This is
    exactly the disciplined replacement for imgpatch_conformal's hand-set global
    tau: instead of scaling fixed bands, the band edge IS this quantile.
    """
    scores = sorted(cal_scores)
    n = len(scores)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:                      # alpha too small for this n -> take everything
        return math.inf
    return scores[k - 1]


# --------------------------------------------------------------------------- #
# adaptive conformal inference (trace-level)
# --------------------------------------------------------------------------- #
@dataclass
class ACIController:
    """One controller per tool f.  Holds the calibration scores (so alpha can be
    mapped to a concrete quantile) and the online alpha_t."""
    cal_scores: List[float]
    target_alpha: float                 # the coverage we want, 1-target = coverage
    gamma: float = 0.05                 # ACI learning rate
    alpha: float = None                 # current alpha_t (init below)

    def __post_init__(self):
        if self.alpha is None:
            # task-level init: start at the split-conformal operating point
            self.alpha = self.target_alpha

    def threshold(self) -> float:
        """Current nonconformity cutoff, from alpha_t."""
        a = min(max(self.alpha, 0.0), 1.0)
        return split_conformal_quantile(self.cal_scores, a)

    def update(self, covered: bool) -> None:
        """ACI step.  err_t = 1 if the true label fell OUTSIDE the set.
        Requires a feedback signal for y_t -- see OPEN note."""
        err = 0.0 if covered else 1.0
        self.alpha = self.alpha + self.gamma * (self.target_alpha - err)
        # clamp so the quantile stays well-defined
        self.alpha = min(max(self.alpha, 0.0), 1.0)
        # OPEN: at deployment y_t for intermediate tool calls is unknown.  GQA's
        # scene-graph annotations DO give per-step y_t at calibration time
        # (object presence, attribute, relation), so ACI can be *fit* offline
        # on labelled traces; live traces then run with frozen or proxy-updated
        # alpha.  Closing this honestly is the main research question.


# --------------------------------------------------------------------------- #
# per-trace bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class Call:
    t: int
    tool: str
    x_repr: str
    pred_set: object           # the AbstractOther / AbstractTuple returned
    alpha: float               # alpha_t in force when the set was produced


@dataclass
class TraceConformal:
    """Owns one ACIController per tool and logs the {(x_t,f_t,C_hat_t,y_t)} trace."""
    controllers: dict                       # tool_name -> ACIController
    calls: List[Call] = field(default_factory=list)

    def alpha_for(self, tool: str) -> float:
        c = self.controllers.get(tool)
        return c.alpha if c else 0.0

    def threshold_for(self, tool: str) -> float:
        c = self.controllers.get(tool)
        return c.threshold() if c else math.inf

    def log(self, tool: str, x_repr: str, pred_set) -> None:
        self.calls.append(Call(len(self.calls) + 1, tool, x_repr,
                               pred_set, self.alpha_for(tool)))

    def feedback(self, tool: str, covered: bool) -> None:
        """Push a realised (covered?) signal to that tool's ACI controller."""
        c = self.controllers.get(tool)
        if c:
            c.update(covered)

    def answer_budget(self, tools_used: Sequence[str]) -> float:
        """Union bound: the answer set built from these tool outputs covers the
        truth with prob >= 1 - sum(alpha_used).  Loose but sound; see README."""
        return sum(self.alpha_for(tool) for tool in set(tools_used))


def make_trace_conformal(cal_by_tool: dict, target_alpha: float, gamma: float = 0.05):
    """cal_by_tool: tool_name -> list of calibration nonconformity scores."""
    controllers = {
        name: ACIController(cal_scores=list(scores), target_alpha=target_alpha, gamma=gamma)
        for name, scores in cal_by_tool.items()
    }
    return TraceConformal(controllers=controllers)


if __name__ == "__main__":
    # runnable check: quantile monotonicity + ACI reacts to misses
    scores = [i / 100 for i in range(100)]           # 0.00 .. 0.99
    q_loose = split_conformal_quantile(scores, alpha=0.30)
    q_tight = split_conformal_quantile(scores, alpha=0.05)
    assert q_tight >= q_loose, (q_loose, q_tight)     # smaller alpha -> bigger set

    aci = ACIController(cal_scores=scores, target_alpha=0.1, gamma=0.1)
    a0 = aci.alpha
    aci.update(covered=False)                          # a miss -> tighten (alpha down)
    assert aci.alpha < a0, (a0, aci.alpha)
    aci.update(covered=True)                           # a hit -> relax (alpha up)
    print("conformal self-check ok: q(0.3)=%.2f q(0.05)=%.2f alpha now=%.3f"
          % (q_loose, q_tight, aci.alpha))
