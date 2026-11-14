"""Selectors (README §4b): a selector phi(x, w) picks at most one tool call
in a trace, and the calibrated threshold of a call is the sum of the offsets
lambda_phi of the selectors that pick it.

A selector here is "the n-th call satisfying pred", where pred only reads
the call's own features and the trace before it.  That makes every selector
stable by construction: it returns nothing until its call appears, then
keeps that index for the rest of the trace.  Every call is also in its tool
family's group (the intercept), so calls no selector picks still get a
threshold.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List

FAMILIES = ("find", "query", "yesno")
QTYPES = ("verify", "choose", "logical", "compare", "query")   # GQA's structural types
YESNO_START = ("is ", "are ", "do ", "does ", "did ", "was ", "were ", "can ", "could ",
               "has ", "have ")


def qtype(question: str) -> str:
    """Question type from the text alone, so it also works off GQA (99.5% agreement
    with GQA's structural types)."""
    q = " " + question.strip().lower().rstrip("?") + " "
    yesno = q[1:].startswith(YESNO_START)
    if any(w in q for w in (" same ", " different ", " than ", "what is common", "what do both")):
        return "compare"
    exist_q = q[1:].startswith(("is there", "are there", "do you see", "does it look", "do you think"))
    if yesno and (" and " in q or " both " in q or " either " in q or (" or " in q and exist_q)):
        return "logical"
    if " or " in q:
        return "choose"
    return "verify" if yesno else "query"


@dataclass(frozen=True)
class Selector:
    name: str
    pred: Callable[[dict], bool]
    n: int = 1


def _sel(name, pred, n=1):
    return Selector(name, pred, n)


def default_selectors() -> List[Selector]:
    S = []
    for fam in FAMILIES:
        S += [_sel(f"{fam}#{n}", lambda f, fam=fam: f["fam"] == fam, n) for n in (1, 2, 3)]
        S += [_sel(f"{fam}|{qt}#1", lambda f, fam=fam, qt=qt: f["fam"] == fam and f["qtype"] == qt)
              for qt in QTYPES]
        S.append(_sel(f"{fam}|image#1", lambda f, fam=fam: f["fam"] == fam and f["region"] == "image"))
        # loop-state drift: the first call after a crowded find (raw detector count, so it
        # doesn't depend on the thresholds themselves)
        S.append(_sel(f"{fam}|after_crowded#1", lambda f, fam=fam: f["fam"] == fam and f["crowded"]))
    for cat in ("color", "material", "name", "other"):
        S.append(_sel(f"query|{cat}#1", lambda f, cat=cat: f["fam"] == "query" and f["attr"] == cat))
    return S


class Selection:
    """Per-trace state: counts pred hits so each selector picks its n-th call."""
    def __init__(self, selectors: List[Selector]):
        self.selectors, self.hits = selectors, Counter()

    def step(self, feats: dict) -> List[str]:
        """Groups of this call: its family + every selector that picks it."""
        out = [f"fam:{feats['fam']}"]
        for s in self.selectors:
            if s.pred(feats):
                self.hits[s.name] += 1
                if self.hits[s.name] == s.n:
                    out.append(s.name)
        return out


if __name__ == "__main__":
    assert qtype("Is the cup white?") == "verify"
    assert qtype("Is the cup red or blue?") == "choose"
    assert qtype("Are there both cups and plates?") == "logical"
    assert qtype("Is there a cup or a plate?") == "logical"
    assert qtype("Are the cup and the plate the same color?") == "compare"
    assert qtype("What color is the cup?") == "query"
    sel = Selection(default_selectors())
    f = dict(fam="find", qtype="query", region="image", crowded=False, attr=None)
    g1 = sel.step(f); g2 = sel.step(f); g3 = sel.step(dict(f, crowded=True))
    assert "find#1" in g1 and "find|query#1" in g1 and "find#1" not in g2       # stable: once
    assert "find#2" in g2 and "find#3" in g3 and "find|after_crowded#1" in g3
    assert sel.step(f) == ["fam:find"]                                          # 4th: intercept only
    print("selector ok")
