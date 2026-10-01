"""The tool set.  Each perception tool returns a prediction *set*, not a point.

The set-construction mirrors the paper's conformal backend
`quasar-colm2026-artifact/epic/imgpatch_conformal.py` almost line-for-line --
    * find        <- imgpatch_conformal.find (certain / uncertain by score band)
    * query_attr  <- imgpatch_conformal.simple_query (VLM enumerates answers+conf)
    * verify      <- imgpatch_conformal.verify_property (3-valued by conf bands)
The only change is *where the band edge comes from*: instead of a global tau
(GQA/conformal.py), the edge is the online conformal quantile from conformal.py.

Raw model access is injected via a `Backends` object so this file stays free of
torch / OpenAI.  The real backend wires to imgpatch_4o_all / imgpatch_local /
imgpatch_conformal; the demo injects a deterministic mock.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List, Tuple

from setsem import (AbstractOther, AbstractTuple, lift, join_many,
                    ABS_BOOL_TRUE, ABS_BOOL_FALSE, ABS_BOOL_TOP)


# --------------------------------------------------------------------------- #
# what a raw backend must provide (real one refs imgpatch_*; demo mocks it)
# --------------------------------------------------------------------------- #
Region = Tuple[int, int, int, int]        # bbox (x0,y0,x1,y1); whole image = full frame


class Backends(Protocol):
    def detect(self, region: Region, phrase: str) -> List[Tuple[Region, float]]:
        """Open-vocab detector (OWL-ViT / GroundingDINO): boxes + confidences.
        cf. imgpatch_local.object_detector / imgpatch_conformal.find."""
    def attr_scores(self, region: Region, attribute_class: str) -> List[Tuple[str, float]]:
        """VLM: candidate values + confidences (should ~sum to 1).
        cf. imgpatch_conformal._do_model_call (the JSON-confidence prompt)."""
    def yesno_score(self, region: Region, predicate: str) -> float:
        """VLM P(yes) for a predicate.  cf. imgpatch_conformal.verify_property."""


# --------------------------------------------------------------------------- #
# the tools
# --------------------------------------------------------------------------- #
@dataclass
class Tools:
    backends: Backends
    conformal: "TraceConformal"           # from conformal.py; supplies thresholds + logging

    # -- find(phrase) -> set of regions -------------------------------------- #
    def find(self, region: Region, phrase: str) -> AbstractTuple:
        """Score band from the conformal quantile.  A box is *certain* if its
        nonconformity (= 1 - detector score) is comfortably inside the set, and
        *uncertain* if it only just makes the cut -- the two-band split of
        imgpatch_conformal.find, but with a calibrated edge."""
        thr = self.conformal.threshold_for("find")          # nonconformity cutoff
        margin = thr * 0.5                                   # inner band = "certain"
        items = []
        for box, score in self.backends.detect(region, phrase):
            nonconf = 1.0 - score
            if nonconf <= margin:
                items.append((box, True))                    # certainly there
            elif nonconf <= thr:
                items.append((box, False))                   # maybe there
            # else: below the set -> dropped
        out = AbstractTuple(tuple(items), alpha=self.conformal.alpha_for("find"))
        self.conformal.log("find", f"{phrase}@{region}", out)
        return out

    # -- query_attribute(region, class) -> set of values --------------------- #
    def query_attribute(self, region: Region, attribute_class: str) -> AbstractOther:
        thr = self.conformal.threshold_for("query_attribute")
        vals = [v for v, s in self.backends.attr_scores(region, attribute_class)
                if (1.0 - s) <= thr]
        if not vals:                                         # never return empty
            vals = [v for v, _ in self.backends.attr_scores(region, attribute_class)[:1]]
        out = AbstractOther(frozenset(vals), alpha=self.conformal.alpha_for("query_attribute"))
        self.conformal.log("query_attribute", f"{attribute_class}@{region}", out)
        return out

    # -- verify(region, predicate) -> {yes}|{no}|{yes,no} -------------------- #
    def verify(self, region: Region, predicate: str) -> AbstractOther:
        thr = self.conformal.threshold_for("verify")
        p_yes = self.backends.yesno_score(region, predicate)
        # inside the uncertain band from either side -> TOP
        if (1.0 - p_yes) <= thr and p_yes <= thr:
            out = ABS_BOOL_TOP
        elif p_yes >= 0.5:
            out = ABS_BOOL_TRUE
        else:
            out = ABS_BOOL_FALSE
        out = AbstractOther(out._possibilities, alpha=self.conformal.alpha_for("verify"))
        self.conformal.log("verify", f"{predicate}@{region}", out)
        return out

    # -- spatial(a, b) -> set of relations (deterministic geometry) ---------- #
    def spatial(self, a: Region, b: Region) -> AbstractOther:
        """Deterministic from bboxes -> a *set* because several relations can
        hold at once (e.g. {left_of, above}).  Set-valued but alpha=0."""
        ax = (a[0] + a[2]) / 2; ay = (a[1] + a[3]) / 2
        bx = (b[0] + b[2]) / 2; by = (b[1] + b[3]) / 2
        rels = set()
        rels.add("left_of" if ax < bx else "right_of")
        rels.add("above" if ay < by else "below")
        out = AbstractOther(frozenset(rels), alpha=0.0)
        self.conformal.log("spatial", f"{a}|{b}", out)
        return out

    # -- count(region_set) -> integer (set-valued if input uncertain) -------- #
    def count(self, regions: AbstractTuple) -> AbstractOther:
        """Deterministic count, but AbstractTuple.card() yields the [lb, ub]
        integer set when membership is uncertain."""
        out = regions.card()
        self.conformal.log("count", "region_set", out)
        return out
