"""The five perception tools.  Each call: its groups (selector.py) -> raw
scores -> set at the threshold -> program line -> scene-graph label -> update
(README §2-4b).  The threshold comes from `cp`: an ACI (Q_k(1 - alpha_t), updated
after every labelled call) or a conformal.Fitted (group-aware, static).

`cp=None` is the point-estimate agent used in the calibrate phase: top-1
values / detector score > 0.1 (quasar's thresholds), no alpha, but the true-label score
S(y_t) and the call's groups are still logged so they can become calibration data.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from answer_dsl import NO, WILD, YES, Box, DslError, Program, fmt, match, norm
from conformal import CAL_KEY
from labels import COLORS, MATERIALS, category, same_object
from selector import Selection, default_selectors, qtype

POINT_FIND = 0.1          # quasar ABSTRACT_FIND_LOW
CROWDED = 10              # a find with this many raw detector boxes > POINT_FIND is "crowded"
SPATIAL = {"left", "right", "above", "below", "top", "bottom", "side"}


def query_prompt(attribute: str, obj: str, cat) -> str:
    """Name the object: an open "main object" prompt got "double decker bus" for
    height.  A closed category is scored over all its label words by
    Backends.choose (Decisions API), so every option gets a probability."""
    if attribute.strip().endswith("?"):
        return attribute
    if cat == "name":
        return "What is the main object in this image? Answer with its name."
    return f"What is the {attribute} of the {obj} in this image?"


class Env:
    def __init__(self, image_id: str, image, backends, labeler, cp, question: str = ""):
        self.image_id, self.image = image_id, image
        self.backends, self.labeler, self.cp = backends, labeler, cp
        self.selection = Selection(default_selectors())
        self.state = dict(qtype=qtype(question), crowded=False)   # features selectors read
        self._groups, self._thr = [], None
        self.whole: Box = (0, 0, image.width, image.height)
        self.program = Program()
        self.program.whole = self.whole
        self.calls: List[dict] = []
        self.line_covered: Dict[str, Optional[bool]] = {}   # line id -> covered (None: unlabeled)

    # ---------------------------------------------------------------- plumbing
    def _region(self, rid):
        if rid in (None, "", "image"):
            return self.whole, [], "image"
        if rid not in self.program.regions:
            raise DslError(f"unknown region {rid!r}; use 'image' or an r-id from a find line")
        box, line_id, name = self.program.regions[rid]
        return box, [line_id], name

    def _name(self, rid):
        """The name a region was found under, so labels match objects of that name.
        None for the whole image and for "anywhere" regions (no specific object)."""
        if rid in self.program.regions and rid not in self.program.anywhere:
            return self.program.regions[rid][2]
        return None

    def _select(self, tool: str, region, attr=None) -> Optional[float]:
        """Fix this call's groups (before its set is built) and return its threshold."""
        feats = dict(self.state, tool=tool, fam=CAL_KEY[tool], attr=attr,
                     region="image" if region in (None, "", "image") else "box")
        self._groups = self.selection.step(feats)
        self._thr = self.cp.threshold(tool, self._groups) if self.cp else None
        return self._thr

    def _finish(self, tool, expr, values, deps, label, covered, true_score, extra=None):
        alpha = self.cp.alpha if self.cp else None
        line = self.program.add(expr, values, alpha, deps)
        labeled = label is not None
        self.line_covered[line.id] = covered if labeled else None
        self.calls.append(dict(t=len(self.calls) + 1, line=line.id, tool=tool, expr=expr,
                               values=sorted(line.values), alpha=alpha,
                               threshold=self._thr, groups=self._groups, labeled=labeled,
                               label=label, covered=covered if labeled else None,
                               true_score=true_score if labeled else None, **(extra or {})))
        if self.cp and labeled:            # ACI: immediate feedback, lands before the next set
            for ok in ([ok for _, ok in extra["instances"]] if extra and "instances" in extra else [covered]):
                self.cp.update(ok)
        return line

    def call(self, name: str, args: dict) -> str:
        fn = {"find": self.find, "query": self.query, "verify": self.verify,
              "exists": self.exists, "relate": self.relate}.get(name)
        if fn is None:
            return f"error: unknown tool {name!r}"
        args = {k: v for k, v in args.items() if k != "thought"}
        try:
            line = fn(**args)
        except (DslError, TypeError) as e:
            return f"error: {e}"
        return "Program so far:\n" + self.program.render() + \
            f"\n\nNew line: {self.program.render_line(line)}"

    # ---------------------------------------------------------------- tools
    def find(self, name: str, region: str = "image"):
        box, deps, _ = self._region(region)
        q = self._select("find", region)
        cands = self.backends.detect(self.image_id, self.image, box, name)
        if self.cp:
            kept = [(b, s) for b, s in cands if 1 - s <= q]
        else:
            kept = [(b, s) for b, s in cands if s > POINT_FIND]
        # q >= 1 admits every label with S <= 1, including boxes the detector never proposed.
        # The full label space must cover (ACI's alpha <= 0 case), so add the
        # search region itself: "somewhere in here".
        anywhere = q is not None and q >= 1
        gt = self.labeler.find(name, box, cands)
        covered = true_score = None
        instances = []
        if gt is not None:
            # each instance is its own coverage event (README §6): its score is the
            # cutoff needed to catch it, 1 if no proposed box finds it (g None:
            # the grader saw an instance no box frames)
            for g in gt:
                hits = [] if g is None else [s for b, s in cands if same_object(g, b)]
                instances.append([1 - max(hits, default=0.0),
                                  anywhere or (g is not None and any(same_object(g, b) for b, _ in kept))])
            covered = all(ok for _, ok in instances)
            true_score = max((sc for sc, _ in instances), default=0.0)
        line = self._finish("find", f'find("{name}", {region or "image"})', [], deps,
                            None if gt is None else len(gt), covered, true_score,
                            dict(boxes=[[round(v) for v in b] + [round(s, 3)] for b, s in kept],
                                 anywhere=anywhere, instances=instances))
        rids = [self.program.new_region(b, line.id, name) for b, _ in kept]
        if anywhere:
            rids.append(self.program.new_region(box, line.id, name, anywhere=True))
        line.values = frozenset(rids)
        self.calls[-1]["values"] = sorted(rids, key=lambda r: int(r[1:]))
        self.state["crowded"] = sum(s > POINT_FIND for _, s in cands) >= CROWDED
        return line

    def query(self, region: str, attribute: str):
        box, deps, name = self._region(region)
        cat = category(attribute)
        q = self._select("query", region, attr="color" if cat is COLORS else "material"
                         if cat is MATERIALS else "name" if cat == "name" else "other")
        question = query_prompt(attribute, "main object" if name == "image" else name, cat)
        answers = self.backends.choose(self.image_id, self.image, box, question, cat) \
            if isinstance(cat, set) else self.backends.ask(self.image_id, self.image, box, question)
        if self.cp:
            values = {a for a, p in answers if 1 - p <= q}
            if q >= 1:                      # every unlisted label has S = 1 <= q too
                values.add(WILD)
        else:
            values = {answers[0][0]} if answers else set()
        truths = self.labeler.query(box, attribute, self._name(region))
        covered = true_score = None
        if truths is not None:
            # names and open answers match either way round: "laptop" names a "laptop computer"
            hit = (lambda t, a: match(t, a) or match(a, t)) if cat == "name" or cat is None else match
            covered = WILD in values or any(hit(t, v) for t in truths for v in values)
            true_score = min(1 - max((p for a, p in answers if hit(t, a)), default=0.0)
                             for t in truths)
        return self._finish("query", f'query({region}, "{attribute}")', values, deps,
                            sorted(truths) if truths else None, covered, true_score,
                            dict(scores=answers))

    def _yesno(self, tool, expr, box, deps, question, truth):
        p_yes = self.backends.predicate(self.image_id, self.image, box, question)
        p_yes = 0.5 if p_yes is None else p_yes          # refusal: no information
        score = {YES: 1 - p_yes, NO: p_yes}
        if self.cp:
            values = {v for v, s in score.items() if s <= self._thr}
        else:
            values = {YES if p_yes >= 0.5 else NO}
        covered = None if truth is None else truth in values
        return self._finish(tool, expr, values, deps, truth, covered,
                            None if truth is None else score[truth], dict(p_yes=p_yes))

    def verify(self, region: str, property: str):
        box, deps, name = self._region(region)
        obj = "object" if name == "image" else name
        q = f'Does the {obj} in the image have the property "{property}" (yes/no)?'
        self._select("verify", region)
        return self._yesno("verify", f'verify({region}, "{property}")', box, deps, q,
                           self.labeler.verify(box, property, self._name(region)))

    def exists(self, name: str, region: str = "image"):
        box, deps, _ = self._region(region)
        q = f"Is there a {name} in the image (yes/no)?"
        self._select("exists", region)
        return self._yesno("exists", f'exists("{name}", {region or "image"})', box, deps, q,
                           self.labeler.exists(name, box))

    def relate(self, region_a: str, region_b: str, relation: str):
        if set(norm(relation).split()) & SPATIAL:
            raise DslError("left/right/above/below come from box geometry: use side(a, b) or "
                           "above(a, b) in the answer spec, e.g. same(side(o2, o1), 'left')")
        box_a, deps_a, name_a = self._region(region_a)
        box_b, deps_b, name_b = self._region(region_b)
        union = (min(box_a[0], box_b[0]), min(box_a[1], box_b[1]),
                 max(box_a[2], box_b[2]), max(box_a[3], box_b[3]))
        q = f"In this image, is the {name_a} {relation} the {name_b} (yes/no)?"
        self._select("relate", region_a)
        return self._yesno("relate", f'relate({region_a}, {region_b}, "{relation}")', union,
                           deps_a + deps_b, q, self.labeler.relate(box_a, box_b, relation,
                                                                   self._name(region_a),
                                                                   self._name(region_b)))

    # ---------------------------------------------------------------- answer
    def answer(self, spec: str) -> dict:
        values, closure = self.program.evaluate(spec)
        return dict(spec=spec, values=sorted(values), shown=fmt(values),
                    closure=sorted(closure, key=lambda x: int(x[1:])),
                    bound=self.program.bound(closure) if self.cp else None,
                    closure_all_covered=all(self.line_covered.get(i) is True for i in closure),
                    closure_all_labeled=all(self.line_covered.get(i) is not None for i in closure))
