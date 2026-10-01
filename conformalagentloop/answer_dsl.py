"""The constrained answer action.

`answer(spec)` does NOT take a free-form string; it takes a one-line expression
in a tiny DSL that says *how to derive the answer from the trajectory*.  We then
evaluate the spec with set-valued semantics (setsem.py) over the tool outputs.

Why this shape (vs ViperGPT synthesising a whole program up front):
  * the "program" is one line, written at the END, after all uncertainty is
    observed -- no up-front structural commitment, easy to recover from;
  * soundness is by construction: every leaf is a conformal set with a known
    miscoverage budget, the combinators are deterministic set-lifts, so the
    result set covers the truth with prob >= 1 - (sum of leaf budgets)  [union bound].

DSL surface (all operate on / return AbstractOther | AbstractTuple):
    find(phrase)                      -> region set          (re-uses Tools.find)
    attribute_of(regions, attr)       -> value set           (query_attribute, lifted over regions)
    verify(regions, predicate)        -> bool set
    relate(regions_a, regions_b)      -> relation set
    count(regions)                    -> integer set
    union(x, ...)                     -> join of value sets
    yesno(bool_set)                   -> {"yes"}|{"no"}|{"yes","no"}

The evaluator dispatches leaves back through the SAME Tools instance, so the
sets used in the answer are exactly the (conformally-cached) sets the agent
already saw during the trace.
"""
from __future__ import annotations
from typing import List

from setsem import AbstractOther, AbstractTuple, lift, join_many, bool_value


class DslContext:
    """Whitelisted namespace + running union-bound budget.  One per answer eval."""
    def __init__(self, tools, whole_image):
        self.tools = tools
        self.whole = whole_image
        self.tools_used: List[str] = []

    # -- leaves -------------------------------------------------------------- #
    def find(self, phrase: str) -> AbstractTuple:
        self.tools_used.append("find")
        return self.tools.find(self.whole, phrase)

    def attribute_of(self, regions: AbstractTuple, attr: str) -> AbstractOther:
        self.tools_used.append("query_attribute")
        # lift query_attribute over every possible region, then join the value sets
        return join_many(
            self.tools.query_attribute(box, attr) for box in regions.values()
        ) if regions.values() else AbstractOther(frozenset({"__none__"}))

    def verify(self, regions: AbstractTuple, predicate: str) -> AbstractOther:
        self.tools_used.append("verify")
        return join_many(self.tools.verify(box, predicate) for box in regions.values())

    def relate(self, a: AbstractTuple, b: AbstractTuple) -> AbstractOther:
        self.tools_used.append("spatial")
        return join_many(
            self.tools.spatial(ba, bb) for ba in a.values() for bb in b.values()
        )

    def count(self, regions: AbstractTuple) -> AbstractOther:
        self.tools_used.append("count")
        return self.tools.count(regions)

    # -- combinators (pure, alpha carried by setsem) ------------------------- #
    def union(self, *xs) -> AbstractOther:
        return join_many(xs)

    def yesno(self, b: AbstractOther) -> AbstractOther:
        return lift(lambda v: "yes" if v else "no", b)

    def namespace(self):
        return {
            "find": self.find, "attribute_of": self.attribute_of,
            "verify": self.verify, "relate": self.relate, "count": self.count,
            "union": self.union, "yesno": self.yesno,
        }


def evaluate(spec: str, tools, whole_image):
    """Evaluate an answer spec.  Returns (answer_set, union_bound_alpha).

    ponytail: restricted eval with an empty __builtins__ and a fixed whitelist.
    Fine for a sketch; a deployed system wants a real parser + validator so a
    misbehaving LLM cannot smuggle anything past the whitelist.
    """
    ctx = DslContext(tools, whole_image)
    result = eval(spec, {"__builtins__": {}}, ctx.namespace())   # noqa: S307 (sketch)
    if not isinstance(result, (AbstractOther, AbstractTuple)):
        result = AbstractOther(frozenset({result}))
    budget = tools.conformal.answer_budget(ctx.tools_used)
    # coverage of the derived set is >= 1 - budget (union bound over sets used)
    return result, budget
