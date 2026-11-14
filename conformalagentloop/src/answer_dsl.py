"""The trace as a straight-line program (README section 5).

Every tool call appends a line  o_k = tool(args) -> SET  with the ACI level
alpha_t in force and its direct deps (the find lines behind any region ids it
used).  A line's running bound is the union bound over its dependency closure:
    P[line covers] >= 1 - sum_{o in closure} alpha_o
The final answer is one more line: a combinator spec over existing lines,
parsed with `ast` against a whitelist (no eval, no tool re-calls).
"""
from __future__ import annotations
import ast
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

WILD = "*"                       # the full label space (shown as ⊤)
YES, NO = "yes", "no"
Box = Tuple[float, float, float, float]

_IRREGULAR = {"men": "man", "women": "woman", "people": "person", "children": "child",
              "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose"}


def singular(w: str) -> str:
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith(("ses", "xes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return " ".join("gray" if w == "grey" else w for w in s.split())


def match(truth: str, value: str) -> bool:
    """truth's words appear contiguously in value ("blue" matches "light blue")."""
    t = " ".join(singular(w) for w in norm(truth).split())
    v = " ".join(singular(w) for w in norm(value).split())
    return bool(t) and f" {t} " in f" {v} "


def covers(values: FrozenSet[str], truth: str) -> bool:
    return WILD in values or any(match(truth, v) for v in values)


class DslError(ValueError):
    pass


@dataclass
class Line:
    id: str
    expr: str
    values: FrozenSet[str]
    alpha: Optional[float]           # None: point estimate or combinator
    deps: FrozenSet[str]


class Program:
    def __init__(self):
        self.lines: Dict[str, Line] = {}
        self.regions: Dict[str, Tuple[Box, str, str]] = {}   # rid -> (box, line id, name)
        self.anywhere = set()      # rids meaning "somewhere in this search region" (full set)
        self.whole: Optional[Box] = None   # the image box, for side(o1, image)

    def add(self, expr, values, alpha, deps) -> Line:
        line = Line(f"o{len(self.lines) + 1}", expr, frozenset(values), alpha, frozenset(deps))
        self.lines[line.id] = line
        return line

    def new_region(self, box: Box, line_id: str, name: str, anywhere: bool = False) -> str:
        rid = f"r{len(self.regions) + 1}"
        self.regions[rid] = (box, line_id, name)
        if anywhere:
            self.anywhere.add(rid)
        return rid

    def closure(self, ids) -> FrozenSet[str]:
        seen, stack = set(), list(ids)
        while stack:
            i = stack.pop()
            if i not in seen:
                seen.add(i)
                stack.extend(self.lines[i].deps)
        return frozenset(seen)

    def bound(self, ids) -> float:
        return 1 - sum(self.lines[i].alpha or 0.0 for i in self.closure(ids))

    def render_line(self, line: Line) -> str:
        s = f"{line.id} = {line.expr} -> {fmt(line.values)}"
        if line.alpha is not None:
            needs = sorted(self.closure([line.id]) - {line.id}, key=lambda x: int(x[1:]))
            s += f"   α={line.alpha:.3f}  bound≥{self.bound([line.id]):.2f}"
            if needs:
                s += f"  (needs {', '.join(needs)})"
        for rid in sorted(line.values & self.anywhere):
            s += f"\n     {rid} = anywhere in the searched region: no box ruled out"
        return s

    def render(self) -> str:
        return "\n".join(self.render_line(l) for l in self.lines.values())

    # ------------------------------------------------------------------ answer
    def evaluate(self, spec: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        """-> (answer set, dependency closure).  Raises DslError on anything
        outside the whitelist."""
        try:
            tree = ast.parse(spec.strip(), mode="eval").body
        except SyntaxError as e:
            raise DslError(f"cannot parse spec: {e.msg}")
        deps = set()

        def ev(n):
            if isinstance(n, ast.Name):
                if n.id in self.lines:
                    deps.add(n.id)
                    return self.lines[n.id].values
                if n.id in self.regions:
                    deps.add(self.regions[n.id][1])
                    return frozenset({n.id})
                if n.id == "image" and self.whole is not None:
                    return frozenset({"image"})          # the whole photo: a constant, no deps
                raise DslError(f"unknown id {n.id!r}: reference lines (o1, ...) or regions (r1, ...). "
                               f"To answer with a line's set, write answer('o1').")
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                return norm(n.value)
            # Python's not/and/or are accepted as not_/and_/or_ (same three-valued meaning)
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
                return _not(self, ev(n.operand))
            if isinstance(n, ast.BoolOp):
                return (_and if isinstance(n.op, ast.And) else _or)(self, *[ev(v) for v in n.values])
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in COMBINATORS and not n.keywords):
                args = [ev(a) for a in n.args]
                try:
                    return COMBINATORS[n.func.id](self, *args)
                except TypeError:
                    raise DslError(f"wrong number of arguments to {n.func.id}()")
            raise DslError(f"not allowed in spec: {ast.unparse(n)!r}")

        values = ev(tree)
        if not isinstance(values, frozenset):
            raise DslError(f"answer({spec!r}) is a literal, not a derived set. Reference the line "
                           f"that supports it, e.g. answer('o1').")
        return values, self.closure(deps)


def _key(v: str):
    return (0, int(v[1:]), v) if re.fullmatch(r"r\d+", v) else (1, 0, v)


def fmt(values) -> str:
    vals = sorted(("⊤" if v == WILD else v for v in values), key=_key)
    return "{" + ", ".join(vals) + "}"


# ---------------------------------------------------------------- combinators
def _set(x) -> FrozenSet[str]:
    if not isinstance(x, frozenset):
        raise DslError(f"expected a line id, got literal {x!r} "
                       f"(literals are only allowed in same/different/choose)")
    return x


def _set_or_lit(x) -> FrozenSet[str]:
    """A literal is a constant: a singleton set with no uncertainty (alpha 0)."""
    return frozenset({x}) if isinstance(x, str) else _set(x)


def _bools(x) -> FrozenSet[str]:
    x = _set(x)
    if not x <= {YES, NO}:
        raise DslError(f"expected a yes/no set, got {fmt(x)}")
    return x


def _union(p, *xs):
    return frozenset().union(*map(_set, xs))


def _same(p, a, b):
    a, b = _set_or_lit(a), _set_or_lit(b)
    if not a or not b:
        return frozenset()
    if WILD in a or WILD in b:
        return frozenset({YES, NO})
    return frozenset(YES if norm(x) == norm(y) else NO for x in a for y in b)


def _not(p, a):
    return frozenset(NO if v == YES else YES for v in _bools(a))


def _and(p, *xs):
    if len(xs) < 2:
        raise TypeError
    xs = [_bools(x) for x in xs]
    if not all(xs):
        return frozenset()
    # yes is possible iff every input can be yes; no iff any input can be no
    return frozenset(([YES] if all(YES in x for x in xs) else []) +
                     ([NO] if any(NO in x for x in xs) else []))


def _or(p, *xs):
    return _not(p, _and(p, *[_not(p, x) for x in xs]))


def _choose(p, a, *options):
    a = _set(a)
    if not options or any(isinstance(o, frozenset) for o in options):
        raise DslError("choose(o, 'x', 'y'): options must be string literals")
    if a and all(v in p.regions or v == "image" for v in a):
        raise DslError("choose needs a value line (e.g. query), not regions from find. "
                       "For 'is there an X or a Y', use or_ over exists lines")
    if a and a <= {YES, NO} and not set(options) <= {YES, NO}:
        raise DslError(f"choose on a yes/no line can't pick between {list(options)}; "
                       f"choose on the query line that holds those values")
    if WILD in a:
        return frozenset(options)
    return frozenset(o for o in options if any(match(o, v) for v in a))


def _boxes(p, x):
    rids = _set(x)
    if rids == {"image"}:
        return [(p.whole, False)]
    if not rids or not all(r in p.regions for r in rids):
        raise DslError("side/above take find lines, region ids, or image")
    return [(p.regions[r][0], r in p.anywhere) for r in rids]


def _compare(p, a, b, axis, lo, hi):
    out = set()
    for x, x_any in _boxes(p, a):
        for y, y_any in _boxes(p, b):
            if x_any or y_any:              # position unknown: either side is possible
                out |= {lo, hi}
            else:
                out.add(lo if (x[axis] + x[axis + 2]) < (y[axis] + y[axis + 2]) else hi)
    return frozenset(out)


def _side(p, a, b):
    return _compare(p, a, b, 0, "left", "right")


def _above(p, a, b):
    return _compare(p, a, b, 1, "above", "below")


COMBINATORS = {
    "union": _union, "same": _same, "different": lambda p, a, b: _not(p, _same(p, a, b)),
    "and_": _and, "or_": _or, "not_": _not, "choose": _choose,
    "side": _side, "above": _above,
}


if __name__ == "__main__":
    p = Program()
    o1 = p.add('find("cup")', [], 0.1, [])
    r1 = p.new_region((0, 0, 10, 10), o1.id, "cup")
    r2 = p.new_region((50, 0, 60, 10), o1.id, "cup")
    p.lines[o1.id].values = frozenset({r1, r2})
    o2 = p.add('find("plate")', [], 0.1, [])
    r3 = p.new_region((20, 40, 40, 60), o2.id, "plate")
    p.lines[o2.id].values = frozenset({r3})
    o3 = p.add('query(r1, "color")', {"white", "gray"}, 0.1, [o1.id])
    o4 = p.add('query(r3, "color")', {"white"}, 0.05, [o2.id])
    assert abs(p.bound(["o3"]) - 0.8) < 1e-9
    vals, clo = p.evaluate("same(o3, o4)")
    assert vals == {YES, NO} and clo == {"o1", "o2", "o3", "o4"}
    assert abs(p.bound(clo) - 0.65) < 1e-9                       # o1 counted once
    assert p.evaluate("choose(o3, 'white', 'black')")[0] == {"white"}
    assert p.evaluate("side(r1, o2)")[0] == {"left"}
    assert p.evaluate("side(o1, o2)")[0] == {"left", "right"}
    p.whole = (0, 0, 100, 100)
    assert p.evaluate("side(r1, image)")[0] == {"left"} and p.evaluate("above(o2, image)")[0] == {"below"}
    for bad in ["choose(o1, 'cup', 'plate')", "choose(o5, 'white', 'black')"]:
        try:
            p.evaluate(bad); raise AssertionError(bad)
        except DslError:
            pass
    r4 = p.new_region((0, 0, 100, 100), o2.id, "plate", anywhere=True)
    assert p.evaluate(f"side(r1, {r4})")[0] == {"left", "right"}           # anywhere: both sides
    assert p.evaluate("same(o4, 'white')")[0] == {YES}
    assert p.evaluate("not same(o4, 'white') or same(o3, 'gray')")[0] == {YES, NO}
    assert fmt(frozenset({"r10", "r2", "r1"})) == "{r1, r2, r10}"
    y, n, yn = (p.add(e, v, 0.1, []).id for e, v in [("y", {YES}), ("n", {NO}), ("yn", {YES, NO})])
    assert p.evaluate(f"and_({y}, {y}, {yn})")[0] == {YES, NO}
    assert p.evaluate(f"and_({y}, {n}, {yn})")[0] == {NO}
    assert p.evaluate(f"or_({n}, {n}, {yn})")[0] == {YES, NO} and p.evaluate(f"or_({n}, {y})")[0] == {YES}
    for bad in ["__import__('os')", "o3.values", "o9", "union('x')", "choose(o3, o4)", "o3 + o4",
                "'no'", "no", "not_(o5, o6)", "and_(o5)"]:
        try:
            p.evaluate(bad); raise AssertionError(bad)
        except DslError:
            pass
    assert covers(frozenset({"light blue"}), "blue") and not covers(frozenset({"bluegrass"}), "blue")
    assert covers(frozenset({WILD}), "anything") and match("car", "cars")
    print(p.render())
    print("answer_dsl ok")
