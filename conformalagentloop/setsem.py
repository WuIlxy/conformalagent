"""Set-valued semantics: the prediction-set lattice the agent reasons over.

This is a trimmed, standalone copy of the core of
`quasar-colm2026-artifact/epic/conformal_utils.py` (the paper's abstract
interpreter).  Same idea, fewer operators:

  * AbstractOther[T]  -- a prediction *set* (frozenset of possibilities).
                         Every op lifts pointwise (product, then join).
  * AbstractTuple[T]  -- a set-valued sequence from `find`: items tagged
                         (value, certain).  len -> integer range; [0] -> heads.
  * join / join_many  -- least upper bound = set union.
  * ABS_BOOL_*        -- the three-valued bool ({T}, {F}, {T,F}=TOP).

The one thing added here that the paper's version doesn't track is a per-set
*coverage budget* (`alpha`), so the answer DSL can carry a union bound through
a computation (see answer_dsl.py).  It rides alongside the value; the algebra
itself is unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypeVar, Generic, FrozenSet, Tuple, Callable
import itertools

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# core lattice
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AbstractOther(Generic[T]):
    """A prediction set.  `alpha` is the miscoverage budget spent to obtain it
    (0.0 for a deterministic / point value)."""
    _possibilities: FrozenSet[T]
    alpha: float = 0.0

    def possibilities(self) -> FrozenSet[T]:
        return self._possibilities

    def is_certain(self) -> bool:
        return len(self._possibilities) == 1

    def __iter__(self):
        return iter(self._possibilities)

    def __len__(self):
        return len(self._possibilities)

    def __repr__(self):
        body = ", ".join(map(repr, sorted(self._possibilities, key=repr)))
        a = "" if self.alpha == 0.0 else f" @a={self.alpha:.3g}"
        return f"{{{body}}}{a}"


@dataclass(frozen=True)
class AbstractTuple(Generic[T]):
    """Output of `find`: a set-valued list.  Each item is (value, certain);
    certain items are definitely present, uncertain ones may or may not be."""
    _items: Tuple[Tuple[T, bool], ...]
    alpha: float = 0.0

    def values(self):
        return tuple(v for v, _ in self._items)

    def __len__(self):
        """Plain item count (int) -- so builtin len() works.  For the
        *uncertain* cardinality use .card()."""
        return len(self._items)

    def card(self):
        """Cardinality is itself uncertain -> a set of integers [lb, ub].
        (This is what the paper's AbstractTuple.__len__ returns; kept as a named
        method here so Python's builtin len() stays usable elsewhere.)"""
        lb = sum(1 for _, certain in self._items if certain)
        ub = len(self._items)
        return AbstractOther(frozenset(range(lb, ub + 1)), alpha=self.alpha)

    def head(self):
        """The set of possible first elements (empty tuple -> {None})."""
        heads = []
        for v, certain in self._items:
            heads.append(v)
            if certain:
                break
        if not heads:
            heads = [None]
        return AbstractOther(frozenset(heads), alpha=self.alpha)


def _as_set(x):
    return x if isinstance(x, AbstractOther) else AbstractOther(frozenset({x}))


def join(a: AbstractOther, b: AbstractOther) -> AbstractOther:
    """Least upper bound = union of possibilities; alpha = max (same evidence
    line) is the caller's call -- here we keep the larger budget."""
    a, b = _as_set(a), _as_set(b)
    return AbstractOther(a._possibilities | b._possibilities, alpha=max(a.alpha, b.alpha))


def join_many(xs) -> AbstractOther:
    out = AbstractOther(frozenset())
    for x in xs:
        out = join(out, x)
    return out


def lift(f: Callable, *args) -> AbstractOther:
    """Apply a concrete function to the cartesian product of its (possibly
    set-valued) arguments, then join.  This is `do_pairwise`/`do_unary`
    generalised to n-ary f -- the pointwise lifting from the paper."""
    sets = [_as_set(a) for a in args]
    budget = sum(s.alpha for s in sets)          # union bound over inputs
    results = (f(*combo) for combo in itertools.product(*[s._possibilities for s in sets]))
    joined = join_many(AbstractOther(frozenset({r})) for r in results)
    return AbstractOther(joined._possibilities, alpha=budget)


# --------------------------------------------------------------------------- #
# three-valued bool
# --------------------------------------------------------------------------- #
ABS_BOOL_TRUE = AbstractOther(frozenset({True}))
ABS_BOOL_FALSE = AbstractOther(frozenset({False}))
ABS_BOOL_TOP = AbstractOther(frozenset({False, True}))


def bool_value(b: AbstractOther):
    """Collapse a bool-set to True / False / 'TOP'."""
    poss = b._possibilities
    if poss == {True}:
        return True
    if poss == {False}:
        return False
    return "TOP"


if __name__ == "__main__":
    # runnable check: lifting, join, and the tuple cardinality range
    a = AbstractOther(frozenset({2, 3}), alpha=0.02)
    b = AbstractOther(frozenset({10, 20}), alpha=0.03)
    s = lift(lambda x, y: x + y, a, b)
    assert s._possibilities == {12, 13, 22, 23}, s
    assert abs(s.alpha - 0.05) < 1e-9, s.alpha           # union bound adds

    t = AbstractTuple((("dog_box", True), ("cat_box", False)))
    assert t.card()._possibilities == {1, 2}, t.card()   # 1 certain, up to 2

    assert bool_value(join(ABS_BOOL_TRUE, ABS_BOOL_FALSE)) == "TOP"
    print("setsem self-check ok:", s, "| count(find)=", t.card())
