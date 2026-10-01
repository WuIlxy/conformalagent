"""ReAct prompt pieces.  The one change from vanilla ReAct: the Observation the
model sees back is a SET, and the prompt tells it what the set means and how
reliable it is.  This is where the model retrospects on the conformal set as
*evidence with a known reliability*, not as a point estimate.
"""

SYSTEM = """\
You answer a question about an image by calling tools one at a time.

Each tool returns a SET of possibilities, not a single value. A returned set is
a conformal prediction set: it is guaranteed to contain the true value with
probability at least 1 - alpha, where alpha is stated with each observation.

Tools:
  find(phrase)                 -> a set of image regions matching the phrase
  query_attribute(region,attr) -> a set of attribute values (e.g. color, material)
  verify(region,predicate)     -> {yes}, {no}, or {yes,no} (uncertain)
  spatial(region_a,region_b)   -> a set of spatial relations
  count(region_set)            -> an integer, or an integer range if membership is uncertain

Each turn output exactly:
  Thought: <your reasoning about the sets seen so far>
  Action: <one tool call>  OR  Answer: <answer-spec>

When an observation is a set with more than one element you may:
  (a) proceed, treating any element as possibly correct;
  (b) call another tool to disambiguate (e.g. verify a candidate);
  (c) stop and commit a final ANSWER SPEC that covers all still-possible values.

The Answer spec is NOT free text. It is one line in this DSL, evaluated with
set semantics over the tools you called:
  find(phrase), attribute_of(regions,attr), verify(regions,pred),
  relate(a,b), count(regions), union(...), yesno(bool_set)
Example:  Answer: union(attribute_of(find("dog"), "color"))

Prefer the tightest spec that is still guaranteed to cover the true answer given
the sets you have seen. Spend extra tool calls only when disambiguation is worth
it against the alpha budget.
"""


def render_observation(pred_set, alpha: float) -> str:
    """The critical 'set as evidence' message fed back after each tool call."""
    if hasattr(pred_set, "_items"):          # AbstractTuple (region set from find)
        parts = [f"{v}{'' if certain else '?'}" for v, certain in pred_set._items]
        elems = ", ".join(parts) + "   (?=uncertain member)"
    else:
        elems = ", ".join(map(str, getattr(pred_set, "_possibilities", [pred_set])))
    return (
        f"Observation: the tool returned the set {{{elems}}}.\n"
        f"This set is guaranteed to contain the true answer with probability at "
        f"least 1 - alpha, with alpha = {alpha:.3g}.\n"
        f"You may (a) proceed assuming any element could be correct, (b) issue a "
        f"follow-up tool call to disambiguate, or (c) commit a final answer set "
        f"that covers all possibilities consistent with the sets seen so far."
    )


def render_trace(question: str, steps) -> str:
    lines = [f"Question: {question}"]
    for s in steps:
        lines.append(f"Thought: {s.thought}")
        lines.append(f"Action: {s.action_repr}")
        lines.append(s.observation_text)
    return "\n".join(lines)
