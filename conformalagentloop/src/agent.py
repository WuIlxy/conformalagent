"""ReAct loop over OpenAI function calling.  One tool call per turn; every
result is the program so far (README §5), so the agent always reasons over
sets + alpha + running bounds, never point values (except in point mode)."""
from __future__ import annotations
import json
from typing import Callable, Optional

from answer_dsl import NO, YES, DslError

YESNO_START = ("is ", "are ", "do ", "does ", "did ", "was ", "were ", "can ", "could ", "has ", "have ")

SYSTEM = """You answer a question about an image by calling tools, one per turn.

Every tool result is a SET of possibilities, appended as a numbered line to a program:
  o3 = query(r1, "color") -> {white, gray}   α=0.100  bound≥0.80  (needs o1)
- The set contains the true value with probability at least 1-α (a conformal prediction set).
- bound: union-bound guarantee for this line together with every line it depends on.
- ⊤ means the tool could rule nothing out (any value is possible).
- If no α is shown, the set is the tool's single best guess, with no guarantee.

Tools:
- find(name, region): boxes that may be `name`. Each box gets a region id (r1, r2, ...).
  The set is built to include every true instance, so it may include extra boxes.
  A region marked "anywhere" means the object may be somewhere in the searched area with
  no box ruled out; query or verify it like any region, or search again more specifically.
- query(region, attribute): values of an attribute (e.g. "color", "material", "name"),
  or a full question ending in "?".
- verify(region, property): {yes}, {no}, or {yes, no}.
- exists(name, region): is there such an object? {yes}, {no}, or {yes, no}.
- relate(region_a, region_b, relation): does a non-spatial relation hold (e.g. "on", "holding")?
  For left/right/above/below use side/above in the answer spec instead.
region is "image" or a region id from a find line.

Finish with answer(spec). The spec derives the answer from existing lines (o-ids, r-ids);
it never contains the answer value itself:
  union(a, ...)                     same(a, b) / different(a, b) -> yes/no (b may be a literal)
  and_(a, b, ...) / or_(a, b, ...) / not_(a) over yes/no lines
  choose(a, "x", "y") -> the options still possible in a
  side(a, b) -> left/right, above(a, b) -> above/below, from boxes (a, b: find lines, r-ids,
                or image for the photo itself)
Examples:
  "Is there a plate?"                      exists -> o1          answer("o1")
  "What color is the cup?"                 query -> o2           answer("o2")
  "Is the cup white?"                      query -> o2           answer("same(o2, 'white')")
  "Is the cup red or blue?"                query -> o2           answer("choose(o2, 'red', 'blue')")
  "Is the cup left or right of the plate?" find o1, find o2      answer("side(o1, o2)")
  "Is there a car left of the man?"        find o1 (car), o2     answer("same(side(o1, o2), 'left')")
  "Which side of the photo is the man on?" find o1               answer("side(o1, image)")
  "Is there a kite or a balloon?"          exists o1, exists o2  answer("or_(o1, o2)")
Use side/above for left/right/top/bottom; they cover all box pairs at once, so don't check
pairs one by one. The answer set must cover every value still consistent with what you saw;
prefer the smallest such set. Yes/no questions need a yes/no answer set."""

_REGION = {"type": "string", "description": '"image" or a region id like "r1"'}
_THOUGHT = {"type": "string", "description": "brief reasoning for this step"}


def _tool(tool_name, desc, **props):
    props = {"thought": _THOUGHT, **props}
    return {"type": "function", "function": {
        "name": tool_name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(props),
                       "additionalProperties": False}}}


TOOLS = [
    _tool("find", "Find boxes matching an object name.", name={"type": "string"}, region=_REGION),
    _tool("query", "Ask for an attribute of a region.", region=_REGION, attribute={"type": "string"}),
    _tool("verify", "Check whether a region has a property.", region=_REGION, property={"type": "string"}),
    _tool("exists", "Check whether an object exists in a region.", name={"type": "string"}, region=_REGION),
    _tool("relate", "Check a relation between two regions.", region_a=_REGION, region_b=_REGION,
          relation={"type": "string"}),
    _tool("answer", "Commit the final answer spec.", spec={"type": "string"}),
]

# policy(messages, force_answer) -> {"name", "args", "id", "msg"} or None
Policy = Callable[[list, bool], Optional[dict]]
policy_tokens: dict = {}     # model -> [prompt, completion] tokens this process (for cost)


def openai_policy(model: str) -> Policy:
    from openai import OpenAI
    client = OpenAI()
    # gpt-6-luna rejects function tools on chat.completions unless reasoning is off.
    # ponytail: reasoning off; port to the Responses API if the policy needs it
    extra = {"temperature": 0} if model.startswith("gpt-4") else {"reasoning_effort": "none"}
    tokens = policy_tokens.setdefault(model, [0, 0])

    def policy(messages, force):
        r = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, parallel_tool_calls=False,
            tool_choice={"type": "function", "function": {"name": "answer"}} if force else "required",
            **extra)
        m = r.choices[0].message
        tokens[0] += r.usage.prompt_tokens
        tokens[1] += r.usage.completion_tokens
        if not m.tool_calls:
            return None
        tc = m.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {"_bad_json": tc.function.arguments}
        return {"name": tc.function.name, "args": args, "id": tc.id, "msg": {
            "role": "assistant", "content": m.content,
            "tool_calls": [{"id": tc.id, "type": "function", "function": {
                "name": tc.function.name, "arguments": tc.function.arguments}}]}}
    return policy


def run_question(question: str, env, policy: Policy, max_steps: int = 8) -> dict:
    """-> {"answer": env.answer(...) dict or None, "steps", "stopped", "thoughts"}"""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {question}"}]
    thoughts, errors = [], []
    q = question.strip().lower()
    yesno_q = q.startswith(YESNO_START) and " or " not in q   # "Is it red or blue?" is a choice
    side_q = "which side" in q
    open_q = q.startswith(("what", "which", "who", "where", "how"))
    for step in range(max_steps):
        call = policy(messages, step == max_steps - 1)
        if call is None:
            messages.append({"role": "user", "content": "Call a tool or answer."})
            continue
        messages.append(call["msg"])
        thoughts.append(call["args"].get("thought", ""))
        if call["name"] == "answer":
            try:
                ans = env.answer(str(call["args"].get("spec", "")))
                if not ans["values"]:
                    raise DslError("the answer set is empty, so it can't contain the answer. Check "
                                   "that the spec combines lines of the right kind.")
                if yesno_q and not set(ans["values"]) <= {YES, NO}:
                    raise DslError(f"this is a yes/no question but the answer set is {ans['shown']}. "
                                   f"Derive yes/no, e.g. same(o2, 'white') or verify/exists.")
                if open_q and set(ans["values"]) <= {YES, NO}:
                    raise DslError(f"this question asks for a value, not yes/no. Answer with the "
                                   f"line that holds the values, e.g. answer('o2').")
                if side_q and not set(ans["values"]) <= {"left", "right"}:
                    raise DslError(f"this asks which side, but the answer set is {ans['shown']}. "
                                   f"Use side(o1, image) for the side of the photo.")
                return {"answer": ans, "steps": step + 1, "stopped": "answered",
                        "thoughts": thoughts, "errors": errors}
            except DslError as e:
                out = f"error: {e}"
        elif "_bad_json" in call["args"]:
            out = "error: arguments were not valid JSON"
        else:
            out = env.call(call["name"], call["args"])
        if out.startswith("error:"):
            errors.append(f"{call['name']}({call['args'].get('spec', '')}): {out}")
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": out})
    return {"answer": None, "steps": max_steps, "stopped": "max_steps", "thoughts": thoughts,
            "errors": errors}
