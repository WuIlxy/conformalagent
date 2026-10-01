"""The ReAct-style conformal agent loop.

Unifying move: every Action the policy emits is a one-line DSL expression
(answer_dsl).  A "probe" evaluates it just to *observe* the resulting set (and
feed that set, with its alpha, back into the prompt); an "answer" evaluates the
same kind of expression and commits it.  So exploratory tool use and the final
answer run through identical set-valued semantics -- the agent is always
looking at conformal sets, never point estimates.

    question ---> [policy: Thought + Action] ---> evaluate spec (set semantics)
        ^                                              |
        |            Observation: set + alpha  <-------+
        +--- loop until Action is an Answer, or max_steps ---------------------

The policy (the LLM) is injected.  The real one prompts gpt-4o-mini exactly like
imgpatch_4o_all.do_gpt_call and parses "Thought:/Action:/Answer:"; the demo
injects a scripted policy so the loop runs with no API and no torch.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import prompts
import answer_dsl
from setsem import AbstractOther, AbstractTuple


Region = Tuple[int, int, int, int]


@dataclass
class Step:
    thought: str
    action_repr: str
    kind: str                 # "probe" | "answer"
    spec: str
    observation: object       # AbstractOther | AbstractTuple
    alpha: float              # union-bound budget for this spec's sets
    observation_text: str


@dataclass
class Episode:
    question: str
    steps: List[Step] = field(default_factory=list)
    answer_set: Optional[AbstractOther] = None
    answer_alpha: float = 0.0
    stopped_reason: str = "answered"


# policy: sees (system, trace_text, question) -> (thought, kind, spec)
Policy = Callable[[str, str, str], Tuple[str, str, str]]

# feedback oracle (calibration only): given (spec, observed_set) -> covered?
# Used to drive ACI.  None at deployment (see conformal.py OPEN note).
Oracle = Optional[Callable[[str, object], bool]]


def run_episode(question: str,
                whole_image: Region,
                tools,
                policy: Policy,
                oracle: Oracle = None,
                max_steps: int = 6) -> Episode:
    ep = Episode(question=question)

    for _ in range(max_steps):
        trace_text = prompts.render_trace(question, ep.steps)
        thought, kind, spec = policy(prompts.SYSTEM, trace_text, question)

        pred_set, budget = answer_dsl.evaluate(spec, tools, whole_image)
        obs_text = prompts.render_observation(pred_set, budget)

        ep.steps.append(Step(
            thought=thought, action_repr=f"{kind}: {spec}", kind=kind,
            spec=spec, observation=pred_set, alpha=budget, observation_text=obs_text,
        ))

        # calibration-time ACI feedback: did the set cover the truth?
        if oracle is not None:
            covered = oracle(spec, pred_set)
            # attribute the (covered?) signal to each tool the spec touched
            for tool in set(_tools_in(spec)):
                tools.conformal.feedback(tool, covered)

        if kind == "answer":
            ep.answer_set = pred_set if isinstance(pred_set, AbstractOther) \
                else AbstractOther(frozenset(pred_set.values()))
            ep.answer_alpha = budget
            return ep

    ep.stopped_reason = "max_steps"
    return ep


def _tools_in(spec: str) -> List[str]:
    """Crude: which tool leaves a spec references (for feedback attribution)."""
    names = {"find": "find", "attribute_of": "query_attribute", "verify": "verify",
             "relate": "spatial", "count": "count"}
    return [tool for leaf, tool in names.items() if leaf in spec]


# --------------------------------------------------------------------------- #
# reference sketch of the REAL LLM policy (not exercised in the demo)
# --------------------------------------------------------------------------- #
def openai_policy(model="gpt-4o-mini"):
    """Returns a Policy backed by an LLM.  Mirrors imgpatch_4o_all.do_gpt_call.
    Left as a sketch: fill in the client and a robust parser."""
    def policy(system, trace_text, question):
        # response = client.chat.completions.create(
        #     model=model, temperature=0,
        #     messages=[{"role": "system", "content": system},
        #               {"role": "user", "content": trace_text}])
        # text = response.choices[0].message.content
        # return parse_thought_action(text)   # -> (thought, "probe"|"answer", spec)
        raise NotImplementedError("wire to imgpatch_4o_all-style client + parser")
    return policy
