"""End-to-end mock trajectory: "What color is the dog?"  Runs with no API, no
torch -- a deterministic MockBackends stands in for OWL-ViT + the VLM so the
whole loop, set semantics, DSL, and conformal layer exercise together.

    python demo.py
"""
from __future__ import annotations
from typing import List, Tuple

from setsem import AbstractOther, AbstractTuple
from conformal import make_trace_conformal
from tools import Tools, Region
from loop import run_episode
import prompts


# --------------------------------------------------------------------------- #
# a fake image world: one clear brown dog + one faint spurious box
# --------------------------------------------------------------------------- #
DOG_BOX = (10, 10, 90, 90)
SPURIOUS = (200, 200, 230, 230)
GROUND_TRUTH_COLOR = "brown"


class MockBackends:
    def detect(self, region, phrase):
        if "dog" in phrase:
            return [(DOG_BOX, 0.90), (SPURIOUS, 0.30)]   # high-conf dog + faint noise
        return []

    def attr_scores(self, region, attribute_class):
        if attribute_class == "color":
            return [("brown", 0.70), ("black", 0.25), ("white", 0.05)]
        return [("unknown", 1.0)]

    def yesno_score(self, region, predicate):
        return 0.8 if "brown" in predicate else 0.2


# --------------------------------------------------------------------------- #
# calibration scores per tool (offline split-conformal init for ACI)
# nonconformity = 1 - model_score.  These clusters place the thresholds so that
# find keeps the dog box only, and color stays ambiguous {brown, black}.
# --------------------------------------------------------------------------- #
CAL = {
    "find":            [round(0.02 * i, 3) for i in range(1, 21)],   # ~0.02..0.40
    "query_attribute": [round(0.04 * i, 3) for i in range(1, 21)],   # ~0.04..0.80
    "verify":          [round(0.03 * i, 3) for i in range(1, 21)],
    "spatial":         [0.0],
    "count":           [0.0],
}
TARGET_ALPHA = 0.10


# --------------------------------------------------------------------------- #
# scripted policy: stands in for the LLM.  Thought + (kind, spec) each turn.
# --------------------------------------------------------------------------- #
def scripted_policy():
    script = [
        ("Find the dog first.",                       "probe",  'find("dog")'),
        ("Ask its colour; note if the set is ambiguous.",
                                                      "probe",  'attribute_of(find("dog"), "color")'),
        ("Colour set has >1 value; commit a set that covers all of them.",
                                                      "answer", 'union(attribute_of(find("dog"), "color"))'),
    ]
    i = {"n": 0}
    def policy(system, trace_text, question):
        thought, kind, spec = script[min(i["n"], len(script) - 1)]
        i["n"] += 1
        return thought, kind, spec
    return policy


def oracle(spec, pred_set):
    """Calibration feedback for ACI: did the observed set cover the truth?"""
    if isinstance(pred_set, AbstractOther) and pred_set._possibilities and \
       all(isinstance(x, str) for x in pred_set._possibilities):
        # a value set (colors) -> covered iff it contains the true color
        if pred_set._possibilities & {"brown", "black", "white"}:
            return GROUND_TRUTH_COLOR in pred_set._possibilities
    return True   # region/other sets: dog is present, so covered


def main():
    conformal = make_trace_conformal(CAL, target_alpha=TARGET_ALPHA, gamma=0.05)
    tools = Tools(backends=MockBackends(), conformal=conformal)

    ep = run_episode("What color is the dog?", whole_image=(0, 0, 256, 256),
                     tools=tools, policy=scripted_policy(), oracle=oracle)

    # ---- print the trajectory -------------------------------------------- #
    print("=" * 70)
    print(prompts.render_trace(ep.question, ep.steps))
    print("=" * 70)
    print(f"FINAL answer set : {ep.answer_set}")
    print(f"union-bound alpha: {ep.answer_alpha:.3f}  "
          f"(coverage >= {1 - ep.answer_alpha:.2f})")
    print(f"logged calls     : {[c.tool for c in conformal.calls]}")
    print(f"find alpha now    : {conformal.alpha_for('find'):.4f} "
          f"(ACI moved it from {TARGET_ALPHA})")

    # ---- checks ----------------------------------------------------------- #
    assert ep.stopped_reason == "answered"
    assert ep.steps[-1].kind == "answer"
    assert GROUND_TRUTH_COLOR in ep.answer_set._possibilities, ep.answer_set   # COVERAGE
    assert len(ep.answer_set) >= 2, "expected an ambiguous (informative) set"   # {brown,black}
    assert 0.0 < ep.answer_alpha < 0.5, ep.answer_alpha                        # union bound sane
    assert any(len(getattr(s.observation, "_possibilities", [])) > 1
               for s in ep.steps), "agent should have seen a multi-element set"
    assert conformal.alpha_for("find") != TARGET_ALPHA, "ACI should have updated"
    print("\ndemo self-check ok: true answer covered by the committed set.")


if __name__ == "__main__":
    main()
