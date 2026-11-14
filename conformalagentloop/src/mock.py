"""No-key, no-download fixtures for `run.py --mock`: a fake image + scene graph,
fake detector/VLM scores, and a scripted policy.  The test question has a planted
VLM miss so ACI must react (README §2)."""
from __future__ import annotations
from types import SimpleNamespace

from labels import SceneGraphLabeler, iou

CUP1, CUP2, JUNK, PLATE = (10, 10, 30, 30), (60, 10, 80, 30), (85, 85, 95, 95), (40, 60, 70, 90)
SG = {"width": 100, "height": 100, "objects": {
    "1": {"name": "cup", "x": 10, "y": 10, "w": 20, "h": 20, "attributes": ["white"],
          "relations": [{"name": "to the left of", "object": "3"}]},
    "2": {"name": "cup", "x": 60, "y": 10, "w": 20, "h": 20, "attributes": ["gray"], "relations": []},
    "3": {"name": "plate", "x": 40, "y": 60, "w": 30, "h": 30, "attributes": ["white"], "relations": []}}}
DETECT = {"cup": [(CUP1, 0.9), (CUP2, 0.35), (JUNK, 0.12)], "plate": [(PLATE, 0.8)]}
COLOR = {CUP1: [("white", 0.7), ("gray", 0.2), ("silver", 0.1)],
         CUP2: [("black", 0.85), ("gray", 0.15)],          # planted miss: truth is gray
         PLATE: [("white", 0.9), ("beige", 0.1)]}

QUESTIONS = [
    dict(qid="cal1", imageId="img", question="What color is the cup on the left?", answer="white"),
    dict(qid="cal2", imageId="img", question="Is there a plate?", answer="yes"),
    dict(qid="test1", imageId="img", question="Is the cup the same color as the plate?", answer="yes"),
    dict(qid="test2", imageId="img", question="Is the plate to the left or right of the cups?", answer="right"),
]
SCRIPTS = {
    "What color is the cup on the left?": [
        ("find", {"name": "cup", "region": "image"}), ("query", {"region": "r1", "attribute": "color"}),
        ("answer", {"spec": "o2"})],
    "Is there a plate?": [("exists", {"name": "plate", "region": "image"}), ("answer", {"spec": "o1"})],
    "Is the cup the same color as the plate?": [
        ("find", {"name": "cup", "region": "image"}), ("find", {"name": "plate", "region": "image"}),
        ("query", {"region": "r1", "attribute": "color"}), ("query", {"region": "r2", "attribute": "color"}),
        ("query", {"region": "r3", "attribute": "color"}),
        ("answer", {"spec": "same(union(o3, o4), o5)"})],
    "Is the plate to the left or right of the cups?": [
        ("find", {"name": "plate", "region": "image"}), ("find", {"name": "cup", "region": "image"}),
        ("relate", {"region_a": "r1", "region_b": "r2", "relation": "to the right of"}),  # rejected
        ("answer", {"spec": "'right'"}),                    # literal, rejected: agent sees the error...
        ("answer", {"spec": "side(o1, o2)"})],              # ...and recovers
}
# test-phase calibration scores: Q(0.9) = 0.72, so CUP2 (S=.65) is kept, JUNK (S=.88) is not
CAL = {k: [round(0.04 * i, 2) for i in range(20)] for k in ("find", "query", "yesno")}


class MockBackends:
    def detect(self, image_id, image, box, name):
        return DETECT.get(name, [])

    def choose(self, image_id, image, box, question, options):
        return COLOR[max(COLOR, key=lambda b: iou(b, box))]

    def predicate(self, image_id, image, box, question):
        return 0.8


def scripted_policy(messages, force):
    script = SCRIPTS[messages[1]["content"].removeprefix("Question: ")]
    i = sum(m["role"] == "assistant" for m in messages)
    name, args = script[min(i, len(script) - 1)]
    cid = f"call{i}"
    return {"name": name, "args": {"thought": "scripted", **args}, "id": cid,
            "msg": {"role": "assistant", "content": None, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}]}}


def context():
    from PIL import Image
    img = Image.new("RGB", (100, 100))
    return SimpleNamespace(
        backends=MockBackends(), policy=scripted_policy, image=lambda image_id: img,
        labeler=lambda q, image: SceneGraphLabeler(SG, 100, 100, {"cup", "plate"}))


def check(cal_scores, test_traces, test_calls):
    assert cal_scores["find"] and cal_scores["query"] and cal_scores["yesno"], cal_scores
    t1, t2 = test_traces
    assert t1["answer"]["values"] == ["no", "yes"] and t1["answer_covered"]
    assert t1["answer"]["closure"] == ["o1", "o2", "o3", "o4", "o5"]
    c = [x for x in test_calls if x["qid"] == "test1"]
    assert c[0]["values"] == ["r1", "r2"], c[0]                       # JUNK cut by Q(0.9)
    assert c[0]["groups"][:2] == ["fam:find", "find#1"] and "find#2" in c[1]["groups"]   # selectors
    assert c[3]["covered"] is False and c[3]["values"] == ["black"]    # the planted miss
    assert c[4]["alpha"] < c[3]["alpha"] - 0.04, "ACI must lower alpha after a miss"
    assert abs(t1["answer"]["bound"] - (1 - sum(x["alpha"] for x in c))) < 1e-9
    assert t2["answer_covered"] and t2["steps"] == 5 and len(t2["errors"]) == 2   # recovered
    assert t1["alpha_start"] == t2["alpha_start"] == 0.1               # per-question reset
    # threshold 1 (calibration says no cutoff suffices): the full set must cover
    from conformal import ACI
    from tools import Env
    ctx = context()
    env = Env("img", ctx.image("img"), MockBackends(), ctx.labeler(None, None),
              ACI({"find": [1.0] * 20}, 0.1, 0.05))
    env.find("cup")
    assert env.calls[-1]["anywhere"] and env.calls[-1]["covered"]
    assert len(env.program.lines["o1"].values) == 4 and "anywhere" in env.program.render()
