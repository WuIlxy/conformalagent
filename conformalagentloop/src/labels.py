"""Ground truth for intermediate tool calls (README §6).  Two graders with the
same five methods: TerraLabeler (a strong model answers the call itself, the
default) and SceneGraphLabeler (GQA's scene graph).  Every method returns None
when the call can't be labelled honestly.

find returns the instances the call should catch; each is graded on its own
(tools.Env.find).  An instance is a box, or None for one no proposed box frames.
"""
from __future__ import annotations
import re
from typing import List, Optional, Set

from answer_dsl import Box, NO, YES, norm, singular

COLORS = {"white", "black", "blue", "green", "brown", "red", "gray", "yellow", "orange",
          "silver", "pink", "tan", "purple", "gold", "beige", "cream", "maroon", "khaki",
          "teal", "turquoise", "navy"}
MATERIALS = {"wood", "metal", "plastic", "glass", "brick", "concrete", "stone", "leather",
             "cloth", "fabric", "paper", "ceramic", "porcelain", "rubber", "steel",
             "cardboard", "straw", "wicker", "denim", "cotton", "marble", "wire"}
SIZES = {"large", "small"}
HEIGHTS = {"tall", "short"}
LENGTHS = {"long", "short"}
SHAPES = {"round", "square", "rectangular", "triangular", "oval"}
# query text keyword -> label space; also the categories verify uses to label "no"
CATEGORIES = {"color": COLORS, "colour": COLORS, "material": MATERIALS, "made of": MATERIALS,
              "size": SIZES, "height": HEIGHTS, "length": LENGTHS, "shape": SHAPES}
_ALIASES = {"wooden": "wood", "metallic": "metal", "big": "large", "huge": "large",
            "little": "small", "tiny": "small", "circular": "round"}
PERSON = {"person", "man", "woman", "boy", "girl", "child", "kid", "lady", "guy", "player",
          "skier", "surfer", "skateboarder", "snowboarder", "people", "men", "women"}
STOP = {"a", "an", "the", "any", "some", "all"}


def _attr(a: str) -> str:
    a = norm(a)
    return _ALIASES.get(a, a)


def category(text: str):
    """Attribute-question text -> label space, 'name', or None (unlabelable)."""
    t = norm(text)
    for key, space in CATEGORIES.items():
        if key in t:
            return space
    if t in {"name", "object", "type", "kind", "category", "what is it", "what is this"}:
        return "name"
    return None


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(b: Box) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def same_object(obj: Box, box: Box) -> bool:
    """Does detector box `box` find annotated object `obj`?  Annotators and
    detectors draw boxes differently (a pot's body vs pot + handle + lid), so
    IoU alone under-counts: also accept a box holding >=80% of the object
    while being at most 4x its area."""
    a = _area(obj)
    return iou(obj, box) >= 0.3 or (a > 0 and _overlap(obj, box) >= 0.8 * a and _area(box) <= 4 * a)


def _inside(obj: Box, region: Box) -> bool:
    a = _area(obj)
    return a > 0 and _overlap(obj, region) / a >= 0.5


GRADER = "gpt-6.1-terra"
GRADER_EFFORT = "medium"
YN = " Answer with exactly one word: yes or no."
# Lenient on naming and box fit (audit: a red dish rejected as "not a bowl", a
# whole-image box on a tablecloth rejected as "not the table").  The count stays strict:
# a lenient count added ~14% more instances, mostly with no detector box.
FRAME = ('Does the red box frame one {name}? Answer yes if it mainly frames a single object a person could '
         'reasonably call "{name}" (a close synonym or broader/narrower term counts; the box may be loose, '
         'include surroundings, or cut off hidden parts). Answer no only if it frames a different kind of '
         'object, only a small part of one, or several together.' + YN)
COUNT = ('How many separate objects that are clearly a "{name}" are visible in this image? Count only '
         'obvious, visible instances; do not count look-alikes or broader categories. Answer with a number only.')
OPEN = ('What is the {attribute} of the {obj} in this image? List the short answers a person might give, '
        'most likely first, comma-separated, at most four, nothing else.')
NAMES = ("What is the main object in this image? List the names a person might use for it, from most "
         'specific to most general (e.g. "baseball player, player, person"), comma-separated, at most '
         "six, nothing else.")
COLORS_DRAW = ["red", "blue", "lime", "magenta", "orange", "cyan", "yellow", "white"]


def _yn(text: str) -> Optional[str]:
    w = norm(text).split()
    return w[0] if w and w[0] in (YES, NO) else None


def draw_boxes(image, region: Box, boxes, min_side: int = 512):
    """The region's crop, upscaled to >= min_side, with boxes drawn and numbered
    from 1 (image coords in, crop coords drawn)."""
    from PIL import ImageDraw, ImageFont
    crop = image.crop(tuple(round(v) for v in region)).convert("RGB")
    k = max(1.0, min_side / max(1, min(crop.size)))
    crop = crop.resize((round(crop.width * k), round(crop.height * k)))
    d, font = ImageDraw.Draw(crop), ImageFont.load_default(size=max(16, round(min(crop.size) / 25)))
    for i, b in enumerate(boxes, 1):
        xy = [(b[0] - region[0]) * k, (b[1] - region[1]) * k, (b[2] - region[0]) * k, (b[3] - region[1]) * k]
        c = COLORS_DRAW[(i - 1) % len(COLORS_DRAW)]
        d.rectangle(xy, outline=c, width=3)
        d.text((xy[0] + 4, xy[1] + 2), str(i), fill=c, font=font, stroke_width=2, stroke_fill="black")
    return crop


class TerraLabeler:
    """A strong model grades each call by answering it itself, from the same
    crop the tool saw, without seeing the tool's scores."""
    def __init__(self, image_id: str, image, backends, model: str = GRADER):
        self.image_id, self.image, self.b, self.model = image_id, image, backends, model

    def _ask(self, image, box: Box, text: str) -> str:
        return self.b.chat(self.model, [(image, box)], text,
                           [self.image_id, [round(v) for v in box]], reasoning_effort=GRADER_EFFORT)

    def find(self, name: str, region: Box, cands=None) -> Optional[List[Box]]:
        """Count the objects in the region (no boxes drawn), then check proposed
        boxes one at a time, best score first, each drawn alone; stop once as many
        are confirmed as counted.  Objects no box frames come back as None.  A
        near-duplicate of a confirmed box (IoU >= 0.7) is skipped; same_object is
        too loose here and merged two people standing close together."""
        n = _count(self._ask(self.image, region, COUNT.format(name=name)))
        if n is None:
            return None
        found = []
        for b, _ in sorted(cands or [], key=lambda c: -c[1]):
            if len(found) >= n:
                break
            if any(iou(f, b) >= 0.7 for f in found):
                continue
            img = draw_boxes(self.image, region, [b])
            yn = _yn(self.b.chat(self.model, [(img, (0, 0, img.width, img.height))], FRAME.format(name=name),
                                 [self.image_id, [round(v) for v in region], [round(v) for v in b]],
                                 reasoning_effort=GRADER_EFFORT))
            if yn is None:
                return None
            if yn == YES:
                found.append(tuple(b))
        return found + [None] * (n - len(found))

    def exists(self, name: str, region: Box) -> Optional[str]:
        return _yn(self._ask(self.image, region, f"Is there a {name} in this image?" + YN))

    def query(self, box: Box, attribute: str, name: Optional[str] = None) -> Optional[Set[str]]:
        cat = category(attribute)
        obj = name if name and name != "image" else "main object"
        if cat is None:
            # open attribute (shape words outside SHAPES, activity, pose...): every acceptable answer
            ans = {norm(x) for x in self._ask(self.image, box, OPEN.format(attribute=attribute, obj=obj)).split(",")}
            return (ans - {""}) or None
        if cat == "name":
            # every name a person might use: "person" and "baseball player" are both right
            names = {norm(x) for x in self._ask(self.image, box, NAMES).split(",")} - {""}
            return names or None
        a = _attr(self._ask(self.image, box, f"What is the {attribute} of the {obj} in this image? "
                                             f"Answer with exactly one word from: {', '.join(sorted(cat))}."))
        return {a} if a in cat else None

    def verify(self, box: Box, prop: str, name: Optional[str] = None) -> Optional[str]:
        obj = name if name and name != "image" else "object"
        return _yn(self._ask(self.image, box, f'Does the {obj} in this image have the property "{prop}"?' + YN))

    def relate(self, box_a: Box, box_b: Box, relation: str,
               name_a: Optional[str] = None, name_b: Optional[str] = None) -> Optional[str]:
        union = (min(box_a[0], box_b[0]), min(box_a[1], box_b[1]),
                 max(box_a[2], box_b[2]), max(box_a[3], box_b[3]))
        a, b = name_a or "object", name_b or "object"
        img = draw_boxes(self.image, union, [box_a, box_b])
        text = (f"Box 1 (red) marks the {a}; box 2 (blue) marks the {b}. "
                f"Is the {a} {relation} the {b}?" + YN)
        return _yn(self.b.chat(self.model, [(img, (0, 0, img.width, img.height))], text,
                               [self.image_id, [round(v) for v in box_a], [round(v) for v in box_b]],
                               reasoning_effort=GRADER_EFFORT))


def _count(text: str) -> Optional[int]:
    m = re.search(r"\d+", text or "")
    return min(int(m.group()), 20) if m else None


class SceneGraphLabeler:
    def __init__(self, sg: dict, img_w: int, img_h: int, vocab: Set[str],
                 relevant: Optional[Set[str]] = None):
        sx, sy = img_w / sg["width"], img_h / sg["height"]
        self.vocab = vocab               # singular object names across all scene graphs
        self.relevant = relevant         # object ids the question means (None: all instances)
        self.objs = {
            oid: dict(id=oid, name=" ".join(singular(w) for w in norm(o["name"]).split()),
                      box=(o["x"] * sx, o["y"] * sy, (o["x"] + o["w"]) * sx, (o["y"] + o["h"]) * sy),
                      attrs={_attr(a) for a in o["attributes"]},
                      rels=[(norm(r["name"]), r["object"]) for r in o["relations"]])
            for oid, o in sg["objects"].items()
        }

    # ------------------------------------------------------------ matching
    def _phrase(self, phrase: str) -> List[str]:
        return [singular(w) for w in norm(phrase).split() if w not in STOP]

    def _matches(self, words: List[str], o: dict, modifiers: bool = True) -> bool:
        name = o["name"].split()
        if words[-len(name):] == name:
            mods = words[:-len(name)]
        elif words and words[-1] in PERSON and o["name"] in PERSON:
            mods = words[:-1]
        else:
            return False
        return not modifiers or all(_attr(m) in o["attrs"] for m in mods)

    def _known(self, words: List[str]) -> bool:
        return bool(words) and (words[-1] in self.vocab or words[-1] in PERSON)

    def _best(self, box: Box, name: Optional[str] = None) -> Optional[dict]:
        """The annotated object a region shows.  With the name the region was
        found under, match only objects of that name (looser rule); without
        one (region "image"), any object at IoU >= 0.5."""
        if name and name != "image":
            words = self._phrase(name)
            cands = [o for o in self.objs.values()
                     if self._matches(words, o, modifiers=False) and same_object(o["box"], box)]
        else:
            cands = [o for o in self.objs.values() if iou(o["box"], box) >= 0.5]
        return max(cands, key=lambda o: iou(o["box"], box), default=None)

    # ------------------------------------------------------------ the five labels
    def _instances(self, words, region, modifiers=True):
        return [o for o in self.objs.values()
                if self._matches(words, o, modifiers) and _inside(o["box"], region)]

    def find(self, name: str, region: Box, cands=None) -> Optional[List[Box]]:
        """Boxes `find` must catch.  With question annotations: the instances
        the question means (modifiers not required: they describe the object);
        None if the noun is present but none of its instances is meant.
        Without: every instance.  [] if the object is known but absent; None if
        the phrase can't be mapped."""
        words = self._phrase(name)
        if self.relevant is not None:
            named = self._instances(words, region, modifiers=False)
            meant = [o["box"] for o in named if o["id"] in self.relevant]
            if meant:
                return meant
            if named:
                return None
            return [] if self._known(words) else None
        hits = [o["box"] for o in self._instances(words, region)]
        return hits if hits or self._known(words) else None

    def exists(self, name: str, region: Box) -> Optional[str]:
        """A plain fact about every instance, independent of the question."""
        words = self._phrase(name)
        if self._instances(words, region):
            return YES
        return NO if self._known(words) else None

    def query(self, box: Box, attribute: str, name: Optional[str] = None) -> Optional[Set[str]]:
        o, cat = self._best(box, name), category(attribute)
        if o is None or cat is None:
            return None
        truths = {o["name"]} if cat == "name" else o["attrs"] & cat
        return truths or None

    def verify(self, box: Box, prop: str, name: Optional[str] = None) -> Optional[str]:
        o = self._best(box, name)
        if o is None:
            return None
        p = _attr(prop)
        if p in o["attrs"] or " ".join(singular(w) for w in p.split()) == o["name"]:
            return YES
        for cat in {id(c): c for c in CATEGORIES.values()}.values():
            if p in cat and o["attrs"] & cat:
                return NO
        return None

    def relate(self, box_a: Box, box_b: Box, relation: str,
               name_a: Optional[str] = None, name_b: Optional[str] = None) -> Optional[str]:
        a, b = self._best(box_a, name_a), self._best(box_b, name_b)
        if a is None or b is None:
            return None
        r = norm(relation)
        return YES if any(t == b["id"] and (r in n or n in r) for n, t in a["rels"]) else NO


if __name__ == "__main__":
    sg = {"width": 100, "height": 100, "objects": {
        "1": {"name": "cups", "x": 10, "y": 10, "w": 20, "h": 20, "attributes": ["white"],
              "relations": [{"name": "to the left of", "object": "2"}]},
        "2": {"name": "plate", "x": 40, "y": 60, "w": 30, "h": 30, "attributes": ["wooden", "big"], "relations": []},
        "3": {"name": "man", "x": 70, "y": 10, "w": 20, "h": 40, "attributes": [], "relations": []}}}
    L = SceneGraphLabeler(sg, 200, 200, {"cup", "plate", "man", "dog"})       # image is 2x the sg
    whole = (0, 0, 200, 200)
    assert L.find("the white cup", whole) == [(20, 20, 60, 60)]
    assert L.find("red cup", whole) == [] and L.find("dog", whole) == []
    assert L.find("zorblax", whole) is None and L.exists("person", whole) == YES
    assert L.query((20, 20, 60, 60), "color") == {"white"}
    assert L.query((80, 120, 140, 180), "material") == {"wood"}
    assert L.query((80, 120, 140, 180), "size") == {"large"}
    assert L.query((140, 20, 180, 100), "color") is None                    # no color attr
    assert L.verify((20, 20, 60, 60), "white") == YES and L.verify((20, 20, 60, 60), "red") == NO
    assert L.verify((80, 120, 140, 180), "small") == NO
    assert L.verify((20, 20, 60, 60), "shiny") is None
    assert L.relate((20, 20, 60, 60), (80, 120, 140, 180), "left of") == YES
    assert L.relate((80, 120, 140, 180), (20, 20, 60, 60), "left of") == NO
    # loose detector box around the cup (IoU 0.25, holds the whole cup, 4x its area):
    loose = (20, 20, 100, 100)
    assert iou((20, 20, 60, 60), loose) < 0.3 and same_object((20, 20, 60, 60), loose)
    assert L.query(loose, "color") is None                                   # no name: strict IoU
    assert L.query(loose, "color", name="cup") == {"white"}                  # found as "cup": matched
    assert L.query(loose, "color", name="plate") is None                     # wrong name: no match
    assert not same_object((20, 20, 60, 60), (0, 0, 200, 200))              # whole image is not a match
    # question means cup "1" only: find must catch that one; another cup is not required
    sg["objects"]["4"] = {"name": "cup", "x": 85, "y": 85, "w": 5, "h": 5, "attributes": [], "relations": []}
    R = SceneGraphLabeler(sg, 200, 200, {"cup", "plate", "man", "dog"}, relevant={"1"})
    assert R.find("white cup", whole) == [(20, 20, 60, 60)] and R.find("cup", whole) == [(20, 20, 60, 60)]
    assert R.find("plate", whole) is None and R.find("dog", whole) == []    # present-not-meant / absent
    assert R.exists("plate", whole) == YES                                   # exists ignores relevance
    # the strong-model grader, with canned replies
    class FakeBackends:
        calls = []
        def chat(self, model, crops, text, key, **kw):
            self.calls.append(key)
            if "How many" in text:
                return "2"
            if "red box" in text:
                assert crops[0][0].size[0] >= 512                    # the drawn crop is upscaled
                return "Yes." if key[-1] == [20, 20, 60, 60] else "no"
            return {"color": "Wooden", "activity": "Sitting, resting", "names": "Baseball player, player, Person", "Box 1": "no"}.get(
                next(k for k in ("color", "activity", "names", "Box 1", "") if k in text), "Yes.")
    from PIL import Image
    fb = FakeBackends()
    T = TerraLabeler("img", Image.new("RGB", (200, 200)), fb)
    cands = [((0, 0, 10, 10), 0.9), ((22, 22, 60, 60), 0.4), ((20, 20, 60, 60), 0.5)]
    # 2 counted; box 1 no, box (20,20,60,60) yes, (22,22,..) the same cup -> not asked; 1 missed
    assert T.find("cup", whole, cands) == [(20, 20, 60, 60), None]
    assert [k[-1] for k in fb.calls[1:]] == [[0, 0, 10, 10], [20, 20, 60, 60]]
    assert T.query((20, 20, 60, 60), "color") is None                  # "wood" is not a color
    assert T.query((20, 20, 60, 60), "activity") == {"sitting", "resting"}   # open attribute
    assert T.query((20, 20, 60, 60), "name") == {"baseball player", "player", "person"}
    assert T.exists("cup", whole) == YES and T.relate(whole, whole, "on") == NO
    assert _count("about 3 cups") == 3 and _count("none") is None
    print("labels ok")
