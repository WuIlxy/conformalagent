"""Raw model scores, disk-cached by (image id, box, text, model).
  detect     OWLv2 boxes + scores (from quasar's epic/imgpatch_local)
  choose     gpt-6-luna Decisions API: P(option) for every option of a closed label set
  predicate  gpt-6-luna Decisions API: P(yes)
  ask        free-text answers with self-reported scores, for open queries
  chat       a plain cached chat call (the grader uses it)
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

from answer_dsl import Box, norm
from labels import iou

DETECT_FLOOR = 0.05      # OWL candidates below this never enter any set
MAX_BOXES = 20
DECIDE_MODEL = "gpt-6-luna"   # the only model the Decisions API serves

ASK_PROMPT = """You are given an image and a question.
        Your task is to identify *all possible answers* to the question based on the image,
        and assign a confidence score (between 0 and 1) to each answer.
        The scores must sum to 1. For each possibility, keep the description concise---no explanation.

        Return your answer in the following JSON format:
        [
        {{ "answer": "ANSWER_1", "score": 0.65 }},
        {{ "answer": "ANSWER_2", "score": 0.35 }}
        ]

        Now, answer the question: \"{query}\""""


class Cache:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def get_or(self, key_parts, fn):
        key = hashlib.md5(json.dumps(key_parts).encode()).hexdigest()
        path = self.root / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        out = fn()
        if out is not None:                       # don't cache failures
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(out))
            os.replace(tmp, path)
        return out


def _nms(cands, thr=0.5):
    keep = []
    for box, s in sorted(cands, key=lambda c: -c[1]):
        if all(iou(box, k) < thr for k, _ in keep):
            keep.append((box, s))
    return keep


class Backends:
    def __init__(self, cache_dir: Path, vlm_model: str = DECIDE_MODEL):
        from openai import OpenAI
        self.client = OpenAI()
        self.vlm_model = vlm_model
        self.cache = Cache(cache_dir)
        self.tokens = [0, 0]                  # tool [input, output] tokens, uncached calls this process
        self.chat_tokens = {}                 # model -> [input, output] for chat() (the grader)
        self._owl = None
        self._owl_lock = threading.Lock()     # ponytail: one global lock on OWL; fine at ~1 call/s

    def _detector(self):
        if self._owl is None:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            import torch
            from transformers import pipeline
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._owl = pipeline(model="google/owlv2-base-patch16-ensemble",
                                 task="zero-shot-object-detection", device=device)
        return self._owl

    def detect(self, image_id: str, image, box: Box, name: str) -> List[Tuple[Box, float]]:
        """Candidate boxes (image coords) with OWL scores, NMS'd, best first."""
        def run():
            with self._owl_lock:
                preds = self._detector()(image.crop(tuple(round(v) for v in box)), candidate_labels=[name],
                                         threshold=DETECT_FLOOR)
            x0, y0 = box[0], box[1]
            cands = [((p["box"]["xmin"] + x0, p["box"]["ymin"] + y0,
                       p["box"]["xmax"] + x0, p["box"]["ymax"] + y0), float(p["score"]))
                     for p in preds]
            return _nms(cands)[:MAX_BOXES]
        out = self.cache.get_or(["detect", image_id, [round(v) for v in box], name], run)
        return [(tuple(b), s) for b, s in out]

    @staticmethod
    def _data_url(image, box: Box) -> str:
        crop = image.crop(tuple(round(v) for v in box))
        if min(crop.size) < 32:                           # tiny crops: upscale for the VLM
            crop = crop.resize((max(crop.width, 32), max(crop.height, 32)))
        buf = BytesIO()
        crop.convert("RGB").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @classmethod
    def _image_msg(cls, image, box: Box) -> dict:
        return {"type": "image_url", "image_url": {"url": cls._data_url(image, box)}}

    def _decide(self, image, box: Box, question: dict):
        """One Decisions API question about the crop; its answer dict, or None on refusal.
        The installed SDK has no client.decisions yet, so post to the endpoint."""
        r = self.client.post("/decisions", cast_to=object, body={
            "model": DECIDE_MODEL, "questions": [dict(question, name="q")],
            "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": self._data_url(image, box)}]}]})
        self.tokens[0] += r["usage"]["input_tokens"]
        a = r["answers"][0]
        return None if a["type"] == "refusal" else a

    def choose(self, image_id: str, image, box: Box, question: str, options) -> List[Tuple[str, float]]:
        """[(option, P)] over every option, best first ([] on refusal).  The API
        reports probabilities to 0.01, so options under 0.005 tie at 0."""
        opts = sorted(options)
        def run():
            a = self._decide(image, box, {"type": "choice", "instructions": question,
                                          "choices": [{"value": o} for o in opts]})
            if a is None:
                return None
            p = {x["value"]: x["probability"] for x in a["probabilities"]}
            return sorted(([o, p.get(o, 0.0)] for o in opts), key=lambda x: -x[1])
        out = self.cache.get_or(["choose", DECIDE_MODEL, image_id, [round(v) for v in box], question, opts], run)
        return [(a, p) for a, p in (out or [])]

    def predicate(self, image_id: str, image, box: Box, question: str):
        """P(yes) for a yes/no question about the crop (None on refusal)."""
        def run():
            a = self._decide(image, box, {"type": "predicate", "instructions": question})
            return None if a is None else {"p": a["probability"]}
        out = self.cache.get_or(["predicate", DECIDE_MODEL, image_id, [round(v) for v in box], question], run)
        return out["p"] if out else None

    def chat(self, model: str, crops, text: str, key, **kw) -> str:
        """A cached chat call over [(image, box)] crops; '' if the model said nothing."""
        def run():
            r = self.client.chat.completions.create(model=model, **kw, messages=[{"role": "user", "content": [
                *[self._image_msg(im, b) for im, b in crops], {"type": "text", "text": text}]}])
            t = self.chat_tokens.setdefault(model, [0, 0])
            t[0] += r.usage.prompt_tokens
            t[1] += r.usage.completion_tokens
            return {"text": (r.choices[0].message.content or "").strip()}
        return self.cache.get_or(["chat", model, *key, text], run)["text"]

    def ask(self, image_id: str, image, box: Box, question: str,
            model: str = None) -> List[Tuple[str, float]]:
        """VLM answers with confidences, normalized and summing to 1 ([] on parse failure)."""
        model = model or self.vlm_model
        def run():
            r = self.client.chat.completions.create(
                model=model, **_sampling(model),
                messages=[{"role": "user", "content": [
                    self._image_msg(image, box),
                    {"type": "text", "text": ASK_PROMPT.format(query=question)}]}])
            self.tokens[0] += r.usage.prompt_tokens       # cache misses only: what was paid
            self.tokens[1] += r.usage.completion_tokens
            return parse_answers(r.choices[0].message.content or "")
        out = self.cache.get_or(["ask", model, image_id, [round(v) for v in box], question], run)
        return [(a, p) for a, p in (out or [])]


def _sampling(model: str) -> dict:
    """gpt-4.x take temperature; gpt-5.x/6 need reasoning off (for logprobs and tools)."""
    return {"temperature": 0} if model.startswith("gpt-4") else {"reasoning_effort": "none"}


def option_probs(top, options):
    """First-token logprobs -> [(option, p)] summing to 1.  A token counts for the
    option it is a prefix of (shortest such option on ties: "tan" before "teal")."""
    import math
    probs = {}
    for t in top:
        tok = t.token.strip().lower()
        hits = [o for o in options if tok and o.startswith(tok)]
        if hits:
            o = min(hits, key=len)
            probs[o] = probs.get(o, 0.0) + math.exp(t.logprob)
    total = sum(probs.values())
    return sorted(([o, p / total] for o, p in probs.items()), key=lambda x: -x[1]) if total else None


def parse_answers(text: str):
    m = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    merged = {}
    for it in items:
        a = norm(it.get("answer", ""))
        if a:
            merged[a] = merged.get(a, 0.0) + max(0.0, float(it.get("score", 0)))
    total = sum(merged.values())
    if total <= 0:
        return None
    return sorted(([a, p / total] for a, p in merged.items()), key=lambda x: -x[1])


if __name__ == "__main__":
    assert parse_answers('x [ {"answer": "Grey", "score": 0.5}, {"answer": "gray.", "score": 0.3},'
                         ' {"answer": "white", "score": 0.2} ] y') == [["gray", 0.8], ["white", 0.2]]
    assert parse_answers("no json") is None
    from types import SimpleNamespace as T
    top = [T(token="Red", logprob=-0.1), T(token=" red", logprob=-3.0), T(token="ta", logprob=-2.0),
           T(token="xyz", logprob=-1.0)]
    out = dict(option_probs(top, ["red", "tan", "teal"]))
    assert set(out) == {"red", "tan"} and out["red"] > 0.8 and abs(sum(out.values()) - 1) < 1e-9
    assert _nms([((0, 0, 10, 10), 0.9), ((1, 1, 10, 10), 0.5), ((50, 50, 60, 60), 0.3)]) == \
        [((0, 0, 10, 10), 0.9), ((50, 50, 60, 60), 0.3)]
    print("backends ok")
