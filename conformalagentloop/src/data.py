"""GQA val: the 1000 questions quasar sampled (val_n1000_seed2025, copied here),
images from lmms-lab/GQA, scene graphs from the official GQA release."""
from __future__ import annotations
import json
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

from answer_dsl import norm, singular

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW, CLEAN, IMAGES, CACHE = DATA / "raw", DATA / "clean", DATA / "images", DATA / "cache"
QUESTIONS = CLEAN / "gqa_val_n1000_seed2025.json"
SG_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"


def questions():
    """[{qid, imageId, question, answer}] in the file's (already random) order."""
    return [dict(qid=k, **v) for k, v in json.loads(QUESTIONS.read_text()).items()]


@lru_cache(maxsize=1)
def scene_graphs() -> dict:
    path = RAW / "val_sceneGraphs.json"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        zpath = RAW / "sceneGraphs.zip"
        print(f"downloading {SG_URL} (~45 MB)")
        urllib.request.urlretrieve(SG_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extract("val_sceneGraphs.json", RAW)
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def object_vocab() -> frozenset:
    """Singular object names (and their head nouns) across all val scene graphs."""
    out = set()
    for g in scene_graphs().values():
        for o in g["objects"].values():
            words = [singular(w) for w in norm(o["name"]).split()]
            if words:
                out.update({" ".join(words), words[-1]})
    return frozenset(out)


def _instructions():
    """GQA val questions with annotations (lmms-lab/GQA val_all_instructions, ~266 MB, HF cache)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    return pa.concat_tables(
        pq.read_table(hf_hub_download("lmms-lab/GQA", f"val_all_instructions/val-0000{i}-of-00002.parquet",
                                      repo_type="dataset"),
                      columns=["id", "imageId", "question", "answer", "annotations"])
        for i in range(2))


def _objects(a) -> list:
    return sorted({oid.strip() for part in ("question", "answer", "fullAnswer")
                   for x in (a.get(part) or []) for oid in x["value"].split(",")})


def question_objects() -> dict:
    """qid -> scene-graph object ids the question/answer refer to (GQA's own
    annotations), for the 1000-question file.  Labeller-only knowledge: the
    agent never sees these.  (cal_pool questions carry theirs as "objects".)"""
    path = CLEAN / "question_objects.json"
    if not path.exists():
        import pyarrow as pa
        import pyarrow.compute as pc
        t = _instructions()
        t = t.filter(pc.is_in(t["id"], value_set=pa.array(sorted(q["qid"] for q in questions()))))
        out = {r["id"]: _objects(r["annotations"]) for r in t.to_pylist()}
        CLEAN.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return json.loads(path.read_text())


def cal_pool(n: int, seed: int = 2025) -> list:
    """n more val questions for calibration, none on an image the 1000-question
    file uses, drawn the same way (from all of val, not only "balanced").
    Cached in data/clean/."""
    path = CLEAN / f"cal_pool_n{n}_seed{seed}.json"
    if not path.exists():
        import random
        import pyarrow as pa
        import pyarrow.compute as pc
        t = _instructions()
        used = pa.array(sorted({q["imageId"] for q in questions()}))
        t = t.filter(pc.invert(pc.is_in(t["imageId"], value_set=used)))
        idx = random.Random(seed).sample(range(t.num_rows), n)
        out = [dict(qid=r["id"], imageId=r["imageId"], question=r["question"], answer=r["answer"],
                    objects=_objects(r["annotations"])) for r in t.take(idx).to_pylist()]
        CLEAN.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return json.loads(path.read_text())


def ensure_images(image_ids) -> None:
    """Save the needed val images to data/images/<id>.jpg (one-time ~1.5 GB HF download)."""
    img_dir = IMAGES
    missing = {i for i in image_ids if not (img_dir / f"{i}.jpg").exists()}
    if not missing:
        return
    import datasets
    img_dir.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset("lmms-lab/GQA", "val_all_images", split="val")
    for idx, i in enumerate(ds["id"]):
        if i in missing:
            ds[idx]["image"].convert("RGB").save(img_dir / f"{i}.jpg", quality=95)
            missing.discard(i)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not in lmms-lab/GQA val: {sorted(missing)[:5]}")


def load_image(image_id: str):
    from PIL import Image
    return Image.open(IMAGES / f"{image_id}.jpg").convert("RGB")
