"""BLINK (Fu et al. 2024, arXiv 2404.12390) as a one-step test of the batch
conformal layer: every question is a single multiple-choice call, its label is
the answer itself, and its groups are the task, the option count and the image
count.  The score is S(option) = 1 - P(option), from the first answer token's logprobs.

Only val has answers (test's are "hidden"), so val is split in half over many
seeds, half to fit and half to measure.

    python src/blink.py --check
    python src/blink.py --per_task 50       # random 50 val questions per task
    python src/blink.py                     # all 1,901 val questions
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from conformal import Fitted, fit_gcp, fit_mvp

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence", "IQ_Test",
         "Jigsaw", "Multi-view_Reasoning", "Object_Localization", "Relative_Depth",
         "Relative_Reflectance", "Semantic_Correspondence", "Spatial_Relation",
         "Visual_Correspondence", "Visual_Similarity")
FAM = "fam:blink"
INSTRUCTION = "\nAnswer with the letter of your choice only, no parentheses or explanation."


def score(ex, task, backends, model):
    """One question -> a row with P(option) for every option (None if no option token came back)."""
    from backends import _sampling, option_probs
    letters = "abcdefgh"[:len(ex["choices"])]
    images = [ex[f"image_{i}"] for i in range(1, 5) if ex[f"image_{i}"] is not None]

    def run():
        content = [backends._image_msg(im, (0, 0, im.width, im.height)) for im in images]
        r = backends.client.chat.completions.create(
            model=model, max_completion_tokens=16, logprobs=True, **_sampling(model),
            top_logprobs=20 if model.startswith("gpt-4") else 5,
            messages=[{"role": "user", "content": content + [
                {"type": "text", "text": ex["prompt"] + INSTRUCTION}]}])
        backends.tokens[0] += r.usage.prompt_tokens
        backends.tokens[1] += r.usage.completion_tokens
        return option_probs(r.choices[0].logprobs.content[0].top_logprobs, list(letters))

    out = backends.cache.get_or(["blink", model, ex["idx"]], run)
    if not out:
        return None
    return dict(idx=ex["idx"], task=task, answer=ex["answer"].strip("() ").lower(),
                probs={o: dict(out).get(o, 0.0) for o in letters}, n_images=len(images))


def groups(row):
    return [FAM, row["task"], f"options={len(row['probs'])}", f"images={row['n_images']}"]


def evaluate(rows, target=0.1, m=20, splits=20):
    """method -> group -> [coverage, avg set size, % singleton, % full set] on the
    held-out halves, pooled over `splits` random splits."""
    fits = dict(raw=lambda r: Fitted({FAM: 1 - target}, target),      # trust the model's probabilities
                split=lambda r: fit_gcp([(s, gs[:1]) for s, gs in r], target),
                gcp=lambda r: fit_gcp(r, target),
                mvp=lambda r: fit_mvp(r, target, m))
    tally = {name: defaultdict(lambda: [0, 0, 0, 0, 0]) for name in fits}
    for seed in range(splits):
        shuffled = random.Random(seed).sample(rows, len(rows))
        half = len(shuffled) // 2
        cal = [(1 - r["probs"][r["answer"]], groups(r)) for r in shuffled[:half]]
        for name, fit in fits.items():
            f = fit(cal)
            for r in shuffled[half:]:
                gs = groups(r)
                q = f.threshold("blink", gs)
                kept = [o for o, p in r["probs"].items() if 1 - p <= q]
                for g in gs:
                    t = tally[name][g]
                    t[0] += 1
                    t[1] += r["answer"] in kept
                    t[2] += len(kept)
                    t[3] += len(kept) == 1
                    t[4] += len(kept) == len(r["probs"])
    return {name: {g: [t[1] / t[0], t[2] / t[0], t[3] / t[0], t[4] / t[0]] for g, t in by.items()}
            for name, by in tally.items()}


def report(rows, res):
    by = defaultdict(list)
    for r in rows:
        for g in groups(r):
            by[g].append(r)
    top1 = lambda rs: sum(max(r["probs"], key=r["probs"].get) == r["answer"] for r in rs) / len(rs)
    names = sorted(by, key=lambda g: (g != FAM, ":" in g or "=" in g, g))
    head = ["group", "n", "top-1", "guess"] + [f"{n} cov" for n in res] + [f"{n} size" for n in res] + ["options"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in names:
        rs = by[g]
        guess = sum(1 / len(r["probs"]) for r in rs) / len(rs)
        opts = sum(len(r["probs"]) for r in rs) / len(rs)
        lines.append("| " + " | ".join(
            [g, str(len(rs)), f"{top1(rs):.2f}", f"{guess:.2f}"]
            + [f"{res[n][g][0]:.2f}" for n in res] + [f"{res[n][g][1]:.2f}" for n in res]
            + [f"{opts:.2f}"]) + " |")
    lines += ["", "| method | singleton | full set | lowest group coverage |", "|---|---|---|---|"]
    for n in res:
        low = min(res[n], key=lambda g: res[n][g][0])
        lines.append(f"| {n} | {res[n][FAM][2]:.2f} | {res[n][FAM][3]:.2f} | {res[n][low][0]:.2f} ({low}) |")
    return "\n".join(lines)


def check():
    rng = random.Random(0)
    rows = []
    for i in range(1000):
        easy = i % 2 == 0
        p = rng.uniform(0.7, 1.0) if easy else rng.uniform(0.0, 1.0)
        rows.append(dict(task="easy" if easy else "hard", answer="a", n_images=1,
                         probs=dict(a=p, b=1 - p)))
    res = evaluate(rows, splits=3)
    assert abs(res["gcp"]["hard"][0] - 0.9) < 0.05 and abs(res["gcp"]["easy"][0] - 0.9) < 0.05, res["gcp"]
    assert abs(res["mvp"]["hard"][0] - 0.9) < 0.05, res["mvp"]
    assert res["split"]["hard"][0] < 0.88 < res["split"]["easy"][0], res["split"]   # pooled: miscovers both ways
    print("blink ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--per_task", type=int, default=0, help="random N val questions per task (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mvp_m", type=int, default=20)
    args = ap.parse_args()
    if args.check:
        return check()

    from datasets import load_dataset
    from dotenv import load_dotenv
    from backends import Backends
    load_dotenv(ROOT / ".env")
    backends = Backends(ROOT / "data" / "cache")
    rows, failed = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        for task in TASKS:
            ds = load_dataset("BLINK-Benchmark/BLINK", task, split="val")
            if args.per_task:
                ds = ds.shuffle(seed=2025).select(range(min(args.per_task, len(ds))))
            got = list(pool.map(lambda ex: score(ex, task, backends, args.model), ds))
            rows += [r for r in got if r]
            failed += sum(r is None for r in got)
            print(f"{task}: {sum(r is not None for r in got)}/{len(got)}  tokens {backends.tokens}", flush=True)

    out = ROOT / "runs" / "blink" / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} scored, {failed} without an option token")
    text = report(rows, evaluate(rows, m=args.mvp_m))
    (out / "review.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
