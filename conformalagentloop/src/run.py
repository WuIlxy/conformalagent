"""Calibrate (point-estimate agent -> true-label scores + each call's groups),
then test with the conformal agent.  --method picks the thresholds: per-question
ACI from alpha_1 = alpha* (aci), or selector-aware BatchGCP / BatchMVP fitted on
the calibration calls (gcp, mvp).  README §4, §4b, §8-9.

Every call is graded by gpt-6.1-terra (--grader terra) or GQA's scene graph
(--grader scene_graph).  A find call grades each instance on its own.

    python src/run.py --mock                      # no key, no downloads; asserts
    python src/run.py --n_cal 200 --n_test 800    # real run (needs OPENAI_API_KEY in .env)
    python src/run.py --method mvp --cal_pool --n_cal 1000 --n_test 1000
    python src/run.py --method mvp --cal runs/gqa/<models>/<ts> --n_test 800    # reuse a calibration
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from agent import run_question
from answer_dsl import WILD, covers
from conformal import ACI, CAL_KEY, fit_gcp, fit_mvp
from tools import Env

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ["find", "query", "verify", "exists", "relate"]


def units(c):
    """A labelled call's coverage events as [(true score, covered)]: one per
    instance for find (empty if it has none), one per call otherwise.  Runs
    made before per-instance find grading have no "instances": one event."""
    return [tuple(i) for i in c["instances"]] if "instances" in c else [(c["true_score"], c["covered"])]


def run_one(q, ctx, cp, phase, max_steps):
    t0 = time.time()
    trace = dict(phase=phase, qid=q["qid"], question=q["question"], gt=q["answer"],
                 alpha_start=cp.alpha if cp else None)
    env = None
    try:
        image = ctx.image(q["imageId"])
        env = Env(q["imageId"], image, ctx.backends, ctx.labeler(q, image), cp, q["question"])
        res = run_question(q["question"], env, ctx.policy, max_steps)
        ans = res["answer"]
        vals = frozenset(ans["values"]) if ans else frozenset()
        covered = bool(ans) and covers(vals, q["answer"])
        trace.update(res, answer_covered=covered,
                     correct=covered and len(vals) == 1 and WILD not in vals,
                     program=env.program.render())
    except Exception as e:                       # one bad question must not kill a long run
        trace.update(answer=None, answer_covered=False, correct=False, stopped="error",
                     error=f"{type(e).__name__}: {e}", tb=traceback.format_exc(limit=3))
    calls = [dict(phase=phase, qid=q["qid"], **c) for c in (env.calls if env else [])]
    labeled = [c for c in calls if c["labeled"]]
    trace.update(n_calls=len(calls), n_labeled=len(labeled),
                 any_miss=any(not c["covered"] for c in labeled) if labeled else None,
                 alpha_end=cp.alpha if cp else None, seconds=round(time.time() - t0, 1))
    return trace, calls


def run_phase(phase, qs, ctx, args, make_cp=None):
    """make_cp=None: point-estimate agent.  Otherwise make_cp() -> the ACI or
    Fitted thresholds for one question (one shared ACI with --no_reset)."""
    out = Path(args.out_dir)
    shared = make_cp() if (make_cp and args.no_reset) else None

    def job(q):
        cp = shared if shared else (make_cp() if make_cp else None)
        return run_one(q, ctx, cp, phase, args.max_steps)

    workers = 1 if shared else args.workers
    traces, calls = [], []
    with ThreadPoolExecutor(workers) as pool, \
            open(out / "traces.jsonl", "a") as ft, open(out / "calls.jsonl", "a") as fc:
        for i, (tr, cs) in enumerate(pool.map(job, qs), 1):
            traces.append(tr); calls.extend(cs)
            ft.write(json.dumps(tr) + "\n"); ft.flush()
            fc.writelines(json.dumps(c) + "\n" for c in cs); fc.flush()
            ok = "✓" if tr["answer_covered"] else ("!" if tr["stopped"] == "error" else "✗")
            print(f"[{phase} {i}/{len(qs)}] {ok} {tr['qid']} {tr['question'][:60]!r} "
                  f"-> {tr['answer']['shown'] if tr.get('answer') else tr['stopped']} (gt {tr['gt']})")
    return traces, calls


def fit_thresholds(cal_calls, args):
    """Fit GCP and MVP on most calibration questions; report per-group coverage
    on the held-out rest (no agent shift here: same calls, unseen questions)."""
    lab = [c for c in cal_calls if c["labeled"]]
    if lab and "groups" not in lab[0]:
        raise SystemExit("this calibration has no call groups (made before selectors); rerun it")
    held = lambda c: zlib.crc32(c["qid"].encode()) % 1000 < args.holdout * 1000
    fit_c, hold_c = [c for c in lab if not held(c)], [c for c in lab if held(c)]
    rows = [(sc, c["groups"]) for c in fit_c for sc, _ in units(c)]
    if not rows:
        raise SystemExit("no labelled calibration calls to fit on (did every question error?)")
    fits = {"split": ACI(cal_scores_from(fit_c), args.alpha, 0.0),
            "gcp": fit_gcp(rows, args.alpha, args.min_count),
            "mvp": fit_mvp(rows, args.alpha, args.mvp_m, args.mvp_tol, args.min_count)}
    groups = sorted({g for c in lab for g in c["groups"]})
    L = [f"## threshold fit ({len(fit_c)} fit / {len(hold_c)} held-out labelled calls, {len(rows)} fit rows, "
         f"MVP: {len(fits['mvp'].patches)} patches)", "",
         "held-out coverage per group (target " + f"{1 - args.alpha:.2f}" + "):", "",
         "| group | fit n | held-out n | split (per tool) | GCP | MVP |", "|---|---|---|---|---|---|"]
    for g in groups:
        hs = [c for c in hold_c if g in c["groups"]]
        if len(hs) < 10:
            continue
        cov = lambda f: _mean([float(sc <= f.threshold(c["tool"], c["groups"])) for c in hs for sc, _ in units(c)])
        L.append(f"| {g} | {sum(g in c['groups'] for c in fit_c)} | {len(hs)} | "
                 f"{cov(fits['split'])} | {cov(fits['gcp'])} | {cov(fits['mvp'])} |")
    return fits[args.method], "\n".join(L) + "\n"


def cal_scores_from(calls):
    cal = {}
    for c in calls:
        if c["labeled"]:
            cal.setdefault(CAL_KEY[c["tool"]], []).extend(sc for sc, _ in units(c))
    return cal


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return f"{statistics.mean(xs):.3f}" if xs else "–"


def summarize(phase, traces, calls, args):
    L = [f"## {phase} ({len(traces)} questions)", ""]
    if phase == "test":
        how = (f"γ = {args.gamma}, " + ("one ACI stream (--no_reset)" if args.no_reset else
                                         "ACI reset per question")) if args.method == "aci" else \
            f"static {args.method.upper()} thresholds over tool families + selectors"
        L.append(f"target per-call coverage 1-α* = {1 - args.alpha:.2f}, {how}")
        L.append("")
    L += ["coverage counts each find instance on its own", "",
          "| tool | calls | labeled | coverage | mean set size | ⊤ sets |", "|---|---|---|---|---|---|"]
    for tool in TOOLS + ["all"]:
        cs = [c for c in calls if tool in ("all", c["tool"])]
        if not cs:
            continue
        lab = [c for c in cs if c["labeled"]]
        L.append(f"| {tool} | {len(cs)} | {len(lab)} | {_mean([float(ok) for c in lab for _, ok in units(c)])} | "
                 f"{_mean([len(c['values']) for c in cs if WILD not in c['values']])} | "
                 f"{sum(WILD in c['values'] for c in cs)} |")
    answered = [t for t in traces if t.get("answer")]
    L += ["", "| trace metric | value |", "|---|---|",
          f"| answered | {len(answered)}/{len(traces)} |",
          f"| errors | {sum(t['stopped'] == 'error' for t in traces)} |",
          f"| answer coverage (GQA answer ∈ set) | {_mean([float(t['answer_covered']) for t in traces])} |",
          f"| accuracy (singleton & correct) | {_mean([float(t['correct']) for t in traces])} |",
          f"| mean answer set size | {_mean([len(t['answer']['values']) for t in answered])} |",
          f"| mean answer bound | {_mean([t['answer']['bound'] for t in answered])} |",
          f"| P(any labeled miss in trace) | {_mean([float(t['any_miss']) for t in traces if t['any_miss'] is not None])} |",
          f"| policy-logic errors (closure all covered, answer missed) | "
          f"{sum(t['answer']['closure_all_covered'] and not t['answer_covered'] for t in answered)} |",
          f"| mean steps | {_mean([t['steps'] for t in traces if 'steps' in t])} |"]
    if phase == "test":
        L.append(f"| mean α at end of trace | {_mean([t['alpha_end'] for t in traces])} |")
    # the selector objective: P(true label in set | selector picked a call)
    groups = sorted({g for c in calls for g in c.get("groups", [])})
    L += ["", "| group | calls | labeled | coverage | mean set size |", "|---|---|---|---|---|"]
    for g in groups:
        cs = [c for c in calls if g in c["groups"]]
        lab = [c for c in cs if c["labeled"]]
        if len(lab) >= 10:
            L.append(f"| {g} | {len(cs)} | {len(lab)} | {_mean([float(ok) for c in lab for _, ok in units(c)])} | "
                     f"{_mean([len(c['values']) for c in cs if WILD not in c['values']])} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--n_cal", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=800)
    ap.add_argument("--method", choices=["aci", "gcp", "mvp"], default="aci")
    ap.add_argument("--cal", help="reuse the calibrate phase of an earlier run dir (same --cal_pool); "
                                  "skips calibrate")
    ap.add_argument("--cal_pool", action="store_true",
                    help="calibrate on n_cal other val questions (images not in the 1000-question "
                         "file) and test on that file's first n_test")
    ap.add_argument("--grader", choices=["terra", "scene_graph"], default="terra",
                    help="who labels each call: a strong model answering it, or GQA's scene graph")
    ap.add_argument("--grader_model", default="gpt-6.1-terra")
    ap.add_argument("--find_label", choices=["all", "relevant"], default="all",
                    help="scene_graph grader only: find's instances are every instance of the name "
                         "(all) or the ones the question means (relevant, uses GQA's annotations)")
    ap.add_argument("--holdout", type=float, default=0.2, help="share of calibration questions held "
                                                                "out to check the GCP/MVP fit")
    ap.add_argument("--min_count", type=int, default=20, help="groups with fewer rows get no offset")
    ap.add_argument("--mvp_m", type=int, default=20, help="MVP threshold grid k/m")
    ap.add_argument("--mvp_tol", type=float, default=0.002, help="MVP calibration-error tolerance")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--no_reset", action="store_true", help="one ACI stream across all test questions")
    ap.add_argument("--policy_model", default="gpt-6-luna")
    ap.add_argument("--vlm_model", default="gpt-6-luna",
                    help="open queries only; closed queries and yes/no use gpt-6-luna's Decisions API")
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    models = "mock" if args.mock else "__".join(
        [args.policy_model, args.vlm_model, args.grader_model if args.grader == "terra" else "scene-graph"])
    args.out_dir = str(ROOT / "runs" / "gqa" / models / time.strftime("%Y%m%d-%H%M%S"))
    Path(args.out_dir).mkdir(parents=True)
    (Path(args.out_dir) / "config.json").write_text(json.dumps(vars(args), indent=1))

    if args.mock:
        import mock
        ctx, qs = mock.context(), mock.QUESTIONS
        cal_qs, test_qs = qs[:2], qs[2:]
    else:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        import data
        from agent import openai_policy
        from backends import Backends
        from labels import SceneGraphLabeler, TerraLabeler
        qs = data.questions()
        if args.cal_pool:
            cal_qs = [] if args.cal else data.cal_pool(max(args.n_cal, 2500))[:args.n_cal]   # same questions as earlier 1000-cal runs
            test_qs = qs[:args.n_test]
        else:
            cal_qs = [] if args.cal else qs[:args.n_cal]
            test_qs = qs[args.n_cal:args.n_cal + args.n_test]
        data.ensure_images({q["imageId"] for q in cal_qs + test_qs})
        backends = Backends(data.CACHE, args.vlm_model)
        if args.grader == "terra":
            labeler = lambda q, img: TerraLabeler(q["imageId"], img, backends, args.grader_model)
        else:
            sgs, vocab = data.scene_graphs(), data.object_vocab()
            meant = {**data.question_objects(), **{q["qid"]: q["objects"] for q in cal_qs}} \
                if args.find_label == "relevant" else {}
            labeler = lambda q, img: SceneGraphLabeler(sgs[q["imageId"]], img.width, img.height, vocab,
                                                       set(meant[q["qid"]]) if meant else None)
        ctx = SimpleNamespace(backends=backends, policy=openai_policy(args.policy_model),
                              image=data.load_image, labeler=labeler)

    summary = ""
    if args.cal:
        cal_calls = [c for c in map(json.loads, open(Path(args.cal) / "calls.jsonl"))
                     if c["phase"] == "calibrate"]
    else:
        cal_traces, cal_calls = run_phase("calibrate", cal_qs, ctx, args)
        summary += summarize("calibrate", cal_traces, cal_calls, args)
    cal = cal_scores_from(cal_calls)
    (Path(args.out_dir) / "cal_scores.json").write_text(json.dumps(cal))
    print("calibration scores per family:", {k: len(v) for k, v in cal.items()})

    if args.mock:
        cal_mock = cal
        cal = mock.CAL                     # fixed scores so the test phase is deterministic
    if args.method == "aci":
        make_cp = lambda: ACI(cal, args.alpha, args.gamma)
    else:
        fitted, report = fit_thresholds(cal_calls, args)
        summary += "\n" + report
        (Path(args.out_dir) / "thresholds.json").write_text(json.dumps(vars(fitted), indent=1))
        make_cp = lambda: fitted
    test_traces, test_calls = run_phase("test", test_qs, ctx, args, make_cp)
    summary += "\n" + summarize("test", test_traces, test_calls, args)

    if not args.mock:
        from agent import policy_tokens
        summary += (f"\ntokens (input, output), uncached calls only: policy {policy_tokens}, "
                    f"tools {ctx.backends.tokens}, grader {ctx.backends.chat_tokens}\n")
    (Path(args.out_dir) / "summary.md").write_text(summary)
    print("\n" + summary + f"\nlogs: {args.out_dir}")
    if args.mock and args.method == "aci":
        mock.check(cal_mock, test_traces, test_calls)
        print("mock self-check ok")


if __name__ == "__main__":
    main()
