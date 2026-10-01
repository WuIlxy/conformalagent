# conformalagentloop (v2)

A ReAct agent for GQA whose perception tools return **conformal prediction
sets**, with the set level adapted *within each question* by **Adaptive
Conformal Inference** (ACI; Gibbs & Candès, *Adaptive Conformal Inference Under
Distribution Shift*, NeurIPS 2021), and whose final answer is a **one-line DSL
spec over the observed sets**, so answer coverage follows from tool coverage.

> Status: **design spec**. This README is the contract; code lands after it is
> reviewed. v1 (mocked skeleton) is at commit `779cd7f`.
> Setting: GQA val, ground-truth labels for intermediate tool calls come from
> the GQA scene graphs. Later datasets: a strong model does that labelling
> (§6 is the single swap point).

---

## 1. One question, end to end

```
question + image
   │
   ▼
policy LLM (function calling) ── tool call ──► backend scores ──► set Ĉ_t = {y : S(y) ≤ Q̂_k(1-α_t)}
   ▲                                                                 │
   │   observation o_t: the set, its id, α_t                        │ scene-graph label y_t
   └─────────────────────────────────────────────────────────────────┤ err_t = 1[y_t ∉ Ĉ_t]
                                                                     ▼
                                                  α_{t+1} = α_t + γ(α* − err_t)   (ACI, per question)
   ...
policy ── answer("yesno(same(o2, o4))") ──► deterministic eval over o2, o4 ──► answer set + α budget
```

Every tool call is one ACI step. Feedback is **immediate**: the label for call
t is known before call t+1, so the update lands before the next set is built.

## 2. ACI, as used here

From the paper (§2, §4), for a stream of calls t = 1..T:

- Set at level α_t: `Ĉ_t(α_t) = { y : S_t(X_t, y) ≤ Q̂_t(1 − α_t) }`, where
  `Q̂_t(p) = inf{ s : (1/|D_cal|) Σ_r 1{S_r ≤ s} ≥ p }` over calibration scores.
- Update: `α_{t+1} = α_t + γ(α* − err_t)`, with `err_t = 1[Y_t ∉ Ĉ_t(α_t)]`.
  A miss lowers α (bigger sets next); a hit raises it slowly.
- **No clamping.** α_t is allowed to leave [0, 1] (it stays in [−γ, 1+γ],
  Lemma 4.1). Convention: α_t ≤ 0 ⇒ Ĉ_t = full label space (⊤);
  α_t ≥ 1 ⇒ Ĉ_t = ∅. (v1 clamped; that changes the dynamics, so it's gone.)
- Guarantee (Prop. 4.1), deterministic, any sequence:
  `| (1/T) Σ err_t − α* | ≤ (max(α_1, 1 − α_1) + γ) / (γ T)`.

The paper lets `S_t` and `Q̂_t` change with t. We use that: **one shared level
α_t per question, one score function + calibration set per tool k**. Call t
uses tool k_t and threshold `Q̂_{k_t}(1 − α_t)`. So a miss on `find` widens the
next `query`. A tool is called only once or twice per question, so separate
per-tool levels would barely move.

**Per question, reset.** Questions are treated as iid, so each (question, image)
runs its own ACI from α_1. With T ≈ 3–6 calls, Prop. 4.1 is vacuous inside one
trace. What it provides there is the behaviour: the trace gets more
conservative after an observed error on *this* image. The guarantee comes
from the outer layer (§4), which is valid *because* questions are iid.

Calls whose label can't be resolved (§6) are **unlabeled**: there's no update
and α_t carries forward. They are logged and counted.

Defaults: `α* = 0.1`, `γ = 0.05`. The paper uses 0.005 for long streams; a
trace this short needs a larger step to react at all. γ is swept in experiments.

## 3. Tools and scores

Every tool call produces observation `o_t` with a conformal set and records
`α_t` and its dependencies (the region ids it used). Backends are copied from
the quasar artifact. The quasar folder isn't tracked, so nothing is imported
from it.

| tool | set over | score `S(y)` | backend (copied from) |
|---|---|---|---|
| `find(name, region="image")` | boxes → new region ids `r_i` | `1 − owl_score(box)` | OWLv2 (`imgpatch_local`, `imgpatch_conformal.find`) |
| `query(region, attribute)` | values, `attribute ∈ {color, material, name, …}` | `1 − p_vlm(y)` (0 if not listed → S = 1) | gpt-4o-mini JSON answers+scores (`imgpatch_conformal._do_model_call_real`) |
| `verify(region, property)` | `{yes, no}` | `S(yes) = 1 − p_yes`, `S(no) = p_yes` | same prompt (`verify_property`) |
| `exists(name, region="image")` | `{yes, no}` | as `verify` | same prompt |
| `relate(region_a, region_b, relation)` | `{yes, no}` | as `verify`, on the union-box crop | same prompt |

- `find`'s set covers iff **every** ground-truth instance is matched (IoU ≥ 0.5)
  by a box in the set. Its true-label score is `max over instances of
  (1 − best matching owl_score)` (1 if an instance was never detected). So
  `Q̂(1−α)` is "the cutoff needed to catch all instances".
- `exists` is its own yes/no tool rather than "is the find set non-empty".
  Conformal find sets are supersets, so non-empty proves nothing.
- `⊤` (full set) appears when α_t ≤ 0: for `query` it means "any value".
- VLM calls are disk-cached by `(image crop, prompt)`, like `imgpatch_conformal`.
- GQA has no counting questions, so there's no `count`.

## 4. Choosing α_1: outer layer over iid questions

`α_1` is chosen on a calibration split **of this agent's own traces**. The
agent's sets change which calls it makes next, so calibration from fixed
annotations wouldn't reflect the test-time call distribution.

Three disjoint splits of the 1000 sampled GQA val questions
(`val_n1000_seed2025` ids, reused from quasar):

| phase | split | what runs | produces |
|---|---|---|---|
| 0. scores | `score` (200) | point-estimate agent (top-1 values, OWL > 0.1), no sets | per-tool `D_cal,k` = true-label scores; also **the non-conformal baseline** |
| 1. calibrate | `cal` (300) | conformal agent with per-question ACI, for each `α_1 ∈ {0.02, 0.05, 0.1, 0.2}` | chosen `α̂_1` |
| 2. test | `test` (500) | conformal agent at `α̂_1` | reported numbers |

Phase 1 is a one-parameter **Learn-then-Test** with fixed-sequence testing.
The trace-level risk is `R(α_1) = P(some labeled call in the trace is
uncovered)`. Test grid values from most to least conservative, each with a
binomial p-value for `R ≤ δ` at level ε; stop at the first failure; `α̂_1` =
the last one that passed. This gives `P(R(α̂_1) ≤ δ) ≥ 1 − ε`. It doesn't need
`R` to be monotone in `α_1`, and it's valid even though `α_1` shapes the traces.
Defaults: `δ = 0.1`, `ε = 0.1`. `--risk answer` uses "final answer set misses
GQA answer" as the event instead.

Per-trace coverage over ~5 calls is coarse (0, 0.2, …, 1). So "≥ 90%
per-trace coverage" effectively means "all calls covered", which is why the
event above is *any miss in the trace*.

Baselines from the same code: `--gamma 0` (fixed-α split conformal),
`--no_reset` (one ACI stream across all questions, where Prop. 4.1 is
meaningful over T in the thousands), phase 0 (point estimates).

## 5. The answer DSL

The policy ends by calling `answer(spec)`. The spec is one line that combines
**observation ids**. It never calls tools again: under ACI, α has moved since
the observation, so a re-call would produce a different set than the one the
agent reasoned over.

```
answer("o3")                              # what color is the cup?        -> o3 = query(r1, color)
answer("union(o3, o5)")                   # two candidate cups queried
answer("yesno(same(o3, o5))")             # same color?
answer("choose(o3, 'red', 'blue')")       # red or blue?
answer("and_(o2, not_(o4))")              # logical questions
answer("side(o1, o6)")                    # left or right? (from boxes of two find sets)
```

| combinator | semantics (lifted pointwise over sets, then joined) |
|---|---|
| `union(a, …)` | set union |
| `same(a, b)`, `different(a, b)` | `{yes}` / `{no}` / `{yes, no}` over all pairs |
| `and_`, `or_`, `not_` | three-valued over `{yes, no}` sets |
| `choose(a, x, y)` | `a ∩ {x, y}` |
| `side(f_a, f_b)` | `{left, right}` from box centres, over all box pairs |

- **Soundness.** Combinators are deterministic, so if every observation in the
  spec's **dependency closure** covers, the answer set covers. The closure
  includes the `find` behind each region id, and each observation is counted
  once. So `P[y* ∈ answer] ≥ 1 − Σ_{o ∈ closure} α_o` (union bound, with α_o =
  the α_t in force at that call). This is reported alongside each answer,
  not enforced.
- What it doesn't cover: a spec with the wrong logic (policy error). That's
  measured separately: "all closure observations covered but answer missed".
- Parsed with `ast` against a whitelist (names = combinators, `o<k>` ids,
  string literals). No `eval`.

## 6. Labels (`labels.py`, the one swap point)

GQA val scene graphs (`val_sceneGraphs.json`) give objects with name, box,
attributes and relations. `label(call, scene_graph) -> y_t | None`:

| tool | label | unlabeled when |
|---|---|---|
| `find(name)` | sg objects whose name matches the phrase's head noun (lowercase, singular, small synonym table), with any other words required among their attributes | no sg object matches the name *and* the name isn't in the GQA object vocab |
| `query(r, attr)` | region → sg object with max IoU ≥ 0.5; truth = its attributes ∩ vocab[attr] (`name` → its name). Several true values ⇒ covered iff any is in the set; score = min over them | no IoU match, or no attribute of that category |
| `verify(r, p)` | yes if `p` is among the object's attributes/name; no if the object has *another* value from `p`'s category | `p`'s category absent on the object (scene graphs are incomplete; absence isn't "no") |
| `exists(name)` | yes iff a matching sg object is in the region | — |
| `relate(a, b, rel)` | yes iff the sg has `rel` from A's object to B's | either region unmatched |
| final answer | GQA `answer`, matched like quasar's `eval_match` (`gt in pred`, lowercased) | — |

The scene graphs are noisy and incomplete, so measured miscoverage partly
reflects label error. A strong-model labeller replaces this function later
with the same signature.

## 7. Files

```
data.py        GQA val questions (lmms-lab/GQA) + images + val_sceneGraphs.json; split by phase
backends.py    OWLv2 + gpt-4o-mini JSON-confidence call + disk cache    (copied from quasar)
conformal.py   Q̂_k from D_cal,k; per-question ACI level; LTT selection of α_1
tools.py       the five tools: scores -> set at Q̂_k(1-α_t); observation registry (o_k, r_i, deps)
labels.py      scene-graph oracle (§6)
answer_dsl.py  ast-parsed combinators over observation ids + closure budget (§5)
agent.py       OpenAI function-calling loop; observation text = set + α_t
run.py         phases 0/1/2, jsonl logs, summary
```

v1's `setsem.py` (lattice, two-band find) and `demo.py` go away: membership
is binary under a conformal threshold, and `run.py --mock` replaces the demo.

## 8. Run

```bash
cd conformalagentloop
cp .env.example .env          # put OPENAI_API_KEY=... in it
uv sync
uv run python run.py --mock   # no key, no downloads: fake backends, checks the loop/ACI/DSL
uv run python run.py --phase all --n_score 200 --n_cal 300 --n_test 500
```

The first real run downloads GQA val images from HuggingFace (~3 GB), OWLv2
weights (~1.8 GB) and `sceneGraphs.zip` from Stanford (~40 MB), into `data/`
(gitignored). OWLv2 uses MPS if available, else CPU.

Flags: `--alpha 0.1 --gamma 0.05 --delta 0.1 --eps 0.1 --risk {trace,answer}
--no_reset --policy_model gpt-4o-mini --max_steps 8`.

**Cost:** roughly 5–10 gpt-4o-mini calls per trace. Phase 1 runs the cal
split once per grid value (4 × 300 traces). The whole pipeline costs a few
dollars; VLM calls are cached across phases.

## 9. Outputs

`runs/<timestamp>/`:
- `calls.jsonl`: one row per tool call with question id, t, tool, args,
  set, α_t, threshold, label, covered, unlabeled
- `traces.jsonl`: one row per question with spec, answer set, closure
  budget, answer covered, all-calls covered, steps
- `summary.md`, with:
  - per tool: coverage vs 1 − α*, mean set size, % unlabeled
  - per trace: P(any miss) vs δ
  - answers: answer coverage, accuracy (singleton & correct), set size,
    policy-logic errors
  - comparison against the baselines

## 10. Open / known limits

- Per-question ACI has no within-trace theorem (T too small); the guarantee
  is the LTT one over iid questions (§4).
- The union bound ignores correlation between calls on the same image.
- Labels are noisy (§6); free-text tool arguments are mapped to scene-graph
  objects heuristically.
- No ground truth at deployment: within-trace ACI needs the labeller. In GQA
  it is the scene graph; elsewhere it is the strong-model labeller.
- Delayed feedback (`paper/architecture.md` §3.4) isn't needed for GQA, since
  feedback is immediate. Revisit for datasets where labels arrive late.
