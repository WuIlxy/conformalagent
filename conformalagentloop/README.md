# conformalagentloop (v2)

A ReAct agent for GQA whose perception tools return **conformal prediction
sets**, with the set level adapted *within each question* by **Adaptive
Conformal Inference** (ACI; Gibbs & Candès, *Adaptive Conformal Inference Under
Distribution Shift*, NeurIPS 2021), and whose final answer is a **one-line DSL
spec over the observed sets**, so answer coverage follows from tool coverage.

---

## 1. One question, end to end

```
question + image
   │
   ▼
policy LLM (function calling) ── tool call ──► backend scores ──► set Ĉ_t = {y : S(y) ≤ Q̂_k(1-α_t)}
   ▲                                                                 │
   │   observation o_t: the set, its id, α_t                        │ grader label y_t
   └─────────────────────────────────────────────────────────────────┤ err_t = 1[y_t ∉ Ĉ_t]
                                                                     ▼
                                                  α_{t+1} = α_t + γ(α* − err_t)   (ACI, per question)
   ...
policy ── answer("same(o2, o4)") ──► deterministic eval over o2, o4 ──► answer set + running bound
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
next `query`. The calibration families are `find`, `query`, and `yesno`: `verify`,
`exists` and `relate` share one VLM yes/no prompt and pool their scores, since
each alone is called too rarely to calibrate. A tool is called only once or twice per question, so separate
per-tool levels would barely move.

**Per question, reset to α_1 = α\*.** This follows the paper: start at the
target level. Questions are treated as iid, so each (question, image) runs
its own ACI from α_1 = α\*. Each tool's *baseline* is its calibrated threshold
map `Q̂_k` (§4), not a different starting α. With T ≈ 3–6 calls, Prop. 4.1 is
vacuous inside one trace. What ACI provides there is the behaviour: the trace
gets more conservative after an observed error on *this* image.

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
| `find(name, region="image")` | boxes → new region ids `r_i` | `1 − det_score(box)` | Grounding DINO base, every box above 0.05 after NMS (no cap) |
| `query(region, attribute)` | values, `attribute ∈ {color, material, name, …}` | `1 − p(y)`. Closed categories (color, material, size, …): gpt-6-luna's Decisions API `choice` question gives `p` for every label word (`Backends.choose`; reported to 0.01, so words under 0.005 tie at S = 1). Open ones (name, free questions): JSON answers + self-reported scores, 0 if not listed → S = 1 | closed: Decisions API (gpt-6-luna); open: `--vlm_model` (gpt-6-luna) |
| `verify(region, property)` | `{yes, no}` | `S(yes) = 1 − p_yes`, `S(no) = p_yes`; `p_yes` from a Decisions API `predicate` question (`Backends.predicate`) | Decisions API (gpt-6-luna) |
| `exists(name, region="image")` | `{yes, no}` | as `verify` | same |
| `relate(region_a, region_b, relation)` | `{yes, no}` | as `verify`, on the union-box crop | same |

- **`find` grades each instance on its own.** For every instance the grader
  says the call should catch, the true-label score is `1 − best matching
  det_score` (1 if no proposed box finds it), and that instance is covered iff a
  box in the set finds it. Each instance is one calibration row and one ACI
  update; the call counts as covered (for the trace's closure) iff all its
  instances are. Requiring *all* instances in one event pushed the 90% cutoff
  to "keep every box" for both OWL and Grounding DINO (150 replayed calls:
  OWL 0.89 / Grounding DINO 0.91 of calls had every instance among the
  proposals). "Found" (`labels.same_object`): IoU ≥ 0.3, or the box holds ≥ 80%
  of the object while being at most 4× its area.
- **`find`'s full set.** When the threshold reaches 1, the set also gets the
  searched region itself, shown as "anywhere in the searched region". The
  proposals alone aren't the full label space, and ACI needs the full set to
  cover.
- **Why Grounding DINO, uncapped.** On 150 random terra-graded calibration find
  calls, the share of counted objects no proposed box finds was OWLv2 (20-box
  cap) 20%, DINO capped 14%, DINO uncapped 6%. Above 10%, the 90% cutoff is 1
  (keep every box); uncapped DINO's is 0.91, keeping ~7 of ~15 boxes. `side`/`above` treat that region as
  "either side".
- `relate` rejects left/right/above/below: those come from box geometry via
  `side`/`above` in the answer spec, over all box pairs at once.
- `exists` is its own yes/no tool rather than "is the find set non-empty".
  Conformal find sets are supersets, so non-empty proves nothing.
- `⊤` (full set) appears when α_t ≤ 0: for `query` it means "any value".
- VLM calls are disk-cached by `(image crop, prompt)`, like `imgpatch_conformal`.
- GQA has no counting questions, so there's no `count`.

## 4. Per-tool baseline: calibrate, then test

There are two disjoint sets of questions from the 1000 sampled GQA val
questions (`val_n1000_seed2025` ids, reused from quasar). Testing on the
calibration questions would overstate coverage.

| phase | questions | what runs | produces |
|---|---|---|---|
| calibrate | 200 | point-estimate agent (top-1 values, detector score > 0.1), no sets | per tool k: `D_cal,k` = scores of the true labels → baseline threshold `Q̂_k(1 − α*)`. Also **the non-conformal baseline** |
| test | 800 | conformal agent, per-question ACI from α_1 = α\* | reported numbers |

What α_1 = α\* gives you:
- **First call of each question:** threshold `Q̂_k(1 − α*)`, i.e. split conformal.
  This covers with probability ≥ 1 − α\* *if* the call is exchangeable with
  tool k's calibration calls.
- **Why it might not be:** the test agent sees sets, while the calibration
  agent saw point estimates, so it makes different calls. This is the
  endogenous shift (`paper/architecture.md` §1). ACI is the correction: when
  the baseline under-covers, observed misses lower α_t.
- **Formal ACI guarantee:** with `--no_reset`, one ACI stream runs across all
  test questions, and Prop. 4.1 gives `|miscoverage − α*| ≤ (1 − α* + γ)/(γT)`
  with T in the thousands. With per-question reset, the same quantity is
  reported empirically.

Baselines from the same code: `--gamma 0` (fixed split conformal, no
adaptation), `--no_reset` (above), and the calibrate phase (point estimates).

Possible later addition, not built: choose α_1 by Learn-then-Test over a
grid, which would give a formal per-trace guarantee ("P(any miss in a trace)
≤ δ").

## 4b. Selector-aware batch thresholds (`--method gcp|mvp`)

Path A of two. The batch step sets group-aware starting thresholds; online
adaptation against the agent's own shift (path B) comes later. ACI (§2)
stays as `--method aci`, the default.

**Selectors** (`selector.py`). A selector `φ(x, w) ∈ {1..k} ∪ {∅}` picks at
most one tool call in a trace. It must be stable: `∅` until its call
appears, then that index for the rest of the trace. Here every selector is
"the n-th call satisfying `pred`", and `pred` reads only the call's own
features and the trace before it, so stability holds by construction. A
call's threshold is

  `λ(x, w, i) = λ_fam(i) + Σ_φ 1{φ(x, w) = i} λ_φ`

where the family intercept gives calls that no selector picks a threshold.
The objective, per selector: `P(f*(z_i) ∈ C(z_i) | φ selects i) ≥ 1 − α`.
A selector picks at most one call per question, so its calibration rows are
one per question (iid), unlike pooling every call of a trace.

Default selectors, for each family in {find, query, yesno}:
- `fam#n`, n = 1..3: the n-th call of the family (step depth)
- `fam|qtype#1`: the first call on a question of type
  verify/choose/logical/compare/query. `qtype()` reads only the text, and
  matches GQA's structural types on 99.5% of 20k val questions.
- `fam|image#1`: the first call on the whole image
- `fam|after_crowded#1`: the first call after a find with ≥ 10 raw detector boxes
  > 0.1 (loop-state drift; uses raw counts, so it doesn't depend on λ)
- and `query|{color,material,name,other}#1`

**Fitting** (`conformal.py`). Both algorithms come from Jung, Noarov,
Ramalingam & Roth, *Batch Multivalid Conformal Prediction* (2022), and use
the labelled calibration calls as rows `(true-label score, groups)`:
- `fit_gcp` (Alg. 1, BatchGCP): an LP for the λ that minimises pinball loss
  at q = 1 − α. Gives coverage q in every group with ≥ `--min_count` rows.
  Family intercepts are always fitted; a family with no data gets the full set.
- `fit_mvp` (Alg. 2, BatchMVP): starts from GCP, snaps to the grid k/m, then
  repeatedly patches the (group, threshold level) cell with the worst
  coverage until each group's quantile calibration error is ≤ `tol · P(g)`.
  The result also holds conditional on the threshold value. It differs from
  the paper in two ways, both for small samples: a patch takes the closest
  shift with coverage ≥ q when one exists, and a cell that can't be improved
  is skipped instead of stopping the algorithm.
- A held-out share of calibration questions (`--holdout`) reports per-group
  coverage for per-tool split / GCP / MVP before the test phase runs (no
  agent shift in that check).

**Calibration data.** `--cal_pool` draws `n_cal` more GQA val questions
(from all 2,011,853, like the test file) on images the 1000-question test
file doesn't use. The scene graph is only needed to grade calibration calls.
At test time GCP/MVP thresholds are static and need no labels; only ACI
needs them.

Known limits: the calibration agent (point estimates) still differs from the
test agent, so selector membership shifts between them. That is path B's
job. GCP/MVP make no finite-sample correction, so small groups over-fit; the
held-out table shows it. On 280 old calibration rows the per-selector
held-out n was only 10–37.

## 5. The trace is a program (DSL)

Each loop step appends one line to a straight-line program. A line is a tool
call or a combinator over earlier lines, and every line holds a set, its α,
and its **running bound**. The agent sees the whole program so far as its
context:

```
Q: Is the cup the same color as the plate?

o1 = find("cup")                 -> {r1, r2}        α=0.10   bound ≥ 0.90
o2 = find("plate")               -> {r3}            α=0.10   bound ≥ 0.90
o3 = query(r1, "color")          -> {white, gray}   α=0.10   bound ≥ 0.80   (needs o1)
o4 = query(r2, "color")          -> {white}         α=0.10   bound ≥ 0.80   (needs o1)
o5 = query(r3, "color")          -> {white}         α=0.10   bound ≥ 0.80   (needs o2)
answer = same(union(o3, o4), o5) -> {yes, no}                bound ≥ 0.50   (needs o1..o5)
```

- **Dependencies.** A line depends on every line whose output it uses: `o3`
  queries region `r1`, which came from `o1`, so `o3` is only meaningful if
  `o1` also covered. A line's closure is all of its ancestors, each counted
  once (`o1` is shared by `o3` and `o4`).
- **Running bound.** A line misses only if some line in its closure missed,
  so `P[line covers] ≥ 1 − Σ_{o ∈ closure} α_o` (union bound; α_o is the α_t
  in force at that call). Every line shows this, so mid-trace the agent can
  see what a candidate answer would be worth and decide whether to probe
  more or commit. The final line's bound is the answer's guarantee.
- **No re-calls.** Combinators only read sets that are already in the
  program. Re-calling a tool after α has moved would give a different set
  from the one the agent reasoned over.
- The agent writes tool calls through function calling and the final line
  through `answer(spec)`, for example:

```
answer("o3")                              # what color is the cup?
answer("union(o3, o4)")                   # two candidate cups
answer("same(o3, o5)")                    # same color?
answer("choose(o3, 'red', 'blue')")       # red or blue?
answer("and_(o2, not_(o4))")              # logical questions
answer("side(o1, o6)")                    # left or right? (from boxes of two find sets)
```

| combinator | semantics (lifted pointwise over sets, then joined) |
|---|---|
| `union(a, …)` | set union |
| `same(a, b)`, `different(a, b)` | `{yes}` / `{no}` / `{yes, no}` over all pairs; `b` may be a literal (a constant, α = 0) |
| `and_(a, b, …)`, `or_(a, b, …)`, `not_` | three-valued over `{yes, no}` sets; Python `and`/`or`/`not` also accepted |
| `choose(a, x, y)` | `a ∩ {x, y}`; rejects find lines, and yes/no lines unless the options are yes/no |
| `side(f_a, f_b)`, `above(f_a, f_b)` | `{left, right}` / `{above, below}` from box centres, over all box pairs; `f_b` may be `image` (side of the photo) |

- **Soundness.** Combinators are deterministic, so the running bound above
  is sound for the answer line too. It's reported, not enforced.
- What it doesn't cover: a spec with the wrong logic (policy error). That's
  measured separately: "all closure observations covered but answer missed".
- Parsed with `ast` against a whitelist (names = combinators, `o<k>`/`r<k>`
  ids, string literals). No `eval`. A bare literal answer (`answer("no")`) is
  rejected: the answer has to be derived from lines.
- Answer type checks (an error goes back to the agent): the answer set can't
  be empty; yes/no questions (Is/Are/Do/Does/… with no " or ") need a set
  inside `{yes, no}`; what/which/who/where/how questions can't get one;
  "which side" questions need `{left, right}`.

## 6. Labels (`labels.py`, the one swap point)

Two graders implement the same five methods; `--grader` picks one.

**`terra` (default): `TerraLabeler`.** gpt-6.1-terra (`--grader_model`,
reasoning effort medium) answers each call itself, from the crop the tool saw,
without seeing the tool's scores. Coverage then means "the set contains the
grader's answer", so the grader's own error rate bounds what that coverage
says about the truth. Replies are disk-cached.

| tool | grader asks | unlabeled when |
|---|---|---|
| `find(name)` | first, how many `name`s are in the region (clean crop, no boxes). Then the proposed boxes one at a time, best detector score first, each drawn **alone** on the region: "does the red box frame one `name`?"; a box framing an already confirmed object is skipped, and checking stops once as many are confirmed as counted. Instances = the confirmed boxes + one empty instance (score 1) per counted object no box framed | the count or a box answer isn't a number / yes / no |
| `query(r, attr)` | closed category: "What is the `attr` of the `name`? Answer with exactly one word from: …". `name`: every name a person might use for the object, specific to general ("baseball player, player, person"); a set covers if any of its names matches any of these either way round ("laptop" ↔ "laptop computer") | open attributes; a reply outside the category |
| `verify` / `exists` | the tool's yes/no question | reply isn't yes/no |
| `relate(a, b, rel)` | the union crop with box 1 (red) and box 2 (blue) drawn: is 1 `rel` 2? | reply isn't yes/no |

**`scene_graph`: `SceneGraphLabeler`.** GQA val scene graphs (`val_sceneGraphs.json`) give objects with name, box,
attributes and relations. GQA's question annotations give, per question, the
object ids the question and answer refer to (`data.question_objects()`, from
lmms-lab/GQA `val_all_instructions`). **Only the labeller uses them; the agent
never sees them.** A strong-model labeller would supply the same thing ("which
object does this question mean") on datasets without annotations.
`label(call) -> y_t | None`:

| tool | label | unlabeled when |
|---|---|---|
| `find(name)` | default (`--find_label all`): **every** object in the region whose name matches the phrase's head noun (lowercase, singular, small synonym table), independent of the question. With `--find_label relevant`: only the object(s) the question means. `[]` if no object of that name is in the region (vacuously covered) | the phrase isn't in the GQA object vocab; with `relevant`, also when the noun is in the image but none of its instances is meant |
| `query(r, attr)` | region → the sg object *with the name it was found under* that it finds (`same_object`); for `image`, any object at IoU ≥ 0.5. Truth = its attributes ∩ the category (color, material, size, height, length, shape; `name` → its name). Several true values ⇒ covered iff any is in the set; score = min over them | no matching object, or no attribute of that category |
| `verify(r, p)` | same region matching; yes if `p` is among the object's attributes/name; no if the object has *another* value from `p`'s category | `p`'s category absent on the object (scene graphs are incomplete; absence isn't "no") |
| `exists(name)` | yes iff any matching sg object is in the region (a plain fact; ignores which object is meant) | phrase not in the GQA object vocab |
| `relate(a, b, rel)` | same region matching; yes iff the sg has `rel` from A's object to B's | either region unmatched |
| final answer | GQA `answer`, matched like quasar's `eval_match` (`gt in pred`, lowercased) | — |

Two `find` labels. `relevant` grades find on the object the question means.
OWL's recall on those is 92% (top-3: 85%, top-1: 64%; 372 objects from 300
questions). But the grade then depends on the question, so a misleading or
adversarial question changes it, and context searches go unlabelled (a
biased sample). `all` (default) grades find the same way as the other tools:
on what it was asked, never on what the question wanted. Its cost: OWL,
keeping nearly every box, catches all instances in only 81% of find calls,
and object size barely matters (instance recall 0.68 below 0.3% of the image,
0.88 above 10%). So reaching 90% needs the full set ("anywhere"), whatever the
threshold.

The scene graphs are noisy and incomplete, so measured miscoverage partly
reflects label error. A strong-model labeller replaces this function later
with the same signature.

## 7. Files

```
src/
  data.py        GQA val questions + images + scene graphs; calibration pool
  backends.py    Grounding DINO, gpt-6-luna Decisions API (choose / predicate), free-text ask, grader chat; disk cache
  conformal.py   Q̂_k from D_cal,k; per-question ACI level; BatchGCP / BatchMVP fits (§4b)
  selector.py    stable selectors, question type, per-trace selection state (§4b)
  tools.py       the five tools: groups -> scores -> set at the threshold -> program line -> label -> update
  labels.py      graders: gpt-6.1-terra and scene graph (§6)
  answer_dsl.py  the trace program: lines, deps, running bound; ast-parsed combinators (§5)
  agent.py       OpenAI function-calling loop; observation text = set + α_t
  run.py         GQA calibrate + test phases, jsonl logs, summary
  mock.py        fake image/scene graph/scores + scripted policy for --mock (planted miss)
  blink.py       BLINK one-step test of the batch conformal layer (one call per question)
data/
  clean/         tracked: gqa_val_n1000_seed2025.json (the 1000 test questions, from quasar),
                 cal_pool_n*_seed2025.json, question_objects.json
  raw/           sceneGraphs.zip, val_sceneGraphs.json            (gitignored)
  images/        GQA val images <imageId>.jpg                      (gitignored)
  cache/         model and detector call cache                     (gitignored)
runs/            gitignored: runs/<dataset>/<policy>__<tool model>__<grader>/<timestamp>/
```

Self-checks: `uv run python src/conformal.py` (also `answer_dsl.py`, `labels.py`,
`selector.py`, `backends.py`, `blink.py --check`).

v1's `setsem.py` (lattice, two-band find), `demo.py`, `loop.py` and `prompts.py`
are gone: membership is binary under a conformal threshold, `run.py --mock`
replaces the demo, and `agent.py` replaces the loop and prompts.

## 8. Run

```bash
cd conformalagentloop
cp .env.example .env          # put OPENAI_API_KEY=... in it
uv sync
uv run python src/run.py --mock   # no key, no downloads: fake backends, checks the loop/ACI/DSL
uv run python src/run.py --n_cal 200 --n_test 800
uv run python src/run.py --method mvp --cal_pool --n_cal 1000 --n_test 1000
```

The first real run downloads GQA val images from HuggingFace (~3 GB), Grounding DINO
weights (~0.9 GB, HF cache) and `sceneGraphs.zip` from Stanford (~45 MB) into
`data/images/` and `data/raw/`. The detector uses MPS if available, else CPU. Tool and
grader calls are cached in `data/cache/`.

Flags: `--method aci|gcp|mvp --alpha 0.1 --gamma 0.05 --no_reset
--policy_model gpt-6-luna --vlm_model gpt-6-luna --grader terra|scene_graph
--grader_model gpt-6.1-terra --max_steps 12 --workers 8
--cal runs/gqa/<models>/<ts> --cal_pool --find_label all|relevant --holdout 0.2
--min_count 20 --mvp_m 20 --mvp_tol 0.002`.
`--cal` reuses an earlier run's calibrate phase (its `calls.jsonl`) and skips
it; pass the same `--cal_pool` as that run. GCP/MVP need a calibration
made after selectors existed. Without `--cal_pool`, test questions are
`[n_cal : n_cal + n_test]` of the file; with it, the file's first `n_test`.
`--no_reset` runs sequentially.

**Cost:** roughly 5–10 policy calls (gpt-6-luna) per trace, a few Decisions API
calls ($0.10 / M input tokens, output free) and one grader call (gpt-6.1-terra)
per tool call. Tool and grader calls are disk-cached; policy calls are not.

## 9. Outputs

`runs/gqa/<policy>__<tool model>__<grader>/<timestamp>/` (mock: `runs/gqa/mock/`):
- `calls.jsonl`: one row per tool call with question id, t, line, expr,
  set, α_t, threshold, label, covered, true-label score, raw scores; find
  also has `instances` ([score, covered] per instance)
- `traces.jsonl`: one row per question with the rendered program, spec,
  answer set, closure, bound, answer covered, any miss, thoughts, α start/end
- `cal_scores.json`: the per-family calibration scores
- `thresholds.json` (gcp/mvp): λ per group, MVP grid and patches
- `summary.md`, with:
  - per tool: coverage vs 1 − α*, mean set size, % unlabeled
  - per group (family, selector): coverage given the selector picked a call
  - gcp/mvp: held-out per-group coverage of the fit (§4b)
  - per trace: P(any miss in the trace)
  - answers: answer coverage, accuracy (singleton & correct), set size,
    policy-logic errors
  - comparison against the baselines
