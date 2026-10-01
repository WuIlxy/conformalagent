# conformalagentloop

An **initial sketch** of a ReAct-style agent whose tools return *conformal
prediction sets* instead of point estimates, and which reasons about those sets
as evidence with a known reliability — then commits a final answer through a
tiny set-valued DSL that gives soundness by construction.

> Status: a runnable skeleton, not a system. `python demo.py` exercises the
> whole pipeline end-to-end on a mocked image (no API, no torch). Everything
> marked `# OPEN:` / “stubbed” below is a deliberate hole, not an oversight.
> These are initial ideas — change the shapes freely.

---

## 1. Why not synthesize a program up front (ViperGPT / Quasar)

ViperGPT’s program-synthesis step buys compositionality but costs a lot: you
need a soundness argument for the *whole* synthesized program, the LLM has to
commit to structure before it has seen any evidence, and a bad generation is
hard to recover from.

Flip it. The LLM sees the question, picks a tool, **sees the result (now a
prediction set)**, picks the next tool, and so on until it decides to answer.
Every step is a discrete decision the model can **condition on the actual
observed uncertainty** — adaptive control flow instead of a fixed program. The
programmatic guarantee doesn’t disappear; it moves to the *end*, into a one-line
answer spec written once all uncertainty has been observed (§4).

## 2. Architecture

```
   question + image
        │
        ▼
  ┌──────────────┐   Thought + Action (a one-line DSL expr)
  │  policy (LLM) │ ───────────────────────────────────────┐
  └──────────────┘                                          ▼
        ▲                                        ┌────────────────────┐
        │  Observation: SET + alpha              │ answer_dsl.evaluate │  set-valued
        │  "contains truth w.p. >= 1-alpha"      │  (setsem lifting)   │  semantics
        │                                        └─────────┬──────────┘
        │                                                  │ leaves dispatch to
        │                                                  ▼
        │                                        ┌────────────────────┐
        └──────────────── observe ───────────────│       tools        │ find/query/verify/…
                                                 └─────────┬──────────┘
                                                           │ raw scores
                                     threshold q̂(α_t)      ▼
                                                 ┌────────────────────┐
                                                 │  conformal layer   │ split-conformal init
                                                 │  ACIController/tool │ + adaptive (ACI) α_t
                                                 └─────────┬──────────┘
                                                           ▼
                                                 raw backend (OWL-ViT + VLM)
                                                 = imgpatch_4o_all / imgpatch_conformal
```

Loop-time tool calls and the final answer run through the **same** evaluator, so
the agent is *always* looking at conformal sets — never a collapsed point value.

## 3. The tool set (`tools.py`)

Small and GQA-shaped; GQA questions decompose into these the way scene-graph
annotations do. Each perception tool builds its set exactly like the paper’s
[`imgpatch_conformal.py`](../quasar-colm2026-artifact/epic/imgpatch_conformal.py),
except the band edge is a **calibrated conformal quantile**, not a global τ.

| tool | returns | set from | α? |
|---|---|---|---|
| `find(phrase)` | region set (`AbstractTuple`, certain/uncertain) | OWL-ViT scores, two-band split | ✓ |
| `query_attribute(region, attr)` | value set (`AbstractOther`) | VLM enumerates values+confidence | ✓ |
| `verify(region, predicate)` | `{yes}` / `{no}` / `{yes,no}` | VLM P(yes), 3-valued bands | ✓ |
| `spatial(a, b)` | relation set | bbox geometry (deterministic) | ✗ |
| `count(region_set)` | integer, or **integer range** | `AbstractTuple.card()` | inherits |
| `answer(spec)` | terminal | evaluates the DSL spec | union bound |

`count` shows the key contagion: deterministic on a certain input, but
**set-valued** the moment its input membership is uncertain (`{1,2}`).

## 4. The constrained answer (`answer_dsl.py`)

`answer` does **not** take a free-form string. It takes a one-line spec saying
*how to derive the answer from the trajectory*, evaluated with set semantics:

```
answer( union( attribute_of( find("dog"), "color" ) ) )
```

- Leaves (`find`, `attribute_of`, `verify`, `relate`, `count`) dispatch back
  through the same `Tools` (conformally cached), so the answer uses exactly the
  sets the agent already saw.
- Combinators (`union`, `yesno`) are pure set-lifts from `setsem.py`.

**Soundness by construction.** Each leaf set covers its truth with prob ≥ 1−α_leaf;
the combinators are deterministic. So the derived answer set covers the true
answer with probability

```
  P[ y* ∈ answer_set ]  ≥  1 − Σ α_leaf         (union bound over the sets used)
```

This keeps ViperGPT’s programmatic-guarantee flavour, but the “program” is one
line, written at the end, with all uncertainty already resolved. The union bound
is **loose** — tightening it (shared calibration, Bonferroni→better) is open.

## 5. The conformal layer (`conformal.py`) — the core research problem

Model a run as a sequence of calls
**{(xₜ, fₜ, Ĉₜ, yₜ)}₍ₜ₌₁..T₎**: input xₜ, the conformal predictor fₜ called,
its predicted set Ĉₜ = fₜ(xₜ), ground truth yₜ. The hard part: **xₜ is produced by
the agent’s own prior actions**, so it is *not* exchangeable with fₜ’s offline
calibration data. The agent induces covariate shift on its own tools.

Two guarantees, two mechanisms:

- **Trace-level** — coverage along one trace. *Adaptive Conformal Inference*
  (Gibbs & Candès 2021): after each call, nudge the level,
  `α_{t+1} = α_t + γ·(α_target − err_t)`, where `err_t = 1[ yₜ ∉ Ĉₜ ]`. It reacts
  to realised error rather than assuming exchangeability, so it tolerates the
  shift. `ACIController` holds the calibration scores and maps `α_t` to a
  concrete nonconformity quantile `q̂(α_t)` (`split_conformal_quantile`).

- **Task-level** — 95% of *traces* achieve 95% coverage. Run ACI within each
  trace, but pick its **initial** `α_1` (equivalently the starting quantile) by
  ordinary split conformal over a held-out set of whole traces. Outer
  split-conformal chooses the init; inner ACI adapts from there.

This is the disciplined version of the paper’s single global τ: τ becomes the
**online conformal quantile `q̂(α_t)`**, per tool, adapting through the trace.

### Open problems (where the sketch is naive on purpose)

1. **The ACI update needs `err_t`, i.e. yₜ for intermediate calls.** At
   deployment you don’t have it. *But GQA ships scene-graph annotations* — object
   presence, attributes, relations — which give per-step yₜ at **calibration**
   time. So ACI can be *fit* offline on labelled traces; live traces then run
   with frozen (or proxy-updated) α. Closing this honestly is the main question.
2. **Short traces.** ACI’s guarantee is long-run in T; a GQA trace is a handful
   of calls. Trace-level coverage over one short trace is aspirational — the
   task-level (across-traces) framing is the load-bearing one.
3. **Dependence / union-bound looseness.** §4’s bound ignores correlation between
   tool errors and double-counts a tool used twice.
4. **The answer DSL uses restricted `eval`.** Fine for a sketch (empty builtins +
   whitelist); a deployed system needs a real parser/validator.

## 6. Where this reuses the paper

| here | paper |
|---|---|
| `setsem.AbstractOther/AbstractTuple/join/lift` | `epic/conformal_utils.py` (trimmed copy) |
| `tools.find` two-band split | `epic/imgpatch_conformal.find` |
| `tools.query_attribute` (VLM value+conf set) | `imgpatch_conformal.simple_query` / `_do_model_call` |
| `tools.verify` 3-valued | `imgpatch_conformal.verify_property` |
| `loop.openai_policy` | `imgpatch_4o_all.do_gpt_call` |
| `conformal.q̂(α_t)` replaces global τ | `GQA/conformal.py` scale_up/scale_down |

## 7. Files

```
setsem.py       prediction-set lattice (AbstractOther/Tuple, join, lift)   [self-check]
conformal.py    split-conformal init + ACI (trace/task level)              [self-check]
tools.py        the 6 tools; raw backend injected (Backends protocol)
answer_dsl.py   constrained answer spec + set-valued evaluator + union bound
prompts.py      ReAct system prompt + the "set as evidence" observation text
loop.py         the ReAct episode loop; pluggable policy (LLM injected)
demo.py         mocked end-to-end trajectory "What color is the dog?"       [runnable]
```

## 8. Run it

```bash
python demo.py        # end-to-end mock trace + coverage check
python setsem.py      # lattice self-check
python conformal.py   # quantile + ACI self-check
```

Wiring a real run means: implement `Backends` against OWL-ViT + a VLM (copy from
`imgpatch_local` / `imgpatch_4o_all`), fill in `loop.openai_policy` with an LLM
client and a `Thought/Action/Answer` parser, and build `CAL` (per-tool
calibration scores) from labelled GQA traces.
