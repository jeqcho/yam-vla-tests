# bimanual_easy_bench_5 — diagnostic 5-task bimanual suite

A short, diagnostic eval covering **five distinct bimanual manipulation
primitives** chosen to be tractable on the bimanual YAM rig. The "easier
than IKEA, more informative than LIBERO-Object" canonical benchmark for
this hardware.

Renamed from `bimanual_easy_bench_8` after three tasks were dropped
(see CHANGELOG section below) and one new task was added.

## Quickstart

```bash
# 1. Bring up your inference server (Terminal A)
./scripts/run_server.sh molmoact2     # or pi05 / gr00t-n17

# 2. Run the eval (Terminal B). Samples = attempts per task.
uv run scripts/run_eval.py --policy molmoact2 --eval bimanual_easy_bench_5 \
    --samples 5

# Same eval with a longer reset window:
uv run scripts/run_eval.py --policy pi05 --eval bimanual_easy_bench_5 \
    --samples 3 --reset-seconds 60

# Disable the countdown (operator-driven, advance only on Enter / right-arrow):
uv run scripts/run_eval.py --policy gr00t-n17 --eval bimanual_easy_bench_5 \
    --reset-seconds 0
```

## Operator flow (one attempt)

Per iteration, the harness drives you through five stages. **→ (right
arrow) is the universal "advance" key.** Enter works the same.

```
1. BANNER         shows TASK, iteration N of X, the exact prompt sent
                  to the policy.
   →  start the rollout.

2. ROLLOUT        arms execute the prompt. You watch.
   →  end the rollout immediately (e.g. you saw success, or it's
      clearly off-rails).
   (auto-ends if the policy emits max_chunks chunks OR if the
   per-attempt wall-clock timeout — default 60 s — expires.)

3. ARM RESET      arms ramp back to the canonical training-mean ready
                  pose. Automatic — no input needed.

4. SCORE          s=success  f=failure  u=unclear  r=redo  Enter=skip.

5. RESET WINDOW   countdown (default 30 s); you reset the physical scene
                  during this window.
   →  skip the countdown and go straight to the next iteration.
   's' to skip the remaining iterations of this task.
   'q' to abort the entire eval.
```

## The five tasks

Each task isolates a distinct **bimanual primitive** — a motor skill
that has no single-arm sequential reduction.

| #  | Task                  | Primitive                                        | ALOHA-family ancestor                       |
|----|-----------------------|--------------------------------------------------|---------------------------------------------|
| 1  | cube into gray box    | Bimanual insertion (held receptacle)             | ALOHA peg-in-socket                         |
| 2  | wire rack lift        | Coordinated wide-rigid lift                      | ALOHA-2 bimanual tray                       |
| 3  | pour cubes            | Asymmetric tool + receiver, flow                 | ALOHA chip-bag pour                         |
| 4  | rail tap              | Mid-air bimanual convergence                     | refined from ALOHA bimanual insertion       |
| 5  | marker uncap          | Stabilize + axial pull                           | ALOHA cup uncap                             |

**Why these five and not the original eight?** Two filters were applied
to every task:

1. **Truly bimanual.** Counterfactual: could a single-arm robot do the
   task by working sequentially? If yes, dropped.
2. **Atomic.** Each task tests one primitive, not a chain. Success rate
   reflects skill, not endurance.

Plus a third filter added after early hardware runs:

3. **Within reach of the policy's training distribution.** Tasks where
   the core verb is 0/591 in the BimanualYAM training corpus AND the
   physical primitive is genuinely outside the rigid-pick-place
   distribution were dropped or replaced.

## CHANGELOG: what's different from bimanual_easy_bench_8

**Dropped:**
- `bag_fold` — deformable fold task. "Fold" verb is 0/591 in the
  BimanualYAM training corpus. Empirically observed 10/10 failure rate
  on early hardware run. Associated CSVs deleted.
- `cube_handoff` — discrete mid-trajectory transfer. "Pass" verb is
  0/591 in corpus; bimanual handoff is genuinely a hard ALOHA-canonical
  primitive that wasn't represented in molmoact2's training mix.
- `allen_wrench_turn` — stabilize + rotational torque. Rotation verbs
  ("turn", "rotate", "twist") all 0/591 in corpus — this primitive
  is entirely absent from training.
- `velcro_pull` — replaced (see below). Original setup was infeasible:
  velcro strips too small to be picked up off the table by the YAM
  gripper.

**Added:**
- `cube_into_gray_box` — bimanual insertion into a held receptacle.
  One arm holds a gray box in the air, the other places an orange cube
  into the box. Directly matches the corpus's dominant "Pick up + Place
  X into Y" template — 263 "into" occurrences in 591 annotations.
  This is the most corpus-aligned task in the suite and should produce
  the highest single-task success rate.

## VLA-prompting principles applied

Instruction strings in `tasks.yaml` were edited to match the phrasings
that VLAs (pi0, MolmoAct, OpenVLA) handle most reliably:

- **Imperative present tense, no metalanguage.**
- **`"your left arm"` / `"your right arm"`** — used in most published
  checkpoints' phrasings, though the BimanualYAM training corpus
  specifically has 0 occurrences of arm-side language (an open
  question whether dropping the arm-side qualifier improves results
  on molmoact2 specifically; see Sources below).
- **Every object disambiguated by visible attribute.** `"the red
  marker"`, `"the orange cube"`, `"the gray box"` — never just
  `"the cube"`.
- **Restate nouns when ambiguity is plausible.** Modern VLAs do not
  reliably resolve anaphora across long instructions.
- **Clean grammar, no typos.** pi0 specifically [freezes on typos or
  ambiguous phrasing](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/).
- **Single atomic verb-result.** No "then" chaining unless the second
  action is the only verifiable end-state.

## Sources

- Tony Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
  Low-Cost Hardware" (ALOHA), RSS 2023.
- Aldaco et al., "ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual
  Teleoperation," 2024.
- Black et al., "π₀: A Vision-Language-Action Flow Model for General
  Robot Control," 2024.
- AI2, "MolmoAct 2: An open foundation for robots that work in the real
  world," 2026.
- "Evaluating π₀ in the Wild," Penn PAL Lab, 2025.
- AllenAI BimanualYAM training corpus annotations
  (`allenai/<date>-<cat>-<NN>` datasets on HuggingFace), accessed via
  `meta/tasks_annotated.parquet`.
