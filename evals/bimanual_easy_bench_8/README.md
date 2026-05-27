# bimanual_easy_bench_8 — diagnostic 8-task bimanual suite

A short, diagnostic eval covering **eight distinct bimanual manipulation
primitives**. Designed as the canonical "easier than IKEA, more
informative than LIBERO-Object" benchmark for the bimanual YAM rig.

## Quickstart

```bash
# 1. Bring up your inference server (Terminal A)
./scripts/run_server.sh molmoact2     # or pi05 / gr00t-n17

# 2. Run the eval (Terminal B). Samples = attempts per task.
./scripts/run_eval.py --policy molmoact2 --eval bimanual_easy_bench_8 \
    --samples 5

# Same eval with a longer reset window:
./scripts/run_eval.py --policy pi05 --eval bimanual_easy_bench_8 \
    --samples 3 --reset-seconds 60

# Disable the countdown (operator-driven, advance only on Enter / right-arrow):
./scripts/run_eval.py --policy gr00t-n17 --eval bimanual_easy_bench_8 \
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
   (auto-ends if the policy emits max_chunks chunks.)

3. ARM RESET      arms ramp back to the training-mean ready pose.
                  Automatic — no input needed.

4. SCORE          s=success  f=failure  u=unclear  r=redo  Enter=skip.

5. RESET WINDOW   countdown (default 30s); you reset the physical scene
                  during this window.
   →  skip the countdown and go straight to the next iteration.
   's' to skip the remaining iterations of this task.
   'q' to abort the entire eval.
```

## Why eight tasks?

Each task isolates a distinct **bimanual primitive** — a motor skill
that has no single-arm sequential reduction. Picking N variations of
one primitive (the LIBERO-Object pattern) measures one skill at high
SNR but one axis. Picking N primitives measures eight axes, one each.

| #  | Task              | Primitive                              | ALOHA-family ancestor                       |
|----|-------------------|----------------------------------------|---------------------------------------------|
| 1  | shirt fold        | Symmetric coordination on deformable   | Mobile ALOHA t-shirt fold                   |
| 2  | velcro pull       | Opposing impulsive force on a bond     | ALOHA velcro cable tie (reversed)           |
| 3  | cube hand-off     | Discrete mid-trajectory transfer        | ALOHA cube transfer                         |
| 4  | wire rack lift    | Coordinated wide-rigid lift             | ALOHA-2 bimanual tray                       |
| 5  | pour cubes        | Asymmetric tool + receiver, flow        | ALOHA chip-bag pour                         |
| 6  | rail tap          | Mid-air bimanual convergence            | refined from ALOHA bimanual insertion       |
| 7  | marker uncap      | Stabilize + axial pull                  | ALOHA cup uncap                             |
| 8  | allen wrench turn | Stabilize + torque                      | ALOHA cup uncap (rotational variant)        |

**Why these eight and not others?** Two filters applied to every task:

1. **Truly bimanual.** Counterfactual: could a single-arm robot do the
   task by working sequentially? If yes, dropped. (This is what
   eliminated "stack two cubes from a shared box": the simultaneous
   grasp is bimanual but the place+stack reduces to single-arm.)
2. **Atomic.** Each task tests one primitive, not a chain. (This is
   what eliminated chained "lift, transport to rack, place"-style
   tasks: success is the product of three primitives, so a 50% failure
   on each gives 12.5% end-to-end — measures endurance, not skill.)

## VLA-prompting principles applied

The instruction strings in `tasks.yaml` were edited to match the
phrasing that VLAs (pi0, MolmoAct, OpenVLA) handle most reliably:

- **Imperative present tense, no metalanguage.**
  - ✗ `"stabilize the block"`   → ✓ `"hold the wooden block"`
  - ✗ `"use both grippers"`      → ✓ `"using both arms"`
- **`"your left arm"` / `"your right arm"` over `"gripper"`.** Most
  ALOHA / pi0 / MolmoAct training data uses the arm-as-body phrasing,
  not robotics-jargon "gripper" / "manipulator."
- **Every object disambiguated by visible attribute.** `"the red
  marker"`, `"the orange cube"`, `"the green plate"` — never just
  `"the cube"`.
- **No pronouns.** Restate the noun each time: `"pass the orange cube
  to your right arm"` not `"pass it to your right arm"`. Modern VLAs
  do not reliably resolve anaphora across an instruction.
- **Clean grammar, no typos.** pi0 specifically [freezes on typos or
  ambiguous phrasing](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/),
  and articulated-object tasks may need multiple prompt rephrasings to
  find the in-distribution one.
- **Single atomic verb-result.** No "then" chaining unless the second
  action is the only verifiable end-state (task #3 keeps "then place
  the cube on the green plate" only because that's the score criterion).

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
