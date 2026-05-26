# yam-vla-tests

Unified testing grounds for vision-language-action (VLA) policies on the
bimanual YAM rig. Three policies, one interface, one place to add new
evals.

| Policy | HF repo | Wire | Default port |
|---|---|---|---|
| **MolmoAct2** | [`allenai/MolmoAct2-BimanualYAM`](https://huggingface.co/allenai/MolmoAct2-BimanualYAM) | HTTP + json_numpy | 8202 |
| **GR00T N1.7** | [`jeqcho/gr00t-n17-yam-bimanual`](https://huggingface.co/jeqcho/gr00t-n17-yam-bimanual) | ZMQ + msgpack-numpy | 5556 |
| **π₀.₅** | [`jeqcho/pi05-yam-bimanual`](https://huggingface.co/jeqcho/pi05-yam-bimanual) | WebSocket + msgpack (openpi) | 8000 |

## Quick start

```bash
# 1. One-time: download checkpoints into hf-cache/ (~50 GB for all 3)
./scripts/download_checkpoints.sh molmoact2     # just one — or omit for all 3

# 2. Bring up an inference server (Terminal A)
./scripts/run_server.sh pi05                    # or molmoact2 / gr00t-n17

# 3. Run an eval against it (Terminal B)
./scripts/run_eval.py --policy pi05 --eval ikea_10

# All flags are flat — no more `--` passthrough:
./scripts/run_eval.py --policy gr00t-n17 --eval andon_10 \
    --attempts 1 --horizon-stride 4 --left-cam-serial 349622072241

# Or open an interactive REPL for prompt iteration:
./scripts/run_repl.py --policy pi05
# > pick up the orange cube
# [arms move...]
# > /quit
```

The three policies are interchangeable — the **only** thing that changes
between them is which YAML in `configs/policy/` gets loaded. Same eval
harness, same control loop, same safety, same journal.

## Layout

```
yam-vla-tests/
├── README.md
├── pyproject.toml                       # single uv-managed venv
├── src/yam_vla/
│   ├── core/                            # POLICY ABC + observation/state codec
│   │   ├── observation.py               # YamObservation, YamStateCodec, ImageRole
│   │   ├── policy.py                    # Policy ABC, Prediction, ServerInfo
│   │   ├── config.py                    # PolicyConfig.from_path(...).build()
│   │   ├── runner.py                    # AsyncPolicyInference + helpers
│   │   └── legacy.py                    # compat shim into _archive/molmoact2-setup
│   └── policies/                        # ONE FILE PER VLA — symmetric
│       ├── molmoact2.py                 # HTTP + json_numpy adapter
│       ├── gr00t_n17.py                 # ZMQ + msgpack-numpy adapter
│       └── pi05.py                      # WebSocket + openpi-client adapter
├── configs/policy/                      # per-policy YAML (drift per-checkpoint)
│   ├── molmoact2.yaml
│   ├── gr00t-n17.yaml
│   └── pi05.yaml
├── servers/                             # SERVER-SIDE LAUNCH — one dir per VLA
│   ├── molmoact2/run.sh
│   ├── gr00t-n17/{run.sh, yam_config.py, offline_shim.py}
│   └── pi05/{run.sh, register_yam_pi05.py}
├── evals/                               # CLIENT-SIDE TASK LISTS — one dir per eval
│   ├── _harness/                        # shared CSV writer + YAML loader
│   ├── andon_10/tasks.yaml              # 10 free-form bimanual tasks
│   └── ikea_10/tasks.yaml               # 10 IKEA 1-page-assembly products
├── scripts/                             # top-level entry points
│   ├── run_server.sh                    # → servers/<policy>/run.sh
│   ├── run_eval.py                      # → start_session(policy, eval_def)
│   └── download_checkpoints.sh
├── docs/handoffs/                       # legacy HANDOFFs preserved for reference
├── _archive/                            # git-subtree-merged history of:
│   ├── molmoact2-setup/                 #   github.com/jeqcho/molmoact2-setup
│   ├── eval-yam/                        #   github.com/jeqcho/eval-yam
│   ├── grootn1.7-exploration/           #   github.com/jeqcho/grootn1.7-exploration
│   ├── dreamzero-exploration/           #   github.com/jeqcho/dreamzero-exploration
│   ├── ikea-10/                         #   github.com/jeqcho/ikea-10
│   └── reports/                         #   github.com/jeqcho/reports
├── hf-cache/                            # gitignored — model weights
└── logs/                                # gitignored — per-run server stdout
```

## The unified inference contract

Every policy implements the same `Policy` ABC. The eval/REPL/control
loop sees ONE interface; per-VLA wire formats live entirely inside the
backend implementation.

```python
from yam_vla.core import PolicyConfig, YamObservation, ImageRole
import numpy as np

# 1) Build a policy from YAML
policy = PolicyConfig.from_path("configs/policy/pi05.yaml").build()

# 2) Construct one canonical observation (HWC uint8 RGB, 14-D state)
obs = YamObservation(
    images={
        ImageRole.TOP:         top_frame,         # (H,W,3) uint8 RGB
        ImageRole.LEFT_WRIST:  left_frame,
        ImageRole.RIGHT_WRIST: right_frame,
    },
    state=arm_state,                              # (14,) float32
    prompt="put the orange cube into the box",
)

# 3) Run inference
pred = policy.predict(obs, timeout_s=5.0)
# pred.actions   -> (N, 14) float32 ABSOLUTE joint targets, canonical YAM layout
# pred.rtt_ms    -> wall-clock RTT including transport
# pred.horizon   -> N
# pred.server_info -> per-backend timing dict
```

The same 3 lines work for all three backends — the only thing that
changes is which YAML you load.

### What lives where (per-VLA quirks)

| Quirk | Where | Why |
|---|---|---|
| GR00T's `(B=1, T=1)` leading dims + 4-key state split | `policies/gr00t_n17.py` | Server expects it; YamStateCodec.split() owns the layout |
| π₀.₅'s CHW image transpose + `base_0_rgb` canonical keys | `policies/pi05.py` | Agilex fork's AlohaInputs zero-fills unknown keys silently |
| π₀.₅'s `(50, 32) → (:, :14)` pad-strip | `policies/pi05.py` | Model returns padded internal shape |
| GR00T's relative-arm decode | server-side (already absolute on wire) | `StateActionProcessor.unapply_action` runs in-server |
| MolmoAct's `num_steps` flow-matching knob | `**opts` keyword | Only MolmoAct uses it; others ignore |
| Image-key renames (top → top_cam / top / base_0_rgb) | `configs/policy/<name>.yaml` | Changes per-checkpoint, not per-code |

## Adding a new VLA

1. Drop `src/yam_vla/policies/<name>.py` implementing `Policy.predict`.
2. Add it to the `_LAZY` registry in `src/yam_vla/policies/__init__.py`.
3. Write `configs/policy/<name>.yaml` (image keys, transport, defaults).
4. Add `servers/<name>/run.sh` to bring up the server.
5. `./scripts/run_eval.py --policy <name> --eval <existing_eval>` works.

## Adding a new eval

1. Create `evals/<name>/tasks.yaml` (see existing for schema).
2. Optionally add `evals/<name>/eval.py` if you need custom scoring.
3. `./scripts/run_eval.py --policy <any> --eval <name>` works.

No code changes are required if you only need the standard per-attempt
CSV + journal logging — which is the same across all 3 policies (the
journal entry is tagged `[policy=…]` so cross-policy comparison is
trivial later).

## Hardware prereqs

Same as the legacy per-VLA repos:

```bash
# Cameras + arms reachable (uses i2rt's venv)
/home/andon/yam-tests/i2rt/.venv/bin/python \
    _archive/molmoact2-setup/scripts/list_cams.py

/home/andon/yam-tests/i2rt/.venv/bin/python \
    _archive/molmoact2-setup/scripts/preflight.py \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --skip-server
```

The legacy `molmoact2-setup/scripts/` tree still lives under
`_archive/` and is imported via `yam_vla.core.legacy` — see
`src/yam_vla/core/legacy.py` for the rationale (don't duplicate ~1500
lines of validated hardware glue).

## What's deliberately NOT done in this refactor

The three big follow-ups from the original design landed in subsequent
commits and are no longer pending:

- ✅ **Lifted hardware/safety/journal into `core/`.** Now in
  `core/{hardware,safety,journal,observability,control_loop}.py`. The
  old `core/legacy.py` shim and its monkey-patches are deleted.
- ✅ **New REPL** at `scripts/run_repl.py` — interactive prompt loop on
  top of the same control loop the eval harness uses.
- ✅ **Eval harness on pure new-code.** `evals/_harness/runner.py` calls
  `core.run_attempt` directly, no more `_yc.main()` indirection.

Remaining smaller TODOs:

- **More eval task lists.** Only `andon_10` + `ikea_10` ship today; the
  "easy-10" diagnostic suite is designed but not implemented as a
  `tasks.yaml`.
- **CSV columns are stable but minimal.** No per-attempt score-by-atomic
  field for IKEA partial credit — operator scores the attempt as one
  unit. Easy to extend.

## See also

- `docs/handoffs/` — original HANDOFFs from each subsumed repo
- `_archive/<name>/HANDOFF.md` — legacy bring-up docs (still accurate
  for the server side)
- `_archive/molmoact2-setup/journal.md` — the cumulative research
  journal; all three policies append here with `[policy=…]` tags
