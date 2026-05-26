# Handoff: eval-yam — multi-VLA eval on bimanual YAM

The short version. The long version (architecture rationale, file tree,
backend internals) is `README.md` in this directory.

## Bring the system online

Same hardware checklist as `molmoact2-setup/HANDOFF.md`:

```bash
# Cameras + arms visible
/home/andon/yam-tests/i2rt/.venv/bin/python \
    /home/andon/yam-tests/molmoact2-setup/scripts/list_cams.py

/home/andon/yam-tests/i2rt/.venv/bin/python \
    /home/andon/yam-tests/molmoact2-setup/scripts/preflight.py \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --skip-server
```

## One-time client-side install

The i2rt venv (which the eval-yam clients run from) needs two extra
packages -- ZMQ for the gr00t backend, openpi-client for the pi05 backend:

```bash
VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv uv pip install \
    pyzmq msgpack-numpy

VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv uv pip install \
    'openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client'
```

Skip the `openpi-client` line if you'll never use `--policy pi05`.

## One-time per-policy server setup

### molmoact2

Nothing new. The eval-yam server wrapper just delegates to
`../molmoact2-setup/scripts/run_server.sh`. The molmoact2-setup venv must
already be working (see `molmoact2-setup/HANDOFF.md`).

```bash
./download_checkpoints.sh molmoact2     # ~21 GB into hf-cache/
```

### gr00t-n17

Reuses the `Isaac-GR00T` clone from `../grootn1.7 exploration/`. If you
don't have it, follow `../grootn1.7 exploration/HANDOFF.md` Step 0
(important: HF auth for the gated `nvidia/Cosmos-Reason2-2B` repo) and
let `uv sync --all-extras` produce that 15-GB venv. After that:

```bash
./download_checkpoints.sh gr00t-n17     # ~6 GB into hf-cache/
```

To point at a different Isaac-GR00T clone, set `GR00T_DIR=/path/...`.

### pi05

The big one. Three steps:

1. **Clone + sync openpi** (or point `OPENPI_DIR` at an existing clone --
   ideally your training fork, which already has the real `yam_pi05`
   config registered):

   ```bash
   git clone https://github.com/Physical-Intelligence/openpi.git \
       servers/pi05/openpi
   cd servers/pi05/openpi && uv sync
   # ~10 GB venv: jax[cuda12], torch 2.7.1, flax, orbax, transformers, ...
   ```

2. **Reconcile `servers/pi05/register_yam_pi05.py`** with the actual
   `TrainConfig(name="yam_pi05", ...)` used to train
   `jeqcho/pi05-yam-bimanual`. The template uses pi05_aloha-style
   defaults that are correct in most respects but may differ in
   image-key naming or `use_delta_joint_actions`. The fastest fix: copy
   the actual entry from your training fork's
   `src/openpi/training/config.py`.

   If the trained config used image keys OTHER than the Aloha defaults
   (`cam_high / cam_left_wrist / cam_right_wrist`), also update
   `scripts/yam_backends.py:Pi05WebsocketBackend.IMG_KEY_*` to match.

3. **Download the checkpoint:**

   ```bash
   ./download_checkpoints.sh pi05         # ~12 GB Orbax JAX into hf-cache/
   ```

## Running

Each policy has three thin wrapper scripts. Pick one row:

| What you want                | Command (Terminal A: server, Terminal B: client)                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| MolmoAct2 REPL               | `./scripts/run_server_molmoact2.sh`  and  `./scripts/run_repl_molmoact2.sh  --left-cam-serial ... `   |
| MolmoAct2 10-task eval       | `./scripts/run_server_molmoact2.sh`  and  `./eval-10-tasks/run_eval_molmoact2.sh ...`             |
| Pi-0.5 REPL                  | `./scripts/run_server_pi05.sh`       and  `./scripts/run_repl_pi05.sh  --left-cam-serial ... `        |
| Pi-0.5 10-task eval          | `./scripts/run_server_pi05.sh`       and  `./eval-10-tasks/run_eval_pi05.sh ...`                  |
| GR00T-N1.7 REPL              | `./scripts/run_server_gr00t-n17.sh`  and  `./scripts/run_repl_gr00t-n17.sh  --left-cam-serial ... `   |
| GR00T-N1.7 10-task eval      | `./scripts/run_server_gr00t-n17.sh`  and  `./eval-10-tasks/run_eval_gr00t-n17.sh ...`             |

CLI flags after the wrapper name are forwarded to the underlying
`repl_yam.py` / `eval_yam_tasks.py`, so anything that works for the
molmoact2 REPL works for the others (same `--left-cam-serial`,
`--horizon-stride`, `--attempts`, `--tasks 1,3,5`, etc.).

The wrapper's choice of `--policy` and server URL/host/port is hardcoded.
Override via env var if you need to:

```bash
YAM_SERVER_HOST=remote-box.lan ./scripts/run_repl_gr00t-n17.sh ...
YAM_SERVER_URL=http://other:8202/act ./scripts/run_repl_molmoact2.sh ...
```

## Files to look at when something breaks

- `scripts/yam_backends.py` -- THE place where wire formats are defined.
  Every per-policy bug shows up here first (key names, shape transposes,
  relative-vs-absolute action conversion, msgpack codec wiring).
- `scripts/repl_yam.py` and `eval-10-tasks/eval_yam_tasks.py` -- the
  multi-backend main()s. They share argparse + hardware-bringup
  scaffolding; bugs in the inner control loop are NOT here, they are in
  `molmoact2-setup/scripts/yam_repl.run_one_attempt`.
- `servers/pi05/register_yam_pi05.py` -- if pi05 inference produces
  garbage actions or shape errors, the most likely cause is this
  template diverging from the actual training config (especially the
  image-key set).
- `../molmoact2-setup/scripts/yam_client.py` -- the source of truth for
  safety clipping (`safe_command`), the SDK lock fix, async inference
  fetcher, and the journal format. Don't edit those for backend reasons;
  use the `Backend` abstraction.

## Troubleshooting

### Server-side

| Symptom                                       | Likely cause                                                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `GatedRepoError: 401 ... Cosmos-Reason2-2B`   | gr00t-n17: HF auth not done. `hf auth login` + accept the model EULA on huggingface.co.                             |
| `CUDA error: no kernel image is available`    | gr00t-n17: Isaac-GR00T venv torch is too old for Blackwell sm_120. Run `uv sync` in the Isaac-GR00T clone.            |
| `mat1 and mat2 must have the same dtype`      | molmoact2: bf16 patches didn't apply -- look at the `[INFO] Applied patches` log.                                   |
| `ValueError: configuration 'yam_pi05' not found` | pi05: `register_yam_pi05.py` wasn't imported before serve_policy parsed `--policy.config`. Check `PYTHONPATH`.       |
| pi05 returns shape (16, 7) actions            | Your fork's `Pi0Config(action_dim=...)` is wrong, or the wrong norm_stats are being used. Verify in register_yam_pi05.py. |

### Client-side

| Symptom                                  | Likely cause                                                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError: openpi_client`     | pi05 client deps not installed. See "One-time client-side install" above.                                          |
| `zmq.error.Again` after first call       | gr00t REQ socket got into a bad state. The backend auto-resets; if it persists, restart the gr00t server.          |
| pi05 actions look random / jittery       | Image keys mismatch. Confirm `Pi05WebsocketBackend.IMG_KEY_*` matches what the fork's `yam_pi05` data transform expects. |
| gr00t arms jump at every chunk boundary  | Backend isn't adding the current state to relative arm deltas. Check `Gr00tZmqBackend.predict` (it does this).     |
| arms behave incoherently                 | left/right or top/wrist camera ordering off. **Same root cause across all three policies** -- run `--dry-run` and physically move only the LEFT arm to verify which feed shows it. |

## What's local-only

This folder is a sibling of `molmoact2-setup/` and `grootn1.7 exploration/`
under `~/yam-tests/`. It expects them to exist as siblings; relative paths
are baked into the scripts. If you `git init` this folder separately, you
need `.gitignore` to skip:

```
hf-cache/
logs/
eval-10-tasks/results/
servers/pi05/openpi/    # if you cloned it under this folder
```

## Things deliberately NOT done

- ❌ `git push` -- no remote configured.
- ❌ Actually launched a pi05 server end-to-end. The wire-format research is
  thorough but `yam_pi05`'s image keys depend on jeqcho's fork (the model
  card doesn't pin them publicly). First run will likely produce one
  shape-mismatch error pointing at the right key set.
- ❌ Verified the gr00t-n17 safetensors checkpoint loads cleanly via
  Isaac-GR00T's `Gr00tPolicy(--model-path=...)`. The existing
  grootn1.7-exploration setup loaded an experiment-cfg directory; the HF
  checkpoint may have a slightly different on-disk layout
  (model.safetensors vs sharded). If `--model-path` errors, point
  `Gr00tPolicy` at the snapshot dir under `hf-cache/` and check for
  `experiment_cfg/conf.yaml` vs `config.json`.
- ❌ Pre-installed `pyzmq + msgpack-numpy + openpi-client` into the i2rt
  venv. Listed as explicit one-time steps in this HANDOFF.
- ❌ Move the real arm with any of the three policies (the molmoact2 path
  has been validated; pi05 and gr00t-n17 paths await first hardware-side
  smoke test).
