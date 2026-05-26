# eval-yam — multi-VLA evaluation on bimanual YAM

Drop-in replacement for `molmoact2-setup/eval-10-tasks/run_eval.sh` and
`molmoact2-setup/scripts/run_repl.sh` that lets you run the same Andon
10-task eval and interactive REPL against **three** vision-language-action
policies on the bimanual YAM rig:

| Policy      | Source repo                           | Wire format                   | Port |
| ----------- | ------------------------------------- | ----------------------------- | ---- |
| `molmoact2` | `allenai/MolmoAct2-BimanualYAM`       | HTTP POST `/act` + json_numpy | 8202 |
| `pi05`      | `jeqcho/pi05-yam-bimanual`            | WebSocket + msgpack (openpi)  | 8000 |
| `gr00t-n17` | `jeqcho/gr00t-n17-yam-bimanual`       | ZeroMQ REQ/REP + msgpack-numpy | 5556 |

The hardware path (cameras, arms, safety clipping, async fetcher, journal,
EnterStopWatcher, return-on-exit ramp) is **shared, single-sourced** —
this folder imports `yam_client`, `yam_repl`, and the helpers from
`../molmoact2-setup/scripts/` and just routes the inference call site
through a `Backend` abstraction.

## File tree

```
eval-yam/
├── README.md                              # this file
├── HANDOFF.md                             # bring-up checklist + troubleshooting
├── download_checkpoints.sh                # pulls the 3 HF repos into hf-cache/
├── scripts/
│   ├── yam_backends.py                    # Backend ABC + 3 concrete impls
│   ├── repl_yam.py                        # multi-backend REPL (main)
│   ├── run_repl_molmoact2.sh              # thin wrapper -> --policy molmoact2
│   ├── run_repl_pi05.sh                   # thin wrapper -> --policy pi05
│   ├── run_repl_gr00t-n17.sh              # thin wrapper -> --policy gr00t-n17
│   ├── run_server_molmoact2.sh            # delegates to molmoact2-setup
│   ├── run_server_pi05.sh                 # delegates to servers/pi05/run_server.sh
│   └── run_server_gr00t-n17.sh            # delegates to servers/gr00t/run_server.sh
├── eval-10-tasks/
│   ├── eval_yam_tasks.py                  # multi-backend 10-task harness (main)
│   ├── run_eval_molmoact2.sh              # thin wrapper
│   ├── run_eval_pi05.sh
│   ├── run_eval_gr00t-n17.sh
│   └── results/                           # per-policy CSVs land here (gitignored)
│       ├── molmoact2/
│       ├── pi05/
│       └── gr00t-n17/
├── servers/
│   ├── gr00t/
│   │   ├── yam_config.py                  # NEW_EMBODIMENT YAM modality (4 keys)
│   │   └── run_server.sh                  # GR00T inference server on :5556
│   └── pi05/
│       ├── register_yam_pi05.py           # runtime shim: registers yam_pi05 config
│       ├── run_server.sh                  # openpi serve_policy on :8000
│       └── README.md                      # openpi setup notes
└── hf-cache/                              # local HF cache (gitignored, big)
    └── checkpoints/
        ├── allenai_MolmoAct2-BimanualYAM/
        ├── jeqcho_pi05-yam-bimanual/
        └── jeqcho_gr00t-n17-yam-bimanual/
```

## Quick start

```bash
cd /home/andon/yam-tests/eval-yam

# 1. One-time: download checkpoints (~50 GB total -- run only what you need)
./download_checkpoints.sh                    # all three
./download_checkpoints.sh molmoact2          # just one

# 2. One-time per-policy server setup -- see HANDOFF.md for prereqs:
#    - molmoact2: nothing extra; reuses molmoact2-setup's venv
#    - gr00t-n17: reuses ../grootn1.7\ exploration/Isaac-GR00T/.venv
#                 (HF auth for nvidia/Cosmos-Reason2-2B required)
#    - pi05:      git clone openpi into servers/pi05/openpi + `uv sync`
#                 + reconcile register_yam_pi05.py with your training fork

# 3. One-time client-venv deps (i2rt venv, ~/yam-tests/i2rt/.venv):
VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv uv pip install pyzmq msgpack-numpy
VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv uv pip install \
    'openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client'

# 4. Run!
# Terminal A: server (pick one)
./scripts/run_server_molmoact2.sh
# ...or...
./scripts/run_server_pi05.sh
# ...or...
./scripts/run_server_gr00t-n17.sh

# Terminal B: REPL or 10-task eval against the same policy
./scripts/run_repl_molmoact2.sh   --left-cam-serial AAAA  ...
./eval-10-tasks/run_eval_pi05.sh  --left-cam-serial AAAA  ...
```

## Selecting a policy

Three independent ways (use whichever is convenient):

1. **The wrapper script you invoke** -- `run_repl_pi05.sh` hardcodes
   `--policy pi05` and the right server-host/port. **This is the
   recommended UX** (and what `HANDOFF.md` documents).

2. **Direct invocation with `--policy`** if you want to override server
   defaults inline:
   ```bash
   /home/andon/yam-tests/i2rt/.venv/bin/python \
       eval-10-tasks/eval_yam_tasks.py \
       --policy gr00t-n17 \
       --server-host 192.168.1.42 --server-port 5556 \
       --tasks 1,3,5 --attempts 5
   ```

3. **Environment variable overrides** without editing the wrapper:
   ```bash
   YAM_SERVER_HOST=192.168.1.42 ./scripts/run_repl_gr00t-n17.sh ...
   ```

## What `Backend` does (and why this design works)

Every VLA we care about exposes a different wire format -- MolmoAct2
(HTTP+json_numpy), Pi-0.5 (WebSocket+msgpack via openpi), GR00T (ZMQ+
msgpack-numpy). They also use different observation schemas:

| Concern             | molmoact2                    | pi05 (Aloha-style)                  | gr00t-n17                                |
| ------------------- | ---------------------------- | ----------------------------------- | ---------------------------------------- |
| Image keys          | `top_cam / left_cam / right_cam` | `images.cam_high / cam_left_wrist / cam_right_wrist` (CHW) | `video.top / left / right` (with B,T dims)             |
| State key           | `state(14,)` flat            | `state(14,)` flat                   | 4 keys: `left_arm(6) / left_gripper(1) / right_arm(6) / right_gripper(1)` |
| Language key        | `instruction`                | `prompt`                            | `language.annotation.human.task_description` |
| Action layout       | `(N, 14)` ABSOLUTE           | `(16, 14)` ABSOLUTE                 | 4 keys: `(1, 16, *)`, **arms RELATIVE / grippers ABSOLUTE** |

Each `Backend` subclass in `scripts/yam_backends.py` knows how to:
- pack the canonical YAM 14-D state + 3 HWC RGB frames into its server's
  native observation schema
- decode the server's action chunk back into `(N, 14)` ABSOLUTE joint
  positions (gr00t backend adds the current state to relative arm deltas)
- transport: HTTP requests / openpi-client websocket / pyzmq REQ socket

The factory `install_backend(backend)` monkey-patches
`yam_client.post_actions` (and `yam_repl.post_actions` -- a separate
binding via `from yam_client import post_actions`) so the existing
control loop (~2169 lines of arm/camera/safety/journal code in
`molmoact2-setup`) doesn't need to know about any of this.

## What `eval-yam` deliberately does NOT do

- **It doesn't install heavy server dependencies.** OpenPI is JAX + cu12
  (~10 GB venv) and Isaac-GR00T is similar. We point at existing clones
  rather than duplicating.
- **It doesn't modify `molmoact2-setup/`.** Imports flow one way:
  eval-yam -> molmoact2-setup (helpers). The MolmoAct2 path is unchanged.
- **It doesn't ship a `yam_pi05` config.** That config is in jeqcho's
  private openpi fork; `servers/pi05/register_yam_pi05.py` is a
  best-effort template -- reconcile it with the training fork before
  serving for real.
