# servers/pi05 — OpenPI Pi-0.5 server for `jeqcho/pi05-yam-bimanual`

This is the heaviest server to bring up in eval-yam, because the
`yam_pi05` training config is **not in upstream openpi** — it lives in
the trainer's private fork. We work around that with a runtime
registration shim (`register_yam_pi05.py`) that appends a matching
`TrainConfig` to openpi's `_CONFIGS` list right before `serve_policy.py`
looks it up.

## Files

| File                    | What it does                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `run_server.sh`         | Launches openpi's `scripts/serve_policy.py` against `--policy.config=yam_pi05 --policy.dir=<ckpt>` |
| `register_yam_pi05.py`  | Imported as a side-effect; appends `TrainConfig(name="yam_pi05", ...)` to `_CONFIGS`               |
| `openpi/` (untracked)   | Place an openpi clone here, or set `OPENPI_DIR=/elsewhere`                                         |

## Why register at runtime

The HF model card says to load via:

```python
from openpi.training import config as oc
train_cfg = next(c for c in oc._CONFIGS if c.name == "yam_pi05")
```

But upstream `_CONFIGS` only contains `pi05_aloha / pi05_droid / pi05_libero`.
The `yam_pi05` entry must be added before that `next(...)` runs. The
cleanest way (without forking openpi) is a module that, at import,
appends a `TrainConfig`. `run_server.sh` imports the shim before invoking
`serve_policy.py` (via a small `runpy` wrapper) so the registration is
in place when serve_policy parses `--policy.config`.

## Reconciling with your training fork

The template in `register_yam_pi05.py` is the most-likely guess from the
HF model card + `assets/jeqcho/yam-bimanual-merged-v2-train/norm_stats.json`
shape. The fields you may need to change to match the actual training
config:

```python
data=LeRobotAlohaDataConfig(
    repo_id="jeqcho/yam-bimanual-merged-v2-train",   # may differ
    assets=AssetsConfig(
        assets_dir=None,                              # leave None to use bundled
        asset_id="jeqcho/yam-bimanual-merged-v2-train",
    ),
    adapt_to_pi=False,                                # almost certainly False for YAM
    use_delta_joint_actions=True,                     # check the fork
)
```

**Especially watch for:** if the fork defined a custom `YamInputs/YamOutputs`
transform pair (instead of reusing `aloha_policy.AlohaInputs`), the
observation **image keys are different** — typically something like
`top / left_wrist / right_wrist` instead of `cam_high / cam_left_wrist /
cam_right_wrist`. Mirror the change in
`../../scripts/yam_backends.py:Pi05WebsocketBackend.IMG_KEY_*`.

## Server side requirements

- **VRAM**: Pi-0.5 (~3B + paligemma 3B = ~6B params) needs ~14 GB in bf16.
  Sharing the RTX 5090 (32 GB) with the MolmoAct2 server (~21 GB) is not
  possible — only run one at a time.
- **Python**: 3.11 (openpi's pin).
- **Wheels**: `jax[cuda12]==0.5.3`, `torch==2.7.1`, `flax==0.10.2`,
  `orbax-checkpoint==0.11.13`. `uv sync` inside the openpi clone resolves
  all of it.
- **Port**: 8000 (openpi default; override with `PORT=8001 ./run_server.sh`).

## Client side requirements

Only one package, into the i2rt venv:

```bash
VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv uv pip install \
    'openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client'
```

This installs `openpi_client.websocket_client_policy.WebsocketClientPolicy`
(used by `Pi05WebsocketBackend`) and openpi's vendored msgpack-numpy codec.
No JAX on the client.

## First-run sanity check (no robot)

You can verify the server-side stack independent of hardware by sending a
synthetic observation. The simplest check is to confirm the server is up
and the policy loaded:

```bash
# Terminal A
./run_server.sh   # wait for "Listening on 0.0.0.0:8000"

# Terminal B (i2rt venv)
/home/andon/yam-tests/i2rt/.venv/bin/python <<'EOF'
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
p = WebsocketClientPolicy(host="127.0.0.1", port=8000)
obs = {
    "state": np.zeros(14, dtype=np.float32),
    "images": {
        "cam_high":        np.zeros((3, 240, 320), dtype=np.uint8),
        "cam_left_wrist":  np.zeros((3, 240, 320), dtype=np.uint8),
        "cam_right_wrist": np.zeros((3, 240, 320), dtype=np.uint8),
    },
    "prompt": "warmup",
}
out = p.infer(obs)
print("actions shape:", out["actions"].shape)   # should be (16, 14)
print("server_timing:", out.get("server_timing"))
EOF
```

If you get `KeyError` on something other than `actions`, the image keys
in `register_yam_pi05.py` likely don't match what the data transform
expects.
