# experimental/rtc — Real-Time Chunking benchmark for BimanualYAM

This folder contains an A/B-benchmarkable RTC inference stack that runs
alongside the existing synchronous + async-time-aligned setup. Nothing here
modifies the main `scripts/` or `molmoact2/` trees; the legacy :8202 server
and `scripts/yam_client.py` are untouched.

See [INVESTIGATION.md](./INVESTIGATION.md) for the design rationale (why
option b: server-side lerobot policy, client keeps the existing safety
harness).

## What's in here

| File                  | Purpose                                                        |
|-----------------------|----------------------------------------------------------------|
| `INVESTIGATION.md`    | Phase RTC-1 — integration path analysis, decision, blockers   |
| `host_server_rtc.py`  | RTC-enabled inference server on port 8203                     |
| `run_server_rtc.sh`   | Launches the server with sensible defaults                    |
| `yam_client_rtc.py`   | Client that drives the arms using RTC's leftover queue        |
| `run_client_rtc.sh`   | Sister of `scripts/run_client.sh`, points at :8203            |
| `requirements.txt`    | Extra pip install: Ai2's lerobot fork (molmoact2-policy branch) |

## One-time setup

The RTC server needs a SEPARATE Python 3.12 venv (`.venv-rtc`) because
Ai2's lerobot fork (`molmoact2-policy` branch) requires Python ≥3.12,
while the main `molmoact2-setup/.venv` is pinned to 3.11. The two
servers (`:8202` legacy, `:8203` RTC) run side-by-side from different
venvs.

Bootstrap the RTC venv from scratch (~5 min, ~5 GB):

```bash
cd /home/andon/yam-tests/molmoact2-setup

# 1. Create the venv on Python 3.12
uv venv --python 3.12 .venv-rtc

# 2. Install torch+CUDA from PyTorch's index (cu128 = RTX 5090 / Blackwell)
VIRTUAL_ENV=$PWD/.venv-rtc uv pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install the inference stack
VIRTUAL_ENV=$PWD/.venv-rtc uv pip install \
    transformers fastapi 'uvicorn[standard]' json-numpy \
    huggingface_hub hf-transfer pillow numpy accelerate \
    safetensors einops requests scipy

# 4. Install lerobot fork (--no-deps to avoid re-resolving torch)
VIRTUAL_ENV=$PWD/.venv-rtc uv pip install --no-deps \
    'lerobot @ git+https://github.com/allenai/lerobot.git@molmoact2-policy'

# 5. Install lerobot's runtime deps that --no-deps skipped
VIRTUAL_ENV=$PWD/.venv-rtc uv pip install \
    draccus==0.10.0 opencv-python-headless gymnasium \
    termcolor tqdm packaging
```

`scipy` is needed by the FAST action tokenizer that lerobot pulls in.
`draccus`/`opencv-python-headless`/`gymnasium`/`termcolor`/`tqdm` are
declared deps of `lerobot` that `--no-deps` skipped.

No new install is needed in the i2rt venv — the RTC client runs from
that venv (it needs the i2rt SDK to drive the arms) and reuses
`scripts/yam_client.py`'s safety harness via `sys.path.insert(...)`.

## Running the stack

### 1. Start the RTC server (port 8203)

Open a fresh terminal:

```bash
./experimental/rtc/run_server_rtc.sh
```

This launches `host_server_rtc.py` with `bfloat16`, RTC enabled,
`execution_horizon=10` (boot-time default; the client overrides per request),
`max_guidance_weight=10.0` (β; matches lerobot's RTCConfig default — the
paper does not pin β to a specific value), EXP schedule (paper Eq. 5). The
server runs a two-pass warmup (with and without leftover) so both code
paths capture CUDA graphs before the first real `/act`. First load is
~30 s; warmup adds another ~10–20 s.

Health check:
```bash
curl -s http://127.0.0.1:8203/act | python -m json.tool
```
Should report `"rtc": {"enabled": true, "execution_horizon": 10, ...}`.

The legacy `:8202` server (`scripts/run_server.sh`) can run simultaneously;
both share the GPU. Expect ~26 GB VRAM total if you run both.

### 2. Run the RTC client

In another terminal:

```bash
./experimental/rtc/run_client_rtc.sh \
    --top-cam-serial 349622072241 \
    --left-cam-serial 427622271914 \
    --right-cam-serial 352122272708 \
    --move-to-ready \
    --execution-horizon 8 \
    --max-step-rad 0.05
```

(Top camera changed from a SONix V4L2 USB cam to a RealSense D435 — hence
`--top-cam-serial` instead of `--top-cam-v4l2`. Left/right D405 wrist cams
unchanged.)

Same CLI surface as `scripts/run_client.sh` plus the RTC-specific knobs:

**Horizon and delay control** (decoupled, paper-faithful):

- `--execution-horizon` (default 8): paper's `s` — the number of actions
  played between consecutive inference calls. Fixed for the whole rollout.
  The paper hints `s ≈ H/2` but `H/4` (=8) leaves more room for `d` to
  spike without underflowing the queue. Paper's constraint is
  `d ≤ s ≤ H - d`; with `H=30`, set `s ≤ 14` to keep room for `d`.
- `--rtc-rtt-buffer-size` (default 8): size of the buffer used to
  predict `inference_delay = ceil(max(buffer) / dt)` per paper Alg. 1
  line 18 (`d = max(Q)`). Max-of-buffer (NOT EMA) is the paper's choice
  because it is conservative under spikes.
- `--rtc-inference-delay-fixed INT`: pin `inference_delay` to a constant
  (ticks) instead of estimating from RTT. Useful for ablation.
- `--rtc-zero-delay`: diagnostic ablation that forces `inference_delay = 0`
  (empty frozen prefix in the mask).
- `--rtc-min-inference-delay` (default 1) and `--rtc-max-inference-delay`
  (default auto = `(H - s) // 2`): clamps on the predicted `d`.

**RTC sampler tuning** (per-request overrides, no server restart needed):

- `--rtc-max-guidance-weight FLOAT`: override `RTCConfig.max_guidance_weight`
  (β) for every request this session. The RTC paper does not pin β to a
  specific value; the server default 10.0 matches lerobot's
  `RTCConfig.max_guidance_weight`. Higher = tighter prefix anchoring;
  lower = more model freedom near chunk boundaries.
- `--rtc-schedule {linear,exp,zeros,ones}`: override
  `prefix_attention_schedule`. `exp` is the paper's Eq. 5 default;
  `linear` is lerobot's RTCConfig default (simpler, slightly different
  fade curve).
- `--rtc-debug`: turn on `RTCConfig.debug=True` per request (records
  per-step intermediate state). Off by default for speed.
- `--seed INT`: seed the flow-matching noise initialization for
  deterministic chunk generation. Useful for reproducing a specific
  rollout; leave unset for production.

Same safety as `yam_client.py`: per-tick clip, return-on-exit ramp,
journal prompt, SDK lock fix, cameras-before-arms.

### 3. Benchmark RTC against the existing async-time-aligned client

Easiest path:

1. Leave the `:8202` server running.
2. Start the `:8203` server alongside it.
3. Run trials alternately:
   ```bash
   # Existing async-time-aligned:
   ./scripts/run_client.sh --inference-mode async-time-aligned ...

   # New RTC:
   ./experimental/rtc/run_client_rtc.sh --execution-horizon 10 ...
   ```
4. Answer the journal prompt at end of each trial. The journal entries
   capture the full invocation including which runner was used; you can
   diff success/failure rates after a session.

Things to look at in the journal:

- `[boundary]` (legacy client) vs `[rtc-boundary]` (RTC client) — the
  RTC client should keep `tail_vs_a0(arm)` consistently smaller, especially
  when RTT is high.
- `horizon_span` — if RTC is doing its job, this should be comparable to
  the async-time-aligned baseline; a big delta hints at a config bug
  (wrong `norm_tag`, wrong `image_keys`, etc.).
- `clip:` lines — if the per-tick clip fires more often under RTC, the
  policy is either commanding bigger jumps (regression) or the leftover
  isn't being honored (degenerate-to-zero-delay case).

## Wire-format diff: `:8202` vs `:8203`

Both servers accept identical mandatory fields (top_cam / left_cam /
right_cam / instruction / state / num_steps). The RTC server additionally
accepts three optional fields:

```python
{
    ...
    "prev_chunk_left_over": ndarray(L, 14) float32,  # optional, NORMALIZED
    "inference_delay":      int,                     # optional, default 0
    "execution_horizon":    int,                     # optional, default 10
}
```

`prev_chunk_left_over` MUST be in the model's NORMALIZED latent space
(same scale as `actions_raw` returned by the server), not the
de-normalized joint space the robot executes. The RTC inpainting mixes
the leftover into the flow-matching trajectory before any
de-normalization, so the round-trip has to be in raw space.

Response: `actions` (de-normalized joint space, for execution),
`actions_raw` (normalized latent space, to send back as leftover),
`dt_ms`, and an `"rtc": {...}` dict that echoes leftover_len_in, exec
horizon, inference_delay, num_steps, max_guidance_weight, schedule, and
seed. The non-RTC client will ignore the extra fields; the RTC client
uses `actions_raw` for the leftover round-trip and `rtc` for sanity
logging.

## Bring-up notes (issues encountered and how they were fixed)

For posterity, here's what didn't work on the first try and how it was
resolved. All fixes are committed in the current `host_server_rtc.py`.

1. **Python version**: lerobot fork requires `>=3.12`; the main `.venv`
   is pinned to `==3.11`. Solution: separate `.venv-rtc` (see Setup).

2. **`TransitionKey` import path**: lives at `lerobot.types` on this
   branch, not `lerobot.configs.types`.

3. **`json_numpy.patch()` globally patches stdlib `json`**: breaks
   `numpy.testing` import (it calls `json.loads(..., object_hook=…
   SimpleNamespace…)`; json_numpy chains its hook around the caller's
   and fails on `"__numpy__" in SimpleNamespace`). Solution: remove the
   global patch; handlers already use `json_numpy.loads/dumps` directly.

4. **Preprocessor batch shape**: pipeline's `to_transition` defaults to
   `batch_to_transition`, which expects a FLAT dict keyed by
   `"observation.*"` and `"task"`, not a `TransitionKey`-keyed dict.

5. **`inference_action_mode` required**: must be set explicitly (we set
   `"continuous"` at config construction).

6. **`output_features` required with positive shape**: needed by
   `_output_action_dim()`. Set both `input_features` (state only) and
   `output_features` (action) with explicit `PolicyFeature(type=…,
   shape=…)`. DO NOT include image features in `input_features` — the
   normalizer iterates over them and calls `torch.as_tensor(PIL.Image)`,
   which fails.

7. **`@torch.inference_mode()` breaks RTC**: RTC's denoise_step uses
   `torch.autograd.grad` for its correction term. inference_mode kills
   grad tracking and the autograd call raises. Removed the decorator;
   model is still in `.eval()`.

8. **`inference_delay` clamping**: the client clamps the predicted delay
   to `[--rtc-min-inference-delay, --rtc-max-inference-delay]`, with the
   upper bound defaulting to `(H - s) // 2` so paper's constraint
   `d ≤ s ≤ H - d` is satisfied. If RTT is so high that the clamp fires
   regularly, the queue starts underflowing and we degenerate to
   near-sync behavior. The `underflow_ticks` counter in the periodic
   `clip:` log line shows how often the queue ran dry.

9. **Wire-API `execution_horizon` is NOT paper's `s`**: lerobot's
   `predict_action_chunk(execution_horizon=…)` kwarg is the END of the
   prefix-mask fade region, not the paper's execution horizon `s`. The
   client sends `execution_horizon = len(leftover)` so the mask is
   `ones(d), fade(d → len(leftover)), zeros(len(leftover) → H)` — which
   matches the paper's Eq. 5 with `H - s` aligned to the actual leftover
   boundary. Confusingly named in lerobot, but consistent in behavior.

## Reverting

This entire folder is self-contained. To roll back, `rm -rf experimental/rtc/`
and `git restore experimental/`. The legacy `:8202` server and client are
unaffected.
