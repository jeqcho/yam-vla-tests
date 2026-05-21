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

The RTC server needs Ai2's lerobot fork in the existing server venv. Install:

```bash
VIRTUAL_ENV=/home/andon/yam-tests/molmoact2-setup/.venv \
  uv pip install -r experimental/rtc/requirements.txt
```

This adds the lerobot package only; torch / transformers / fastapi / etc.
are already pinned in the setup venv. The lerobot fork's heavy deps (
diffusers, peft, draccus etc.) will land alongside; expect ~200 MB of new
packages and ~30 s of resolver time.

No new install is needed in the i2rt venv — the client reuses
`scripts/yam_client.py`'s safety harness via `sys.path.insert(...)`.

## Running the stack

### 1. Start the RTC server (port 8203)

Open a fresh terminal:

```bash
./experimental/rtc/run_server_rtc.sh
```

This launches `host_server_rtc.py` with `bfloat16`, RTC enabled,
`execution_horizon=10`, `max_guidance_weight=10.0`, LINEAR schedule. The
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
    --top-cam-v4l2 /dev/v4l/by-id/usb-SONix_Technology_Co.__Ltd._Streaming_Camera_SN0001-video-index0 \
    --left-cam-serial 427622271914 \
    --right-cam-serial 352122272708 \
    --move-to-ready \
    --execution-horizon 10 \
    --max-step-rad 0.05
```

Same CLI surface as `scripts/run_client.sh` plus three RTC-specific flags:

- `--execution-horizon` (default 10): how many actions per chunk get
  executed before re-querying. The remainder is sent back as
  `prev_chunk_left_over` so the server inpaints the next chunk's prefix
  smoothly.
- `--inference-delay-mode` (default `ema-rtt`): how to compute the
  `inference_delay` field sent to the server. `ema-rtt` rounds
  `EMA(RTT) / dt` up; `fixed` uses a constant; `zero` degenerates to
  vanilla chunked inference.
- `--inference-delay-ema-alpha` (default 0.5): EMA smoothing factor.

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
    "prev_chunk_left_over": ndarray(L, 14) float32,  # optional
    "inference_delay":      int,                     # optional, default 0
    "execution_horizon":    int,                     # optional, default 10
}
```

Response: same shape (`actions` + `dt_ms`), with one extra dict
`"rtc": {...}` that echoes the leftover length the server saw, exec
horizon, inference_delay, and num_steps used. The non-RTC client will
ignore that field; the RTC client uses it for sanity logging.

## Known unknowns

These are flagged in INVESTIGATION.md as well; copying here for runtime
visibility:

1. **First-run discovery**: `MolmoAct2Policy(config)` may complain about a
   missing `dataset_stats` argument despite `make_molmoact2_pre_post_processors`
   falling back to `norm_stats.json`. If it does, the fix is to hand-load
   `norm_stats.json` and pass the matching dict.

2. **`image_keys` exact names**: we use the strings spelled out in
   `norm_stats.json[metadata_by_tag][yam_dual_molmoact2][camera_keys]`,
   i.e. `observation.images.{top,left,right}`. If the lerobot processor
   has hardcoded different keys (e.g. `left_wrist`), the server will
   error at startup; this is a 1-line `CAMERA_KEYS` fix in
   `host_server_rtc.py`.

3. **`inference_delay` clamping**: the client clamps the computed delay
   to `[0, 15]` (half a chunk). If RTT is so high that this clamp fires
   regularly, RTC's prefix-attention has effectively no leftover left to
   inpaint and we degenerate to async-time-aligned behavior. The
   `horizon_arm_span` log should help spot this.

## Reverting

This entire folder is self-contained. To roll back, `rm -rf experimental/rtc/`
and `git restore experimental/`. The legacy `:8202` server and client are
unaffected.
