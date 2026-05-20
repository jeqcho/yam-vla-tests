# Handoff: MolmoAct2 on Bimanual YAM

This is the short version. The long version is `/home/andon/yam-tests/reports/molmoact2-setup.md`.

## Bring the system online (assume both arms + 3 cameras now plugged in)

```bash
# 1. Confirm everything is visible
/home/andon/yam-tests/i2rt/.venv/bin/python scripts/list_cams.py
# Note the three serial numbers.

/home/andon/yam-tests/i2rt/.venv/bin/python scripts/preflight.py \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --skip-server                        # server isn't started yet

# Expect: cameras OK x3, CAN OK x2, arms init OK with sensible joint positions.

# 2. Start the inference server (Terminal A)
./scripts/run_server.sh
# Wait for "Listening on 0.0.0.0:8202" — first run also downloads any
# remaining HF blobs and patches modeling_molmoact2.py.

# 3. Health check (Terminal B)
curl http://127.0.0.1:8202/act
# Expect num_cameras=3, state_dim=14, norm_tag="yam_dual_molmoact2".

# 4. Dry run the policy — NO arm movement
./scripts/run_client.sh \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --top-cam-serial   <T> \
    --left-cam-serial  <L> \
    --right-cam-serial <R> \
    --dry-run

# 5. Eyeball the printed actions. They should be (14,) floats in radians
#    in the rough vicinity of the current joint positions (deltas << 1 rad).
#    Then drop --dry-run to actually run.
```

## If the server crashes at startup

- `RuntimeError: CUDA error: no kernel image is available` → the cu128 torch
  swap didn't take effect; redo `uv sync` from `molmoact2-setup/`.
- `mat1 and mat2 must have the same dtype` → the bf16 patches in
  `host_server_yam.py` didn't apply; look at the `[INFO] Applied patches`
  log line; if it says "needle not found" repeatedly, fall back to
  `--dtype float32` (RTX 5090 has the VRAM for it).
- `extra_special_tokens` error → bump transformers within 4.57.x.

## If the arms behave incoherently

Most likely cause: **left/right swap, or "top" camera being mounted as something else**. The model was trained with a fixed camera order. To check:

1. Run `--dry-run` and physically move only the LEFT arm a few cm.
2. Watch which of `left_cam` / `right_cam` shows it (open in image viewer or save frames).
3. Re-assign serials if needed.

## What's local-only

The `molmoact2-setup/` git repo has one commit on `main`, no remote. Push when ready:

```bash
gh auth login
gh repo create yam-molmoact2 --source=/home/andon/yam-tests/molmoact2-setup --private --push
```

(Same goes for `/home/andon/yam-tests/reports/`.)

## Stuff I didn't do (because hardware-or-judgment call)

- ❌ Move the real arm with the policy (deliberately deferred to your supervision)
- ❌ Identify which gripper variant is on each arm (need eyes on the hardware)
- ❌ Confirm left ↔ can0 vs left ↔ can1 mapping (need eyes on the hardware)
- ❌ `git push` (no `gh auth`)
- ❌ Verify the second arm exists — `can1` had zero packets when I probed
