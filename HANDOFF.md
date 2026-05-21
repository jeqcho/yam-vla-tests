# DreamZero exploration — HANDOFF

**Working dir:** `/home/andon/yam-tests/dreamzero exploration/`
**Sibling reference:** `molmoact2-setup/` (the same shape but for Ai2 MolmoAct2)
**Status:** scaffolding done; no Modal app deployed yet; no YAM checkpoint exists.

## What's runnable right now

1. `./scripts/run_modal_server.sh droid` — deploys
   [`GEAR-Dreams/DreamZero-DROID`](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
   onto Modal H100:2 and prints a `wss://` URL. **Costs ~$10/hr while up;
   stop with `modal app stop dreamzero-droid` when done.**
2. `uv run python scripts/smoke_test_remote.py --url <wss://…> --schema droid`
   — confirms inference returns `(N, 8)` finite actions. This is the
   single-command "does the thing actually work" test.

## What's blocked

3. **DreamZero-YAM-bimanual inference.** No public checkpoint. You either:
   - Wait for NVIDIA/Ai2 to publish one,
   - Or fine-tune via `./scripts/run_finetune_modal.sh` (see below).
4. **`scripts/dreamzero_yam_client.py` against a real bimanual YAM.** Code is
   in place but refuses to run against a DROID server (wrong embodiment); it
   needs a YAM checkpoint.

## Manual prerequisites for fine-tuning

The Modal fine-tune launcher needs a HF dataset id pointing at a LeRobot v2
YAM tree with `meta/modality.json` declaring the DreamZero schema. None of
the public Ai2 datasets ship that exact `modality.json` — they ship the
MolmoAct2 schema (state(14,), action(14,), but with index slices laid out as
in `host_server_yam.py`, not as the 32–46 layout DreamZero's
`scripts/open_loop_yam.py` expects).

**To prepare data:**
1. Pick an Ai2 BimanualYAM dataset from
   <https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset-69f81e17b140ec34f430a35e>
   or use your own teleop in LeRobot v2 format.
2. Write/copy a `meta/modality.json` matching DreamZero's expectations.
   See `REPORT_dreamzero_setup.md` → "Preparing YAM data" for the exact JSON.
3. Set `meta/embodiment.json` → `{"embodiment_tag": "yam"}`.
4. Push to a private HF dataset repo.
5. `./scripts/run_finetune_modal.sh <that-repo-id> run1 200` — short smoke
   run, ~$5–$10. If it converges, re-run with `--max-steps 100000`.

## Cost expectations

| Step | Hardware | Wallclock | Modal $ |
|---|---|---|---|
| Cold-start DROID server (first call) | H100:2 | ~10 min | ~$1.7 |
| Each subsequent inference | H100:2 | ~3 s | ~$0.01 |
| Server idle (until scaledown @ 600 s) | H100:2 | varies | ~$10/hr |
| Fine-tune 200-step smoke | H100:4 | ~30 min | ~$10 |
| Fine-tune full 100k steps | H100:4 | ~12 hr | ~$200 |

## Outstanding decisions for the user

1. **Do you have YAM bimanual play data already?** If yes — point
   `run_finetune_modal.sh` at it; that unblocks task 3.
2. **GPU budget for fine-tuning.** Default is H100:4 (~$10/hr). Override with
   `FT_GPU=H100:8` for half-wallclock, double cost. Or `FT_GPU=A100-80gb:4`
   for lower cost (slower; bf16 only with --no-tf32 might be needed).
3. **DROID-on-bimanual-YAM probe?** I've kept this blocked behind
   `--allow-droid-server` because sending bimanual state to a single-arm
   model is undefined behaviour. Useful only if you want a "model loads,
   server returns something" confirmation.

## Pointers

- Wire format details: `REPORT_dreamzero_setup.md` → "Wire protocol"
- Modality schema: `dreamzero/groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml:267-291`
- Offline open-loop YAM eval (no server needed, but needs a YAM checkpoint):
  `dreamzero/scripts/open_loop_yam.py`
- DreamZero paper: <https://arxiv.org/abs/2602.15922>
