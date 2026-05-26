# DreamZero exploration — journal

## 2026-05-21 — initial scaffolding + Modal Tunnels pivot (Claude, autonomous)

**Goal:** scaffold a DreamZero (WAM, NVIDIA GEAR Lab) exploration in
`dreamzero exploration/`, mirroring `molmoact2-setup/`. Per user request,
re-engineer the inference path around the [Physical Intelligence × Modal
"real-time inference for robots" blog](https://modal.com/blog/physical-intelligence-runs-real-time-remote-inference-for-robotic-control-on-modal),
prep the Ai2 BimanualYAM data for a fine-tune, and launch.

### Constraints discovered

1. **DreamZero needs ≥2 GPUs** (NCCL via `torch.distributed.run --nproc_per_node 2`).
   14B BF16 weights ~28 GB. Local RTX 5090 (single, 32 GB) cannot run it.
   → All inference and training on Modal.
2. **No public DreamZero-YAM checkpoint exists.** Only `DreamZero-DROID`
   (deployable, single-arm Franka) and `DreamZero-AgiBot` (LoRA base).
3. **Ai2 BimanualYAM data is LeRobot v3.0**, DreamZero loader expects **v2**.
   Conversion is non-trivial (sharded → per-episode, per-camera asymmetric
   shards, per-episode task strings).
4. **Modal Tunnels are the right transport.** PI's QUIC/UDP portal isn't a
   public API; `modal.forward(port)` is what the blog references first.

### Decisions

- **Inference path:** `modal.forward(8000)` HTTPS tunnel forwarding to a
  containerized `socket_test_optimized_AR.py` on H100:2. WebSocket upgrades
  pass through `wss://`. Print URL to stdout for the local launcher to grep.
- **Data prep:** custom Modal job (`modal/prepare_yam_data.py`) that does
  the full v3→v2 conversion:
    1. parquet shards split per-`episode_index` to v2 per-episode parquets
    2. per-camera video shards cut by timestamp via ffmpeg (using
       `meta/episodes/chunk-NNN/file-NNN.parquet` for the file_index +
       from_timestamp + to_timestamp per (episode, camera))
    3. `annotation.task` injected per row from `meta/tasks_annotated.parquet`
       (Ai2 keys by `episode_index`, not `task_index`)
    4. upstream `convert_lerobot_to_gear.py` runs to compute stats +
       relative_stats + modality.json
    5. video keys post-edited to add `_camera-images-rgb` suffix so the
       DreamZero YAM YAML resolves them
- **One-command inference deliverable:** `scripts/run_inference.sh droid`
  brings up the server, watches the log for the URL, runs a smoke test,
  reports ✅, tears down. Pass `--keep` to leave running.
- **Did NOT auto-launch the full 100k fine-tune** — that's ~$200 of credits.
  The smoke fine-tune (200 steps, ~$10) is a separate command the user can
  run after eyeballing the prep deliverable.

### Bugs surfaced during the work

1. **`assert DREAMZERO_DIR.is_dir()` at module top-level** crashed inside
   Modal containers (where the local-clone path doesn't exist). Moved to
   local-entrypoint only.
2. **Ai2 uses v3 (`file-NNN.parquet`)**, my initial code expected v2
   (`episode_NNNNNN.parquet`). Rewrote the splitter to handle v3.
3. **Per-camera video shards are asymmetric** (top: 3, left: 2, right: 2;
   size-capped per-camera). Naive file-NNN↔file-NNN mapping broke. Switched
   to `meta/episodes/` metadata for per-camera file_index + timestamps.
4. **Per-episode task strings are in `tasks_annotated.parquet` keyed by
   `episode_index`**, not in `tasks.parquet` keyed by `task_index`. Rewrote
   the loader.
5. **pandas chokes on pyarrow nested-list dtypes** in `meta/episodes/`
   sidecars (huge stats blobs). Read via pyarrow Table API directly with
   `columns=[...]` to skip the heavy stats columns.
6. **flash-attn with `--no-build-isolation`** needs `packaging`, `ninja`,
   `setuptools` pre-installed. Added them to the Modal images.

### Files

```
modal/dreamzero_server.py            # modal.forward() Tunnels, H100:2
modal/dreamzero_finetune.py          # H100:4 yam_training.sh wrapper
modal/prepare_yam_data.py            # v3→v2 + ffmpeg cuts + GEAR meta
scripts/run_inference.sh             # ★ one-command e2e inference
scripts/run_finetune_modal.sh
scripts/inspect_prep.sh
scripts/smoke_test_remote.py
scripts/dreamzero_yam_client.py
pyproject.toml                       # uv-managed local venv
README.md, HANDOFF.md
```

Full report at `/home/andon/yam-tests/reports/dreamzero-setup.md`.

### Open at session-end

- Prep job running on Modal (Stage 2/5, ffmpeg cuts ~30 min ETA).
- Inference server cold-start retrying after a flash-attn build fix.
- Once both clear, smoke fine-tune (~$10) is the next gate before the full
  100k run (~$200).
