# Rerun (.rrd) recordings

`.rrd` is Rerun's binary recording format — frames + state + actions
captured per attempt. Used to replay a run after the fact (e.g. to
extract first frames for the eval report) and to spot-check what the
arms saw vs what the policy commanded.

**This dir is `.gitignore`d** — recordings live on local disk only.

## Where to find existing recordings

18 recordings from yesterday's IKEA-10 + bring-up runs live at:

```
/home/andon/yam-tests/eval-yam/logs/rrd/        (5.6 GB, 18 files)
```

Filename convention: `YYYY-MM-DD_HHMMSS_<eval-or-policy>.rrd`. The 9
prefixed with `ikea10_eval_molmoact2` are the recordings used to
extract first-frame images for `reports/ikea_10/ikea-10_vla_eval.pdf`.

To replay:

```bash
# Live viewer
rerun /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_185120_ikea10_eval_molmoact2.rrd

# Extract first frames as PNGs (used by the report build)
python reports/ikea_10/extract_images.py \
    --rrd /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_185120_ikea10_eval_molmoact2.rrd \
    --out /tmp/frames/
```

## Where new recordings land

When `scripts/run_eval.py` (or the legacy `run_repl_*.sh --rerun`) runs
with `--rerun`, the .rrd file is auto-saved here (`logs/rrd/`). See
`evals/_harness/runner.py` for the wire-up.

## Why this isn't symlinked from the old location

The legacy logs dir at `~/yam-tests/eval-yam/logs/rrd/` is the
authoritative location until you run a new eval from the new repo.
After your first new-repo run, this dir will start accumulating
recordings here. If you want everything in one place, manually
`mv ~/yam-tests/eval-yam/logs/rrd/*.rrd ./` once.
