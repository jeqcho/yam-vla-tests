# reports/

Long-form documentation and published artifacts. Three categories:

| File | What |
|---|---|
| `molmoact2-setup.md` | Original MolmoAct2 bring-up writeup (2026-05-04 → 2026-05-20) |
| `dreamzero-setup.md` | DreamZero (NVIDIA GEAR) exploration writeup |
| `system-resources-2026-05-21.md` | Workstation snapshot — GPU/VRAM/disk inventory at the time the multi-backend work started |
| `ikea_10/` | **Published IKEA 10-furniture eval report** — PDF, HTML (with self-hosted fonts + per-task first frames), build scripts |

## IKEA-10 report

`ikea_10/ikea-10_vla_eval.pdf` is the 10-page PDF report covering the
2026-05-25 IKEA assembly eval — MolmoAct2 only (the GR00T-n17 and pi05
finetunes hadn't been hardware-verified yet). Each page shows one
furniture: Swedish + English name, the instruction sent to the model,
first frames of each attempt, and partial-credit scoring against the
atomic_actions list (which is now in `evals/ikea_10/tasks.yaml`).

To rebuild from `report.html`:

```bash
cd reports/ikea_10
./publish_to_docs.sh
# Or directly:
python build.py
```

The build pulls first frames from `.rrd` recordings under
`logs/rrd/` — see that dir's README.
