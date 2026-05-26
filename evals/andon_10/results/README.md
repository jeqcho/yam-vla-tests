# andon_10 results

Per-policy CSVs from each `./scripts/run_eval.py --eval andon_10` run.

Schema is the 17-column `AttemptRow` defined in
`evals/_harness/results.py` — stable across all evals so cross-policy
comparison is straightforward downstream (pandas / SQL ingest).

## Historical results

The two CSVs under `molmoact2/results_2026-05-22_*.csv` are from the
original MolmoAct-only Andon 10-task eval (May 22 hardware session, run
via the legacy `molmoact2-setup/eval-10-tasks/eval_andon_tasks.py`).
They predate the multi-backend refactor — the column set is the old
10-column variant, NOT the new 17-column schema. Treat as reference,
not benchmark baseline.
