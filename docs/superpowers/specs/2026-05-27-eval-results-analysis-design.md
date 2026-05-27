# Eval results analysis tooling — design

**Date:** 2026-05-27
**Status:** Approved, ready for implementation plan
**Author:** brainstormed with Claude (Opus 4.7)

## Goal

After each eval run, one command produces a multi-panel PNG + a markdown table that answers: **for this (eval, policy), where did the policy fail by task and by primitive?** Keep the data pipeline METR-compatible so we can bolt on the time-horizon fit later without rewiring ingestion.

The next run will produce ~40 attempts (4 tasks × 10 reps × 1 policy `molmoact2`) under `evals/bimanual_easy_bench_4/results/molmoact2/`. The tool needs to be working before that data lands.

## Non-goals (v1)

These are deferred — see "Implementation that remains" at the bottom.

1. The METR logistic fit / halvings / time-horizon plot.
2. A real `instance_id` axis in the writer (today `instance_id = task_id`).
3. Categorical `failure_mode` column (today free-text `notes`).
4. Per-policy `release_date` metadata.
5. Cross-policy side-by-side comparison plots.
6. Migrating the legacy `andon_10` CSV schema (the shim loads it; that's enough).

## Architecture

### File tree (new/modified)

```
yam-vla-tests/
├── src/yam_vla/analysis/           # NEW
│   ├── __init__.py
│   ├── aggregate.py                # load_results, schema shim, primitive join
│   ├── plots.py                    # make_summary_figure
│   └── report.py                   # render_markdown
├── scripts/
│   └── analyze.py                  # NEW — thin argparse CLI
├── plots/<eval>/<policy>/          # NEW output dir
│   └── summary_<runs-tag>.png
├── reports/<eval>/                 # NEW output dir
│   └── <policy>_<runs-tag>.md
└── tests/analysis/                 # NEW
    ├── test_aggregate.py
    └── fixtures/
        ├── new_format.csv
        ├── old_format.csv          # andon_10-style
        └── tasks.yaml
```

### Data flow

```
evals/<eval>/results/<policy>/*.csv ──┐
                                       ├─→ aggregate.load_results(eval, policy)
evals/<eval>/tasks.yaml ──────────────┘                │
                                                       ▼
                                          canonical pandas DataFrame
                                                       │
              ┌─────────────────────────────┬──────────┴──────────────────┐
              ▼                             ▼                             ▼
        plots.make_summary_figure    report.render_markdown      (future: metr.py)
              │                             │
              ▼                             ▼
  plots/<eval>/<policy>/        reports/<eval>/<policy>_*.md
  summary_<tag>.png
```

The DataFrame is the contract. Plots and reports each consume it independently; the future METR fitter consumes the same shape.

### Canonical DataFrame columns

One row per attempt.

| column        | type    | notes                                                                  |
|---------------|---------|------------------------------------------------------------------------|
| `eval`        | str     | e.g., `bimanual_easy_bench_4`                                          |
| `policy`      | str     | e.g., `molmoact2`                                                      |
| `model_id`    | str     | e.g., `allenai/MolmoAct2-BimanualYAM`                                  |
| `task_id`     | str     | from CSV                                                               |
| `primitive`   | str?    | from `tasks.yaml meta.primitive`, NaN if eval doesn't define one       |
| `instance_id` | str     | **v1: equals `task_id`** — placeholder for future per-task variations  |
| `attempt`     | int     | from CSV                                                               |
| `trial_id`    | int     | **v1: equals `attempt`**                                               |
| `status`      | str     | `success` / `failure` / `incomplete` / `crash` / `skip`                |
| `success`     | bool    | derived: `status == "success"`                                         |
| `duration_s`  | float   | from CSV                                                               |
| `chunks`      | int     | from CSV                                                               |
| `rtt_ms_mean` | float   | from CSV                                                               |
| `prompt_kind` | str     | from CSV (`full` / `atomic_N` / `edited`)                              |
| `notes`       | str     | from CSV (free text)                                                   |
| `timestamp`   | str     | from CSV (ISO format)                                                  |
| `csv_source`  | str     | filename — for debugging which run produced this row                   |

## Component contracts

### `aggregate.load_results(eval_name: str, policy: str, results_root: Path | None = None) -> pd.DataFrame`

- Globs `evals/<eval_name>/results/<policy>/results_*.csv`, concatenates.
- **Schema shim**: detects old `andon_10` columns (`task_num, task_name, n_chunks, mean_rtt_ms, max_rtt_ms, mean_horizon_span, clip_pct, …`) and renames to canonical column names. Missing columns become NaN.
- **Primitive join**: loads `evals/<eval_name>/tasks.yaml`, builds `{task_id → meta.primitive}` map. If a task lacks `meta.primitive` (e.g., `ikea_10`), `primitive` is NaN for those rows.
- Adds `success: bool = (status == "success")`.
- Returns empty DataFrame (with correct columns) if no CSVs found — caller decides if that's an error.
- **No dedup**: if a session was resumed and re-ran the same attempt number with a fresh timestamp, both rows are kept. The markdown report flags duplicate `(task_id, attempt)` pairs so the user can decide.

### `plots.make_summary_figure(df: pd.DataFrame, eval_name: str, policy: str) -> matplotlib.figure.Figure`

3-panel horizontal layout (~14×4.5 in, 150 DPI):

1. **Per-task success rate** — horizontal bar chart with **95% Wilson CIs**.
   Tasks sorted by point estimate descending.
   Annotated with `k/n` next to each bar.
2. **Per-primitive success rate** — horizontal bar chart, pooled across tasks sharing the primitive.
   Annotated with n.
   Hidden if `df["primitive"].isna().all()` (e.g., `ikea_10`).
3. **Status mix** — stacked horizontal bar showing fraction of `success / failure / incomplete / crash` across all attempts.

Title: `{eval}  ·  {policy}  ·  N={n_attempts}  ·  {date_min}—{date_max}`.

Wilson CI implementation: prefer `statsmodels.stats.proportion.proportion_confint(k, n, method='wilson')`. If statsmodels not installed, inline the closed-form formula (well-known, ~10 lines).

### `report.render_markdown(df: pd.DataFrame, eval_name: str, policy: str) -> str`

Returns a markdown string with:

- **Headline** — `23/40 success = 57.5% [Wilson 95%: 42.0%–71.7%]`.
- **Per-task table** — id, primitive, n, success rate, Wilson CI, mean duration_s, mean rtt_ms, top note keyword(s).
- **Per-primitive table** — primitive, n, success rate, Wilson CI, member task_ids.
- **Failure notes section** — grouped by `task_id`, lists non-empty `notes` strings (raw; categorical tagging is deferred).
- **Footer** — list of CSV files included; warnings for duplicate `(task_id, attempt)` rows.

### `scripts/analyze.py`

```
uv run scripts/analyze.py --eval bimanual_easy_bench_4 --policy molmoact2
uv run scripts/analyze.py --eval bimanual_easy_bench_4 --policy molmoact2 --out-dir custom/path/
uv run scripts/analyze.py --eval bimanual_easy_bench_4 --policy molmoact2 --no-report
```

Arguments: `--eval` (required), `--policy` (required), `--out-dir` (default `plots/<eval>/<policy>/`), `--report-dir` (default `reports/<eval>/`), `--no-plot`, `--no-report`.

Writes `summary_<runs-tag>.png` and `<policy>_<runs-tag>.md`. `<runs-tag>` is the latest CSV's timestamp suffix (e.g., `20260527_112110`), or `combined_<date>` if multiple CSVs were merged.

Prints absolute paths to stdout on success. Exit 1 with a clear message if no CSVs found.

## Error handling

| Situation                          | Behavior                                                                |
|------------------------------------|-------------------------------------------------------------------------|
| No CSVs found                      | Exit 1 with `no results found at <glob>`                                |
| `tasks.yaml` missing               | Warn, continue with `primitive=NaN`, primitive panel auto-hides         |
| Unknown CSV columns                | Log warning, ignore (forward-compat)                                    |
| Mixed-format CSVs                  | Schema shim runs per-file, then concat                                  |
| Duplicate `(task_id, attempt)`     | Both rows kept; markdown footer flags the duplicates                    |
| `statsmodels` not installed        | Fall back to inline Wilson formula                                      |

## Testing

`pytest tests/analysis/test_aggregate.py`:

- **`test_load_new_format`** — loads a 2-row new-format CSV, asserts canonical columns present and `success` derived.
- **`test_load_old_format_shim`** — loads an `andon_10`-style CSV, asserts columns renamed correctly and `primitive=NaN` (since `andon_10` lacks `meta.primitive`).
- **`test_mixed_format_concat`** — both formats together.
- **`test_primitive_join`** — fixture `tasks.yaml` with `meta.primitive`, asserts join works.
- **`test_missing_tasks_yaml`** — warns and proceeds.
- **`test_make_summary_figure_smoke`** — returns a `Figure`, no exception.
- **Edge cases**: all-fail, all-success, single attempt, primitive-missing.

No pixel-diff tests on the figure — too brittle, not worth it.

## CLAUDE.md compliance

- `uv run` for the script.
- Plots are slide-quality (150 DPI, ≥11pt fonts) but not paper-figure-sized.
- Script finishes in seconds — no tmux needed.
- Output dirs (`plots/`, `reports/`) match the user's standard file layout.

## Implementation that remains (deferred)

Future Claude sessions: do these in roughly this order when the corresponding data exists.

1. **README "Analyzing results" section** — small block in `README.md` showing the CLI invocation. (Trivial, defer until after first happy-path use.)
2. **Real `instance_id` column** — `_harness/results.py:AttemptRow` adds `instance_id`, `_harness/runner.py` plumbs it through. `tasks.yaml` gains an optional `instances:` field listing setup variations per task. Aggregator no longer falls back to `instance_id = task_id`. Needed before METR fit is meaningful.
3. **Categorical `failure_mode`** — either a manual `.yaml` mapping from `notes`-keyword → category, or an LLM tagger that runs once per CSV. Adds a stacked-by-failure-mode panel to the figure.
4. **`configs/policies.yaml`** — `policy → {model_id, release_date, params_b, training_data_notes}`. Required for the time-horizon x-axis.
5. **`src/yam_vla/analysis/metr.py`** — instance-level logistic fit (per-cell α, shared β across policies in a job), pTQ at thresholds, halvings $h = -\log_2(1-\text{pTQ})$. Mirror jeqcho's `plot_logistic_per_cell_instance.py` and `plot_halvings_vs_release_instance.py`, but consume the canonical DataFrame from `aggregate.load_results`. Two new plots: `logistic_per_cell.png` (per-policy fit overlay) and `halvings_vs_release.png` (the headline time-horizon plot).
6. **`scripts/metr_fit.py`** — thin CLI: takes a list of (eval, policy) pairs, calls `aggregate` for each, runs the joint logistic, writes plots and a report.
7. **Cross-policy comparison plot** — `plots.make_comparison_figure(dfs_by_policy)` — side-by-side per-task bars across policies. Useful even before METR is wired.
8. **Migrate `andon_10` CSVs to canonical schema** — only if you want `andon_10` to be a first-class corpus alongside `bimanual_easy_bench_4`. Probably not worth it.

## Open questions worth re-checking later

- Should `incomplete` count as a failure for the rate, or be excluded from the denominator? Current decision: **include as failure** (the policy didn't accomplish the task; the operator's reasons are in `notes`). Revisit if many runs are aborted for non-policy reasons.
- "Runs-tag" naming: if multiple CSVs are combined, `combined_<date>` is fine, but if a single CSV is being analyzed, mirror its timestamp. Watch for collisions.
- Wilson CIs are exact in the binomial sense but don't account for **task heterogeneity** — when you aggregate across 4 tasks, the overall success rate has a wider true CI than Wilson reports. Acceptable for a per-(eval, policy) summary; not acceptable for cross-policy comparisons (which would need a clustered or hierarchical model — deferred with METR).
