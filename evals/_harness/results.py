"""Per-attempt CSV writer for VLA evals.

One row per attempt, columns chosen to support cross-policy comparison
later (which is why `policy` and `model_id` are always present):

    timestamp, policy, model_id, eval, task_id, attempt, status,
    duration_s, chunks, rtt_ms_mean, rtt_ms_p95, rtt_ms_max,
    horizon_arm_mean, clip_rate, prompt_kind, prompt_text, notes

Files land at `evals/<name>/results/<policy>/results_<timestamp>.csv`,
which keeps resume support simple (per-policy subdir).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Stable column order. Adding a new column is fine; reordering is not
# (it'd break any downstream pandas/sql ingest pinned to this layout).
CSV_FIELDS: tuple[str, ...] = (
    "timestamp",
    "policy",
    "model_id",
    "eval",
    "task_id",
    "attempt",
    "status",
    "duration_s",
    "chunks",
    "rtt_ms_mean",
    "rtt_ms_p95",
    "rtt_ms_max",
    "horizon_arm_mean",
    "clip_rate",
    "prompt_kind",
    "prompt_text",
    "notes",
)


@dataclass
class AttemptRow:
    timestamp:        str
    policy:           str
    model_id:         str
    eval:             str
    task_id:          str
    attempt:          int
    status:           str           # success | failure | skip | crash
    duration_s:       float = 0.0
    chunks:           int = 0
    rtt_ms_mean:      float = 0.0
    rtt_ms_p95:       float = 0.0
    rtt_ms_max:       float = 0.0
    horizon_arm_mean: float = 0.0
    clip_rate:        float = 0.0
    prompt_kind:      str = "full"  # full | atomic_N | edited
    prompt_text:      str = ""
    notes:            str = ""

    def asdict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in CSV_FIELDS}


class ResultsWriter:
    """Append-only per-policy CSV writer with header-on-create."""

    def __init__(self, base_dir: str | Path, policy: str, eval_name: str):
        self.base_dir = Path(base_dir)
        self.policy = policy
        self.eval_name = eval_name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = self.base_dir / policy
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"results_{ts}.csv"
        self._opened = False

    def write(self, row: AttemptRow) -> None:
        new_file = not self._opened
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
            if new_file:
                w.writeheader()
                self._opened = True
            w.writerow(row.asdict())
