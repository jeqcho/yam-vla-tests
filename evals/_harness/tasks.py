"""Task definition + YAML loader.

An eval YAML looks like:

    name: ikea_10
    description: 10 IKEA 1-page-assembly products
    n_attempts_default: 3
    tasks:
      - id:           LACK
        english:      side table
        instruction:  "Flip the table upside down. Place the screw on the hole ..."
        atomic_actions:
          - "Flip the table upside down."
          - "Place the screw on the hole and screw it in place lightly."
          - ...
      - id:           FISKBO
        ...

`id` is the stable handle stored in CSVs. `english` and `instruction`
are human-facing. `atomic_actions` is optional and enables per-subtask
scoring in evals that support it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvalTask:
    """One row in a tasks.yaml."""
    id:             str
    instruction:    str
    english:        str | None = None
    atomic_actions: tuple[str, ...] = field(default_factory=tuple)
    meta:           dict[str, Any]  = field(default_factory=dict)

    @property
    def has_atomic(self) -> bool:
        return bool(self.atomic_actions)


@dataclass(frozen=True)
class EvalDefinition:
    """Top-level eval = name + description + ordered task list.

    `reset_seconds_default` is the inter-attempt scene-reset countdown
    used when the operator hasn't passed --reset-seconds on the CLI.
    0 keeps the legacy operator-driven flow (no countdown between
    attempts, advance only on Enter / right-arrow).
    """
    name:                  str
    description:           str
    n_attempts_default:    int
    tasks:                 tuple[EvalTask, ...]
    reset_seconds_default: float = 0.0


def load_tasks(path: str | Path) -> EvalDefinition:
    """Load and validate an eval YAML."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    if "name" not in data or "tasks" not in data:
        raise ValueError(f"{path}: missing required keys (name, tasks)")

    tasks: list[EvalTask] = []
    for i, row in enumerate(data["tasks"]):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: task[{i}] must be a mapping")
        if "id" not in row or "instruction" not in row:
            raise ValueError(
                f"{path}: task[{i}] missing 'id' or 'instruction'"
            )
        tasks.append(EvalTask(
            id=str(row["id"]),
            instruction=str(row["instruction"]),
            english=row.get("english"),
            atomic_actions=tuple(row.get("atomic_actions") or ()),
            meta={k: v for k, v in row.items()
                  if k not in {"id", "instruction", "english", "atomic_actions"}},
        ))

    return EvalDefinition(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        n_attempts_default=int(data.get("n_attempts_default", 3)),
        tasks=tuple(tasks),
        reset_seconds_default=float(data.get("reset_seconds_default", 0.0)),
    )
