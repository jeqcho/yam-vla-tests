"""Shared eval-harness primitives.

Each `evals/<name>/eval.py` reuses these so new evals declare ONLY
their task list (YAML) + any scoring twists, not the CSV-writer
plumbing.
"""
from evals._harness.tasks import EvalTask, load_tasks
from evals._harness.results import ResultsWriter
from evals._harness.runner import start_session

__all__ = ["EvalTask", "load_tasks", "ResultsWriter", "start_session"]
