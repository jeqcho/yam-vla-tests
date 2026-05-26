"""Append-only research journal (markdown) — one entry per robot run.

The journal lives at `<repo>/journal.md`. Every policy appends to the
same file, with entries tagged by policy name so the timeline is a
single source-of-truth for "what happened on the robot today".

Flow per run:
    journal_start_s = time.time()
    invocation = capture_invocation()
    ...
    # at end of run:
    entry = prompt_journal_entry(journal_start_s, args)
    if entry is not None:
        write_journal_entry(JOURNAL_PATH, entry, args, invocation)
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_JOURNAL_PATH: str = str(
    Path(__file__).resolve().parents[3] / "journal.md"
)


def capture_invocation() -> str:
    """Best-effort recovery of the user's original shell invocation.

    `YAM_INVOCATION` env var is set by the top-level scripts/run_*.sh
    wrappers to preserve the shell command (including the policy/eval
    selectors). Fallback: `sys.argv` joined, which is the python-level
    view and loses shell quoting nuance.
    """
    inv = os.environ.get("YAM_INVOCATION")
    return inv if inv else " ".join(sys.argv)


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _format_args(args) -> str:
    """Render argparse Namespace as a markdown bullet list."""
    if args is None:
        return "_(none)_"
    items = vars(args) if hasattr(args, "__dict__") else dict(args)
    lines = []
    for k, v in sorted(items.items()):
        if v is None or v is False:
            continue
        sv = repr(v) if isinstance(v, str) and len(v) > 120 else str(v)
        lines.append(f"- `{k}`: {sv}")
    return "\n".join(lines) if lines else "_(none)_"


def prompt_journal_entry(start_time_s: float, args) -> Optional[dict]:
    """Interactively ask the operator how the run went.

    Returns a dict with status/notes/purpose/duration/timestamp, or
    None if the operator skipped (or stdin isn't a TTY -- CI-safe).
    """
    if not sys.stdin.isatty():
        print("[journal] stdin is not a TTY, skipping prompt", flush=True)
        return None
    if getattr(args, "no_journal", False):
        return None

    duration_s = time.time() - start_time_s
    print("\n" + "=" * 70, flush=True)
    print("Research journal -- record this run?", flush=True)
    print("=" * 70, flush=True)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Duration:    {_format_duration(duration_s)}", flush=True)
    print()
    print("How did the run go?", flush=True)
    print("  [s] success  -- task completed as intended", flush=True)
    print("  [f] failure  -- task did not complete", flush=True)
    print("  [u] unclear  -- partial / mixed", flush=True)
    print("  [enter or 'skip']  don't record", flush=True)
    sys.stdout.flush()
    try:
        choice = input("> ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n[journal] skipped", flush=True)
        return None
    if not choice or choice.startswith("skip"):
        return None
    status_map = {"s": "success", "f": "failure", "u": "unclear"}
    status = status_map.get(choice[:1])
    if status is None:
        print(f"[journal] unrecognized {choice!r}, skipping", flush=True)
        return None

    try:
        notes = input("\nWhat happened? (one line, optional)\n> ").strip()
        purpose = input("\nPurpose? (what were you testing, optional)\n> ").strip()
    except (KeyboardInterrupt, EOFError):
        notes = locals().get("notes", "")
        purpose = ""
        print("\n[journal] partial entry recorded", flush=True)

    return {
        "status": status,
        "notes": notes,
        "purpose": purpose,
        "duration_s": duration_s,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_journal_entry(path: str, entry: dict, args, invocation: str) -> None:
    """Append a single markdown entry to the journal."""
    md = [""]
    md.append("---")
    md.append(f"## {entry['timestamp']} -- {entry['status']}")
    md.append("")
    if entry.get("purpose"):
        md.append(f"**Purpose**: {entry['purpose']}")
        md.append("")
    if entry.get("notes"):
        md.append(f"**Notes**: {entry['notes']}")
        md.append("")
    md.append(f"**Duration**: {_format_duration(entry['duration_s'])}")
    md.append("")
    md.append("**Command**:")
    md.append("```")
    md.append(invocation)
    md.append("```")
    md.append("")
    md.append("**Configuration**:")
    md.append(_format_args(args))
    md.append("")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[journal] wrote {entry['status']} entry to {path}", flush=True)


__all__ = [
    "DEFAULT_JOURNAL_PATH",
    "capture_invocation",
    "prompt_journal_entry",
    "write_journal_entry",
]
