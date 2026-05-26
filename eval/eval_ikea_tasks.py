"""IKEA 10-furniture assembly eval for bimanual YAM, multi-VLA.

Companion to eval-yam/eval-10-tasks/eval_yam_tasks.py -- same per-attempt
prompt/outcome/journal flow, same backend abstraction (MolmoAct2 / Pi-0.5
/ GR00T-N1.7), but the task list is the 10 IKEA 1-page-assembly-PDF
products curated in:

  ikea-10/reference/robotics-task-horizon/experiments/7-ikea-full-catalog/
    reports/short-list.md

Each task carries:
  - the Swedish product name (the IKEA SKU label, e.g. "LACK")
  - a short English description (e.g. "side table")
  - the natural-language instruction sent to the policy

DRAFT INSTRUCTIONS: the strings below are first-pass drafts based on
typical 1-page IKEA assembly patterns. Reconcile against each product's
actual PDF (links in short-list.md) before running an eval. The Swedish
name is included in the prompt itself because the model may have seen
it during pretraining and the brand-name anchor can help grounding.

N defaults to 3 attempts per task. Runs against any of the 3 VLAs via
--policy {molmoact2,pi05,gr00t-n17} -- delegates to the same backends as
eval-yam.

Results CSV: ikea-10/eval/results/<policy>/results_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# sys.path wiring: pull in eval-yam's backends + the molmoact2-setup
# helpers (cameras, arms, safety, run_one_attempt) without duplicating.
_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_YAM_SCRIPTS = os.path.normpath(
    os.path.join(_HERE, "..", "..", "eval-yam", "scripts")
)
_MOLMOACT_SCRIPTS = os.path.normpath(
    os.path.join(_HERE, "..", "..", "molmoact2-setup", "scripts")
)
for d in (_MOLMOACT_SCRIPTS, _EVAL_YAM_SCRIPTS):
    if not os.path.isdir(d):
        raise RuntimeError(f"required sibling dir missing: {d}")
sys.path.insert(0, _MOLMOACT_SCRIPTS)
sys.path.insert(0, _EVAL_YAM_SCRIPTS)

import numpy as np  # noqa: E402

import yam_client as yc  # noqa: E402  -- applies SDK lock fix at import
from yam_client import (  # noqa: E402
    DEFAULT_GRIPPER_STEP,
    DEFAULT_HORIZON_STRIDE,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MAX_STEP_RAD,
    DEFAULT_TRAIN_FPS,
    init_arm,
    load_saved_config,
    load_training_mean_pose,
    log,
    make_camera,
    ramp_to_pose,
    read_state,
    trace,
)
import yam_repl  # noqa: E402

import yam_backends  # noqa: E402


# ---------------------------------------------------------------------------
# IKEA 10-task list -- DRAFT, REVIEW AGAINST PDFs BEFORE RUNNING
# ---------------------------------------------------------------------------
# Each row: (swedish, english, instruction, atomic_actions).
#
# Instruction-writing principle: the VLA was trained on cube/box/cup/
# tape-roll manipulation, NOT IKEA SKUs. Instructions describe the
# INSTRUMENTAL physical motion using colors + simple object words
# ("the green metal piece", "the brown leg", "the white wedge"), NOT
# the end goal ("set up the napkin holder", "display the photo").
# Swedish + English names stay in metadata for the operator's reference
# (shown at the ready-prompt + recorded in the CSV) but never enter
# the model's prompt -- the model can't decode "SKOGSRÖR".
#
# Verbs the model understands from its training distribution:
#   pick up, place, put X on/in Y, stack, fold, unfold, open, close,
#   rotate, slide, snap, hook, twist, lift
# Avoid: assemble, set up, install, attach, secure, mount
#
# atomic_actions: a list of self-contained sub-task strings, each
# scoreable INDEPENDENTLY by the operator. Phrasing must restate the
# objects (no "it" / "them" anaphora that depends on prior actions)
# so each can be evaluated in isolation. Used for partial-credit
# scoring: operator marks which atomic actions completed at outcome
# time, score = sum_completed / len(atomic_actions). Lets us
# distinguish "the model got 0% of the way" from "the model did 2/3
# and only fluffed the last step".

# Sorted by Swedish name (Python default lexicographic; Ä < Å < Ö all
# sort after Z, so e.g. LÄMPLIG < LÅNESPELARE).
#
# atomic_actions are OUTCOME-ORIENTED: each one describes a state the
# world should reach, not a preparatory motion. "Pick up X" was dropped
# from all task rows because picking is just preparation -- the operator
# scores it implicitly by whether the followup outcome ("Place X on Y")
# succeeded. Tasks with only one outcome step (LACK, LÄMPLIG, etc.) keep
# a single atom; others have 2-3.
TASKS: list[dict] = [
    {
        "swedish": "FISKBO",
        "english": "8x10 picture frame",
        "instruction":
            "Lift up the big flap, then lift up the small flap, then "
            "place the big flap on the small flap, and put up the picture frame.",
        "atomic_actions": [
            "Lift up the big flap, then lift up the small flap, then "
            "place the big flap on the small flap.",
            "Put up the picture frame.",
        ],
    },
    {
        "swedish": "GREJIG",
        "english": "shoe rack (set of 3 with BAGGMUCK drip tray)",
        "instruction":
            "Lift up the left metal end, lift up the right metal end, "
            "and turn the tray table upside down.",
        "atomic_actions": [
            "Lift up the left metal end.",
            "Lift up the right metal end.",
            "Turn the tray table upside down.",
        ],
    },
    {
        "swedish": "KLIPSK",
        "english": "bed tray",
        "instruction":
            "Lift up the left white leg, slide the left white leg into "
            "the table, lift up the right white leg, slide the right "
            "white leg into the table, and flip the table upright.",
        "atomic_actions": [
            "Lift up the left white leg.",
            "Slide the left white leg into the table.",
            "Lift up the right white leg.",
            "Slide the right white leg into the table.",
            "Flip the table upright.",
        ],
    },
    {
        "swedish": "KROKFJORDEN",
        "english": "two-tier shower caddy",
        "instruction":
            "Attach the hook to the top of the metal stand, hook the big "
            "basket to the upper metal rod, hook the small basket to the "
            "lower metal rod, and hang the shower caddy on the rod.",
        "atomic_actions": [
            "Attach the hook to the top of the metal stand.",
            "Hook the big basket to the upper metal rod.",
            "Hook the small basket to the lower metal rod.",
            "Hang the shower caddy on the rod.",
        ],
    },
    {
        "swedish": "LACK",
        "english": "side table",
        "instruction":
            "Flip the table upside down. Place the screw on the hole and "
            "screw it in place lightly. Pick up the wooden leg and place "
            "it on top of the screw and screw it lightly in place. Turn the "
            "wooden leg clockwise five more times to tighten it partway. "
            "Tighten the wooden leg into the table by twisting it clockwise "
            "until tight.",
        "atomic_actions": [
            "Flip the table upside down.",
            "Place the screw on the hole and screw it in place lightly.",
            "Pick up the wooden leg and place it on top of the screw and "
            "screw it lightly in place.",
            "Turn the wooden leg clockwise five more times to tighten it partway.",
            "Tighten the wooden legs into the table by twisting them "
            "clockwise until tight.",
        ],
    },
    {
        "swedish": "LÄMPLIG",
        "english": "stainless steel trivet",
        "instruction":
            "Pick up the black square and put it to a corner, "
            "then press on the black item until it snaps into the metal tray.",
        "atomic_actions": [
            "Pick up the black square and put it on the missing corner.",
            "Press on the black item until it snaps into the metal tray.",
        ],
    },
    {
        "swedish": "LÅNESPELARE",
        "english": "foldable laptop support",
        "instruction":
            "Lift up the flap, slide the flap into the stand, and flip "
            "the laptop stand upside down.",
        "atomic_actions": [
            "Lift up the flap.",
            "Slide the flap into the stand.",
            "Flip the laptop stand upside down.",
        ],
    },
    {
        "swedish": "PATRULL",
        "english": "door stop",
        "instruction":
            "Put the white strap on top of the white square, then press "
            "the white strap onto the white square until it clicks.",
        "atomic_actions": [
            "Put the white strap on top of the white square.",
            "Press the white strap onto the white square until it clicks.",
        ],
    },
    {
        "swedish": "SKOGSRÖR",
        "english": "napkin holder",
        "instruction":
            "Pick up the green metal piece, fold the left flap upwards, "
            "and fold the right flap upwards.",
        "atomic_actions": [
            "Fold the left flap upwards.",
            "Fold the right flap upwards.",
        ],
    },
    {
        "swedish": "VÅRSYREN",
        "english": "tealight lantern",
        "instruction":
            "Fold up the left sheet, fold up the right sheet, fold the "
            "sheets to form a cylinder, slide the hooks to lock the "
            "sheets together, and attach the handle to the lantern.",
        "atomic_actions": [
            "Fold up the left sheet.",
            "Fold up the right sheet.",
            "Fold the sheets to form a cylinder.",
            "Slide the hooks to lock the sheets together.",
            "Attach the handle to the lantern.",
        ],
    },
]
assert len(TASKS) == 10, f"expected 10 tasks, got {len(TASKS)}"


# ---------------------------------------------------------------------------
# Per-policy argparse defaults (mirror eval_yam_tasks)
# ---------------------------------------------------------------------------

def _policy_defaults(policy: str) -> dict:
    return {
        "molmoact2": {"server_url": "http://127.0.0.1:8202/act"},
        "pi05":      {"server_host": "127.0.0.1", "server_port": 8000},
        "gr00t-n17": {"server_host": "127.0.0.1", "server_port": 5556},
    }[policy]


def _build_backend(args) -> yam_backends.Backend:
    if args.policy == "molmoact2":
        return yam_backends.MolmoActHTTPBackend(server_url=args.server_url)
    if args.policy == "pi05":
        return yam_backends.Pi05WebsocketBackend(host=args.server_host,
                                                  port=args.server_port)
    if args.policy == "gr00t-n17":
        return yam_backends.Gr00tZmqBackend(host=args.server_host,
                                             port=args.server_port)
    raise ValueError(f"Unknown --policy {args.policy!r}")


# ---------------------------------------------------------------------------
# Interactive pickers + per-attempt prompts
# ---------------------------------------------------------------------------

def prompt_select_furniture() -> Optional[list[int]]:
    """Show the 10 furniture; let operator pick a subset (or all).
    Returns 0-based task indices, or None if operator quit.
    """
    print("\n" + "=" * 70, flush=True)
    print("Pick furniture to eval (default: all 10):", flush=True)
    print("=" * 70, flush=True)
    for i, t in enumerate(TASKS, 1):
        print(f"  {i:>2}. {t['swedish']:<22} -- {t['english']}", flush=True)
    print(f"\n[enter] = all  |  comma-sep 1-{len(TASKS)} (e.g. 1,3,5)  |  'q' to quit",
          flush=True)
    sys.stdout.flush()
    try:
        ans = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if ans in {"q", "quit", "exit"}: return None
    if not ans:
        return list(range(len(TASKS)))
    try:
        sel = sorted({int(x) - 1 for x in ans.split(",") if x.strip()})
    except ValueError:
        print(f"  ?? unparseable {ans!r}; defaulting to all", flush=True)
        return list(range(len(TASKS)))
    sel = [i for i in sel if 0 <= i < len(TASKS)]
    if not sel:
        print(f"  no valid indices; defaulting to all", flush=True)
        return list(range(len(TASKS)))
    return sel


def prompt_select_prompt(task: dict) -> Optional[tuple[str, str]]:
    """For one furniture, let operator pick which prompt to send to the VLA:
    the full instruction (default) or any one atomic action.

    Returns (prompt_text_sent_to_model, kind_label) where kind_label is:
      'full'       -- run N attempts on the full instruction
      'atomic_N'   -- run N attempts on the Nth atomic action (1-based)
      'back'       -- return to the furniture picker
      'skip'       -- skip this furniture, advance the queue
    Returns None if operator chose to abort the whole eval.
    """
    print("", flush=True)
    print("─" * 70, flush=True)
    print(f"  {task['swedish']}  --  {task['english']}", flush=True)
    print("─" * 70, flush=True)
    print(f"  [enter] full instruction:", flush=True)
    print(f"          {task['instruction']!r}", flush=True)
    for i, a in enumerate(task["atomic_actions"], 1):
        print(f"  {i}. atomic_{i}: {a!r}", flush=True)
    print(f"  b = back to furniture picker  |  s = skip this furniture  |  q = abort eval",
          flush=True)
    sys.stdout.flush()
    try:
        ans = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if ans in {"q", "quit", "exit"}: return None
    if ans in {"b", "back"}: return ("", "back")
    if ans in {"s", "skip"}: return ("", "skip")
    if not ans:
        return (task["instruction"], "full")
    try:
        idx = int(ans) - 1
        if 0 <= idx < len(task["atomic_actions"]):
            return (task["atomic_actions"][idx], f"atomic_{idx+1}")
    except ValueError:
        pass
    print(f"  ?? unrecognized {ans!r}; defaulting to full instruction", flush=True)
    return (task["instruction"], "full")


def prompt_ready(task_idx: int, attempt: int, n_attempts: int,
                 task: dict, prompt_text: str, prompt_kind: str,
                 policy: str) -> tuple[str, str]:
    """Returns (action, sent_prompt_text):
      action          one of 'go' / 'skip' / 'quit'
      sent_prompt_text  the string actually sent to the model. Equals
                        prompt_text by default; 'e' lets the operator
                        type an override for THIS attempt only (next
                        attempt re-shows the original prompt).
    """
    while True:
        print("\n" + "=" * 70, flush=True)
        print(f"[policy={policy}] task {task_idx + 1}/{len(TASKS)}  |  "
              f"attempt {attempt}/{n_attempts}  |  prompt={prompt_kind}", flush=True)
        print(f"  {task['swedish']}  --  {task['english']}", flush=True)
        print(f"  sending: {prompt_text!r}", flush=True)
        print("=" * 70, flush=True)
        print("[enter to start, 'e' to edit instruction, "
              "'s' to skip this attempt, 'q' to abort eval]", flush=True)
        sys.stdout.flush()
        try:
            ans = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit", prompt_text
        if ans in {"q", "quit", "exit"}:
            return "quit", prompt_text
        if ans in {"s", "skip"}:
            return "skip", prompt_text
        if ans in {"e", "edit"}:
            # Inline edit; override applies to THIS attempt only.
            print(f"  current: {prompt_text!r}", flush=True)
            print("  type new instruction (blank to cancel):", flush=True)
            try:
                new_text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not new_text:
                print("  (cancelled, keeping current)", flush=True)
                continue
            # Re-show the ready prompt with the new text so the operator
            # can sanity-check before pressing enter.
            prompt_text = new_text
            continue
        return "go", prompt_text


def prompt_outcome() -> tuple[Optional[str], str]:
    print("\nHow did it go?", flush=True)
    print("  [s]uccess / [f]ailure / [u]nclear  -- log this attempt", flush=True)
    print("  [r] redo this attempt              -- discard rollout, prompt-start same attempt again", flush=True)
    print("  [enter] skip                       -- CSV 'skip' row, no journal", flush=True)
    sys.stdout.flush()
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None, ""
    if not choice:
        return None, ""
    if choice[:1] == "r":
        return "redo", ""
    status_map = {"s": "success", "f": "failure", "u": "unclear"}
    status = status_map.get(choice[:1])
    if status is None:
        print(f"[journal] unrecognized {choice!r}, skipping", flush=True)
        return None, ""
    try:
        notes = input("Notes (one line, optional)\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        notes = ""
    return status, notes


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _load_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_resumable_session(
    results_dir: str | os.PathLike,
    selected_tasks: list[int],
    n_attempts: int,
) -> Optional[tuple[Path, list[dict], set[tuple[int, int]], set[tuple[int, int]]]]:
    rd = Path(results_dir)
    if not rd.exists():
        return None
    csvs = sorted(rd.glob("results_*.csv"), reverse=True)
    if not csvs:
        return None
    path = csvs[0]
    try:
        rows = _load_csv_rows(path)
    except Exception:
        return None
    done: set[tuple[int, int]] = set()
    for r in rows:
        try:
            done.add((int(r["task_num"]), int(r["attempt"])))
        except (KeyError, ValueError):
            continue
    target = {(t + 1, a) for t in selected_tasks for a in range(1, n_attempts + 1)}
    remaining = target - done
    if not remaining:
        return None
    return path, rows, done, remaining


def prompt_resume_choice(path: Path, done: set, remaining: set) -> str:
    total = len(done) + len(remaining)
    print("\n" + "=" * 70, flush=True)
    print(f"Found previous IKEA-10 session: {path.name}", flush=True)
    print(f"  {len(done)}/{total} attempts done, {len(remaining)} remaining "
          f"under current --tasks / --attempts", flush=True)
    print("=" * 70, flush=True)
    print("[c] continue / [n] new / [q] quit", flush=True)
    sys.stdout.flush()
    while True:
        try:
            ans = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if ans in {"c", "continue"}: return "continue"
        if ans in {"n", "new"}:      return "new"
        if ans in {"q", "quit", "exit"}: return "quit"
        print("Please answer c / n / q.", flush=True)


# ---------------------------------------------------------------------------
# Results CSV (adds swedish/english columns vs eval_yam_tasks)
# ---------------------------------------------------------------------------

class ResultsCsv:
    FIELDS = [
        "timestamp", "policy", "task_num", "swedish", "english",
        "prompt_kind", "prompt_text",
        "attempt", "status",
        "duration_s", "timed_out", "n_chunks", "mean_rtt_ms", "max_rtt_ms",
        "mean_horizon_span", "max_state_vs_a0", "clip_pct", "notes",
    ]

    def __init__(self, path: Path, policy: str):
        self.path = path
        self.policy = policy
        new_file = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def add(self, task_num: int, task: dict, prompt_kind: str,
            prompt_text: str, attempt: int,
            status: str, notes: str, stats: dict) -> None:
        clip_pct = (100.0 * stats["clipped_dim_steps"] /
                    stats["max_possible_clip"]) if stats["max_possible_clip"] else 0.0
        self._writer.writerow({
            "timestamp": stats["timestamp"],
            "policy":    self.policy,
            "task_num":  task_num,
            "swedish":   task["swedish"],
            "english":   task["english"],
            "prompt_kind": prompt_kind,
            "prompt_text": prompt_text,
            "attempt":   attempt,
            "status":    status,
            "duration_s":        f"{stats['duration_s']:.1f}",
            "timed_out":         "1" if stats.get("timed_out") else "0",
            "n_chunks":          stats["n_chunks"],
            "mean_rtt_ms":       f"{stats['mean_rtt_ms']:.0f}",
            "max_rtt_ms":        f"{stats['max_rtt_ms']:.0f}",
            "mean_horizon_span": f"{stats['mean_horizon_span']:.3f}",
            "max_state_vs_a0":   f"{stats['max_state_vs_a0']:.3f}",
            "clip_pct":          f"{clip_pct:.1f}",
            "notes":             notes,
        })
        self._fh.flush()

    def close(self) -> None:
        try: self._fh.close()
        except Exception: pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _cfg = load_saved_config()
    _gripper_default = _cfg.get("gripper", "linear_4310")

    p = argparse.ArgumentParser(
        description="IKEA 10-furniture eval for bimanual YAM "
                    "(MolmoAct2 / Pi-0.5 / GR00T-N1.7)"
    )
    p.add_argument("--policy", required=True,
                   choices=["molmoact2", "pi05", "gr00t-n17"])
    p.add_argument("--server-url", default=None,
                   help="(--policy molmoact2) Full HTTP endpoint URL.")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=None)
    # Hardware
    p.add_argument("--left-can",  default=_cfg.get("left_can",  "can0"))
    p.add_argument("--right-can", default=_cfg.get("right_can", "can1"))
    p.add_argument("--left-gripper",  default=_gripper_default,
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"])
    p.add_argument("--right-gripper", default=_gripper_default,
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"])
    p.add_argument("--top-cam-serial",   default=_cfg.get("top_cam_serial"))
    p.add_argument("--top-cam-v4l2",     default=_cfg.get("top_cam_v4l2"))
    p.add_argument("--left-cam-serial",  default=_cfg.get("left_cam_serial"))
    p.add_argument("--left-cam-v4l2",    default=_cfg.get("left_cam_v4l2"))
    p.add_argument("--right-cam-serial", default=_cfg.get("right_cam_serial"))
    p.add_argument("--right-cam-v4l2",   default=_cfg.get("right_cam_v4l2"))
    p.add_argument("--cam-width",  type=int, default=424)
    p.add_argument("--cam-height", type=int, default=240)
    p.add_argument("--cam-fps",    type=int, default=30)
    # Inference
    p.add_argument("--timeout-s",        type=float, default=15.0)
    p.add_argument("--warmup-timeout-s", type=float, default=60.0)
    p.add_argument("--num-steps", type=int, default=10)
    # Policy execution
    p.add_argument("--train-fps",      type=float, default=DEFAULT_TRAIN_FPS)
    p.add_argument("--horizon-stride", type=int,   default=DEFAULT_HORIZON_STRIDE)
    p.add_argument("--max-step-rad",   type=float, default=DEFAULT_MAX_STEP_RAD)
    p.add_argument("--gripper-step",   type=float, default=DEFAULT_GRIPPER_STEP)
    p.add_argument("--dry-run", action="store_true")
    # Eval-specific
    p.add_argument("-n", "--attempts", type=int, default=3,
                   help="attempts per furniture (default 3)")
    p.add_argument("--tasks", default=None,
                   help="comma-separated 1-based task indices "
                        "(if omitted, an interactive picker shows all 10)")
    p.add_argument("--prompt-kind", default=None,
                   help="which prompt to test per task: 'full' (the full "
                        "instruction) or 'atomic_N' (the Nth atomic action). "
                        "If omitted, prompted interactively per task.")
    p.add_argument("--attempt-timeout-s", type=float, default=120.0,
                   help="wall-clock cap per IKEA attempt (default 120s -- longer than "
                        "the Andon 10-task eval because furniture assembly tasks need "
                        "more time than single pick-place actions)")
    # Reset
    p.add_argument("--ramp-duration-s", type=float, default=5.0)
    p.add_argument("--no-return-on-exit", action="store_true")
    # Observability
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--rerun-connect", default=None, metavar="HOST:PORT")
    p.add_argument("--rerun-save",    default=None, metavar="PATH")
    # Outputs
    p.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
    p.add_argument("--results-dir",
                   default=None,
                   help="per-session CSV results dir. Defaults to results/<policy>/")
    args = p.parse_args()

    # Apply per-policy server defaults.
    pd = _policy_defaults(args.policy)
    if args.policy == "molmoact2":
        if args.server_url is None:
            args.server_url = pd["server_url"]
    else:
        if args.server_port is None:
            args.server_port = pd["server_port"]
    if args.policy != "molmoact2" and args.server_url is None:
        proto = "ws" if args.policy == "pi05" else "tcp"
        args.server_url = f"{proto}://{args.server_host}:{args.server_port}"

    if args.results_dir is None:
        args.results_dir = os.path.join(_HERE, "results", args.policy)

    # Task selection: --tasks for scripted use, interactive picker otherwise
    if args.tasks:
        try:
            selected = sorted({int(x) - 1 for x in args.tasks.split(",") if x.strip()})
        except ValueError:
            print(f"--tasks must be comma-separated integers, got {args.tasks!r}",
                  file=sys.stderr)
            sys.exit(2)
        for idx in selected:
            if not 0 <= idx < len(TASKS):
                print(f"task index {idx+1} out of range 1..{len(TASKS)}",
                      file=sys.stderr)
                sys.exit(2)
    else:
        sel = prompt_select_furniture()
        if sel is None:
            print("Quit before hardware setup. No changes made.", flush=True)
            sys.exit(0)
        selected = sel

    # Resume prompt
    resume_path: Optional[Path] = None
    prior_rows: list[dict] = []
    done_set: set[tuple[int, int]] = set()
    prev = find_resumable_session(args.results_dir, selected, args.attempts)
    if prev is not None:
        path, rows, done, remaining = prev
        choice = prompt_resume_choice(path, done, remaining)
        if choice == "quit":
            print("Quit before hardware setup. No changes made.", flush=True)
            sys.exit(0)
        if choice == "continue":
            resume_path = path
            prior_rows = rows
            done_set = done
            log.info("Resuming %s (%d done, %d remaining)",
                     resume_path.name, len(done), len(remaining))

    invocation = os.environ.get("YAM_INVOCATION") or " ".join(sys.argv)

    # Build backend & install
    backend = _build_backend(args)
    yam_backends.install_backend(backend)

    # Rerun (optional). Mirrors repl_yam.py's auto-save behavior so each
    # ikea-10 session leaves a per-tick .rrd on disk that report/
    # extract_images.py can pull cam/top frames from later, matched per
    # task by application_id="ikea10_eval_*" + journal wall-clock time.
    if args.rerun or args.rerun_save:
        if args.rerun and not args.rerun_save:
            from datetime import datetime as _dt
            rrd_dir = os.path.join(os.path.dirname(_HERE), "..",
                                   "eval-yam", "logs", "rrd")
            rrd_dir = os.path.normpath(rrd_dir)
            os.makedirs(rrd_dir, exist_ok=True)
            args.rerun_save = os.path.join(
                rrd_dir,
                f"{_dt.now().strftime('%Y-%m-%d_%H%M%S')}_ikea10_{args.policy}.rrd",
            )
            log.info("AUTO-SAVING Rerun recording to %s", args.rerun_save)
        try:
            import rerun as rr
            yc._rr = rr
            rr.init(f"ikea10_eval_{args.policy}")
            sinks = []
            if args.rerun_connect:
                host, _, port = args.rerun_connect.partition(":")
                sinks.append(rr.GrpcSink(url=f"rerun+http://{host}:{port}/proxy"))
            else:
                rr.spawn(connect=False)
                sinks.append(rr.GrpcSink())
            if args.rerun_save:
                sinks.append(rr.FileSink(args.rerun_save))
            rr.set_sinks(*sinks)
        except ImportError:
            log.error("--rerun requested but rerun-sdk not installed.")
            sys.exit(2)

    # Health check
    try:
        meta = backend.health_check(timeout_s=3.0)
        log.info("[%s] server health: %s", args.policy, meta)
    except Exception as e:
        log.error("[%s] server health check failed: %s", args.policy, e)
        sys.exit(2)

    # Stash server identity onto args for the journal.
    args.server_repo_id  = meta.get("repo_id",  "unknown")
    args.server_dtype    = meta.get("dtype",    "unknown")
    args.server_norm_tag = meta.get("norm_tag", "unknown")
    args.server_meta     = repr(meta)

    # Cameras before arms (USB-storm-vs-CAN ordering).
    top = cam_l = cam_r = None
    left = right = None
    try:
        cam_kw = dict(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
        trace(f"building cameras at {args.cam_width}x{args.cam_height}/{args.cam_fps}fps")
        top   = make_camera("top",   args.top_cam_serial,   args.top_cam_v4l2,   **cam_kw)
        cam_l = make_camera("left",  args.left_cam_serial,  args.left_cam_v4l2,  **cam_kw)
        cam_r = make_camera("right", args.right_cam_serial, args.right_cam_v4l2, **cam_kw)
        for c in (top, cam_l, cam_r):
            c.start()
        for _ in range(3):
            for c in (top, cam_l, cam_r):
                try: c.grab()
                except Exception as e: log.warning("settle: %s.grab() failed: %s", c.name, e)
        trace("cameras streaming -- safe to init arms")
    except Exception:
        for c in (top, cam_l, cam_r):
            if c is not None:
                try: c.stop()
                except Exception: pass
        raise

    left = init_arm(args.left_can, args.left_gripper)
    right = init_arm(args.right_can, args.right_gripper)

    startup_pose = read_state(left, right)
    log.info("Captured startup pose: %s",
             np.array2string(startup_pose, precision=3))

    ready_pose = load_training_mean_pose()
    ready_pose[6]  = startup_pose[6]
    ready_pose[13] = startup_pose[13]
    log.info("Ramping to training-mean ready pose (%.1fs)...",
             args.ramp_duration_s)
    ramp_to_pose(left, right, ready_pose, duration_s=args.ramp_duration_s,
                 label="initial move-to-ready")

    # Server warmup
    try:
        state = read_state(left, right)
        log.info("Warming up server (timeout=%.0fs)...", args.warmup_timeout_s)
        _wu_actions, _wu_rtt = yc.post_actions(
            args.server_url, top.grab(), cam_l.grab(), cam_r.grab(), state,
            "warmup", args.num_steps, args.warmup_timeout_s,
        )
        log.info("Server warmup OK (rtt=%.0f ms)", _wu_rtt)
    except Exception as e:
        log.error("Server warmup failed: %s. Continuing anyway.", e)

    session_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if resume_path is not None:
        results_path = resume_path
    else:
        results_path = Path(args.results_dir) / f"results_{session_ts}.csv"
    results = ResultsCsv(results_path, policy=args.policy)
    log.info("Writing per-attempt results to %s", results_path)

    loop_t0 = time.perf_counter()
    global_attempt = 0
    tallies: dict[int, dict[str, int]] = {
        i: {"success": 0, "failure": 0, "unclear": 0, "skip": 0}
        for i in selected
    }
    for r in prior_rows:
        try:
            t_idx = int(r["task_num"]) - 1
        except (KeyError, ValueError):
            continue
        if t_idx not in tallies:
            continue
        status = r.get("status", "")
        if status in tallies[t_idx]:
            tallies[t_idx][status] += 1
    aborted = False

    total_target = len(selected) * args.attempts
    remaining_target = total_target - len(done_set)
    print("\n" + "#" * 70, flush=True)
    print(f"# IKEA-10 eval  |  policy={args.policy}  |  "
          f"{len(selected)} furniture × {args.attempts} attempt(s) "
          f"= {total_target} total", flush=True)
    if done_set:
        print(f"# Resuming: {len(done_set)} done, {remaining_target} remaining",
              flush=True)
    print(f"# Session: {session_ts}", flush=True)
    print("#" * 70, flush=True)

    # Interactive mode = no --tasks. In interactive mode, the eval loops
    # continuously: after N=3 attempts on a chosen prompt, we re-prompt
    # the appropriate picker (atomic-level if the last prompt was atomic,
    # furniture-level if the last prompt was the full instruction). With
    # --tasks (scripted), we exit after iterating the queue once.
    interactive = not args.tasks
    interactive_prompt = not args.prompt_kind

    def _resolve_prompt_from_cli(task: dict) -> tuple[str, str]:
        """Map --prompt-kind to (prompt_text, prompt_kind) for scripted mode."""
        if args.prompt_kind == "full":
            return task["instruction"], "full"
        if args.prompt_kind.startswith("atomic_"):
            try:
                atom_idx = int(args.prompt_kind.split("_", 1)[1]) - 1
            except ValueError:
                print(f"bad --prompt-kind: {args.prompt_kind!r}", file=sys.stderr)
                sys.exit(2)
            if not (0 <= atom_idx < len(task["atomic_actions"])):
                log.warning("%s has no atomic_%d (only %d atoms); falling back to 'full'",
                            task["swedish"], atom_idx + 1, len(task["atomic_actions"]))
                return task["instruction"], "full"
            return task["atomic_actions"][atom_idx], f"atomic_{atom_idx+1}"
        print(f"--prompt-kind must be 'full' or 'atomic_N', got {args.prompt_kind!r}",
              file=sys.stderr)
        sys.exit(2)

    def _run_n_attempts(task_idx: int, task: dict,
                        prompt_text: str, prompt_kind: str) -> None:
        """Run args.attempts attempts of (task, prompt) inline.
        Updates global_attempt, tallies, and writes to CSV + journal.
        Raises KeyboardInterrupt if operator aborts the eval.
        """
        nonlocal global_attempt, aborted
        attempt = 1
        while attempt <= args.attempts:
            if (task_idx + 1, attempt) in done_set:
                log.info("task %d attempt %d already done; skipping",
                         task_idx + 1, attempt)
                attempt += 1
                continue
            action, sent_text = prompt_ready(
                task_idx, attempt, args.attempts,
                task, prompt_text, prompt_kind, args.policy,
            )
            if action == "quit":
                aborted = True
                raise KeyboardInterrupt
            if action == "skip":
                log.info("task %d (%s) attempt %d skipped",
                         task_idx + 1, task["swedish"], attempt)
                tallies[task_idx]["skip"] += 1
                attempt += 1
                continue

            # If the operator edited the instruction at the ready prompt,
            # `sent_text` differs from `prompt_text`. Use sent_text for
            # this attempt's run + CSV + journal so the record reflects
            # what the model ACTUALLY received. `prompt_text` (the loop
            # variable) is unchanged so the next attempt re-shows the
            # original prompt.
            effective_prompt = sent_text
            effective_kind = prompt_kind if sent_text == prompt_text else f"{prompt_kind}+edit"

            global_attempt += 1
            stats = yam_repl.run_one_attempt(
                args, left, right, top, cam_l, cam_r,
                effective_prompt, global_attempt, loop_t0,
                attempt_timeout_s=args.attempt_timeout_s,
            )

            log.info("Resetting arms to ready pose (%.1fs)...", args.ramp_duration_s)
            ramp_to_pose(left, right, ready_pose,
                         duration_s=args.ramp_duration_s, label="reset")

            status, notes = prompt_outcome()
            if status == "redo":
                log.info("operator REDO of task %d (%s) attempt %d",
                         task_idx + 1, task["swedish"], attempt)
                continue
            if status is None:
                log.info("attempt skipped from journal")
                tallies[task_idx]["skip"] += 1
                results.add(task_idx + 1, task, effective_kind, effective_prompt,
                            attempt, "skip", notes, stats)
                attempt += 1
            else:
                tallies[task_idx][status] += 1
                tagged_notes = (
                    f"[ikea-10 task {task_idx + 1}/{len(TASKS)}: "
                    f"{task['swedish']} / {task['english']}] "
                    f"[prompt={effective_kind}] "
                    f"[policy={args.policy}] "
                ) + (notes or "")
                yam_repl.write_attempt_entry(
                    args.journal_path, global_attempt,
                    effective_prompt, status, tagged_notes, stats, args,
                    invocation,
                )
                results.add(task_idx + 1, task, effective_kind, effective_prompt,
                            attempt, status, notes, stats)
                attempt += 1

    try:
        # Outer loop: queue of furniture. In interactive mode, after the
        # queue is exhausted we re-prompt the furniture picker; this
        # loops until the user picks 'q' (then prompt_select_furniture
        # returns None and we break).
        outer_session_count = 0
        while True:
            outer_session_count += 1
            if outer_session_count > 1:
                # Re-prompt furniture picker between sessions in interactive mode.
                sel = prompt_select_furniture()
                if sel is None:
                    log.info("operator chose to quit at furniture picker")
                    break
                selected = sel

            back_to_furniture_picker = False
            for task_idx in selected:
                task = TASKS[task_idx]

                # Inner loop: for one furniture, keep re-prompting the
                # prompt picker until the operator picks 'full' (advances
                # to next furniture), 'b' (back to furniture picker), or
                # 's' (skip this furniture). Each iteration runs N attempts.
                while True:
                    if not interactive_prompt:
                        # Scripted: one prompt per task, no inner loop.
                        prompt_text, prompt_kind = _resolve_prompt_from_cli(task)
                        _run_n_attempts(task_idx, task, prompt_text, prompt_kind)
                        break

                    choice = prompt_select_prompt(task)
                    if choice is None:
                        aborted = True
                        raise KeyboardInterrupt
                    prompt_text, prompt_kind = choice
                    if prompt_kind == "skip":
                        log.info("furniture %s skipped", task["swedish"])
                        break  # next furniture in queue
                    if prompt_kind == "back":
                        log.info("back to furniture picker from %s", task["swedish"])
                        back_to_furniture_picker = True
                        break

                    _run_n_attempts(task_idx, task, prompt_text, prompt_kind)

                    if prompt_kind.startswith("atomic_"):
                        # Stay on this furniture; re-show the prompt picker
                        # so operator can pick another atomic / full / b / s.
                        continue
                    # prompt_kind == "full": advance to next furniture.
                    break

                if back_to_furniture_picker:
                    break  # exit for-loop, fall through to outer re-prompt

            if not interactive:
                # Scripted: one pass through --tasks, then done.
                break
            # Interactive: outer while True loops back to the furniture
            # picker (next iteration's outer_session_count > 1 branch).
    except KeyboardInterrupt:
        if aborted:
            log.info("Eval aborted by operator. Tearing down.")
        else:
            log.info("Ctrl-C -- shutting down")
    finally:
        # Per-task summary
        print("\n" + "#" * 70, flush=True)
        print(f"# IKEA-10 summary  |  policy={args.policy}", flush=True)
        print("#" * 70, flush=True)
        print(f"{'#':>3}  {'swedish':<22} {'english':<32} {'S':>3} {'F':>3} {'U':>3} {'Sk':>3} {'rate':>6}",
              flush=True)
        total_s = total_f = total_u = total_sk = 0
        for task_idx in selected:
            t = tallies[task_idx]
            task = TASKS[task_idx]
            ran = t["success"] + t["failure"] + t["unclear"]
            rate = (100.0 * t["success"] / ran) if ran else 0.0
            print(f"{task_idx+1:>3}  {task['swedish'][:22]:<22} "
                  f"{task['english'][:32]:<32} "
                  f"{t['success']:>3} {t['failure']:>3} {t['unclear']:>3} "
                  f"{t['skip']:>3} {rate:>5.0f}%",
                  flush=True)
            total_s += t["success"]; total_f += t["failure"]
            total_u += t["unclear"]; total_sk += t["skip"]
        ran_total = total_s + total_f + total_u
        rate_total = (100.0 * total_s / ran_total) if ran_total else 0.0
        print(f"{'':>3}  {'':<22} {'TOTAL':<32} "
              f"{total_s:>3} {total_f:>3} {total_u:>3} "
              f"{total_sk:>3} {rate_total:>5.0f}%",
              flush=True)
        print(f"\nResults CSV: {results_path}", flush=True)
        print(f"Journal:     {args.journal_path}", flush=True)

        results.close()

        # Teardown
        abort = {"abort": False, "ctrlc_count": 0}
        def _cleanup_sigint(_sig, _frame):
            abort["ctrlc_count"] += 1
            if abort["ctrlc_count"] == 1:
                log.warning("Ctrl-C in cleanup: aborting return-ramp. "
                            "ARMS WILL DROP. Ctrl-C again to hard-exit.")
                abort["abort"] = True
            else:
                os._exit(130)
        try:
            signal.signal(signal.SIGINT, _cleanup_sigint)
        except Exception:
            pass

        if left is not None and right is not None and not args.no_return_on_exit:
            try:
                log.info("Returning arms to startup pose (%.1fs ramp)...",
                         args.ramp_duration_s)
                ramp_to_pose(left, right, startup_pose,
                             duration_s=args.ramp_duration_s,
                             abort_flag=abort, label="return-on-exit")
            except BaseException as e:
                log.warning("return-to-startup ramp failed: %s. ARMS MAY DROP.", e)

        log.info("Stopping cameras")
        for c in (top, cam_l, cam_r):
            if c is None: continue
            try: c.stop()
            except BaseException as e: log.warning("cam %s stop failed: %s", c.name, e)

        log.info("Closing arm SDKs")
        for arm in (left, right):
            if arm is None: continue
            try: arm.close()
            except BaseException as e: log.warning("arm.close() failed: %s", e)

        log.info("Eval done. Ran %d attempt(s) with policy=%s.",
                 global_attempt, args.policy)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
