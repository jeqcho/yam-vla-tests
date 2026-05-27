"""Operator-UX keyboard helpers shared across the eval harness and REPL.

Provides:
  RawTerm           -- context manager: stdin in cbreak + non-blocking.
  read_key          -- single keypress reader (arrows, Enter, chars).
  wait_for_advance  -- block until operator presses ->/Enter/s/q/r.
  reset_countdown   -- inter-attempt scene-reset prompt with optional
                       soft countdown that never auto-advances.
  AdvanceWatcher    -- background thread: flip a flag on ->/Enter for
                       early-stop of a running rollout.

All helpers degrade gracefully on non-TTY stdin (piped / CI): the raw-
mode context is a no-op and `read_key` returns None. Callers should
have a sensible non-TTY fallback path (e.g. `input()`).
"""
from __future__ import annotations

import os
import select
import sys
import time
from typing import Optional


class RawTerm:
    """Switch stdin to cbreak (line-buffer off, echo off) + non-blocking
    for the duration of the `with` block. Restores on exit. No-op if
    stdin isn't a TTY.

    The raw-mode flags are PROCESS-WIDE: nothing else in the process
    should attempt blocking `input()` while we hold this context, or
    behavior is undefined.
    """

    def __enter__(self):
        if not sys.stdin.isatty():
            self.fd = None
            return self
        import fcntl
        import termios
        import tty
        self.fd = sys.stdin.fileno()
        self._termios = termios
        self._fcntl = fcntl
        self.old_term = termios.tcgetattr(self.fd)
        self.old_flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        tty.setcbreak(self.fd)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.old_flags | os.O_NONBLOCK)
        return self

    def __exit__(self, *exc):
        if self.fd is None:
            return False
        try:
            self._termios.tcsetattr(self.fd, self._termios.TCSADRAIN, self.old_term)
            self._fcntl.fcntl(self.fd, self._fcntl.F_SETFL, self.old_flags)
        except Exception:
            pass
        return False


def read_key(timeout: float = 0.1) -> Optional[str]:
    """Read one keypress from stdin. Returns a normalized name or None
    on timeout. Must be called inside a `with RawTerm():` block.

    Returns:
      'right' / 'left' / 'up' / 'down'  -- arrow keys (ANSI \\x1b[A-D)
      'enter'                            -- newline / carriage return
      'esc'                              -- bare escape
      one-character str (lowercased)     -- any other printable
      None                               -- timeout, or non-TTY stdin
    """
    if not sys.stdin.isatty():
        return None
    try:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        return None
    if not r:
        return None
    try:
        buf = os.read(sys.stdin.fileno(), 16)
    except (BlockingIOError, OSError):
        return None
    if not buf:
        return None
    if buf[:1] == b"\x1b":
        if buf[:3] == b"\x1b[A":
            return "up"
        if buf[:3] == b"\x1b[B":
            return "down"
        if buf[:3] == b"\x1b[C":
            return "right"
        if buf[:3] == b"\x1b[D":
            return "left"
        return "esc"
    if buf in (b"\n", b"\r"):
        return "enter"
    try:
        return buf.decode("utf-8", errors="replace")[:1].lower()
    except Exception:
        return None


def wait_for_advance(prompt: str = "") -> str:
    """Block until the operator presses right-arrow, Enter, 's', 'q', 'r'.

    Returns: 'go' (-> or Enter) | 'skip' (s) | 'quit' (q) | 'redo' (r).
    Echoes the prompt once. Falls back to stdlib `input()` if stdin is
    not a TTY.
    """
    if prompt:
        print(prompt, flush=True)
    if not sys.stdin.isatty():
        try:
            ans = (input("> ").strip().lower() or "")
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if ans in ("q", "quit"): return "quit"
        if ans in ("s", "skip"): return "skip"
        if ans in ("r", "redo"): return "redo"
        return "go"
    with RawTerm():
        while True:
            key = read_key(timeout=0.5)
            if key is None:
                continue
            if key in ("right", "enter"):
                return "go"
            if key == "s":
                return "skip"
            if key == "q":
                return "quit"
            if key == "r":
                return "redo"


def reset_countdown(seconds: float, label: str = "next attempt") -> str:
    """Show a single-line scene-reset prompt and wait for the operator to
    advance with -> or Enter (or skip with 's', quit with 'q').

    `seconds` controls the DISPLAY only -- a soft target the operator can
    use to pace themselves. The countdown counts down to zero, then the
    display flips to "READY (waited Xs extra)" and the prompt continues
    waiting indefinitely. The function never auto-advances on timer
    expiry -- only an operator keypress moves forward.

    `seconds <= 0` skips the whole reset-prompt entirely (returns
    immediately with 'auto').

    Returns: 'auto' (only when seconds <= 0)
             | 'go' (-> or Enter pressed)
             | 'skip' ('s' pressed)
             | 'quit' ('q' pressed).
    """
    if seconds <= 0:
        return "auto"
    if not sys.stdin.isatty():
        time.sleep(seconds)
        return "auto"

    start_t = time.monotonic()
    with RawTerm():
        while True:
            elapsed = time.monotonic() - start_t
            remaining = seconds - elapsed
            if remaining > 0:
                msg = (
                    f"\r[reset] {label} in {remaining:5.1f}s  "
                    f"(-> or Enter advance, 's' skip task, 'q' quit) "
                )
            else:
                extra = -remaining
                msg = (
                    f"\r[reset] {label} READY  (+{extra:5.1f}s)  "
                    f"(-> or Enter advance, 's' skip task, 'q' quit) "
                )
            print(f"{msg:<78s}", end="", flush=True)
            key = read_key(timeout=0.1)
            if key is None:
                continue
            print("\r" + " " * 78 + "\r", end="", flush=True)
            if key in ("right", "enter"):
                return "go"
            if key == "s":
                return "skip"
            if key == "q":
                return "quit"


class AdvanceWatcher:
    """Background thread that flips a flag when the operator presses
    right-arrow (->) or Enter.

    Used as the `stop` predicate for run_attempt -- gives the operator
    a way to end an in-flight rollout the moment they see it succeed
    or go off-rails, instead of waiting for max_chunks.

    The watcher holds raw-mode on stdin while it's running, so the
    main thread MUST NOT do blocking stdin reads (`input()`) during
    that window. Call `.stop()` to cleanly tear down raw mode before
    the next stdin read.
    """

    def __init__(self):
        self._stopped = False
        self._stop_thread = False
        self._thread = None

    def start(self) -> None:
        import threading

        def _watch():
            # Non-TTY stdin (piped / CI): no keys available, so sit
            # idle until the main thread calls stop(). The rollout
            # will run to max_chunks in this case.
            if not sys.stdin.isatty():
                while not self._stop_thread:
                    time.sleep(0.1)
                return
            with RawTerm():
                while not self._stop_thread:
                    key = read_key(timeout=0.1)
                    if key in ("right", "enter"):
                        self._stopped = True
                        return

        self._thread = threading.Thread(target=_watch, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the watcher thread to exit and wait for it. Call this
        before any blocking stdin read so raw mode is restored first."""
        self._stop_thread = True
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    @property
    def stopped(self) -> bool:
        return self._stopped

    def predicate(self):
        return lambda: self._stopped


def silence_root_logger() -> None:
    """Cap the unnamed `root` logger at WARNING.

    i2rt's motor-control threads log to the unnamed `root` logger at
    INFO every 10-30 s ("Grav Comp Control Frequency", "Total rate",
    "[PATCHED ... step_time > 0.007s"). They flood stdout during
    score prompts and interleave with operator typed input. Our own
    `yam_vla.*` loggers stay at INFO because they have explicit
    module names and aren't affected by the root cap.

    Call once at the start of any operator-facing entry script.
    """
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("yam_vla", "yam_vla.evals.runner", "yam_vla.hardware",
                 "yam_vla.control_loop", "yam_vla.repl"):
        logging.getLogger(name).setLevel(logging.INFO)


__all__ = [
    "RawTerm",
    "read_key",
    "wait_for_advance",
    "reset_countdown",
    "AdvanceWatcher",
    "silence_root_logger",
]
