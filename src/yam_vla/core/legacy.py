"""Compatibility shim re-exporting validated hardware/safety/journal
helpers from the legacy molmoact2-setup tree (under _archive/).

This module exists because the legacy `yam_client.py` is ~1500 lines of
hard-won, on-hardware-tested code (i2rt SDK lock fix, gripper auto-cal,
RealSense stream, V4L2 stream, per-tick safety clip, async chunk
fetcher, research-journal writer). Copying it into the new package
would be a regression risk; instead we expose it via a clean import
surface so new code reads:

    from yam_vla.core.legacy import init_arm, read_state, safe_command

rather than reaching into _archive/ directly.

NOTE: This is a transitional layer. A future refactor pass will lift
the truly shared parts (cameras, arms, safety, journal) into
`src/yam_vla/core/{hardware,safety,journal}.py` and the shim goes away.
What lives here is what's stable enough to import as-is today.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _legacy_scripts_dir() -> Path:
    """Find _archive/molmoact2-setup/scripts/ relative to this file.

    Layout: <repo>/src/yam_vla/core/legacy.py
            <repo>/_archive/molmoact2-setup/scripts/
    """
    return Path(__file__).resolve().parents[3] / "_archive" / "molmoact2-setup" / "scripts"


_legacy = _legacy_scripts_dir()
if not _legacy.is_dir():
    raise ImportError(
        f"yam_vla.core.legacy: legacy module dir not found at {_legacy}. "
        "Did the _archive/ subtree get deleted?"
    )
if str(_legacy) not in sys.path:
    sys.path.insert(0, str(_legacy))

# Re-exports. Names match the legacy module's public surface.
# Some of these have heavy optional deps (pyrealsense2, i2rt SDK) — the
# imports will fail loudly if those aren't installed, which is the
# right behavior at the boundary between the new code and the legacy
# hardware glue.

from yam_client import (  # type: ignore  # noqa: E402
    # SDK / config
    install_sdk_lock_fix,
    load_saved_config,
    DEFAULT_SETUP_CONFIG_PATH,

    # Cameras
    CameraStream,
    RealSenseStream,
    V4L2Stream,
    make_camera,

    # Arms
    init_arm,
    read_state,

    # Safety
    safe_command,

    # Motion
    load_training_mean_pose,
    ramp_to_pose,

    # Async inference (LEGACY: hardcoded to post_actions; new code should
    # use src/yam_vla/core/runner.py AsyncPolicyInference instead).
    AsyncInferenceFetcher,

    # Research journal (markdown append-only). The default journal path
    # is intentionally shared across all policies — single timeline of
    # "every robot run", with [policy=...] tags identifying provenance.
    prompt_journal_entry,
    write_journal_entry,
    DEFAULT_JOURNAL_PATH,
)

__all__ = [
    "install_sdk_lock_fix",
    "load_saved_config",
    "DEFAULT_SETUP_CONFIG_PATH",
    "CameraStream",
    "RealSenseStream",
    "V4L2Stream",
    "make_camera",
    "init_arm",
    "read_state",
    "safe_command",
    "load_training_mean_pose",
    "ramp_to_pose",
    "AsyncInferenceFetcher",
    "prompt_journal_entry",
    "write_journal_entry",
    "DEFAULT_JOURNAL_PATH",
]
