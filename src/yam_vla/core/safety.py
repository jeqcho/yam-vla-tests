"""Per-tick safety clip for arm commands.

`safe_command(left, right, current_state, desired_action, max_step_rad,
gripper_step)` clips a single 14-D action so no joint moves more than
`max_step_rad` (and no gripper more than `gripper_step`) from the current
state in this tick, then sends the clipped command to both arms.

This is the difference between "policy plans something bad" and "arms
move dangerously". At 30 Hz with max_step_rad=0.15, joint velocity is
capped at ~4.5 rad/s (~260 deg/s) -- well above any in-distribution
policy output, still bounded enough that a single bad action chunk
can't slam an arm.

Set max_step_rad <= 0 to disable joint clipping; gripper_step <= 0 to
disable gripper clipping. Both 0 = raw policy output, only i2rt's 400ms
motor watchdog remains.
"""
from __future__ import annotations

import logging

import numpy as np

from yam_vla.core.observation import (
    LEFT_GRIP, RIGHT_GRIP, STATE_DIM,
)

log = logging.getLogger("yam_vla.safety")


# Sensible per-step caps at 30 Hz inner loop:
#   0.15 rad/step * 30 Hz = 4.5 rad/s ~ 260 deg/s joint velocity ceiling
#   0.15 grip-normalized/step at 30 Hz = jaw moves full open->close in ~0.22s
DEFAULT_MAX_STEP_RAD: float = 0.15
DEFAULT_GRIPPER_STEP: float = 0.15


def safe_command(
    left,
    right,
    current_state: np.ndarray,
    desired_action: np.ndarray,
    *,
    max_step_rad: float = DEFAULT_MAX_STEP_RAD,
    gripper_step: float = DEFAULT_GRIPPER_STEP,
) -> tuple[np.ndarray, int]:
    """Clip + send. Returns (cmd_actually_sent, n_clipped_dims).

    n_clipped_dims is how many of the 14 dimensions hit the cap this tick;
    eval harnesses tally this across all chunks to compute clip_rate.
    """
    if desired_action.shape != (STATE_DIM,):
        raise ValueError(f"action shape {desired_action.shape} != ({STATE_DIM},)")
    delta = desired_action - current_state

    # +inf cap = no clip for that dimension.
    caps = np.full(STATE_DIM,
                   max_step_rad if max_step_rad > 0 else np.inf,
                   dtype=np.float32)
    caps[LEFT_GRIP]  = gripper_step if gripper_step > 0 else np.inf
    caps[RIGHT_GRIP] = gripper_step if gripper_step > 0 else np.inf

    clipped_delta = np.clip(delta, -caps, caps)
    n_clipped = int(np.sum(clipped_delta != delta))
    cmd = (current_state + clipped_delta).astype(np.float32)

    # 7-D each (6 joints + 1 gripper). LEFT_GRIP=6, RIGHT_GRIP=13 -> split at 7.
    left.command_joint_pos(cmd[:7])
    right.command_joint_pos(cmd[7:])
    return cmd, n_clipped


__all__ = ["safe_command", "DEFAULT_MAX_STEP_RAD", "DEFAULT_GRIPPER_STEP"]
