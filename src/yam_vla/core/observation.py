"""Canonical observation + state representation for bimanual YAM.

Every policy backend reads the SAME `YamObservation` and produces a
`Prediction` (see policy.py). The per-policy quirks (CHW vs HWC, B/T dims,
agilex image-key names, num_steps, embodiment tags) live entirely inside
each backend.

This module owns:
  * `YamObservation`     — canonical input dict at the abstraction boundary
  * `YamStateCodec`      — slices/packers for the 14-D YAM state vector
  * `ImageRole`          — the three canonical camera roles

Reading order if you're new: ImageRole -> YamStateCodec -> YamObservation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Camera roles
# ---------------------------------------------------------------------------
# All three VLAs see the same three cameras, but each renames them
# (MolmoAct: top_cam/left_cam/right_cam; GR00T: top/left/right;
# openpi-pi05: base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb). The client
# side only knows the three abstract ROLES; the rename lives in the
# per-policy YAML config and is applied inside each Policy.predict().

class ImageRole:
    TOP: Final[str]         = "top"
    LEFT_WRIST: Final[str]  = "left_wrist"
    RIGHT_WRIST: Final[str] = "right_wrist"

    ALL: Final[tuple[str, ...]] = ("top", "left_wrist", "right_wrist")


# ---------------------------------------------------------------------------
# YAM 14-D state codec
# ---------------------------------------------------------------------------
# Canonical layout: [left_q0..5, left_grip, right_q0..5, right_grip].
# This vocabulary is shared by all three backends (because that's how the
# i2rt SDK reads the arms); per-policy state PACKING (e.g. GR00T's 4-key
# split) is done by composing slices, not by re-inventing the layout.

STATE_DIM: Final[int] = 14
ARM_DOF: Final[int]   = 6
TOTAL_PER_ARM: Final[int] = ARM_DOF + 1  # joints + gripper

# Slices used by the state codec. Exposed as module constants so backends
# can `from yam_vla.core.observation import LEFT_ARM, LEFT_GRIP, ...` rather
# than reach into the class.
LEFT_ARM:   Final[slice] = slice(0, 6)
LEFT_GRIP:  Final[int]   = 6
RIGHT_ARM:  Final[slice] = slice(7, 13)
RIGHT_GRIP: Final[int]   = 13


class YamStateCodec:
    """Split and stitch the 14-D YAM state vector.

    Used by backends that need to repack the state into per-arm keys
    (GR00T) or pass it through (MolmoAct, π₀.₅). All methods are static
    so the codec can be used as a namespace without instantiation.
    """

    STATE_DIM = STATE_DIM
    LEFT_ARM = LEFT_ARM
    LEFT_GRIP = LEFT_GRIP
    RIGHT_ARM = RIGHT_ARM
    RIGHT_GRIP = RIGHT_GRIP

    @staticmethod
    def validate(state: np.ndarray) -> np.ndarray:
        """Coerce to float32 and assert shape (14,). Returns the coerced array."""
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.shape != (STATE_DIM,):
            raise ValueError(
                f"state must be shape ({STATE_DIM},), got {s.shape}"
            )
        return s

    @staticmethod
    def split(state: np.ndarray) -> dict[str, np.ndarray]:
        """Split (14,) into the 4 per-arm streams GR00T expects."""
        s = YamStateCodec.validate(state)
        return {
            "left_arm":      s[LEFT_ARM].copy(),
            "left_gripper":  np.array([s[LEFT_GRIP]], dtype=np.float32),
            "right_arm":     s[RIGHT_ARM].copy(),
            "right_gripper": np.array([s[RIGHT_GRIP]], dtype=np.float32),
        }

    @staticmethod
    def stitch(la: np.ndarray, lg: np.ndarray,
               ra: np.ndarray, rg: np.ndarray) -> np.ndarray:
        """Reassemble (T, 14) actions from 4 per-arm action streams.

        Each input has shape (T, dim). Used by the GR00T backend to invert
        its 4-key action response into the canonical (T, 14) format.
        """
        la, lg, ra, rg = (np.asarray(x, dtype=np.float32) for x in (la, lg, ra, rg))
        T = la.shape[0]
        for name, arr in (("left_gripper", lg), ("right_arm", ra), ("right_gripper", rg)):
            if arr.shape[0] != T:
                raise ValueError(
                    f"action horizon mismatch: left_arm T={T}, {name} T={arr.shape[0]}"
                )
        out = np.zeros((T, STATE_DIM), dtype=np.float32)
        out[:, LEFT_ARM]   = la
        out[:, LEFT_GRIP]  = lg.reshape(-1)
        out[:, RIGHT_ARM]  = ra
        out[:, RIGHT_GRIP] = rg.reshape(-1)
        return out


# ---------------------------------------------------------------------------
# YamObservation — the abstraction boundary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YamObservation:
    """One snapshot of (cameras + arm state + language) at the client side.

    The control loop builds one of these per inference tick and hands it to
    `Policy.predict(obs)`. The policy is responsible for renaming/repacking
    into its server's native schema.

    Invariants enforced at construction (via `__post_init__`):
      * `images` keys are a subset of `ImageRole.ALL` (and must include all 3
        for bimanual policies — relaxed only if a backend explicitly opts in).
      * Each image is HWC uint8 RGB, shape (H, W, 3).
      * `state` is shape (14,) float32 in canonical YAM layout.
    """
    images:    Mapping[str, np.ndarray]  # role -> HxWx3 uint8 RGB
    state:     np.ndarray                # (14,) float32
    prompt:    str

    def __post_init__(self) -> None:
        # state shape + dtype
        s = YamStateCodec.validate(self.state)
        # frozen dataclass — bypass __setattr__ guard
        object.__setattr__(self, "state", s)

        # image schema
        for role, img in self.images.items():
            if role not in ImageRole.ALL:
                raise ValueError(
                    f"unknown image role {role!r}. Must be one of {ImageRole.ALL}"
                )
            arr = np.asarray(img)
            if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
                raise ValueError(
                    f"image[{role!r}] must be HxWx3 uint8 RGB, got shape={arr.shape} "
                    f"dtype={arr.dtype}"
                )

    def has_all_cameras(self) -> bool:
        return set(self.images) >= set(ImageRole.ALL)


__all__ = [
    "ImageRole", "YamStateCodec", "YamObservation",
    "STATE_DIM", "ARM_DOF", "LEFT_ARM", "LEFT_GRIP", "RIGHT_ARM", "RIGHT_GRIP",
]
