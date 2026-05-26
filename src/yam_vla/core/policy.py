"""Policy abstraction: one interface, three backends.

The interface is intentionally small. Per-policy quirks (transport,
observation packing, action unpacking, normalization) live in concrete
subclasses; the eval/REPL/control loop sees only this surface.

Reading order: ServerInfo -> Prediction -> Policy.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from yam_vla.core.observation import STATE_DIM, YamObservation


# ---------------------------------------------------------------------------
# ServerInfo — what every backend tells us about itself at handshake time
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerInfo:
    """Backend self-description returned by `Policy.info()`.

    Fields chosen as the minimum the eval harness + journal need to record
    *provenance* of a run. Optional fields (model_id, action_horizon_hint,
    control_hz_hint) are best-effort — backends fill what they know.
    """
    backend:               str                      # "molmoact2" / "gr00t-n17" / "pi05"
    transport:             str                      # human-readable, e.g. "http://127.0.0.1:8202/act"
    model_id:              str | None    = None     # HF repo_id if known
    action_dim:            int           = STATE_DIM
    action_horizon_hint:   int | None    = None     # checkpoint's training horizon, advisory
    control_hz_hint:       float | None  = None
    raw:                   dict          = field(default_factory=dict)  # backend-native metadata


# ---------------------------------------------------------------------------
# Prediction — what every backend returns from `Policy.predict()`
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    """Canonical inference result.

    `actions` is always (N, 14) float32 ABSOLUTE joint targets in the YAM
    canonical layout. Per-backend conversions (GR00T's relative-arm decode,
    π₀.₅'s 32-dim pad-strip) happen inside the backend before this object
    is constructed.
    """
    actions:     np.ndarray               # (N, 14) float32 absolute
    rtt_ms:      float                    # wall-clock round-trip including transport
    horizon:     int                      # N (== actions.shape[0])
    server_info: dict[str, Any] = field(default_factory=dict)  # server-side timing etc.

    def __post_init__(self) -> None:
        a = np.asarray(self.actions, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != STATE_DIM:
            raise ValueError(
                f"actions must be (N, {STATE_DIM}) float32, got shape {a.shape}"
            )
        object.__setattr__(self, "actions", a)
        if self.horizon != a.shape[0]:
            object.__setattr__(self, "horizon", int(a.shape[0]))


# ---------------------------------------------------------------------------
# Policy — the abstraction
# ---------------------------------------------------------------------------

class Policy(abc.ABC):
    """Abstract base for a VLA inference client.

    The unifying contract:

        predict(obs: YamObservation, **opts) -> Prediction

    `**opts` is the escape hatch for per-policy knobs (e.g. MolmoAct's
    `num_steps` flow-matching steps). Unknown opts MUST be ignored, not
    rejected, so client code can pass a superset.

    Lifecycle:
        info()  -> handshake / health check, fills ServerInfo
        predict -> blocking inference (timeout_s is mandatory)
        reset() -> tell the server we're starting a new attempt (optional;
                   no-op default suits MolmoAct's stateless server)
        close() -> release network sockets / processes
    """

    #: short name registered in policies/__init__.py REGISTRY (see _LAZY)
    name: str = "abstract"

    # ----- handshake / lifecycle -----

    @abc.abstractmethod
    def info(self, timeout_s: float = 3.0) -> ServerInfo:
        """Connect (if needed) and return server self-description."""

    def reset(self, options: dict | None = None) -> None:
        """Signal start-of-attempt. Default: no-op (stateless servers)."""
        del options  # most servers ignore options

    def close(self) -> None:
        """Release transport resources. Default: no-op."""

    # ----- inference -----

    @abc.abstractmethod
    def predict(self, obs: YamObservation, *,
                timeout_s: float = 5.0,
                **opts) -> Prediction:
        """Run one inference. Must return (N, 14) absolute action chunk.

        Implementations should:
          1. validate `obs` (the YamObservation dataclass already enforces
             shape + dtype, so this is mostly type-narrowing)
          2. pack obs into the server's native schema
          3. send + receive (respecting `timeout_s`)
          4. unpack the server's response back to (N, 14) absolute
          5. construct a `Prediction` and return it

        `**opts`: backend-specific knobs. Unknown keys MUST be silently
        ignored (a logger.debug is fine; a raise is not).
        """

    # ----- helpers backends can use -----

    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000.0


__all__ = ["Policy", "Prediction", "ServerInfo"]
