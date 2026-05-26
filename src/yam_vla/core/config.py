"""Per-policy YAML config loader.

Each YAML in `configs/policy/<name>.yaml` describes the deployment
parameters of a single VLA backend (transport, image-key mapping,
backend-specific options, control-loop hints). The loader returns a
fully-constructed `Policy` instance ready to call.

YAML schema (informal):

    policy:    str                   # must match a key in policies/__init__.py REGISTRY
    model_id:  str                   # HF repo_id (provenance only; no auto-download)
    transport:
        scheme: http|zmq|ws          # informational
        host:   str
        port:   int
        path:   str (optional, http only)
        timeout_ms: int (optional, zmq only)
    image_keys:
        top:         <wire-key>
        left_wrist:  <wire-key>
        right_wrist: <wire-key>
    options:                         # backend-specific knobs, e.g. num_steps
        ...
    control:
        action_horizon_hint:     int
        horizon_stride_default:  int

The loader is intentionally tolerant: unknown top-level keys are stored
on the returned PolicyConfig.raw and ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from yam_vla.core.policy import Policy


@dataclass(frozen=True)
class PolicyConfig:
    """Parsed contents of a configs/policy/<name>.yaml file."""
    policy:      str
    model_id:    str | None
    transport:   dict[str, Any]
    image_keys:  dict[str, str]
    options:     dict[str, Any]
    control:     dict[str, Any]
    raw:         dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> "PolicyConfig":
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict) or "policy" not in data:
            raise ValueError(f"{path}: missing required key 'policy'")
        return cls(
            policy=data["policy"],
            model_id=data.get("model_id"),
            transport=dict(data.get("transport") or {}),
            image_keys=dict(data.get("image_keys") or {}),
            options=dict(data.get("options") or {}),
            control=dict(data.get("control") or {}),
            raw=data,
        )

    def build(self, **overrides: Any) -> Policy:
        """Instantiate the concrete Policy. Overrides win over YAML."""
        # Deferred import to avoid circular import at module load time.
        from yam_vla.policies import get_policy_class
        cls = get_policy_class(self.policy)

        kw: dict[str, Any] = {}
        scheme = (self.transport.get("scheme") or "").lower()
        host = self.transport.get("host", "127.0.0.1")
        port = self.transport.get("port")

        if scheme == "http":
            base = f"{scheme}://{host}:{port}{self.transport.get('path', '')}"
            kw["server_url"] = base
            if "num_steps" in self.options:
                kw["default_num_steps"] = int(self.options["num_steps"])
        else:
            # zmq / ws backends share host/port plumbing.
            kw["host"] = host
            if port is not None:
                kw["port"] = int(port)
            if scheme == "zmq" and "timeout_ms" in self.transport:
                kw["zmq_timeout_ms"] = int(self.transport["timeout_ms"])

        # image_keys mapping is uniform across all backends.
        if self.image_keys:
            kw["image_keys"] = dict(self.image_keys)
        # gr00t uses video_keys, not image_keys; rename here without leaking
        # this detail into the YAML schema.
        if self.policy == "gr00t-n17" and "image_keys" in kw:
            kw["video_keys"] = kw.pop("image_keys")
            if "language_key" in self.raw:
                kw["language_key"] = self.raw["language_key"]

        kw.update(overrides)
        return cls(**kw)


__all__ = ["PolicyConfig"]
