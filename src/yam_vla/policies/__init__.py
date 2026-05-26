"""Concrete Policy implementations, one per VLA wire format.

To add a new VLA, drop a `<name>.py` here that exports a single Policy
subclass and register it in `REGISTRY`. Imports are deferred so installing
the package doesn't require every backend's optional deps.
"""
from importlib import import_module
from typing import Type

from yam_vla.core.policy import Policy

# name -> (module_path, class_name) -- import-on-demand
_LAZY: dict[str, tuple[str, str]] = {
    "molmoact2": ("yam_vla.policies.molmoact2",  "MolmoAct2Policy"),
    "gr00t-n17": ("yam_vla.policies.gr00t_n17", "Gr00tN17Policy"),
    "pi05":      ("yam_vla.policies.pi05",       "Pi05Policy"),
}


def get_policy_class(name: str) -> Type[Policy]:
    """Resolve a policy name to its concrete class. Raises KeyError if unknown."""
    try:
        mod_path, cls_name = _LAZY[name]
    except KeyError:
        raise KeyError(
            f"Unknown policy {name!r}. Known: {sorted(_LAZY)}"
        ) from None
    return getattr(import_module(mod_path), cls_name)


def list_policies() -> list[str]:
    return sorted(_LAZY)
