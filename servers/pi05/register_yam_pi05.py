"""Side-effect-on-import shim that registers the ``yam_pi05`` training config
with OpenPI's runtime, so ``serve_policy.py policy:checkpoint --policy.config=yam_pi05``
can find it.

WHY THIS EXISTS
---------------
The ``yam_pi05`` config is NOT in upstream openpi
(https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
only ships pi05_aloha, pi05_droid, pi05_libero). It lives in jeqcho's private
training fork. To serve the public checkpoint, we register a matching
``TrainConfig`` at runtime so the serve script can look it up.

HOW TO USE
----------
1. Edit the body below so it matches the EXACT TrainConfig used to train
   ``jeqcho/pi05-yam-bimanual``. The template below is the agent's best guess
   from public artifacts (HF model card, norm_stats.json). The fields most
   likely to differ from upstream pi05_aloha are:
     - ``data.repo_id``         -- the LeRobot dataset id used in training
     - ``data.assets.asset_id`` -- norm_stats key (already pinned)
     - ``data.adapt_to_pi``     -- almost certainly False (YAM != Trossen Aloha)
     - whether the fork uses ``LeRobotAlohaDataConfig`` or a custom
       ``YamInputs/YamOutputs`` transform pair with different image keys

   The fastest way to get this right is to copy the actual
   ``TrainConfig(name="yam_pi05", ...)`` from your training fork (look in
   ``src/openpi/training/config.py``).

2. ``run_server.sh`` imports this module before invoking serve_policy, which
   appends the config to ``openpi.training.config._CONFIGS``.

WIRE FORMAT IMPLICATIONS
------------------------
The image key names AT THE PYTHON-DATA LEVEL (cam_high, cam_left_wrist,
cam_right_wrist for AlohaInputs; or top/left_wrist/right_wrist if the fork
defined custom YamInputs) determine what the *client* must send. If you
change the data transform here, also update yam_backends.Pi05WebsocketBackend's
IMG_KEY_* constants to match.

References:
- Upstream pi05_aloha TrainConfig:
    https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
- LeRobotAlohaDataConfig:
    https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
- HF model card: https://huggingface.co/jeqcho/pi05-yam-bimanual
  (norm stats key: jeqcho/yam-bimanual-merged-v2-train; action_horizon=16,
   action_dim=14; state_dim=14; trained on 8xB200 for 19,300 steps)
"""
from __future__ import annotations

# Local imports are deferred so that running with --help on the wrapper
# script doesn't require openpi to be importable yet.

def register() -> None:
    """Append the ``yam_pi05`` TrainConfig to openpi's _CONFIGS list AND
    _CONFIGS_DICT (the latter is built ONCE at module-import time --
    appending only to the list is not enough, get_config() looks the dict).
    """
    from openpi.models import pi0_config
    from openpi.policies import aloha_policy  # noqa: F401  -- transforms imported via cfg
    from openpi.training import config as _cfg_mod
    from openpi.training.config import (
        AssetsConfig,
        LeRobotAlohaDataConfig,
        TrainConfig,
    )

    # Bail out if already registered (re-imports are safe).
    if "yam_pi05" in _cfg_mod._CONFIGS_DICT:
        return

    # Pi0Config defaults are action_dim=32, action_horizon=50 (the MODEL's
    # internal padded dimensions, not the data's). All agilex pi05_*
    # configs use bare Pi0Config(pi05=True) and rely on the input/output
    # transforms to map between the data's real action shape (here: 14-D
    # over a 16-step horizon, per the HF model card) and the model's
    # internal (50, 32). Overriding action_dim/horizon here causes a
    # checkpoint shape mismatch -- this was the original bug.
    cfg = TrainConfig(
        name="yam_pi05",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # The LeRobot dataset repo used in training. Confirmed from
            # the checkpoint's `assets/<asset_id>/norm_stats.json`.
            repo_id="jeqcho/yam-bimanual-merged-v2-train",
            assets=AssetsConfig(
                # When set to None, the assets bundled inside the
                # checkpoint dir's ``assets/`` subtree are used. This
                # matches what serve_policy.py expects when loading
                # by --policy.dir.
                assets_dir=None,
                asset_id="jeqcho/yam-bimanual-merged-v2-train",
            ),
            # adapt_to_pi: MUST match the training fork.
            # - False (correct): YAM joint angles passed through as-is.
            # - True  (wrong here): _joint_flip_mask sign-flips joints
            #   [1,2] of each arm on input AND output, mangling the
            #   model's view of state. Tested empirically: with True,
            #   arms immediately retracted to rest pose (model saw
            #   garbage state and planned a "safe" recovery). Reverted.
            #
            # Training fork's actual setting was False.
            adapt_to_pi=False,
            # use_delta_joint_actions: MUST match the training fork's
            # setting. If they differ, the openpi server's AbsoluteActions
            # output_transform either double-adds state (config=True but
            # trained=False -> 2x commanded motion, dangerously fast) or
            # never converts (config=False but trained=True -> tiny
            # near-zero deltas, arm barely moves).
            #
            # Started at True (agilex default). Symptom-based diagnosis
            # against the running checkpoint pointed at the double-add
            # pathology: pi-0.5's a[0] sat ~0.8 rad off a sensible state
            # while molmoact2's a[0] sat ~0.05 rad off the same state,
            # and arm motion felt "like gr00t pre-fix". Flipped to False
            # to match what appears to be the actual training-fork setting.
            use_delta_joint_actions=False,
        ),
    )

    # Both list and dict must be updated -- get_config() reads the dict.
    _cfg_mod._CONFIGS.append(cfg)
    _cfg_mod._CONFIGS_DICT[cfg.name] = cfg


# Side-effect on import so `python -c "import register_yam_pi05"` works as
# the activation hook.
register()
