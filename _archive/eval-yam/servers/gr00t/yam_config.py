"""YAM bimanual modality config for Isaac-GR00T N1.7.

Registers a NEW_EMBODIMENT modality matching the trained checkpoint
``jeqcho/gr00t-n17-yam-bimanual`` -- the keys here are read directly from
the checkpoint's ``experiment_cfg/conf.yaml`` and ARE authoritative.

    state[0..5]   left arm joints  (q0..q5)        -> obs["state"]["left_arm"]
    state[6]      left gripper      (normalized)    -> obs["state"]["left_gripper"]
    state[7..12]  right arm joints (q0..q5)        -> obs["state"]["right_arm"]
    state[13]     right gripper     (normalized)    -> obs["state"]["right_gripper"]

Three RGB cameras:
    - top    (overhead, e.g. D435)             -> obs["video"]["top"]
    - left   (close-up on left arm,  e.g. D405) -> obs["video"]["left"]
    - right  (close-up on right arm, e.g. D405) -> obs["video"]["right"]

NOTE: an earlier version of this file (copied from
``grootn1.7 exploration/scripts/yam_config.py``) used image keys
``[top, left_wrist, right_wrist]`` -- that pre-dated the actual finetune
and never had to match a real checkpoint. The names above come from the
trained config; if you change them, also update
``eval-yam/scripts/yam_backends.py:Gr00tZmqBackend.VIDEO_KEY_*``.

Action horizon: 16 steps (delta_indices 0..15). Arms RELATIVE, grippers
ABSOLUTE -- the GR00T N1.7 N17-YAM training default.

Import this module to register the config. The server picks it up via
``--modality-config-path servers/gr00t/yam_config.py``.
"""
from __future__ import annotations

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ACTION_HORIZON = 16


yam_bimanual_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["top", "left", "right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, ACTION_HORIZON)),
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(
    yam_bimanual_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
