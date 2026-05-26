"""YAM bimanual modality config for Isaac-GR00T N1.7.

Registers a NEW_EMBODIMENT modality with the schema that matches the YAM
14-D state vector used everywhere else in this repo:

    state[0..5]   left arm joints  (q0..q5)
    state[6]      left gripper      (normalized [0, 1])
    state[7..12]  right arm joints (q0..q5)
    state[13]     right gripper     (normalized [0, 1])

Three RGB cameras:
    - top         (overhead, e.g. D435)
    - left_wrist  (close-up on left arm, e.g. D405)
    - right_wrist (close-up on right arm, e.g. D405)

Action horizon: 16 steps.

Joint actions are RELATIVE (delta from current state) per the N1.7 default —
this is what the EgoScale pretraining is built around. Grippers are ABSOLUTE
because a binary open/close is easier to model that way.

Import this module to register the config. The server picks it up via
``--modality-config-path scripts/yam_config.py``.
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
        modality_keys=["top", "left_wrist", "right_wrist"],
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
