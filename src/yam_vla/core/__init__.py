"""yam_vla.core — abstractions + hardware + safety + journal + control loop."""

# Abstractions (what every backend implements)
from yam_vla.core.observation import (
    ImageRole,
    YamObservation,
    YamStateCodec,
    STATE_DIM,
    ARM_DOF,
    LEFT_ARM,
    LEFT_GRIP,
    RIGHT_ARM,
    RIGHT_GRIP,
)
from yam_vla.core.policy import Policy, Prediction, ServerInfo
from yam_vla.core.config import PolicyConfig

# Inference runner (async wrapping of Policy.predict)
from yam_vla.core.runner import (
    AsyncPolicyInference,
    AttemptStats as RunnerAttemptStats,   # kept for back-compat; control_loop owns canonical
    build_observation,
)

# Hardware (cameras + arms + state + i2rt SDK glue)
from yam_vla.core.hardware import (
    install_sdk_lock_fix,
    load_setup_config,
    CameraHealthWatcher,
    CameraStream,
    RealSenseStream,
    V4L2Stream,
    make_camera,
    init_arm,
    read_state,
    ramp_to_pose,
    DEFAULT_TRAIN_FPS,
    DEFAULT_HORIZON_STRIDE,
    DEFAULT_CAM_WIDTH,
    DEFAULT_CAM_HEIGHT,
    DEFAULT_CAM_FPS,
    DEFAULT_SETUP_CONFIG_PATH,
)

# Safety (per-tick action clip)
from yam_vla.core.safety import (
    safe_command,
    DEFAULT_MAX_STEP_RAD,
    DEFAULT_GRIPPER_STEP,
)

# Journal (markdown research log)
from yam_vla.core.journal import (
    DEFAULT_JOURNAL_PATH,
    capture_invocation,
    prompt_journal_entry,
    write_journal_entry,
)

# Observability (optional Rerun .rrd streaming)
from yam_vla.core.observability import RerunRecorder

# Control loop (per-attempt orchestrator)
from yam_vla.core.control_loop import (
    AttemptKnobs,
    AttemptStats,
    run_attempt,
)

# Embodiment registry (yam | trlc_dk1 | ...). Imported here so callers
# can do `from yam_vla.core import get_backend` without reaching into the
# subpackage. The Protocols themselves live in yam_vla.embodiments.base.
from yam_vla.embodiments import get_backend, known_embodiments


__all__ = [
    # observation / state / abstractions
    "ImageRole", "YamObservation", "YamStateCodec",
    "STATE_DIM", "ARM_DOF", "LEFT_ARM", "LEFT_GRIP", "RIGHT_ARM", "RIGHT_GRIP",
    "Policy", "Prediction", "ServerInfo", "PolicyConfig",
    # async inference + obs builder
    "AsyncPolicyInference", "build_observation",
    # hardware
    "install_sdk_lock_fix", "load_setup_config",
    "CameraStream", "RealSenseStream", "V4L2Stream", "make_camera",
    "CameraHealthWatcher",
    "init_arm", "read_state", "ramp_to_pose",
    "DEFAULT_TRAIN_FPS", "DEFAULT_HORIZON_STRIDE",
    "DEFAULT_CAM_WIDTH", "DEFAULT_CAM_HEIGHT", "DEFAULT_CAM_FPS",
    "DEFAULT_SETUP_CONFIG_PATH",
    # safety
    "safe_command", "DEFAULT_MAX_STEP_RAD", "DEFAULT_GRIPPER_STEP",
    # journal
    "DEFAULT_JOURNAL_PATH", "capture_invocation",
    "prompt_journal_entry", "write_journal_entry",
    # observability
    "RerunRecorder",
    # control loop
    "AttemptKnobs", "AttemptStats", "run_attempt",
    # embodiments
    "get_backend", "known_embodiments",
]
