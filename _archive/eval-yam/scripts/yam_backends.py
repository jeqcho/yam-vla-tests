"""Pluggable inference backends for bimanual YAM evaluation.

The eval/REPL client speaks ONE canonical wire format internally:

    inputs:  top, left_wrist, right_wrist (each HxWx3 uint8 RGB)
             state          (14,) float32 = [left_q0..5, left_grip,
                                              right_q0..5, right_grip]
             instruction    str
             num_steps      int      (flow-matching steps, advisory)
             timeout_s      float
    outputs: actions        (N, 14) float32  -- absolute joint positions
             rtt_ms         float

Each VLA's server has its own native schema (json_numpy/HTTP, msgpack/ZMQ,
msgpack/WebSocket); a Backend wraps that schema so the rest of the client
is server-agnostic. Install one with `install_backend(backend)` and the
existing `yam_client.post_actions` / `yam_repl.post_actions` calls route
through it.

Each backend documents:

  * the underlying server's transport, port, repo_id
  * how YAM 14-D state is split / packed into the server's native keys
  * how the server's action chunk is reassembled into (N, 14) absolute joint
    positions (some servers return relative deltas — we convert)
"""
from __future__ import annotations

import abc
import logging
import time
from typing import Optional

import numpy as np

log = logging.getLogger("yam.backends")

STATE_DIM = 14
ARM_DOFS = 7

# Per-arm indexing within the 14-D YAM state vector.
#   [0..5] left_arm    [6] left_gripper
#   [7..12] right_arm  [13] right_gripper
LEFT_ARM = slice(0, 6)
LEFT_GRIP = 6
RIGHT_ARM = slice(7, 13)
RIGHT_GRIP = 13


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class Backend(abc.ABC):
    """Inference-server adapter. One subclass per VLA wire format."""

    name: str = "abstract"

    @abc.abstractmethod
    def health_check(self, timeout_s: float = 3.0) -> dict:
        """Return a small dict describing the server (or raise)."""

    @abc.abstractmethod
    def predict(
        self,
        top: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        state: np.ndarray,
        instruction: str,
        num_steps: int,
        timeout_s: float,
    ) -> tuple[np.ndarray, float]:
        """Returns (actions[N, 14] float32 absolute joint positions, rtt_ms)."""

    def warmup(self, *args, **kw) -> None:
        """Optional model warmup; default no-op."""


# ---------------------------------------------------------------------------
# MolmoAct2 — HTTP + json_numpy on port 8202
# ---------------------------------------------------------------------------
# Wire-format reference: molmoact2-setup/molmoact2/examples/yam/host_server_yam.py
#
# Schema:
#   POST  /act    json_numpy body {top_cam, left_cam, right_cam, state(14,),
#                                  instruction, num_steps?}
#   Reply json_numpy {actions: (N, 14) absolute joint pos, dt_ms}
#
# Camera order in the request MUST be [top, left, right] -- the model was
# trained with this ordering; swapping left/right at inference time produces
# incoherent actions.

class MolmoActHTTPBackend(Backend):
    name = "molmoact2"

    def __init__(self, server_url: str = "http://127.0.0.1:8202/act"):
        import json_numpy  # noqa: F401  -- patches stdlib json
        json_numpy.patch()
        import requests  # noqa: F401  -- import-checked here so a missing dep fails loud
        self.server_url = server_url

    def health_check(self, timeout_s: float = 3.0) -> dict:
        import requests
        r = requests.get(self.server_url, timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    def predict(
        self,
        top: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        state: np.ndarray,
        instruction: str,
        num_steps: int,
        timeout_s: float,
    ) -> tuple[np.ndarray, float]:
        import json_numpy
        import requests
        payload = {
            "top_cam":     top,
            "left_cam":    left_wrist,
            "right_cam":   right_wrist,
            "instruction": instruction,
            "state":       np.asarray(state, dtype=np.float32),
            "num_steps":   int(num_steps),
            "timestamp":   time.time(),
        }
        body = json_numpy.dumps(payload)
        t0 = time.perf_counter()
        resp = requests.post(
            self.server_url, data=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        out = json_numpy.loads(resp.text)
        if "actions" not in out:
            raise RuntimeError(f"server response missing 'actions': keys={list(out.keys())}")
        actions = np.asarray(out["actions"], dtype=np.float32)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        return actions, rtt_ms


# ---------------------------------------------------------------------------
# GR00T N1.7 — ZeroMQ REQ/REP + msgpack-numpy on port 5556
# ---------------------------------------------------------------------------
# Wire-format reference:
#   grootn1.7 exploration/Isaac-GR00T/gr00t/policy/server_client.py
#   grootn1.7 exploration/scripts/yam_client.py    (the slim GrootPolicyClient)
#
# Observation schema (note the (B=1, T=1, ...) leading dims).
# Video keys [top, left, right] come from the trained checkpoint's
# experiment_cfg/conf.yaml (data.modality_configs.new_embodiment.video.modality_keys)
# -- NOT [top, left_wrist, right_wrist] as in the speculative
# grootn1.7-exploration setup.
#   obs["state"]["left_arm"]      (1, 1, 6)    raw joint rad
#   obs["state"]["left_gripper"]  (1, 1, 1)    normalized [0, 1]
#   obs["state"]["right_arm"]     (1, 1, 6)
#   obs["state"]["right_gripper"] (1, 1, 1)
#   obs["video"]["top"]           (1, 1, H, W, 3) uint8 RGB
#   obs["video"]["left"]          (1, 1, H, W, 3)
#   obs["video"]["right"]         (1, 1, H, W, 3)
#   obs["language"]["annotation.human.task_description"]  [[str]]  (1, 1)
#
# Response schema (16-step action chunk per key):
#   actions["left_arm"]      (1, 16, 6)   ABSOLUTE joint targets (rad)
#   actions["left_gripper"]  (1, 16, 1)   ABSOLUTE gripper [0, 1]
#   actions["right_arm"]     (1, 16, 6)   ABSOLUTE
#   actions["right_gripper"] (1, 16, 1)   ABSOLUTE
#
# IMPORTANT: although the *training-time* action_config says rep=RELATIVE
# for the arms, the SERVER decodes relative->absolute before returning the
# chunk. Gr00tPolicy.predict_action calls processor.decode_action which
# calls StateActionProcessor.unapply_action(..., state=current_state),
# and that's where the addition happens. By the time the wire payload
# arrives at the client, the arm values are already absolute joint
# targets in the same space as the YAM state.
#
# Earlier versions of this backend ADDED current state on top, which
# resulted in 2x commanded motion -- arms swung dangerously fast. Do
# NOT add state here.

class Gr00tZmqBackend(Backend):
    name = "gr00t-n17"

    # GR00T checkpoint's modality keys -- read from
    # jeqcho/gr00t-n17-yam-bimanual/experiment_cfg/conf.yaml.
    VIDEO_KEY_TOP = "top"
    VIDEO_KEY_LEFT = "left"
    VIDEO_KEY_RIGHT = "right"
    LANG_KEY = "annotation.human.task_description"

    def __init__(self, host: str = "127.0.0.1", port: int = 5556,
                 timeout_ms: int = 30000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._client = None  # lazy init so import doesn't require pyzmq

    def _client_or_init(self):
        if self._client is None:
            # Local import keeps the dep optional unless this backend is used.
            import msgpack_numpy as _mnp
            import zmq as _zmq
            self._mnp = _mnp
            self._zmq = _zmq
            self._ctx = _zmq.Context.instance()
            self._client = self._ctx.socket(_zmq.REQ)
            self._client.setsockopt(_zmq.RCVTIMEO, self.timeout_ms)
            self._client.setsockopt(_zmq.SNDTIMEO, self.timeout_ms)
            self._client.connect(f"tcp://{self.host}:{self.port}")
        return self._client

    def _call(self, endpoint: str, data: Optional[dict] = None,
              requires_input: bool = True):
        sock = self._client_or_init()
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            sock.send(self._mnp.packb(req))
            msg = sock.recv()
        except self._zmq.error.Again:
            # Socket got into a bad REQ/REP state -- recreate.
            self._client = None
            raise
        resp = self._mnp.unpackb(msg, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"gr00t server error: {resp['error']}")
        return resp

    def health_check(self, timeout_s: float = 3.0) -> dict:
        # PolicyServer.ping endpoint -- returns server metadata.
        # Override RCVTIMEO temporarily for a faster healthcheck failure.
        sock = self._client_or_init()
        old = sock.getsockopt(self._zmq.RCVTIMEO)
        sock.setsockopt(self._zmq.RCVTIMEO, int(timeout_s * 1000))
        try:
            resp = self._call("ping", requires_input=False)
        finally:
            sock.setsockopt(self._zmq.RCVTIMEO, old)
        if not isinstance(resp, dict):
            resp = {"ping": resp}
        resp.setdefault("backend", self.name)
        resp.setdefault("transport", f"zmq tcp://{self.host}:{self.port}")
        return resp

    @staticmethod
    def _bt1(x: np.ndarray) -> np.ndarray:
        """Add leading (B=1, T=1) dims expected by GR00T's PolicyServer."""
        return np.expand_dims(np.expand_dims(np.asarray(x), 0), 0)

    def predict(
        self,
        top: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        state: np.ndarray,
        instruction: str,
        num_steps: int,  # GR00T doesn't use this; horizon is baked into the checkpoint
        timeout_s: float,
    ) -> tuple[np.ndarray, float]:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.shape != (STATE_DIM,):
            raise ValueError(f"state must be ({STATE_DIM},), got {s.shape}")

        observation = {
            "state": {
                "left_arm":      self._bt1(s[LEFT_ARM].astype(np.float32)),
                "left_gripper":  self._bt1(np.asarray([s[LEFT_GRIP]], dtype=np.float32)),
                "right_arm":     self._bt1(s[RIGHT_ARM].astype(np.float32)),
                "right_gripper": self._bt1(np.asarray([s[RIGHT_GRIP]], dtype=np.float32)),
            },
            "video": {
                self.VIDEO_KEY_TOP:   self._bt1(top),
                self.VIDEO_KEY_LEFT:  self._bt1(left_wrist),
                self.VIDEO_KEY_RIGHT: self._bt1(right_wrist),
            },
            "language": {
                self.LANG_KEY: [[instruction]],
            },
        }

        # Server-side timeout already governs the per-call wait; bump the
        # ZMQ recv timeout to roughly match so we don't disconnect early.
        sock = self._client_or_init()
        old = sock.getsockopt(self._zmq.RCVTIMEO)
        sock.setsockopt(self._zmq.RCVTIMEO, int(timeout_s * 1000))

        t0 = time.perf_counter()
        try:
            action_dict, _info = self._call(
                "get_action",
                {"observation": observation, "options": None},
            )
        finally:
            sock.setsockopt(self._zmq.RCVTIMEO, old)
        rtt_ms = (time.perf_counter() - t0) * 1000.0

        # Each key: (B=1, T=horizon, dim). Strip batch, stitch into (T, 14).
        def _strip(name: str) -> np.ndarray:
            a = np.asarray(action_dict[name], dtype=np.float32)
            if a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
            return a
        la = _strip("left_arm")
        lg = _strip("left_gripper")
        ra = _strip("right_arm")
        rg = _strip("right_gripper")
        # Sanity: all four streams must share T.
        T = la.shape[0]
        for name, arr in (("left_gripper", lg), ("right_arm", ra), ("right_gripper", rg)):
            if arr.shape[0] != T:
                raise RuntimeError(
                    f"gr00t: action horizon mismatch: left_arm T={T}, {name} T={arr.shape[0]}"
                )

        # Stitch the 4 per-modality streams into a (T, 14) chunk. ALL FOUR
        # streams are absolute -- the server already converted relative arm
        # deltas to absolute via StateActionProcessor.unapply_action. Do NOT
        # add the current state here; doing so was a real bug that doubled
        # all commanded arm motion and made gr00t feel "too fast / dangerous".
        actions = np.zeros((T, STATE_DIM), dtype=np.float32)
        actions[:, LEFT_ARM]   = la                # absolute joint targets
        actions[:, LEFT_GRIP]  = lg.reshape(-1)    # absolute gripper
        actions[:, RIGHT_ARM]  = ra                # absolute joint targets
        actions[:, RIGHT_GRIP] = rg.reshape(-1)    # absolute gripper
        return actions, rtt_ms


# ---------------------------------------------------------------------------
# Pi-0.5 — WebSocket + msgpack on port 8000 (openpi convention)
# ---------------------------------------------------------------------------
# Wire-format reference:
#   openpi/src/openpi/serving/websocket_policy_server.py
#   packages/openpi-client/src/openpi_client/websocket_client_policy.py
#
# OpenPI's serve_policy.py exposes a `websockets`-library server. On connect
# the server sends one msgpack frame of metadata, then per-call:
#   client -> msgpack(obs); server -> msgpack({"actions": (T, 14), "server_timing": {...}})
# Codec is openpi's vendored msgpack-numpy: ndarrays serialize as
# {__ndarray__, data:bytes, dtype, shape}.
#
# Observation schema -- the AGILEX fork's openpi differs from upstream:
#
#   * Upstream openpi's AlohaInputs accepts cam_high/cam_left_wrist/
#     cam_right_wrist and remaps to canonical base_0_rgb/left_wrist_0_rgb/
#     right_wrist_0_rgb internally.
#   * Agilex's AlohaInputs is a passthrough -- expects images ALREADY in
#     canonical form, zero-fills any canonical key not present (see
#     openpi-agilex/src/openpi/policies/aloha_policy.py: STANDARD_IMAGE_KEYS).
#   * create_trained_policy(repack_transforms=None) defaults to an empty
#     Group AT INFERENCE -- the config's repack mapping (which would
#     translate cam_* -> base_0_rgb in the agilex AlohaInputs world) is
#     NEVER applied at serve time. It only matters during training.
#
# Consequence: clients must send canonical keys directly. Sending
# cam_high gets the image silently dropped into an "extra" bucket the
# model doesn't read, while base_0_rgb stays zero-filled -- the model
# runs BLIND. (Diagnosed by user reporting "pi-0.5 doesn't pick up the
# cube" -- safe motion but no visual grounding.)
#
#   "state"  (14,) float32  [left_q(6), left_grip(1), right_q(6), right_grip(1)]
#   "images" {
#     "base_0_rgb":        (3, H, W) uint8 RGB   <-- top
#     "left_wrist_0_rgb":  (3, H, W) uint8 RGB
#     "right_wrist_0_rgb": (3, H, W) uint8 RGB
#   }
#   "prompt" str
#
# IMPORTANT: CHW, not HWC. We transpose here so the rest of the codebase
# stays HWC-everywhere.
#
# Response:
#   {"actions": (50, 32) float32; sliced to (50, 14) client-side. ABSOLUTE
#    joint targets after the AbsoluteActions output_transform (or directly
#    absolute if use_delta_joint_actions=False matches the training).}
# Layout: [left_q(6), left_grip(1), right_q(6), right_grip(1)] then 18 pad dims.

class Pi05WebsocketBackend(Backend):
    name = "pi05"

    # Canonical openpi image keys. Despite the "aloha" branding, these
    # are the model-internal names every openpi-pi0/pi05 policy reads --
    # the model was trained on these exact strings. Do NOT change without
    # also looking at the agilex AlohaInputs.STANDARD_IMAGE_KEYS list.
    OBS_KEY_STATE = "state"
    OBS_KEY_IMAGES = "images"
    IMG_KEY_TOP = "base_0_rgb"
    IMG_KEY_LEFT = "left_wrist_0_rgb"
    IMG_KEY_RIGHT = "right_wrist_0_rgb"
    OBS_KEY_PROMPT = "prompt"

    def __init__(self, host: str = "127.0.0.1", port: int = 8000,
                 img_key_top: Optional[str] = None,
                 img_key_left: Optional[str] = None,
                 img_key_right: Optional[str] = None,
                 obs_key_state: Optional[str] = None,
                 obs_key_images: Optional[str] = None,
                 obs_key_prompt: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.host = host
        self.port = port
        self.api_key = api_key
        if img_key_top:    self.IMG_KEY_TOP = img_key_top
        if img_key_left:   self.IMG_KEY_LEFT = img_key_left
        if img_key_right:  self.IMG_KEY_RIGHT = img_key_right
        if obs_key_state:  self.OBS_KEY_STATE = obs_key_state
        if obs_key_images: self.OBS_KEY_IMAGES = obs_key_images
        if obs_key_prompt: self.OBS_KEY_PROMPT = obs_key_prompt

        # openpi-client provides the msgpack codec + websocket client policy.
        # Robot-side venv install:
        #   VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv uv pip install \
        #     'openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client'
        try:
            from openpi_client import websocket_client_policy as _wcp  # type: ignore
            self._wcp_cls = _wcp.WebsocketClientPolicy
        except Exception:
            self._wcp_cls = None
        self._policy = None  # lazy

    def _policy_or_init(self):
        if self._policy is None:
            if self._wcp_cls is None:
                raise RuntimeError(
                    "pi05 backend requires `openpi-client` installed in the "
                    "client venv. Install with:\n"
                    "  VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv uv pip install \\\n"
                    "    'openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client'"
                )
            kw = {}
            if self.api_key:
                kw["api_key"] = self.api_key
            self._policy = self._wcp_cls(host=self.host, port=self.port, **kw)
        return self._policy

    def health_check(self, timeout_s: float = 3.0) -> dict:
        # openpi's WebsocketClientPolicy receives the server's metadata frame
        # on connect, so instantiating + reading metadata serves as a health
        # check. Newer client versions expose get_server_metadata().
        policy = self._policy_or_init()
        meta: dict = {}
        for attr in ("get_server_metadata", "server_metadata"):
            if hasattr(policy, attr):
                val = getattr(policy, attr)
                meta = val() if callable(val) else (val or {})
                break
        return {
            "backend": self.name,
            "transport": f"ws://{self.host}:{self.port}",
            **(meta or {}),
        }

    @staticmethod
    def _hwc_to_chw(img: np.ndarray) -> np.ndarray:
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"expected HWC RGB, got shape {a.shape} dtype {a.dtype}")
        # (H, W, 3) -> (3, H, W); .copy() so msgpack sees a contiguous buffer
        return np.transpose(a, (2, 0, 1)).copy()

    def predict(
        self,
        top: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        state: np.ndarray,
        instruction: str,
        num_steps: int,  # pi05 doesn't expose flow-matching step count via the wire
        timeout_s: float,
    ) -> tuple[np.ndarray, float]:
        policy = self._policy_or_init()
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.shape != (STATE_DIM,):
            raise ValueError(f"state must be ({STATE_DIM},), got {s.shape}")
        obs = {
            self.OBS_KEY_STATE:  s,
            self.OBS_KEY_IMAGES: {
                self.IMG_KEY_TOP:   self._hwc_to_chw(top),
                self.IMG_KEY_LEFT:  self._hwc_to_chw(left_wrist),
                self.IMG_KEY_RIGHT: self._hwc_to_chw(right_wrist),
            },
            self.OBS_KEY_PROMPT: instruction,
        }
        t0 = time.perf_counter()
        out = policy.infer(obs)  # openpi-client's blocking infer call
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        if not isinstance(out, dict) or "actions" not in out:
            keys = list(out.keys()) if isinstance(out, dict) else type(out).__name__
            raise RuntimeError(f"pi05 response missing 'actions': {keys}")
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]

        # PI-0.5 servers return the MODEL's padded internal shape, not the
        # real-data shape. With Pi0Config defaults that's (50, 32). The
        # first 14 dims correspond to the real YAM actions; cols 14..31
        # are model padding from training (always 0 after un-normalize).
        #
        # The agilex AlohaOutputs only slices to [:, :14] when
        # `adapt_to_pi=True`, which would ALSO apply trossen-aloha joint
        # flips that are wrong for YAM. So we register yam_pi05 with
        # adapt_to_pi=False and do the dim-slice here client-side.
        # See servers/pi05/register_yam_pi05.py for the symmetric choice.
        if actions.ndim == 2 and actions.shape[1] > STATE_DIM:
            actions = actions[:, :STATE_DIM]
        if actions.ndim != 2 or actions.shape[1] != STATE_DIM:
            raise RuntimeError(
                f"pi05: expected actions (N, {STATE_DIM}), got shape {actions.shape}"
            )
        return actions, rtt_ms


# ---------------------------------------------------------------------------
# Factory + monkey-patch installer
# ---------------------------------------------------------------------------

_BACKENDS = {
    "molmoact2": MolmoActHTTPBackend,
    "gr00t-n17": Gr00tZmqBackend,
    "pi05":      Pi05WebsocketBackend,
}


def make_backend(policy: str, **kw) -> Backend:
    """Construct a backend by policy name. Extra kwargs forwarded to ctor."""
    try:
        cls = _BACKENDS[policy]
    except KeyError:
        raise ValueError(
            f"Unknown policy {policy!r}. Known: {sorted(_BACKENDS)}"
        ) from None
    return cls(**kw)


def install_backend(backend: Backend) -> None:
    """Monkey-patch the post_actions callsites in molmoact2-setup's yam_client
    and yam_repl so the existing close-loop control routes through `backend`.

    Why this works: both modules call `post_actions(server_url, top, left,
    right, state, instruction, num_steps, timeout_s) -> (actions, rtt_ms)`.
    We replace that single symbol; safety clipping, async fetcher, journal,
    boundary diagnostics, and the EnterStopWatcher all keep working unchanged.

    `yam_repl` does `from yam_client import post_actions` at module import,
    which creates a SEPARATE name binding -- a single patch on yam_client
    isn't enough. We patch both modules.
    """
    def _backend_post(server_url, top, left, right, state, instruction,
                      num_steps, timeout_s):
        # server_url is ignored by non-HTTP backends; kept in the signature
        # for source compatibility with the existing call sites.
        del server_url
        return backend.predict(
            top, left, right, state, instruction, num_steps, timeout_s,
        )

    import yam_client as yc
    yc.post_actions = _backend_post
    try:
        import yam_repl as yr
        yr.post_actions = _backend_post
    except Exception:
        # yam_repl may not be imported yet (e.g. for the bare client). That's
        # fine -- the binding in yam_repl happens at `from yam_client import
        # post_actions`, which will pick up the patched yc.post_actions if we
        # patched yc.post_actions FIRST.
        pass
    log.info("Installed backend %r (routed post_actions)", backend.name)
