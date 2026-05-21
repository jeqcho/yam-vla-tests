"""MolmoAct2-BimanualYAM RTC-enabled inference server.

Port 8203. Drop-in alternative to `scripts/host_server_yam.py` (port 8202) that
uses Ai2's lerobot fork's MolmoAct2Policy with Real-Time Chunking (RTC) enabled.
See ../INVESTIGATION.md for the design rationale.

Wire protocol (extends host_server_yam.py with three RTC fields):

    GET  /act        -> health check
    POST /act        -> action inference
        request body  (json_numpy):
            {
              "top_cam":     ndarray(H, W, 3) uint8 RGB,
              "left_cam":    ndarray(H, W, 3) uint8 RGB,
              "right_cam":   ndarray(H, W, 3) uint8 RGB,
              "instruction": str,
              "state":       ndarray(14,) float32,
              "num_steps":   int   (optional, default 10),
              # RTC-specific (all optional; if absent we behave like non-RTC):
              "prev_chunk_left_over": ndarray(L, 14) float32 (optional),
              "inference_delay":      int   (optional, default 0),
              "execution_horizon":    int   (optional, default 10),
              "timestamp":   float  (optional, ignored),
            }
        response body (json_numpy):
            {"actions": ndarray(N, 14) float32, "dt_ms": float, "rtc": {...}}

Setup (assumes the existing molmoact2-setup .venv has torch + transformers
already installed; we add lerobot from Ai2's fork):

    VIRTUAL_ENV=/home/andon/yam-tests/molmoact2-setup/.venv \\
      uv pip install \\
        "lerobot @ git+https://github.com/allenai/lerobot.git@molmoact2-policy"

Run:

    /home/andon/yam-tests/molmoact2-setup/.venv/bin/python \\
        experimental/rtc/host_server_rtc.py --port 8203
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image

# NOTE: do NOT call json_numpy.patch() here. Globally patching stdlib json
# breaks numpy.testing's import (which uses json.loads with its own
# object_hook returning SimpleNamespace). json_numpy chains its hook around
# the caller's, causing 'argument of type SimpleNamespace is not iterable'
# when its hook then runs `"__numpy__" in dct`. Our FastAPI handlers use
# json_numpy.loads/dumps explicitly, so the global patch is unnecessary.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("molmoact2.yam.rtc.server")


# ---------------------------------------------------------------------------
# Constants for the BimanualYAM checkpoint.
# Mirrors what's in scripts/host_server_yam.py + what we read out of
# norm_stats.json: camera_keys / state_key / action_key / setup_type /
# control_mode all come from there. The lerobot config wants them spelled
# out as Python values; we copy them verbatim from the checkpoint metadata.
# ---------------------------------------------------------------------------
REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
STATE_DIM = 14
ACTION_DIM = 14
NUM_CAMERAS = 3
DEFAULT_NUM_STEPS = 10

# Came from norm_stats.json[metadata_by_tag][yam_dual_molmoact2]:
SETUP_TYPE = "bimanual yam robotic arms in molmoact2"
CONTROL_MODE = "absolute joint pose"
CAMERA_KEYS = [
    "observation.images.top",
    "observation.images.left",
    "observation.images.right",
]
STATE_KEY = "observation.state"
ACTION_KEY = "action"
CHUNK_SIZE = 30  # action_horizon from checkpoint


def _to_pil(arr: Any) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


class RTCPolicy:
    """Wraps lerobot's MolmoAct2Policy with RTC enabled and exposes a single
    `predict(...)` that takes raw images + state + leftover and returns the
    14-D unnormalized action chunk.

    Builds the policy + processors once at startup. predict() is called under
    a coarse lock because the action expert isn't safe under concurrent calls
    (same constraint as the HF server; RTC sampling doesn't change this).
    """

    def __init__(
        self,
        repo_id: str,
        device: str,
        dtype: torch.dtype,
        rtc_execution_horizon: int = 10,
        rtc_max_guidance_weight: float = 10.0,
        rtc_attention_schedule: str = "linear",
    ) -> None:
        # Import lerobot here so a missing install fails with a clear message
        # AT startup rather than at import-time of this module.
        try:
            from lerobot.configs.types import FeatureType, PolicyFeature
            from lerobot.policies.molmoact2.configuration_molmoact2 import (
                MolmoAct2Config,
            )
            from lerobot.policies.molmoact2.modeling_molmoact2 import (
                MolmoAct2Policy,
            )
            from lerobot.policies.molmoact2.processor_molmoact2 import (
                make_molmoact2_pre_post_processors,
            )
            from lerobot.policies.rtc.configuration_rtc import (
                RTCAttentionSchedule,
                RTCConfig,
            )
        except ImportError as e:
            raise RuntimeError(
                "lerobot import failed. Install Ai2's fork into the server venv:\n"
                "  VIRTUAL_ENV=/home/andon/yam-tests/molmoact2-setup/.venv "
                "uv pip install 'lerobot @ "
                "git+https://github.com/allenai/lerobot.git@molmoact2-policy'\n"
                f"Original error: {e}"
            ) from e

        # ---------- Build the RTCConfig --------------------------------
        schedule_map = {
            "linear": RTCAttentionSchedule.LINEAR,
            # If the enum has other options on this fork they'll be added when
            # needed; LINEAR is what the paper recommends as default.
        }
        if rtc_attention_schedule not in schedule_map:
            raise ValueError(
                f"unknown rtc_attention_schedule={rtc_attention_schedule!r}; "
                f"known: {list(schedule_map)}"
            )
        self.rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=schedule_map[rtc_attention_schedule],
            max_guidance_weight=rtc_max_guidance_weight,
            execution_horizon=rtc_execution_horizon,
        )
        log.info(
            "RTCConfig: enabled=True, schedule=%s, exec_horizon=%d, max_guidance=%.1f",
            rtc_attention_schedule, rtc_execution_horizon, rtc_max_guidance_weight,
        )

        # ---------- Build the MolmoAct2Config --------------------------
        # Most of these mirror what norm_stats.json says about the YAM
        # checkpoint (norm_tag yam_dual_molmoact2). When in doubt, keep
        # lerobot defaults — the policy class itself overrides them from
        # the HF checkpoint when it loads.
        # output_features must include the action feature with a positive
        # shape, otherwise _output_action_dim() raises.
        # input_features should ONLY include the state -- if image features
        # are present here, the normalizer iterates over them and calls
        # torch.as_tensor(PIL.Image), which fails with "Could not infer dtype
        # of Image". Image handling is done separately by the MolmoAct2
        # PackInputs processor step (which reads config.image_keys directly).
        action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))
        state_feature = PolicyFeature(type=FeatureType.STATE,  shape=(STATE_DIM,))
        self.config = MolmoAct2Config(
            checkpoint_path=repo_id,
            norm_tag=NORM_TAG,
            image_keys=CAMERA_KEYS,
            setup_type=SETUP_TYPE,
            control_mode=CONTROL_MODE,
            chunk_size=CHUNK_SIZE,
            n_action_steps=CHUNK_SIZE,
            rtc_config=self.rtc_config,
            inference_action_mode="continuous",
            input_features={STATE_KEY: state_feature},
            output_features={ACTION_KEY: action_feature},
        )
        log.info(
            "MolmoAct2Config: repo=%s, norm_tag=%s, image_keys=%s",
            repo_id, NORM_TAG, CAMERA_KEYS,
        )

        # ---------- Build the policy ------------------------------------
        log.info("Loading MolmoAct2Policy (this can take ~30s on first run)...")
        t0 = time.perf_counter()
        # MolmoAct2Policy.__init__ delegates to AutoModelForImageTextToText
        # internally; trust_remote_code is set in the config (default True).
        self.policy = MolmoAct2Policy(self.config)
        self.policy = self.policy.to(device).eval()
        # Cast floating-point parameters to the requested dtype (bf16).
        # We avoid `.to(dtype)` on the whole policy because non-fp buffers
        # (token IDs etc.) shouldn't be touched. The action expert + VLM
        # backbone are what we want in bf16.
        for p in self.policy.parameters():
            if p.is_floating_point():
                p.data = p.data.to(dtype)
        log.info("MolmoAct2Policy loaded in %.1fs", time.perf_counter() - t0)

        # ---------- Build the pre/post processors -----------------------
        log.info("Building MolmoAct2 pre/post processor pipelines...")
        t0 = time.perf_counter()
        # dataset_stats=None triggers fallback to norm_stats.json via norm_tag.
        self.pre, self.post = make_molmoact2_pre_post_processors(
            self.config, dataset_stats=None
        )
        log.info("Processors ready in %.1fs", time.perf_counter() - t0)

        self.device = device
        self.dtype = dtype
        # Coarse serialization: predict_action_chunk isn't safe under
        # concurrent calls (CUDA graph capture / action expert state).
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(
        self,
        top_cam: np.ndarray,
        left_cam: np.ndarray,
        right_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = DEFAULT_NUM_STEPS,
        prev_chunk_left_over: np.ndarray | None = None,
        inference_delay: int = 0,
        execution_horizon: int = 10,
    ) -> tuple[np.ndarray, dict]:
        """Run one RTC-augmented inference. Returns (actions[CHUNK_SIZE,14], meta).

        meta has diagnostic fields the client can log:
            {
              "leftover_len_in": int,    # how many leftover steps we received
              "execution_horizon": int,
              "inference_delay": int,
              "num_steps": int,
            }
        """
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (STATE_DIM,):
            raise ValueError(
                f"state must be shape ({STATE_DIM},), got {state_f32.shape}"
            )

        # The lerobot processor pipeline's default `to_transition` is
        # `batch_to_transition`, which expects a FLAT dict where keys
        # starting with "observation." become the observation dict and
        # "task" goes into complementary_data. (See
        # lerobot.processor.converters.batch_to_transition and
        # _extract_complementary_data.) We were previously building a
        # TransitionKey-keyed dict, which the pipeline doesn't recognize as
        # a batch -- the first ObservationProcessorStep then sees None.
        batch_in: dict[str, Any] = {
            CAMERA_KEYS[0]: _to_pil(top_cam),
            CAMERA_KEYS[1]: _to_pil(left_cam),
            CAMERA_KEYS[2]: _to_pil(right_cam),
            STATE_KEY: torch.from_numpy(state_f32).to(self.device),
            "task": str(instruction),
        }

        with self._lock:
            # Preprocess: tokenize prompt + images, normalize state, etc.
            batch = self.pre(batch_in)

            # Convert leftover to a torch tensor on the policy device. Shape
            # expected by the model: (B, L, action_dim). We have B=1.
            prev_chunk_tensor = None
            if prev_chunk_left_over is not None and len(prev_chunk_left_over) > 0:
                pcl = np.asarray(prev_chunk_left_over, dtype=np.float32)
                if pcl.ndim != 2 or pcl.shape[1] != ACTION_DIM:
                    raise ValueError(
                        f"prev_chunk_left_over must be (L, {ACTION_DIM}), got {pcl.shape}"
                    )
                prev_chunk_tensor = (
                    torch.from_numpy(pcl).to(self.device).unsqueeze(0)
                )

            # Run the policy with RTC kwargs. inference_action_mode must be
            # set explicitly; "continuous" matches what the BimanualYAM
            # checkpoint was trained for (flow-matching expert).
            action_chunk = self.policy.predict_action_chunk(
                batch,
                num_steps=int(num_steps),
                inference_delay=int(inference_delay) if inference_delay else None,
                prev_chunk_left_over=prev_chunk_tensor,
                execution_horizon=int(execution_horizon),
                inference_action_mode="continuous",
            )
            # action_chunk: (B, T, action_dim) before postprocessing. We push
            # it through the postprocessor for unnormalization. The postproc
            # takes a PolicyAction (a tensor).
            actions_t = self.post(action_chunk)

        # Convert back to numpy. Strip the batch dim.
        if torch.is_tensor(actions_t):
            actions_np = (
                actions_t.detach().to(dtype=torch.float32, device="cpu").numpy()
            )
        else:
            actions_np = np.asarray(actions_t, dtype=np.float32)
        if actions_np.ndim == 3 and actions_np.shape[0] == 1:
            actions_np = actions_np[0]
        # Final shape sanity check: should be (chunk_size, 14).
        if actions_np.ndim != 2 or actions_np.shape[1] != ACTION_DIM:
            raise RuntimeError(
                f"unexpected action shape from policy: {actions_np.shape}"
            )

        meta = {
            "leftover_len_in": (
                0 if prev_chunk_left_over is None else int(len(prev_chunk_left_over))
            ),
            "execution_horizon": int(execution_horizon),
            "inference_delay": int(inference_delay),
            "num_steps": int(num_steps),
            "chunk_size": int(actions_np.shape[0]),
        }
        return actions_np, meta


def build_app(policy: RTCPolicy, default_exec_horizon: int) -> FastAPI:
    app = FastAPI(title="MolmoAct2-BimanualYAM RTC server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": REPO_ID,
                "norm_tag": NORM_TAG,
                "device": policy.device,
                "dtype": str(policy.dtype),
                "num_cameras": NUM_CAMERAS,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "chunk_size": CHUNK_SIZE,
                "rtc": {
                    "enabled": True,
                    "execution_horizon": policy.rtc_config.execution_horizon,
                    "max_guidance_weight": policy.rtc_config.max_guidance_weight,
                    "schedule": str(policy.rtc_config.prefix_attention_schedule),
                },
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return _error_response(400, f"failed to decode json_numpy body: {e}")

        try:
            top_cam = payload["top_cam"]
            left_cam = payload["left_cam"]
            right_cam = payload["right_cam"]
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as e:
            return _error_response(400, f"missing required field: {e}")

        num_steps = int(payload.get("num_steps", DEFAULT_NUM_STEPS))
        prev_left = payload.get("prev_chunk_left_over", None)
        if prev_left is not None and not isinstance(prev_left, np.ndarray):
            prev_left = np.asarray(prev_left, dtype=np.float32)
        inference_delay = int(payload.get("inference_delay", 0))
        execution_horizon = int(
            payload.get("execution_horizon", default_exec_horizon)
        )

        t0 = time.perf_counter()
        try:
            actions, meta = policy.predict(
                top_cam=top_cam,
                left_cam=left_cam,
                right_cam=right_cam,
                instruction=instruction,
                state=state,
                num_steps=num_steps,
                prev_chunk_left_over=prev_left,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        body = json_numpy.dumps({"actions": actions, "dt_ms": dt_ms, "rtc": meta})
        return Response(content=body, media_type="application/json")

    return app


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def warmup(policy: RTCPolicy) -> None:
    log.info("Warming up model with dummy frames...")
    dummy_img = np.zeros((180, 320, 3), dtype=np.uint8)
    dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
    t0 = time.perf_counter()
    try:
        # 1: warmup WITHOUT a leftover (the bootstrap chunk's code path).
        policy.predict(
            top_cam=dummy_img,
            left_cam=dummy_img,
            right_cam=dummy_img,
            instruction="warmup",
            state=dummy_state,
            num_steps=DEFAULT_NUM_STEPS,
            prev_chunk_left_over=None,
            inference_delay=0,
            execution_horizon=policy.rtc_config.execution_horizon,
        )
        # 2: warmup WITH a leftover (the steady-state RTC code path). Use a
        # representative leftover length = chunk_size - execution_horizon.
        leftover_len = max(1, CHUNK_SIZE - policy.rtc_config.execution_horizon)
        dummy_leftover = np.zeros((leftover_len, ACTION_DIM), dtype=np.float32)
        policy.predict(
            top_cam=dummy_img,
            left_cam=dummy_img,
            right_cam=dummy_img,
            instruction="warmup",
            state=dummy_state,
            num_steps=DEFAULT_NUM_STEPS,
            prev_chunk_left_over=dummy_leftover,
            inference_delay=2,
            execution_horizon=policy.rtc_config.execution_horizon,
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info("Warmup OK (%.1f s)", time.perf_counter() - t0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MolmoAct2-BimanualYAM RTC inference server"
    )
    p.add_argument("--host", default="0.0.0.0", help="bind address")
    p.add_argument(
        "--port", type=int, default=8203,
        help="bind port (default: 8203, different from :8202 main server)",
    )
    p.add_argument("--repo-id", default=REPO_ID, help=f"HF repo id (default: {REPO_ID})")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument("--no-warmup", action="store_true", help="skip warmup pass")
    p.add_argument(
        "--rtc-execution-horizon", type=int, default=10,
        help="RTC execution horizon (steps before client kicks off next /act)",
    )
    p.add_argument(
        "--rtc-max-guidance-weight", type=float, default=10.0,
        help="RTC max guidance weight (default from the paper)",
    )
    p.add_argument(
        "--rtc-attention-schedule", default="linear",
        choices=["linear"],
        help="RTC prefix attention schedule",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    policy = RTCPolicy(
        repo_id=args.repo_id,
        device=args.device,
        dtype=dtype,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_attention_schedule=args.rtc_attention_schedule,
    )
    if not args.no_warmup:
        warmup(policy)

    app = build_app(policy, default_exec_horizon=args.rtc_execution_horizon)

    import uvicorn

    log.info("RTC server listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
