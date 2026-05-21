"""Three correctness tests for the RTC code path used by host_server_rtc.py.

These are unit-level checks meant to catch the kinds of issues the user
called out:

  1. Vanilla reduction: with prev_chunk_left_over=None, RTC's early
     short-circuit should return the unmodified denoise step. The output
     should match a fresh chunk generated through the vanilla path.

  2. Identity inpaint: pass the previous chunk as the leftover with
     inference_delay=chunk_size and execution_horizon=chunk_size. The
     prefix weights are all-ones across the whole chunk; the new chunk
     should be very close to the leftover (modulo finite num_steps).

  3. Prefix weight shape: vary inference_delay across {0, 5, 10, 15}
     with execution_horizon=10. Measure |chunk[i] - leftover[i]| for
     every position. The expected pattern, per get_prefix_weights:
       i in [0..min(idelay, exhor)):   ~0   (hard inpaint)
       i in [min(idelay, exhor)..exhor): linearly increasing (fade)
       i in [exhor..N):                 free (potentially large)

Run with:

    VIRTUAL_ENV=/home/andon/yam-tests/molmoact2-setup/.venv-rtc \\
    /home/andon/yam-tests/molmoact2-setup/.venv-rtc/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/experimental/rtc/test_rtc_correctness.py

Reuses the in-process MolmoAct2Policy (no HTTP server needed). Each
generate call is ~300-400 ms on a 5090.
"""
from __future__ import annotations

import sys
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

# Same lerobot bring-up as host_server_rtc.py.
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCAttentionSchedule, RTCConfig

REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
STATE_DIM = 14
ACTION_DIM = 14
CHUNK_SIZE = 30
CAMERA_KEYS = [
    "observation.images.top",
    "observation.images.left",
    "observation.images.right",
]
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _build_policy(rtc_enabled: bool, max_guidance_weight: float = 10.0):
    """Construct the policy the same way host_server_rtc.py does."""
    rtc_config = None
    if rtc_enabled:
        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
            max_guidance_weight=max_guidance_weight,
            execution_horizon=10,
        )
    config = MolmoAct2Config(
        checkpoint_path=REPO_ID,
        norm_tag=NORM_TAG,
        image_keys=CAMERA_KEYS,
        setup_type="bimanual yam robotic arms in molmoact2",
        control_mode="absolute joint pose",
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        rtc_config=rtc_config,
        inference_action_mode="continuous",
        input_features={STATE_KEY: PolicyFeature(type=FeatureType.STATE,  shape=(STATE_DIM,))},
        output_features={ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )
    print(f"[setup] Loading policy (rtc_enabled={rtc_enabled}, max_guidance={max_guidance_weight})...")
    policy = MolmoAct2Policy(config).to("cuda:0").eval()
    for p in policy.parameters():
        if p.is_floating_point():
            p.data = p.data.to(torch.bfloat16)
    pre, post = make_molmoact2_pre_post_processors(config, dataset_stats=None)
    print(f"[setup] Policy ready.")
    return policy, config, pre, post


def _build_batch(pre, instruction: str = "test"):
    """Build a single-batch input identical to what the HTTP server would
    construct from a real request."""
    rng = np.random.default_rng(0)
    imgs = {
        k: Image.fromarray(rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8))
        for k in CAMERA_KEYS
    }
    state = torch.zeros(STATE_DIM, dtype=torch.float32, device="cuda:0")
    batch_in = {**imgs, STATE_KEY: state, "task": instruction}
    return pre(batch_in)


def _generate(policy, batch, *, leftover=None, inference_delay=0,
              execution_horizon=10, num_steps=10, seed=12345):
    """Single chunk generation. seed makes the noise initialization deterministic."""
    gen = torch.Generator(device="cuda:0").manual_seed(int(seed))
    prev = None
    if leftover is not None:
        prev = torch.from_numpy(np.asarray(leftover, dtype=np.float32)).cuda().unsqueeze(0)
    actions = policy.predict_action_chunk(
        batch,
        num_steps=num_steps,
        inference_delay=inference_delay,
        prev_chunk_left_over=prev,
        execution_horizon=execution_horizon,
        inference_action_mode="continuous",
        generator=gen,
    )
    if torch.is_tensor(actions):
        actions = actions.detach().to(dtype=torch.float32, device="cpu").numpy()
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    return actions


def _arm_idx():
    """The 12 non-gripper joint indices used for max-deviation checks."""
    return np.r_[0:6, 7:13]


def _max_diff_arm(a, b):
    """Max |a - b| over the 12 arm joints, max-pooled across timesteps."""
    arm = _arm_idx()
    return float(np.max(np.abs(a[..., arm] - b[..., arm])))


# =====================================================================
# Test 1: vanilla reduction
# =====================================================================
def test_vanilla_reduction(policy_rtc, batch):
    """With prev_chunk_left_over=None, the RTC-enabled policy should
    short-circuit and return exactly the same chunk as the vanilla path
    would (the early return at the top of RTCProcessor.denoise_step).

    We can't test this directly without constructing a non-RTC policy
    too, but we CAN test that two calls with leftover=None and the same
    seed produce the same chunk (proving determinism + that the RTC
    branch isn't introducing nondeterminism).
    """
    print("\n[test 1] vanilla reduction (no leftover)")
    a1 = _generate(policy_rtc, batch, leftover=None, seed=1)
    a2 = _generate(policy_rtc, batch, leftover=None, seed=1)
    diff = _max_diff_arm(a1, a2)
    print(f"  determinism check: max-arm |a1 - a2| = {diff:.6f} rad")
    if diff > 1e-3:
        print(f"  [WARN] expected near-zero; the RTC path may be adding noise")
    else:
        print(f"  [OK] outputs are deterministic with leftover=None")


# =====================================================================
# Test 2: identity inpaint
# =====================================================================
def test_identity_inpaint(policy_rtc, batch):
    """Pass a known chunk as the leftover with inference_delay=chunk_size
    and execution_horizon=chunk_size. get_prefix_weights returns all-ones,
    so the entire trajectory should be inpainted to match the leftover.
    The output should be very close to the leftover (modulo finite
    flow-matching num_steps).
    """
    print("\n[test 2] identity inpaint (full-chunk anchoring)")
    # Generate a "previous chunk" we'll use as the target.
    prev_chunk = _generate(policy_rtc, batch, leftover=None, seed=42)
    # Now inpaint EVERY position of a new chunk to match prev_chunk.
    new_chunk = _generate(
        policy_rtc, batch, leftover=prev_chunk,
        inference_delay=CHUNK_SIZE,
        execution_horizon=CHUNK_SIZE,
        seed=99,  # different seed -> different initial noise
    )
    diff = _max_diff_arm(new_chunk, prev_chunk)
    print(f"  inputs: leftover.shape={prev_chunk.shape}, idelay=chunk_size, exhor=chunk_size")
    print(f"  max-arm |new_chunk - prev_chunk| = {diff:.4f} rad")
    if diff < 0.1:
        print(f"  [OK] full inpaint successfully anchored new_chunk to leftover")
    elif diff < 0.3:
        print(f"  [PARTIAL] inpaint is pulling but not all the way -- check max_guidance_weight")
    else:
        print(f"  [FAIL] inpaint isn't anchoring; the time indexing or sign might be wrong")


# =====================================================================
# Test 3: prefix weight shape
# =====================================================================
def test_prefix_weight_shape(policy_rtc, batch):
    """With execution_horizon fixed at 10, sweep inference_delay across
    {0, 5, 10, 15}. Per get_prefix_weights, the weight profile is:
      positions [0, min(idelay, 10)):    weight = 1   (hard inpaint)
      positions [min(idelay, 10), 10):   weight = 1 -> 0 linearly
      positions [10, 30):                weight = 0

    So if RTC is working, |new[i] - leftover[i]| should:
      - stay tiny for i < min(idelay, 10)
      - grow linearly between min(idelay, 10) and 10
      - be unconstrained after position 10
    """
    print("\n[test 3] prefix weight shape (sweep inference_delay)")
    prev_chunk = _generate(policy_rtc, batch, leftover=None, seed=7)
    arm = _arm_idx()

    EXHOR = 10
    for idelay in [0, 5, 10, 15]:
        new_chunk = _generate(
            policy_rtc, batch, leftover=prev_chunk,
            inference_delay=idelay, execution_horizon=EXHOR, seed=8,
        )
        per_pos = np.max(np.abs(new_chunk[:, arm] - prev_chunk[:, arm]), axis=1)
        # Print per-position diff at a few representative indices.
        snap = [(i, per_pos[i]) for i in [0, 4, 9, 10, 19, 29]]
        snap_s = "  ".join(f"i={i}:{v:.3f}" for i, v in snap)
        print(f"  idelay={idelay:2d} exhor={EXHOR}: {snap_s}")

    print("  expected (LINEAR schedule, exhor=10):")
    print("    idelay= 0  -> linear fade i=[0..9], free i>=10")
    print("    idelay= 5  -> ones i=[0..4], fade i=[5..9], free i>=10")
    print("    idelay=10  -> ones i=[0..9], free i>=10")
    print("    idelay=15  -> clamped to idelay=10 (see start = min(idelay, exhor))")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        return 1
    policy_rtc, config, pre, post = _build_policy(rtc_enabled=True, max_guidance_weight=10.0)
    batch = _build_batch(pre)

    test_vanilla_reduction(policy_rtc, batch)
    test_identity_inpaint(policy_rtc, batch)
    test_prefix_weight_shape(policy_rtc, batch)

    print("\nDone. Inspect the outputs above; the expected patterns are documented in each test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
