# Phase RTC-1 — Integration path investigation

Goal: enable Real-Time Chunking (RTC) inference (Black et al. 2025, arXiv:2506.07339)
for the MolmoAct2-BimanualYAM checkpoint, so we can A/B-benchmark it against the
existing async-time-aligned client (`scripts/yam_client.py` mode
`async-time-aligned`).

## What RTC is and what it needs

RTC is an inference-time technique for flow-matching chunked policies. At each
chunk boundary the **new chunk is generated with prefix attention to the
remaining tail of the old chunk** (the "leftover"). The first ≈inference_delay
actions of the new chunk are inpainted to continue smoothly from the leftover,
so chunk boundaries don't produce backward jumps even when inference latency
is comparable to one execution horizon. Key params:

- `prev_chunk_left_over` — the unused tail of the previous chunk, shape
  `(leftover_T, action_dim)`. The first `inference_delay` actions of the new
  chunk are constrained (via prefix attention + soft guidance) to flow from
  this tail.
- `inference_delay` — how many timesteps of the new chunk happen during the
  inference call. In practice ≈ ceil(round-trip-time / dt). The client knows
  this best (it actually executes during inference).
- `execution_horizon` — how many steps from each chunk the client actually
  executes before kicking off the next inference. The leftover is everything
  after this index.

The implementation lives in Ai2's lerobot fork at
`src/lerobot/policies/rtc/` (`configuration_rtc.py` defines `RTCConfig`,
`modeling_rtc.py` runs prefix-attended flow sampling, `action_queue.py` and
`action_interpolator.py` provide the client-side leftover queue).

The MolmoAct2 policy in the same fork already supports it natively:
`src/lerobot/policies/molmoact2/modeling_molmoact2.py` exposes
`predict_action_chunk(batch, *, num_steps, inference_delay,
prev_chunk_left_over, execution_horizon, generator)` and switches between
plain flow-matching and `_generate_actions_from_inputs_with_rtc` when
`config.rtc_config.enabled`.

## What's NOT in the lerobot fork yet

- No YAM/bi_yam robot adapter under `src/lerobot/robots/` (only koch, openarm,
  so100, etc).
- No example/config that points lerobot at the BimanualYAM checkpoint.
- The existing async-inference framework (`policy_server.py` + `robot_client.py`)
  assumes a `lerobot.Robot` adapter — we'd have to write one to use it.

## The three integration options

### (a) Full lerobot stack — write a YAM robot adapter + use lerobot policy_server.py + robot_client.py

Pros:
- Most "blessed" path; gets us future updates from upstream for free.
- The robot_client.py already implements the RTC client logic (action queue,
  leftover tracking).
- Co-located with the async-inference framework so we'd also pick up their
  time-aligned tricks.

Cons:
- Need to implement a `BimanualYAMRobot(Robot)` adapter that wraps i2rt: open
  cameras-before-arms ordering, install the SDK lock-fix monkey patch,
  expose YAM as a lerobot Teleop/Robot subclass with the right observation
  features (3 cameras + 14-D state) and action space.
- The lerobot abstractions (`Teleop`, `Robot`, `RobotConfig`, dataset features,
  the `EnvTransition` shape) are extensive and our existing client has hard-won
  safety patterns (return-on-exit ramp, max-step-rad clip, journal prompt)
  that we'd have to re-port.
- Risk: lerobot's robot interface forces a particular control cadence and
  failure-mode taxonomy that may not match our i2rt semantics.
- Highest implementation effort. ≥1 day to do safely.

### (b) Server-side lerobot policy, client-side keeps our shape — load lerobot MolmoAct2Policy in a new host_server_rtc.py, expose RTC params over our existing JSON wire protocol, and reuse the yam_client.py loop scaffolding for the client

Pros:
- Re-uses the existing safety patterns wholesale — SDK lock fix, camera-before-arms,
  startup-pose ramp, journal prompt, per-tick clip.
- Wire-format extension is small (add three optional fields:
  `prev_chunk_left_over`, `inference_delay`, `execution_horizon`).
- Doesn't touch the running :8202 server; the new server lives at :8203 and
  the user A/B-tests by switching `--server-url`.
- The hard part (loading the model with `rtc_config` set, building the
  preprocessing batch, calling `predict_action_chunk` with RTC kwargs,
  post-processing actions back to YAM joint units) is contained in one file.
- Total effort: ≈2–4 hours.

Cons:
- Won't pick up upstream lerobot improvements automatically.
- Two slightly-different inference paths (HF transformers in :8202, lerobot in
  :8203). Easy to keep them straight as long as you don't fuse them.

### (c) Client-side pure-Python re-implementation of RTC

Cons:
- Needs to monkey-patch into the model's flow-matching sampling loop because
  RTC implements **prefix attention** during the diffusion/flow steps — it
  is not a wrapper over `predict_action`; it lives *inside* the sampler.
- The existing HF `predict_action()` method doesn't expose flow-matching
  internals or accept prefix kwargs.
- Infeasible without re-implementing or monkey-patching the model.

## Choice: (b)

**Rationale**: shortest path to a testable RTC stack that:
- Doesn't disturb the working synchronous and async-time-aligned setups.
- Keeps the safety patterns the user has already validated.
- Lets us swap A/B by changing a URL.

The lerobot stack (option a) is the right long-term home but is out of scope
for a 60-minute investigation. The pieces we'd need are documented at the end
of this file as follow-up work.

## Implementation sketch (option b)

### Server (`host_server_rtc.py`, port 8203)

1. Install the lerobot fork into the existing server venv:

   ```bash
   VIRTUAL_ENV=/home/andon/yam-tests/molmoact2-setup/.venv \
     uv pip install "lerobot @ git+https://github.com/allenai/lerobot.git@molmoact2-policy"
   ```

   The existing `molmoact2-setup/.venv` already has compatible torch (2.8+cu128)
   and transformers (4.57.x). Adding lerobot pulls only the lerobot Python code;
   torch is already pinned.

2. Build `MolmoAct2Config` programmatically (no train-yaml needed):

   ```python
   from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
   from lerobot.policies.rtc.configuration_rtc import RTCConfig, RTCAttentionSchedule

   rtc_config = RTCConfig(
       enabled=True,
       prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
       max_guidance_weight=10.0,
       execution_horizon=10,
   )
   config = MolmoAct2Config(
       checkpoint_path="allenai/MolmoAct2-BimanualYAM",
       norm_tag="yam_dual_molmoact2",
       image_keys=["observation.images.top",
                   "observation.images.left",
                   "observation.images.right"],
       rtc_config=rtc_config,
       chunk_size=30,
       n_action_steps=30,
   )
   ```

3. Build the policy + processors:

   ```python
   from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
   from lerobot.policies.molmoact2.processor_molmoact2 import (
       make_molmoact2_pre_post_processors,
   )
   policy = MolmoAct2Policy(config).to("cuda").eval()
   pre, post = make_molmoact2_pre_post_processors(config, dataset_stats=None)
   ```

   `dataset_stats=None` falls back to the `norm_stats.json` packaged in the HF
   snapshot (resolved via `norm_tag`).

4. Per `/act` request: build an `EnvTransition` dict with keys
   `observation.images.{top,left,right}`, `observation.state`,
   `complementary.task`. Run through `pre`, call
   `policy.predict_action_chunk(batch, num_steps=N, inference_delay=k,
   prev_chunk_left_over=tail, execution_horizon=10)`, run through `post`,
   return `(N, 14)` ndarray.

### Client (`yam_client_rtc.py`)

- Inherit the entire safety harness from `yam_client.py` (lock fix, camera-before-arms,
  return-on-exit ramp, clip telemetry, journal).
- Maintain an action queue: after each prediction, execute `exec_horizon` actions
  and save the remainder as `prev_chunk_left_over` for the next request.
- Compute `inference_delay = ceil(estimated_rtt / dt)` from a rolling EMA of
  the last few RTTs.
- Default `--server-url http://127.0.0.1:8203/act`.
- The CLI surface is otherwise identical to the existing client.

## Blockers / open questions

- **`image_keys` naming**: the lerobot processor reads from `observation.images.<name>`.
  The exact `<name>` strings the BimanualYAM checkpoint was trained with may
  not be `top/left/right` — they could be e.g. `cam_high`, `cam_left_wrist`,
  `cam_right_wrist`. If a wrong name is used the processor will either error
  or silently feed black frames. **Action**: at server startup, log the keys
  it actually expects (parsing `config.input_features`) and fail loudly if
  ours don't match. If they don't match, this is a 1-line config fix.
- **`norm_tag`**: confirmed `yam_dual_molmoact2` in the HF checkpoint's
  `norm_stats.json`. The lerobot config field is `norm_tag`. Should JustWork™.
- **`inference_delay` units**: lerobot expects an integer count of timesteps,
  not seconds. The client computes it from EMA(RTT)/dt rounded up.
- **`dataset_stats` requirement**: `make_molmoact2_pre_post_processors`
  *may* require a non-None `dataset_stats` argument. If it does, we'd load
  `norm_stats.json` ourselves and pass a hand-built dict. Plan to discover
  this at first run and handle in v2 of the server.

## Follow-up work (out of scope for this session)

To pursue option (a) later:
1. Write `src/lerobot/robots/bi_yam/` adapter: `BimanualYAMRobot(Robot)`,
   `BimanualYAMRobotConfig`, ports of camera-before-arms ordering and SDK
   lock fix, observation/action feature definitions.
2. Wire it into lerobot's `policy_server.py` + `robot_client.py`.
3. Compare against this option-b implementation on the same task — the
   inference internals are identical, so any quality delta is purely
   client-side (different exec-horizon logic, different camera-capture
   timing, different smoothing).
