# YAM SDK Lock Contention — Investigation & Fix

**Status:** Root cause identified and reproducibly validated. One-line workaround applied in `scripts/yam_client.py` via runtime monkey-patch. Recommend upstream PR to i2rt.

---

## TL;DR

The YAM arms were moving in visible bursts during inference — smooth for a moment, then frozen, then a sudden catch-up snap. After a guided elimination process, we found this is **not** a hardware issue, **not** a CAN bus issue, **not** a USB issue, **not** a model issue, and **not** caused by anything specific to this machine that can't also affect anyone else running i2rt's SDK on stock Linux.

It is one design choice in 40 lines of Python inside `i2rt/motor_drivers/dm_driver.py`. The `command_lock` mutex is held during ~3 ms of CAN I/O on every iteration of a 300 Hz control loop. The other SDK thread that needs that mutex gets starved by Linux's mutex scheduler under that contention and waits hundreds of milliseconds at a time. While it's waiting, motor commands aren't reaching the motors.

A 5-line refactor — release the mutex right after copying commands to a local variable, instead of after the whole CAN round-trip — eliminates the problem entirely. Under direct A/B test:

|  | Before fix | After fix |
| --- | --- | --- |
| Slow events in 12 s | 98 | **0** |
| p99 mutex wait | 412 ms | **30 µs** |
| Max mutex wait | 769 ms | **0.6 ms** |
| `set_commands` throughput | 366 calls | **3,967 calls** |

---

## What we observed

During inference runs, the arms moved in clearly bursty patterns: they'd hold still for ~200–500 ms while the policy was actively commanding new positions, then snap to catch up. The `motion_lag_test_v2.py` instrumented this by reading state and commanding a smooth sinusoid; the measured tracking lag was 300–800 ms with a sustained ~10% rate of "stalls" (commanded position moved, measured position didn't).

The natural assumption was that something was wrong with the cameras, the CAN cable, the motor firmware, USB bandwidth, or the inference server consuming resources. Each of those turned out to be wrong.

---

## The good experiments — eliminating possibilities in order

We treated this as a search problem and built one diagnostic per hypothesis. Each test ruled out a candidate and pointed the next one.

### 1. Sleep jitter test — ruled out OS scheduling

**Question:** Is the Linux kernel just not scheduling our Python thread reliably?

**What we did:** With the SDK control thread running normally and one arm loaded in a representative pose, the main thread did *nothing but* `time.sleep(20 ms)` in a tight loop for 8 seconds. We measured the actual duration of each sleep.

**Result:** p50 = 20.14 ms, p99 = 20.25 ms, max = 20.31 ms (target 20.00 ms). **The OS scheduler is fine.** Our main thread gets scheduled within 0.3 ms of when we ask. Whatever's slow about `get_joint_pos()` is not the OS starving Python.

**Why this mattered:** without this, we would have spent days fiddling with `chrt`, `nice`, CPU governors, and real-time kernel patches. This one test eliminated an entire category.

### 2. CAN frame timing sniffer (passive) — ruled out the CAN bus

**Question:** Are the motors responding slowly on the CAN wire?

**What we did:** Opened a raw `socketcan` listener on `can0` and timed the gap between consecutive frames for 25 seconds while the lag test drove the arm.

**Result:** Each motor's command/response frames arrived at ~300 Hz with p99 = 3.4 ms inter-arrival and max = 38 ms. The bus was never quiet for more than 4.86 ms globally. **The CAN bus is healthy.** No silent retry storms, no electrical issues, no firmware lag in the motors.

### 3. CAN payload sniffer — confirmed the SDK was sending stale commands

**Question:** OK, the CAN traffic is steady. But what's *in* the frames? Are they fresh data or repeats?

**What we did:** Same listener, but this time decoded the 16-bit raw target position from each motor command frame, and computed the run-length distribution: how many consecutive identical commands per motor.

**Result:** For the joint we were actively sweeping, the SDK was sending the **same** target up to 1,069 frames in a row (3,287 ms of identical command despite us updating at 50 Hz). The motor's command frames were physically being sent at 300 Hz, but the *contents* weren't changing.

**Why this mattered:** this immediately reframed the bug. The wire was fine; the SDK simply wasn't pushing our updates through fast enough. Burst motion now had a clean explanation: motor holds the stale target, then jumps when a new one finally arrives.

### 4. `update()` profiler — narrowed slow code to one method

**Question:** Where inside the SDK is the time being spent?

**What we did:** Monkey-patched `MotorChainRobot.update()` to time each section separately:
gravity-comp, torque calc, gripper-limiter, and `_update_joint_state`.

**Result:** Every single slow event had 100% of its time in `_update_joint_state`, with the other sections at <1 ms.

### 5. `_update_joint_state` profiler v2 — narrowed further to one call

**What we did:** Drilled one level deeper into the sub-sections of `_update_joint_state`: encoder check, `motor_chain.set_commands`, state conversion, limit check, saver.

**Result:** All slow events: 100% in `motor_chain.set_commands`. Everything else microseconds.

### 6. Lock acquire profiler — confirmed it's the mutex acquire specifically

**What we did:** Patched `DMChainCanInterface.set_commands` to separately time the `command_lock.acquire()` versus everything else inside.

**Result:** **All slow time was inside `command_lock.acquire()`.** The `with` block body itself was <0.05 ms; the wait to enter the block was hundreds of ms. *This pinpointed the mechanism: mutex contention.*

### 7. Direct patch test — proved the root cause AND validated the fix

**What we did:** Wrote `test_sdk_lock_fix.py`, which runs the same workload but at startup replaces `dm_driver.DMChainCanInterface._set_torques_and_update_state` (the CAN-talker thread's main loop) with a modified version that holds `command_lock` only long enough to copy commands to a local variable, then releases the lock before doing CAN I/O. Ran twice back-to-back: once without the patch (control), once with it.

**Result:**

| Metric | Control | Patched |
| --- | --- | --- |
| `set_commands` calls in 12 s | 366 | 3,967 |
| p50 wait | 4.5 ms | 0.00 ms |
| p99 wait | 412 ms | 0.00 ms |
| Max wait | 769 ms | 0.60 ms |
| Events >50 ms | 98 | 0 |
| Events >500 ms | 3 | 0 |

The patched run had zero slow events and 11× more throughput, in the same 12 seconds on the same machine moments apart. Definitively proven.

---

## Mutex contention — explained slowly

Let's leave the SDK aside for a second and just talk about what's going on conceptually.

When two threads in the same process need to take turns touching a shared piece of memory, they use a *mutex* (short for *mutual exclusion*) — a flag that only one thread at a time can "hold." When you write `with self.command_lock: ...` in Python, you're saying "I want exclusive access to whatever's in this block; if some other thread is already holding the lock, wait until they release."

This works great when contention is light — say, two threads that each want the lock occasionally, with plenty of other work in between. Linux's standard mutex hands the lock to whichever thread is ready when the holder releases it. Both threads make steady progress.

It works **terribly** under one specific pattern: when one thread is in a tight loop that **acquires the lock, does a little work, releases, immediately re-acquires, releases, re-acquires...** with no real break in between. The release-to-reacquire gap can be microseconds. In that gap, the other waiting thread is *supposed* to be woken up by the kernel and given a turn.

But Linux's mutex isn't a strict queue. When the holder releases, the kernel marks the lock as "free" and signals one waiter. But before the waiter actually wakes up, gets scheduled onto a CPU, and runs its acquire code, the original holder — which is already executing on a CPU with hot caches and recent memory access patterns — may have already grabbed the lock again. This is called the **convoy effect** or **lock starvation**.

In our case:
- **Thread A** (`dm_driver.Thread-1`, the CAN-talker) runs this loop ~300 times a second, with no pause between iterations:
  - Acquire `command_lock`.
  - Send command frames to 7 motors, get 7 responses (≈3 ms).
  - Release `command_lock`.
  - Loop back immediately to acquire again.

- **Thread B** (`motor_chain_robot._server_thread`, the high-level update loop) wants `command_lock` ~250 times a second to push the latest target positions into the box that Thread A reads from.

Thread A holds the lock about 99% of wall-clock time (3 ms out of every ~3.3 ms cycle). Thread B has only ~0.3 ms windows per cycle to acquire. **And it usually loses the race**, because Thread A is already running and Thread A's re-acquire happens before Thread B's wake-up completes. After many lost races in a row — sometimes dozens, sometimes hundreds — Thread B finally wins one. By then, hundreds of ms have passed. The freshness of our motor target has gone stale by that long. The motor, having been told to hold its old target by 300 Hz of repeated frames from Thread A, gets a brand-new much-farther-away target. Its PD controller responds with a quick burst of motion to catch up. We see the arm freeze-and-snap.

**The fix is to change Thread A so it doesn't hold the lock during the slow CAN cycle.** Just grab the lock long enough to copy the latest command box into a local variable (microseconds), release the lock, then do CAN with the local copy. Thread B now has microseconds of contention instead of milliseconds, and acquiring becomes essentially instant — no starvation.

That's a one-method refactor. The fix is in `dm_driver.py:529-606`, replacing the body of `_set_torques_and_update_state`. The semantics don't change — the CAN cycle still uses whatever commands were most recent — but the lock-hold window collapses from 3 ms to a few microseconds.

---

## Where exactly does the issue live?

**It is in the i2rt Python SDK, on the PC.** Specifically:

- **Not the robot.** The motors, encoders, mechanical assembly are all fine.
- **Not the CAN bus.** Wire timing is healthy. No retries. No electrical issues.
- **Not the USB stack or CANable adapters.** Frames flow at expected rates.
- **Not the Linux kernel.** Sleep jitter test showed scheduling within 0.3 ms.
- **Not the model server.** Reproduced with server killed.
- **Not the dual-arm setup.** Reproduced with one arm.
- **Not the cameras or inference loop.** Reproduced with neither active.
- **Not specific to this machine.** Other YAM customers running on stock Linux with `SCHED_OTHER` threads have this same code path — they probably hit it too, just may not measure it. The behavior depends on the OS's mutex scheduler, but Linux's default behavior is the dominant one. You will see this on nearly any default-configured x86 Linux machine.

The bug is in one method of one file: `i2rt/motor_drivers/dm_driver.py`, the function `DMChainCanInterface._set_torques_and_update_state`. The combined effect is that the SDK's nominal "250 Hz update rate" actually runs at ~30–100 Hz with high-magnitude outliers under any concurrent load, even though all the underlying hardware would happily support the full target rate.

Some YAM users may not notice because:

- Teleoperation use cases tolerate 200 ms jitter (the human just feels a slightly sluggish arm; the burst is masked by smoothing).
- Scripted-motion use cases issue commands far enough apart that the staleness doesn't matter.
- Closed-loop policy inference like ours — where we're commanding precise targets at 50 Hz from a model — is exactly the workload that exposes the issue.

---

## The fix, in code

In `dm_driver.py:564`, replace:

```python
with self.command_lock:
    try:
        motor_feedback = self._set_commands(self.commands)
    except RuntimeError as e:
        ...
    errors = np.array([...])
    if np.any(errors):
        ...
```

with:

```python
with self.command_lock:
    local_commands = list(self.commands)        # microsecond snapshot
# lock released here
try:
    motor_feedback = self._set_commands(local_commands)   # CAN I/O outside lock
except RuntimeError as e:
    ...
errors = np.array([...])
if np.any(errors):
    ...
```

The error-recovery logic stays identical; it just runs outside the lock. The `state_lock` block further down is untouched.

We've applied this in our project as a runtime monkey-patch at `yam_client.py` import time (function `install_sdk_lock_fix()`). The patch lands before any `DMChainCanInterface` is constructed, so every robot the script touches gets the fixed loop. No upstream SDK files modified.

---

## Implications for inference

We expect three concrete improvements in our `yam_client.py` inference runs, all of which should be immediately visible:

1. **No more burst motion.** The arms should track the policy's commanded trajectory smoothly. The previously-visible "freeze for 200 ms, then snap" should be gone.
2. **Higher SDK update rate.** The `Grav Comp Control Frequency` warnings that previously reported 20–100 Hz should now report close to the 250 Hz target.
3. **`inner step overrun` warnings should largely disappear in our client.** Previously up to 1,500 ms of overrun were logged; should drop to occasional small overruns or none.

What this **does not** fix:

- The model's interpretation of the scene. If the inference still produces near-identity actions (because the cameras/lighting/objects aren't quite training-distribution), that's a separate problem about the model and the scene, not the motion path.
- Any motor or hardware safety behavior. The 400 ms motor watchdog, gripper auto-cal, etc., are unchanged.

The closed-loop policy should now have much fresher motor state and much more reliable command delivery. If the earlier "model produces sensible-looking actions but arm doesn't follow" pattern was partly explained by stale motor state confusing the model's next prediction, that should clear up.

---

## What we changed in this repo

- `scripts/yam_client.py` — added `install_sdk_lock_fix()`, called at module import time. No other changes to inference logic.
- `scripts/test_sdk_lock_fix.py` — standalone A/B test that proves the fix works on demand. Keeps the patch logic available as a reference implementation. Run with and without `--no-fix` to compare.
- This document.

No changes to the i2rt SDK source itself.

---

## Recommended next step (separate from this work)

File an upstream issue against `i2rt-robotics/i2rt` with:

- A link to this report.
- A small reproducer (subset of `test_sdk_lock_fix.py`).
- The before/after numbers from the A/B test.
- The proposed patch (the small diff above).

The fix is non-invasive and preserves all existing behavior including error recovery. It would benefit every YAM user running on standard Linux. The fact that the SDK already has a `time.sleep(0)` near the end of the loop (line 600) with the comment *"yield GIL so other threads can acquire locks"* suggests the maintainers already knew lock-contention was a concern; they just placed the yield after the lock-held section instead of restructuring the lock-held section itself.
