# sync_stopandgo_working — 2026-05-21

Snapshot of `scripts/yam_client.py` and `scripts/run_client.sh` at the point where:

- The bimanual cube-pick task succeeds end-to-end.
- Motion is **synchronous**: each `/act` POST blocks the inner loop, so the arm holds position for ~320 ms between every chunk and then executes 10 actions (~333 ms), then holds, then executes — the "stop-and-go" pattern visible to the eye.
- D405 cameras open **before** arms init (avoids the gs_usb / xHCI watchdog timeouts).
- Per-step clip at `--max-step-rad 0.15` fires ~7% of the time on average.
- Research journal prompt fires at end of every run.

Measured behavior from a clean run on 2026-05-21:
- RTT mean: ~320 ms (range 290–360 ms)
- Replan rate: ~1.5 Hz (sync stride=10)
- Cycle: ~670 ms (320 ms inference + 333 ms exec + ~17 ms overhead)
- Motion duty cycle: ~51% moving, ~49% holding

Git commit at snapshot time: see `git_commit.txt`.

## When to consult this snapshot

Restore from here if any of these regresses on a later iteration:
- Task no longer succeeds.
- Motor "loss communication" errors return.
- Clip rate jumps to >20% of steps.
- New "first action of chunk" jumps appear that didn't exist before.

To restore:
```bash
cp archive/2026-05-21_sync_stopandgo_working/yam_client.py scripts/yam_client.py
cp archive/2026-05-21_sync_stopandgo_working/run_client.sh scripts/run_client.sh
```
