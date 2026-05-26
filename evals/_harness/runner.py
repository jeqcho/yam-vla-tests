"""High-level eval-session orchestrator.

`start_session(policy, eval_def, args, results_base_dir)` owns the
per-eval orchestration (task picker -> prompt picker -> N attempts per
prompt -> CSV row + journal entry) but delegates the per-attempt
control loop to the legacy `yam_repl.run_one_attempt` for now.

Why delegate: the legacy run_one_attempt is ~500 lines of cameras +
arms + safety_clip + async_fetcher + Rerun + boundary diagnostics.
Reimplementing it here is high-risk; instead we wrap it with a small
`Policy -> post_actions` adapter and let the proven code run.

The adapter is the cleanest possible compatibility layer:

    def post_actions_adapter(server_url, top, left, right, state, instr,
                              num_steps, timeout_s):
        obs = build_observation(top, left, right, state, instr)
        pred = policy.predict(obs, timeout_s=timeout_s, num_steps=num_steps)
        return pred.actions, pred.rtt_ms

The legacy `install_backend()` monkey-patch is GONE -- this adapter is
the only point of contact between new Policy code and the legacy loop.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from yam_vla.core.policy import Policy
from yam_vla.core.runner import build_observation

from evals._harness.tasks import EvalDefinition
from evals._harness.results import ResultsWriter

log = logging.getLogger("yam_vla.evals.runner")


def make_post_actions_adapter(policy: Policy, *, default_num_steps: int = 10):
    """Adapt Policy.predict() to the legacy post_actions(...) signature.

    Returns a callable with the exact shape the legacy yam_client /
    yam_repl modules expect, so the legacy run_one_attempt works
    unchanged.
    """
    def post_actions(server_url, top, left, right, state, instruction,
                     num_steps, timeout_s):
        del server_url  # the Policy knows its own transport
        obs = build_observation(top, left, right, state, instruction)
        pred = policy.predict(
            obs,
            timeout_s=float(timeout_s),
            num_steps=int(num_steps or default_num_steps),
        )
        return pred.actions, pred.rtt_ms
    return post_actions


def start_session(
    policy: Policy,
    eval_def: EvalDefinition,
    args: argparse.Namespace,
    *,
    results_base_dir: str | Path,
) -> None:
    """Orchestrate one eval session against `policy` on `eval_def.tasks`.

    Concretely: install the policy->post_actions adapter on the legacy
    modules, then delegate the per-attempt loop to the legacy main()
    pattern. The legacy main() is the proven combination of:
      - camera bring-up
      - arm init + grip auto-cal
      - per-task / per-attempt operator prompts
      - per-tick safety clip
      - async chunk fetcher
      - Rerun .rrd recording (if --rerun)
      - research-journal append on attempt finish
      - results CSV row

    Args:
        policy: built via PolicyConfig.from_path(...).build()
        eval_def: parsed evals/<name>/tasks.yaml
        args: argparse Namespace from scripts/run_eval.py
        results_base_dir: e.g. evals/<name>/results/
    """
    adapter = make_post_actions_adapter(
        policy,
        default_num_steps=getattr(args, "num_steps", 10),
    )

    # Lazy-import the legacy modules through the shim so the new-code
    # surface stays clean. The shim handles sys.path injection.
    from yam_vla.core import legacy  # noqa: F401 (side-effect: sys.path)
    import yam_client as _yc  # type: ignore[import-not-found]
    import yam_repl as _yr    # type: ignore[import-not-found]

    _yc.post_actions = adapter
    _yr.post_actions = adapter
    log.info("[session] post_actions routed through Policy %r", policy.name)

    results = ResultsWriter(
        base_dir=results_base_dir,
        policy=policy.name,
        eval_name=eval_def.name,
    )
    log.info("[session] CSV: %s", results.path)

    # The legacy main() reads `args.eval_tasks` + `args.results_csv`.
    args.policy_name = policy.name
    args.eval_name = eval_def.name
    args.eval_tasks = list(eval_def.tasks)
    args.results_csv = str(results.path)

    # TODO(migration): once core/hardware.py + core/journal.py exist,
    # this becomes a pure-Python orchestrator and we delete the legacy
    # main() call.
    _yc.main()


__all__ = ["start_session", "make_post_actions_adapter"]
