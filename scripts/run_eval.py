#!/usr/bin/env python3
"""Top-level eval dispatcher.

Usage:
    ./scripts/run_eval.py --policy <name> --eval <name> [knobs...]

Concrete examples:
    # MolmoAct2 on the IKEA 10
    ./scripts/run_eval.py --policy molmoact2 --eval ikea_10

    # Pi-0.5 on the Andon 10, 1 attempt per task, dry-run (no arm motion)
    ./scripts/run_eval.py --policy pi05 --eval andon_10 \\
        --attempts 1 --dry-run

    # GR00T-N1.7 with Rerun streaming on
    ./scripts/run_eval.py --policy gr00t-n17 --eval ikea_10 --rerun

Same hardware-side defaults come from yam_setup_config.json (cameras,
CAN channels, gripper). Override any with --left-cam-serial etc.

Equal-footing claim: the three policies are interchangeable here. The
ONLY difference between running each is which YAML in configs/policy/
gets loaded.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make 'src/' and the repo root importable. Avoids needing `pip install -e .`.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from yam_vla.core import PolicyConfig, DEFAULT_JOURNAL_PATH       # noqa: E402
from evals._harness import load_tasks, start_session              # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Run a YAM-VLA eval against any registered policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- WHICH policy + WHICH eval (the only two flags you usually pick) ---
    grp = p.add_argument_group("selection")
    grp.add_argument("--policy", required=True,
                     help="policy name; matches configs/policy/<name>.yaml")
    grp.add_argument("--eval", dest="eval_name", required=True,
                     help="eval name; matches evals/<name>/tasks.yaml")
    grp.add_argument("--config-dir", default=str(_REPO / "configs" / "policy"),
                     help="dir of per-policy YAMLs")
    grp.add_argument("--evals-dir", default=str(_REPO / "evals"),
                     help="dir of per-eval task lists")

    # --- attempt knobs ---
    grp = p.add_argument_group("attempts")
    grp.add_argument("--attempts", type=int, default=None,
                     help="attempts per task (default: eval's n_attempts_default)")
    grp.add_argument("--max-chunks", type=int, default=200,
                     help="safety bound: max inference chunks per attempt "
                          "(~133 s at stride=6, 30 Hz)")
    grp.add_argument("--horizon-stride", type=int, default=None,
                     help="actions to play per chunk (default: per-policy YAML)")
    grp.add_argument("--train-fps", type=float, default=30.0)
    grp.add_argument("--num-steps", type=int, default=10,
                     help="flow-matching steps (molmoact2 only; others ignore)")
    grp.add_argument("--timeout-s", type=float, default=15.0,
                     help="per-inference HTTP/WS/ZMQ timeout")
    grp.add_argument("--inference-mode", default="sync",
                     choices=["sync", "async-naive", "async-time-aligned"])
    grp.add_argument("--dry-run", action="store_true",
                     help="don't command the arms; print actions only")

    # --- safety ---
    grp = p.add_argument_group("safety")
    grp.add_argument("--max-step-rad", type=float, default=0.15,
                     help="per-tick arm-joint clip (rad). 0 disables.")
    grp.add_argument("--gripper-step", type=float, default=0.15,
                     help="per-tick gripper clip (normalized). 0 disables.")
    grp.add_argument("--no-return-on-exit", action="store_true",
                     help="DANGEROUS: skip return-to-startup ramp on exit. "
                          "Arms will drop when motors disable.")

    # --- hardware overrides (defaults from yam_setup_config.json) ---
    grp = p.add_argument_group("hardware")
    grp.add_argument("--left-can",  default=None)
    grp.add_argument("--right-can", default=None)
    grp.add_argument("--gripper",   default=None,
                     choices=[None, "crank_4310", "linear_3507", "linear_4310", "flexible_4310"])
    grp.add_argument("--top-cam-serial",   default=None)
    grp.add_argument("--top-cam-v4l2",     default=None)
    grp.add_argument("--left-cam-serial",  default=None)
    grp.add_argument("--left-cam-v4l2",    default=None)
    grp.add_argument("--right-cam-serial", default=None)
    grp.add_argument("--right-cam-v4l2",   default=None)
    grp.add_argument("--cam-width",  type=int, default=424)
    grp.add_argument("--cam-height", type=int, default=240)
    grp.add_argument("--cam-fps",    type=int, default=30)

    # --- observability ---
    grp = p.add_argument_group("observability")
    grp.add_argument("--rerun", action="store_true",
                     help="stream observations + actions to a Rerun viewer")
    grp.add_argument("--rerun-save", default=None, metavar="PATH",
                     help="also save the Rerun recording to a .rrd file")
    grp.add_argument("--rerun-connect", default=None, metavar="HOST:PORT",
                     help="connect to existing viewer instead of spawning")
    grp.add_argument("--no-journal", action="store_true",
                     help="skip end-of-session journal prompt")
    grp.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH,
                     help="path to the research journal markdown file")

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")

    p = build_parser()
    args = p.parse_args()

    policy_yaml = Path(args.config_dir) / f"{args.policy}.yaml"
    if not policy_yaml.exists():
        p.error(f"policy config not found: {policy_yaml}")
    eval_yaml = Path(args.evals_dir) / args.eval_name / "tasks.yaml"
    if not eval_yaml.exists():
        p.error(f"eval tasks not found: {eval_yaml}")

    cfg = PolicyConfig.from_path(policy_yaml)

    # Per-policy stride default from YAML if user didn't override
    if args.horizon_stride is None:
        args.horizon_stride = int(cfg.control.get("horizon_stride_default", 6))

    policy = cfg.build()
    eval_def = load_tasks(eval_yaml)

    print(f"[run_eval] policy : {policy.name} ({cfg.model_id})", flush=True)
    print(f"[run_eval] eval   : {eval_def.name} ({len(eval_def.tasks)} tasks)", flush=True)
    print(f"[run_eval] stride : {args.horizon_stride}", flush=True)

    results_base_dir = Path(args.evals_dir) / args.eval_name / "results"
    start_session(policy, eval_def, args, results_base_dir=results_base_dir)


if __name__ == "__main__":
    main()
