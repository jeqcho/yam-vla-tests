#!/usr/bin/env python3
"""Top-level eval dispatcher.

Usage:
    ./scripts/run_eval.py --policy <name> --eval <name> [-- legacy-args...]

Loads:
    configs/policy/<policy>.yaml    via PolicyConfig
    evals/<eval>/tasks.yaml         via load_tasks
then starts an eval session against the constructed Policy.

Examples:
    # MolmoAct2 on the IKEA 10 (server already up at default port):
    ./scripts/run_eval.py --policy molmoact2 --eval ikea_10

    # Pi-0.5 on the Andon 10, fewer attempts:
    ./scripts/run_eval.py --policy pi05 --eval andon_10 \\
        -- --attempts 1 --horizon-stride 8

Anything after `--` is forwarded to the legacy harness argparse (so
existing flags like --tasks 1,3 / --attempts 5 / --left-cam-serial X
keep working).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make 'src/' and the repo root importable. This lets us avoid
# `pip install -e .` for now; the user can later switch to that.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from yam_vla.core import PolicyConfig                     # noqa: E402
from evals._harness import load_tasks, start_session      # noqa: E402


def _split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split on the literal `--` separator; everything after goes to legacy."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def main() -> None:
    own_args, passthrough = _split_passthrough(sys.argv[1:])

    p = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Run a YAM-VLA eval against any registered policy.",
    )
    p.add_argument("--policy", required=True,
                   help="policy name; matches configs/policy/<name>.yaml")
    p.add_argument("--eval", dest="eval_name", required=True,
                   help="eval name; matches evals/<name>/tasks.yaml")
    p.add_argument("--config-dir", default=str(_REPO / "configs" / "policy"),
                   help="dir of per-policy YAMLs (default: configs/policy/)")
    p.add_argument("--evals-dir", default=str(_REPO / "evals"),
                   help="dir of per-eval YAMLs (default: evals/)")
    args = p.parse_args(own_args)

    policy_yaml = Path(args.config_dir) / f"{args.policy}.yaml"
    if not policy_yaml.exists():
        p.error(f"policy config not found: {policy_yaml}")
    eval_yaml = Path(args.evals_dir) / args.eval_name / "tasks.yaml"
    if not eval_yaml.exists():
        p.error(f"eval tasks not found: {eval_yaml}")

    cfg = PolicyConfig.from_path(policy_yaml)
    policy = cfg.build()
    eval_def = load_tasks(eval_yaml)

    print(f"[run_eval] policy : {policy.name} ({cfg.model_id})", flush=True)
    print(f"[run_eval] eval   : {eval_def.name} ({len(eval_def.tasks)} tasks)",
          flush=True)
    print(f"[run_eval] passthrough -> legacy: {passthrough}", flush=True)

    # The legacy main() reads sys.argv for its argparse. Splice in
    # the passthrough args (preserve sys.argv[0]).
    sys.argv = [sys.argv[0]] + passthrough

    results_base_dir = Path(args.evals_dir) / args.eval_name / "results"
    namespace = argparse.Namespace()
    start_session(policy, eval_def, namespace,
                  results_base_dir=results_base_dir)


if __name__ == "__main__":
    main()
