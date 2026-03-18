#!/usr/bin/env python3
"""Grid-search launcher for SLURM jobs.

Usage:
    python scripts/launcher.py --dry           # print generated commands
    python scripts/launcher.py                 # submit all jobs via sbatch
    python scripts/launcher.py --time 4:00:00  # override time limit
"""

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Hyperparameter grid
# Each key is a main.py flag (without leading '--').
# Each value is a list of values to sweep over.
# All combinations will be submitted as separate jobs.
# ---------------------------------------------------------------------------
GRID: Dict[str, List[Any]] = {
    "seed": [0, 1, 2],
    "agent.best_of_n": [1, 32],
    "env_name": [
        "pointmaze-medium-navigate-v0",
        "antmaze-medium-navigate-v0",
        "humanoidmaze-medium-navigate-v0",
        "pointmaze-medium-stitch-v0",
        "antmaze-medium-stitch-v0",
        "humanoidmaze-medium-stitch-v0",
        "antsoccer-arena-navigate-v0",
        "antsoccer-arena-stitch-v0",
    ],
    "agent": [
        "agents/accrl.py",
    ],
    "agent.alpha": [0.3],
    "run_group": ["BestOfN"],
}

# Fixed flags added to every job (not swept over)
FIXED: Dict[str, Any] = {
    "eval_episodes": 50,
}

# ---------------------------------------------------------------------------
# SLURM defaults (override via CLI args)
# ---------------------------------------------------------------------------
SLURM_DEFAULTS = {
    "nodes": 1,
    "ntasks_per_node": 1,
    "cpus_per_task": 16,
    "mem": "64G",
    "gres": "gpu:1",
    "time": "10:00:00",
    "account": "plgcrlreason-gpu-gh200",
    "partition": "plgrid-gpu-gh200",
}


def dict_permutations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Return all combinations of the grid values."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def flags_to_str(flags: Dict[str, Any]) -> str:
    parts = []
    for k, v in flags.items():
        if isinstance(v, bool):
            parts.append(f"--{k}={str(v).lower()}")
        else:
            parts.append(f"--{k}={v}")
    return " ".join(parts)


def job_name(combo: Dict[str, Any]) -> str:
    parts = []
    for k, v in combo.items():
        short_k = k.split(".")[-1]  # e.g. "agent.alpha" -> "alpha"
        parts.append(f"{short_k}={v}")
    return "_".join(parts) if parts else "job"


def make_sbatch_script(
    python_flags: Dict[str, Any],
    slurm: Dict[str, Any],
    log_dir: str,
    name: str,
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    safe_name = name.replace("/", "_").replace("=", "").replace(" ", "_")
    out = os.path.join(log_dir, f"{safe_name}.out")
    err = os.path.join(log_dir, f"{safe_name}.err")

    lines = ["#!/bin/bash -l"]
    lines += [
        f"#SBATCH --nodes={slurm['nodes']}",
        f"#SBATCH --ntasks-per-node={slurm['ntasks_per_node']}",
        f"#SBATCH --cpus-per-task={slurm['cpus_per_task']}",
        f"#SBATCH --mem={slurm['mem']}",
        f"#SBATCH --gres={slurm['gres']}",
        f"#SBATCH --time={slurm['time']}",
        f"#SBATCH --account={slurm['account']}",
        f"#SBATCH --partition={slurm['partition']}",
        f"#SBATCH --job-name={safe_name[:60]}",
        f"#SBATCH --output={out}",
        f"#SBATCH --error={err}",
        "",
        "export XDG_CACHE_HOME=$SCRATCH/.cache",
        "export WANDB_API_KEY=$(cat ~/.wandb_key)",
        "export MUJOCO_GL=egl",
        "",
        "cd $SCRATCH/one-big-beautiful-trajectory",
        "cp -ru ~/one-big-beautiful-trajectory/* .",
        "source .venv/bin/activate",
        "",
        f"python main.py {flags_to_str(python_flags)} &",
        "",
        "wait",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search SLURM launcher")
    parser.add_argument("--dry", action="store_true", help="Print sbatch scripts without submitting")
    parser.add_argument("--log_dir", default="logs/grid", help="Directory for SLURM .out/.err files")
    parser.add_argument("--time", default=SLURM_DEFAULTS["time"], help="SLURM time limit (HH:MM:SS)")
    parser.add_argument("--mem", default=SLURM_DEFAULTS["mem"], help="Memory per job")
    parser.add_argument("--partition", default=SLURM_DEFAULTS["partition"], help="SLURM partition")
    parser.add_argument("--account", default=SLURM_DEFAULTS["account"], help="SLURM account")
    args = parser.parse_args()

    slurm = {**SLURM_DEFAULTS, "time": args.time, "mem": args.mem,
             "partition": args.partition, "account": args.account}

    combos = dict_permutations(GRID)
    print(f"Total jobs: {len(combos)}")

    if not args.dry:
        answer = input(f"About to submit {len(combos)} jobs. Continue? [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return

    for i, combo in enumerate(combos):
        python_flags = {**FIXED, **combo}
        name = job_name(combo)
        script = make_sbatch_script(python_flags, slurm, args.log_dir, name)

        if args.dry:
            print(f"\n{'='*60}")
            print(f"Job {i+1}/{len(combos)}: {name}")
            print(f"{'='*60}")
            print(script)
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                f.write(script)
                tmp_path = f.name
            try:
                result = subprocess.run(["sbatch", tmp_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                if result.returncode == 0:
                    print(f"[{i+1}/{len(combos)}] Submitted {name}: {result.stdout.strip()}")
                else:
                    print(f"[{i+1}/{len(combos)}] FAILED {name}: {result.stderr.strip()}", file=sys.stderr)
            finally:
                os.unlink(tmp_path)


if __name__ == "__main__":
    main()
