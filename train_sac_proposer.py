"""Train a discrete-SAC proposer paired with a perfect validator.

SAC's auto-tuned entropy target prevents the policy from collapsing to
deterministic tie-breaking that PPO fell into on ambiguous lava states.

Saves the best checkpoint to ./checkpoints/sac_proposer_perfect/

Usage:
    source .venv/bin/activate
    python train_sac_proposer.py
    python train_sac_proposer.py --iters 2000
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning,
)

import ray
from ray import tune
from ray.tune import Checkpoint, register_env

from eval_common import set_grid

_REPO_ROOT = Path(__file__).resolve().parent
_ROVER_DIR = _REPO_ROOT
if str(_ROVER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROVER_DIR))

from config import create_rllib_config
from env import GridWorldEnv
from utils import (
    AgentConfig,
    MAX_ENV_STEPS,
    ProposerPolicies,
    ValidatorPolicies,
)

LOCAL_LOG_DIR = _REPO_ROOT / "checkpoints"
EXPERIMENT_NAME = "sac_proposer_perfect"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000,
                        help="number of training iterations, default is 1000")
    parser.add_argument("--grid-size", type=int, default=3,
                        help="determines the size of the square grid")
    args = parser.parse_args()

    grid_size, num_lava = set_grid(args)

    agent_config = AgentConfig(
        proposer_policy=ProposerPolicies.LEARNED,
        validator_policy=ValidatorPolicies.PERFECT,
        algorithm_name="sac",
        proposer_sees_lava=False,
    )

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {
            "PYTHONPATH": str(_ROVER_DIR),
            "RAY_DEDUP_LOGS": "0",
        }},
    )
    register_env("env", lambda _: GridWorldEnv(
        size=grid_size,
        num_lava_tiles=num_lava,
        single_agent=False,
        max_steps=MAX_ENV_STEPS,
        proposer_sees_lava=agent_config.proposer_sees_lava,
        randomize_spawn=True,
    ))

    config = create_rllib_config(agent_config)
    config.env_runners(num_env_runners=8)
    config.validate()

    metric = "env_runners/episode_return_mean"
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": args.iters},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=20,
                checkpoint_score_attribute=metric,
                checkpoint_score_order="max",
                num_to_keep=3,
            ),
            storage_path=str(LOCAL_LOG_DIR / "tune"),
            name=EXPERIMENT_NAME,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric=metric, mode="max", filter_nan_and_inf=True)
    df = best.metrics_dataframe
    valid = df[df[metric].notna()] if df is not None else None
    if valid is not None and len(valid):
        best_row = valid.loc[valid[metric].idxmax()]
        print(f"\nBest iteration: {int(best_row['training_iteration'])} "
              f"return={best_row[metric]:.3f}")
    else:
        print("\nNo non-NaN return recorded during training.")
    checkpoint: Checkpoint = best.checkpoint

    out_dir = LOCAL_LOG_DIR / f"{EXPERIMENT_NAME}_g{grid_size}_i{args.iters}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.to_directory(str(out_dir))
    canonical = LOCAL_LOG_DIR / EXPERIMENT_NAME
    if canonical.is_symlink() or canonical.exists():
        if canonical.is_symlink() or canonical.is_dir():
            import shutil
            if canonical.is_symlink():
                canonical.unlink()
            else:
                shutil.rmtree(canonical)
    canonical.symlink_to(out_dir.name)
    print(f"\nSaved best checkpoint to {out_dir}")
    print(f"Canonical symlink: {canonical} -> {out_dir.name}")
    print(f"Proposer module path: {out_dir / 'learner_group' / 'learner' / 'rl_module' / ProposerPolicies.LEARNED}")

    ray.shutdown()


if __name__ == "__main__":
    main()
