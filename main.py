import argparse
import os.path
import time

import gymnasium as gym
import stable_baselines3 as sb3
import wandb
from rl_zoo3 import ALGOS

from experiment_manager import ExperimentManagerLF
from policies.LeaderFollowerAlgorithm import LeaderFollowerAlgorithm

TRAINING_RUN = False

algorithm_config = {
    "algorithm": "leader_follower",
    "leader": "dqn",
    "follower": "dqn",
}

run_config = {
    "seed": 42,
    "environment_id": "Minigrid-Lava",
    "environment_grid_size": 5,
    "environment_max_steps": 256,

    "wandb_project_name": "Intelligent disobedience - leader-follower-grid",

    "log_folder": "./logs",
    "min_timesteps": 50_000,
    "save_frequency": 10_000,
    "tensorboard_log_folder": "./runs",
    "env_kwargs": {},
    "eval_env_kwargs": {"return_summed_reward": True},
    "n_trials": 500,
    "n_parallel_jobs": 1,
    "n_startup_trials": 10,
    "truncate_last_trajectory": True,
    "log_interval": -1,
    "device": "cpu",
    "show_progress": True,
    "hyperparameters_dir": f"{os.path.dirname(os.path.realpath(__file__))}/hyperparams/{algorithm_config['algorithm']}.yaml",
}


def main():
    gym.register(
        run_config["environment_id"],
        "minigrid_env.environment:LavaEnv",
        1,
        False,
        run_config["environment_max_steps"],
    )

    run_name = f"{run_config["environment_id"]}__{algorithm_config["algorithm"]}__{algorithm_config["leader"]}__{algorithm_config["follower"]}__seed_{run_config["seed"]}"
    tags = [f"v{sb3.__version__}"]
    if TRAINING_RUN:
        run = wandb.init(
            name=run_name,
            project=run_config["wandb_project_name"],
            tags=tags,
            config=algorithm_config,
            # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
        )
        wandb.tensorboard.patch(root_logdir=run_config["tensorboard_log_folder"])

    ALGOS["leader_follower"] = LeaderFollowerAlgorithm

    exp_manager = ExperimentManagerLF(
        args=argparse.Namespace(**algorithm_config, **run_config),
        algo=algorithm_config["algorithm"],
        env_id=run_config["environment_id"],
        log_folder=run_config["log_folder"],
        tensorboard_log=f"{run_config["tensorboard_log_folder"]}/{run_name}",
        n_timesteps=run_config["min_learn_timesteps"],
        save_freq=run_config["save_frequency"],
        env_kwargs=run_config["env_kwargs"],
        eval_env_kwargs=run_config["eval_env_kwargs"],
        optimize_hyperparameters=not TRAINING_RUN,
        n_trials=run_config["n_trials"],
        n_jobs=run_config["n_parallel_jobs"],
        optimization_log_path=f"{run_config["log_folder"]}/{run_name}/hp_optimization/",
        n_startup_trials=run_config["n_startup_trials"],
        truncate_last_trajectory=run_config["truncate_last_trajectory"],
        seed=run_config["seed"],
        log_interval=run_config["log_interval"],
        device=run_config["device"],
        show_progress=run_config["show_progress"],
        config=run_config["hyperparameters_dir"],
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results

        if TRAINING_RUN:
            run_config["saved_hyperparams"] = saved_hyperparams
            assert run is not None
            run.config.setdefaults(run_config | algorithm_config)

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()


if __name__ == '__main__':
    main()
