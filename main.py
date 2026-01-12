import argparse
import os.path

import gymnasium as gym
import stable_baselines3 as sb3
import wandb
from rl_zoo3 import ALGOS
from rl_zoo3.hyperparams_opt import HYPERPARAMS_CONVERTER, convert_onpolicy_params, convert_offpolicy_params

from experiment_manager import ExperimentManagerLF
from policies.LeaderFollowerAlgorithm import LeaderFollowerAlgorithm

TRAINING_RUN = False
TRAINING_STEPS = 50_000

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
    "tensorboard_log_folder": "./runs",
    "min_learn_timesteps": TRAINING_STEPS if TRAINING_RUN else TRAINING_STEPS // 10,
    "save_frequency": TRAINING_STEPS // 5 if TRAINING_RUN else TRAINING_STEPS // 50,
    "env_kwargs": {},
    "eval_env_kwargs": {"return_summed_reward": True},
    "n_trials": 100,
    "n_parallel_jobs": 1,
    "n_startup_trials": 10,
    "truncate_last_trajectory": True,
    "log_interval": -1,
    "device": "cpu",
    "show_progress": True,
    "hyperparameters_dir": f"{os.path.dirname(os.path.realpath(__file__))}/hyperparams/{algorithm_config['algorithm']}.yaml",
}


def main():
    if algorithm_config["leader"] == "ars" or algorithm_config["follower"] == "ars":
        raise ValueError("ARS algorithm is not supported.")

    gym.register(
        run_config["environment_id"],
        "minigrid_env.environment:LavaEnv",
        1,
        False,
        run_config["environment_max_steps"],
    )

    run_name = f"{run_config["environment_id"]}__{algorithm_config["algorithm"]}__leader__{algorithm_config["leader"]}__follower__{algorithm_config["follower"]}__seed_{run_config["seed"]}"

    # Optuna parameters
    run_config["storage"] = os.path.abspath(os.path.join(run_config["log_folder"], run_name, "optuna.log"))
    # reset optuna storage
    if not TRAINING_RUN and os.path.exists(run_config["storage"]):
        os.remove(run_config["storage"])
    run_config["study_name"] = run_name

    tags = [f"v{sb3.__version__}"]
    if TRAINING_RUN:
        wandb.tensorboard.patch(root_logdir=run_config["tensorboard_log_folder"])
        run = wandb.init(
            name=run_name,
            project=run_config["wandb_project_name"],
            tags=tags,
            config=algorithm_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
        )

    ALGOS["leader_follower"] = LeaderFollowerAlgorithm
    HYPERPARAMS_CONVERTER["leader_follower"] = convert_onpolicy_params if algorithm_config["leader"] in [
        "a2c",
        "ppo",
        "ppo_lstm",
        "trpo",
        ] else convert_offpolicy_params

    exp_manager = ExperimentManagerLF(
        args=argparse.Namespace(**algorithm_config, **run_config),
        algo=algorithm_config["algorithm"],
        env_id=run_config["environment_id"],
        log_folder=run_config["log_folder"],
        tensorboard_log=f"{run_config["tensorboard_log_folder"]}/{run_name}" if TRAINING_RUN else "",
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
        storage=run_config["storage"],
        study_name=run_config["study_name"],
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
