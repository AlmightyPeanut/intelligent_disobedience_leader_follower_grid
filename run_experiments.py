import ray
from ray import tune
from ray.tune import register_env, Checkpoint

from config import create_rllib_config
from env import GridWorldEnv
from metrics import CustomTBXLoggerCallback, ActionLoggerCallback
from utils import LOG_DIR, GRID_SIZE, NUM_LAVA_TILES, MAX_ENV_STEPS, AgentConfig, AGENT_CONFIGS, TRAINING_ITERATIONS


def run_experiments(agent_config: AgentConfig):
    config = create_rllib_config(agent_config)
    config.callbacks([ActionLoggerCallback])

    experiment_name = f"{agent_config.algorithm_name}"
    if agent_config.proposer_policy is None and agent_config.validator_policy is None:
        experiment_name += "_single_agent"
    else:
        experiment_name += f"_{agent_config.proposer_policy}_{agent_config.validator_policy}__proposer_sees_lava_{agent_config.proposer_sees_lava}"

    tuner = tune.Tuner(
        config.algo_class,
        param_space=config.to_dict(),
        # maybe add tune config
        run_config=tune.RunConfig(
            stop={"training_iteration": TRAINING_ITERATIONS},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=10,
            ),
            storage_path=LOG_DIR / "tune",
            name=experiment_name,
            callbacks=[CustomTBXLoggerCallback()]
        )
    )

    tuner_results = tuner.fit()
    best_checkpoint: Checkpoint = tuner_results.get_best_result().checkpoint
    best_checkpoint.to_directory(LOG_DIR / "tune" / experiment_name / "best_checkpoint")


def main():
    ray.init(local_mode=True)
    for agent_config in AGENT_CONFIGS[4:5]:
        register_env("env", lambda _: GridWorldEnv(
            size=GRID_SIZE,
            num_lava_tiles=NUM_LAVA_TILES,
            single_agent=False,
            max_steps=MAX_ENV_STEPS,
            proposer_sees_lava=agent_config.proposer_sees_lava,
        ))
        run_experiments(agent_config)

    ray.shutdown()


if __name__ == '__main__':
    main()
