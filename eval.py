import itertools
from pathlib import Path

import numpy as np
import ray
import torch
import tqdm
import tree
from ray.rllib import SampleBatch
from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.core.rl_module import RLModule
from ray.tune import register_env

from env import GridWorldEnv
from utils import AgentConfig, GRID_SIZE, MAX_ENV_STEPS, NUM_LAVA_TILES, LOG_DIR, AGENT_CONFIGS


def sample_valid_env_variations(size: int, num_lava_tiles: int):
    start = (0, 0)
    goal = (size - 1, size - 1)

    # Positions that are "blocked" because they are start, goal, or adjacent start
    blocked_positions = {start, goal}

    # All valid positions for lava
    inner_positions = [(x, y)
                       for x in range(0, size)
                       for y in range(0, size)
                       if (x, y) not in blocked_positions]

    # Generate all combinations of 2 lava tiles
    all_variations = list(itertools.combinations(inner_positions, num_lava_tiles))
    if num_lava_tiles == 2:
        all_variations.remove(((0, 1), (1, 0)))  # prevent start blocking
        all_variations.remove(((size - 2, size - 1), (size - 1, size - 2)))  # prevent goal blocking
    return all_variations


def run_experiment(agent_config: AgentConfig) -> None:
    is_single_agent = agent_config.proposer_policy is None and agent_config.validator_policy is None

    final_rewards = []
    validator_rewards = []

    checkpoint_path = Path(LOG_DIR / "tune")
    if is_single_agent:
        checkpoint_path = checkpoint_path / "ppo_single_agent" / "best_checkpoint"
        checkpoint_path = checkpoint_path / "learner_group" / "learner" / "rl_module" / "single_agent"
        single_agent_module = RLModule.from_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path / f"{agent_config.algorithm_name}_{agent_config.proposer_policy}_{agent_config.validator_policy}__proposer_sees_lava_{agent_config.proposer_sees_lava}" / "best_checkpoint"
        checkpoint_path = checkpoint_path / "learner_group" / "learner" / "rl_module"
        proposer_agent_module = RLModule.from_checkpoint(checkpoint_path / agent_config.proposer_policy)
        validator_agent_module = RLModule.from_checkpoint(checkpoint_path / agent_config.validator_policy)

    env = GridWorldEnv(
        size=GRID_SIZE,
        render=True,
        record_render=True,
        max_steps=2 * MAX_ENV_STEPS,
        single_agent=is_single_agent,
        num_lava_tiles=NUM_LAVA_TILES,
        proposer_sees_lava=agent_config.proposer_sees_lava,
    )

    env_variations = sample_valid_env_variations(GRID_SIZE, NUM_LAVA_TILES)
    for env_variation in tqdm.tqdm(env_variations):
        if not env_variation:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(options={"lava_positions": env_variation})
        obs = tree.map_structure(lambda x: torch.tensor(np.expand_dims(x, axis=0)), obs)
        terminated = {"__all__": False}
        truncated = {"__all__": False}

        rewards = {
            "proposer": 0.0,
            "validator": 0.0,
            "single_agent": 0.0,
        }

        while not terminated["__all__"]:
            actions = {}
            if is_single_agent:
                actions["single_agent"] = single_agent_module.forward_inference({
                    SampleBatch.OBS: obs["single_agent"],
                })
            else:
                if "proposer" in obs:
                    actions["proposer"] = proposer_agent_module.forward_inference({
                        SampleBatch.OBS: obs["proposer"],
                    })
                elif "validator" in obs:
                    actions["validator"] = validator_agent_module.forward_inference({
                        SampleBatch.OBS: obs["validator"],
                    })
                else:
                    raise ValueError("Invalid observation")

            parsed_actions = {}
            for agent_id, action in actions.items():
                if "proposer" == agent_id:
                    if isinstance(proposer_agent_module, DefaultPPORLModule):
                        parsed_actions[agent_id] = torch.argmax(action[SampleBatch.ACTION_DIST_INPUTS], dim=-1).item()
                    else:
                        parsed_actions[agent_id] = action[SampleBatch.ACTIONS].item()
                elif "validator" == agent_id:
                    if isinstance(validator_agent_module, DefaultPPORLModule):
                        parsed_actions[agent_id] = torch.argmax(action[SampleBatch.ACTION_DIST_INPUTS], dim=-1).item()
                    else:
                        parsed_actions[agent_id] = action[SampleBatch.ACTIONS].item()
                else:
                    raise ValueError(f"Invalid agent id {agent_id}")
            actions = parsed_actions

            obs, rewards, terminated, truncated, _ = env.step(actions)
            if "validator" in rewards:
                validator_rewards.append(rewards["validator"])
            obs = tree.map_structure(lambda x: torch.tensor(np.expand_dims(x, axis=0)), obs)
            if truncated["__all__"]:
                break

        if not truncated["__all__"] and "proposer" in rewards:
            final_rewards.append(rewards["proposer"])
        elif not truncated["__all__"] and "single_agent" in rewards:
            final_rewards.append(rewards["single_agent"])
        else:
            final_rewards.append(-1.0)

    if is_single_agent:
        env.save_video(f"{agent_config.algorithm_name}_single_agent.mp4")
    else:
        env.save_video(
            f"{agent_config.algorithm_name}_{agent_config.proposer_policy}_{agent_config.validator_policy}.mp4")

    proposer_rewards = np.array(final_rewards)
    validator_rewards = np.array(validator_rewards)

    print(f"Evaluating {agent_config.algorithm_name} on {GRID_SIZE}x{GRID_SIZE} grid with {NUM_LAVA_TILES} lava tiles")
    print(f"with {agent_config.proposer_policy} and {agent_config.validator_policy} policies.")
    print(" Proposer ".center(50, '='))
    print(f"Reached the goal in {np.sum(proposer_rewards > 0.)} out of {len(proposer_rewards)} "
          f"({np.sum(proposer_rewards > 0.) / len(proposer_rewards) * 100:.2f}%).")

    print(" Validator ".center(50, '='))
    print(f"{agent_config.validator_policy} final rewards mean: {np.mean(validator_rewards)}")
    print(f"{agent_config.validator_policy} wanted behaviour: {np.mean(validator_rewards >= 0.) * 100:.2f}%")
    print(f"{agent_config.validator_policy} good disobedience: {np.mean(validator_rewards > 0.) * 100:.2f}%")
    print(f"{agent_config.validator_policy} bad disobedience: {np.mean(validator_rewards < 0.) * 100:.2f}%")


def main():
    ray.init(ignore_reinit_error=True)

    register_env("env", lambda _: GridWorldEnv(GRID_SIZE))
    for agent_config in AGENT_CONFIGS[4:5]:
        run_experiment(agent_config)
    # agent_config = AgentConfig(algorithm_name="dqn")
    # run_experiment(agent_config)

    ray.shutdown()


if __name__ == '__main__':
    main()
