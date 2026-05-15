import pytest

from env import (
    DISTANCE_SHAPING_COEFF,
    EnvironmentAction,
    GridWorldEnv,
    STEP_PENALTY,
    WALL_BUMP_PENALTY,
)


def _make_env(**kwargs):
    defaults = dict(
        size=3,
        num_lava_tiles=0,
        single_agent=True,
        render=False,
        record_render=False,
    )
    defaults.update(kwargs)
    env = GridWorldEnv(**defaults)
    env.reset(seed=0)
    return env


class TestRewardShaping:
    def test_step_toward_goal_gets_positive_shaping(self):
        # Agent spawns at (1,1) facing right. MOVE_FORWARD goes to (1,2),
        # which reduces Manhattan distance to goal (3,3) by 1.
        # Reward = STEP_PENALTY + DISTANCE_SHAPING_COEFF * 1
        env = _make_env()
        _, reward, _, _, _ = env.step({"single_agent": EnvironmentAction.MOVE_FORWARD})
        expected = STEP_PENALTY + DISTANCE_SHAPING_COEFF
        assert reward["single_agent"] == pytest.approx(expected)

    def test_pure_turn_gets_only_step_penalty(self):
        # Turning doesn't move the agent, so the distance shaping is 0.
        env = _make_env()
        _, reward, _, _, _ = env.step({"single_agent": EnvironmentAction.TURN_LEFT})
        assert reward["single_agent"] == pytest.approx(STEP_PENALTY)

    def test_wall_bump_gets_wall_penalty(self):
        # Agent at (1,1) facing right (dir=1). 
        # Turn left twice -> facing left (dir=3)
        # Then MOVE_FORWARD bumps the wall at column 0. no movement, no shaping change
        env = _make_env()
        env.step({"single_agent": EnvironmentAction.TURN_LEFT})
        env.step({"single_agent": EnvironmentAction.TURN_LEFT})
        _, reward, _, _, _ = env.step({"single_agent": EnvironmentAction.MOVE_FORWARD})
        expected = STEP_PENALTY + WALL_BUMP_PENALTY  # shaping term is 0
        assert reward["single_agent"] == pytest.approx(expected)

    def test_goal_terminal_reward_is_five(self):
        # 2x2 grid: 
        # spawn (1,1) facing right -> move to (1,2) -> turn right -> move to (2,2) which is the goal.
        env = _make_env(size=2)
        env.step({"single_agent": EnvironmentAction.MOVE_FORWARD})
        env.step({"single_agent": EnvironmentAction.TURN_RIGHT})
        _, reward, terminated, _, _ = env.step(
            {"single_agent": EnvironmentAction.MOVE_FORWARD}
        )
        assert reward["single_agent"] == 5
        assert terminated["__all__"]
