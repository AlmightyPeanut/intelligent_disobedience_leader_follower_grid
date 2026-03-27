from env import GridWorldEnv, EnvironmentAction


class TestEnv:
    def test_random_actions(self):
        env = GridWorldEnv(render=True)
        env.reset()
        while True:
            action = env.action_spaces["proposer"].sample()
            env.step({
                "proposer": action
            })
            action = env.action_spaces["validator"].sample()
            _, _, terminated, _, _ = env.step({
                "validator": action
            })
            if terminated["__all__"]:
                break

    def test_single_agent(self):
        env = GridWorldEnv(render=True, single_agent=True, num_lava_tiles=0, size=2)
        env.reset()
        env.step({
            "single_agent": EnvironmentAction.MOVE_FORWARD
        })
        env.step({
            "single_agent": EnvironmentAction.TURN_RIGHT
        })
        _, reward, terminated, _, _ = env.step({
            "single_agent": EnvironmentAction.MOVE_FORWARD
        })
        assert reward["single_agent"] == 1
        assert terminated["__all__"]
