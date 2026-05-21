import pytest

from eval_common import _reachable, learned_proposer_checkpoint, sample_valid_env_variations


class TestReachable:
    def test_empty_grid_is_reachable(self):
        assert _reachable(size=3, lava=set()) is True

    def test_lava_on_start_is_unreachable(self):
        assert _reachable(size=3, lava={(0, 0)}) is False

    def test_lava_on_goal_is_unreachable(self):
        assert _reachable(size=3, lava={(2, 2)}) is False

    def test_lava_wall_cuts_off_goal(self):
        # Full row of lava at row 1 separates start from goal on a 3x3.
        assert _reachable(size=3, lava={(1, 0), (1, 1), (1, 2)}) is False

    def test_partial_obstacle_still_reachable(self):
        # Lava at (1,1) is routable around.
        assert _reachable(size=3, lava={(1, 1)}) is True


class TestSampleValidEnvVariations:
    def test_excludes_start_and_goal(self):
        configs = sample_valid_env_variations(size=3, num_lava_tiles=1)
        # No config should place lava on the start (0,0) or goal (size-1, size-1).
        for lava in configs:
            assert (0, 0) not in lava
            assert (2, 2) not in lava

    def test_all_returned_configs_are_reachable(self):
        configs = sample_valid_env_variations(size=3, num_lava_tiles=2)
        assert len(configs) > 0
        for lava in configs:
            assert _reachable(size=3, lava=set(lava))

    def test_zero_lava_returns_single_empty_config(self):
        configs = sample_valid_env_variations(size=3, num_lava_tiles=0)
        assert configs == [()]


class TestCheckpointPath:
    def test_unknown_proposer_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown proposer kind"):
            learned_proposer_checkpoint("xgboost")

    def test_default_name_for_ppo(self):
        path = learned_proposer_checkpoint("ppo")
        assert "ppo_proposer_perfect" in str(path)
        assert path.name == "learned_proposer"

    def test_override_replaces_dir_name(self):
        path = learned_proposer_checkpoint("ppo", override="custom_dir")
        assert "custom_dir" in str(path)
        assert "ppo_proposer_perfect" not in str(path)
