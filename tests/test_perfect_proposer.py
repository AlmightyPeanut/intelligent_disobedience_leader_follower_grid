from rl_modules.perfect_proposer import PerfectProposerRLM


class TestBFS:
    """Unit tests for the BFS planner inside the PerfectProposer.

    These check the static _bfs_next helper directly and don't
    require constructing an RLModule or a torch obs tensor.
    """

    def test_returns_none_when_at_goal(self):
        assert PerfectProposerRLM._bfs_next((2, 2), (2, 2), size=3, blocked=set()) is None

    def test_returns_next_step_not_full_path(self):
        # On an empty 3x3, from (1,1) to (3,3),
        # the first move must be to a neighbor of (1,1), either (1,2) or (2,1).
        nxt = PerfectProposerRLM._bfs_next((1, 1), (3, 3), size=3, blocked=set())
        assert nxt in {(1, 2), (2, 1)}

    def test_routes_around_lava(self):
        # Direct path (1,1) -> (1,2) -> (1,3) -> ... is blocked by lava at (1,2)
        # BFS should pick (2,1) instead
        nxt = PerfectProposerRLM._bfs_next(
            (1, 1), (3, 3), size=3, blocked={(1, 2)}
        )
        assert nxt == (2, 1)

    def test_returns_none_when_unreachable(self):
        # Wall of lava across row 2 cuts the agent off from the goal
        nxt = PerfectProposerRLM._bfs_next(
            (1, 1), (3, 3), size=3, blocked={(2, 1), (2, 2), (2, 3)}
        )
        assert nxt is None

    def test_respects_grid_bounds(self):
        # On a 1x1 grid the only cell is (1,1), which is also the goal
        assert PerfectProposerRLM._bfs_next((1, 1), (1, 1), size=1, blocked=set()) is None
