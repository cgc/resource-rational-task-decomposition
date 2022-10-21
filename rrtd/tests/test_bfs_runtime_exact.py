import rrtd, pythonland_ppl, bfs_runtime_exact, prior_envs
import numpy as np
import pytest
from frozendict import frozendict

lineworld = rrtd.Graph.from_binary_adjacency(np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
]))

def test_bfs_runtime_exact_basics():
    for s, g, exp in [
        (0, 1, 1),
        (0, 2, 2),
        (0, 3, 3),
        (1, 0, 1.5),
        (1, 2, 1.5),
        (1, 3, 3),
    ]:
        # adding 1 since goals are counted.
        assert bfs_runtime_exact.bfs_exact(lineworld.for_task(s, g))['expected_cost'] == exp + 1

def test_bfs_quirk():
    # this is a 2x3 grid
    mdp = rrtd.Graph.from_binary_adjacency(np.array([
        [1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 1],
    ]))

    # This is a pretty critical case; we don't have a uniform over elements at some depth.
    # In this case, the depth 1 nodes (1, 3) are more likely to queue up 4 ahead of 2 (the depth 2 nodes).
    assert np.isclose(bfs_runtime_exact.bfs_exact(mdp.for_task(0, 4))['expected_cost'], 4.25)

@pytest.mark.parametrize(
    "mdp",
    [lineworld, prior_envs.f2c],
)
def test_bfs_runtime_exact_matches(mdp):
    for s in mdp.state_list:
        allgoals = bfs_runtime_exact.bfs_exact_allgoals(mdp.for_task(s, s))
        allgoals_dedup = bfs_runtime_exact.bfs_exact_allgoals(mdp.for_task(s, s), deduplicate_traces=True)
        for g in mdp.state_list:
            old_ = pythonland_ppl.search_cost(mdp, s, g)
            for new_ in [
                bfs_runtime_exact.bfs_exact(mdp.for_task(s, g))['expected_cost'],
                bfs_runtime_exact.bfs_exact(mdp.for_task(s, g), deduplicate_traces=True)['expected_cost'],
                allgoals[g],
                allgoals_dedup[g],
            ]:
                assert np.isclose(new_, old_), (s, g, new_, old_)
