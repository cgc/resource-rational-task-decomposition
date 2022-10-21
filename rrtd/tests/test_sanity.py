import rrtd, solway_objective, prior_envs, bfs_runtime_exact
import pytest
import numpy as np

@pytest.mark.integrationtest
def test_sanity_bfs_exact():
    add_goal_to_visited_count = False
    for name, mdp, expected in [
        ('f2a', prior_envs.f2a, {3, 4, 8, 9, 13, 14}),
        ('f2c', prior_envs.f2c, {4, 5}),
        ('f2d', prior_envs.f2d, {2, 8, 10, 16}),
    ]:
        res = rrtd.task_decomposition(
            mdp, rrtd.all_pairs_shortest_path_distribution(mdp, remove_trivial_tasks=True),
            lambda mdp: bfs_runtime_exact.make_bfs_runtime_exact_algorithm(mdp, add_goal_to_visited_count=add_goal_to_visited_count),
        )
        actual = set(rrtd.argmaxes(rrtd.res_to_arr(mdp, res)).tolist())

@pytest.mark.integrationtest
def test_sanity_solway():
    for name, mdp, expected in [
        ('f2a', prior_envs.f2a, {3, 4, 8, 9, 13, 14}),
        ('f2c', prior_envs.f2c, {4, 5}),
        ('f2d', prior_envs.f2d, {9}),
    ]:
        res = solway_objective.task_decomposition(mdp, rrtd.all_pairs_shortest_path_distribution(mdp), samples=3)
        assert res.argmax() in expected
