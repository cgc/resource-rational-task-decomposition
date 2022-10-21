import betweenness
import rrtd
import networkx
import prior_envs
import automated_design
import numpy as np

def test_betweenness():
    mdp = prior_envs.f2c
    g = automated_design.rrtd_to_nx(mdp)
    for endpoints, normalized in [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]:
        for gen in [betweenness.make_path_generator_distance, betweenness.make_path_generator_vi]:
            custom = betweenness.betweenness_centrality(
                mdp, endpoints=endpoints, normalized=normalized,
                make_path_generator=gen,
            )

            bc = networkx.algorithms.betweenness_centrality(
                g, endpoints=endpoints, normalized=normalized)
            assert np.allclose([bc[i] for i in range(len(mdp.state_list))], custom if normalized else custom/2)

            bcdir = networkx.algorithms.betweenness_centrality(
                g.to_directed(), endpoints=endpoints, normalized=normalized)
            assert np.allclose([bcdir[i] for i in range(len(mdp.state_list))], custom)

def test_optimal_occupancy():
    mdp = automated_design.parse_g6('G??CFw')
    t = rrtd.all_pairs_shortest_path_distribution(mdp)
    bc = betweenness.betweenness_centrality(mdp, normalized=True, endpoints=True, task_distribution=t)
    occ = betweenness.optimal_occupancy(mdp, task_distribution=t)
    assert np.allclose(bc/bc.sum(), occ)

def test_path_gen():
    for mdp in [
        grid2x3,
        automated_design.parse_g6('G??CFw'),
        prior_envs.f2a,
        prior_envs.f2c,
        prior_envs.f2d,
    ]:
        t = rrtd.all_pairs_shortest_path_distribution(mdp)
        pgvi = betweenness.make_path_generator_vi(mdp, t)
        pgd = betweenness.make_path_generator_distance(mdp, t)
        for task in t.support:
            assert set(pgvi(task)) == set(pgd(task))

grid2x3 = rrtd.Graph.from_binary_adjacency(np.array([
    [1, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 1],
]))

def test_path_gen_example():
    # this is a 2x3 grid:
    # 012
    # 345
    pgd = betweenness.make_path_generator_distance(grid2x3, None)
    assert set(pgd(dict(start=0, goal=5))) == {
        (0, 1, 2, 5),
        (0, 1, 4, 5),
        (0, 3, 4, 5),
    }
