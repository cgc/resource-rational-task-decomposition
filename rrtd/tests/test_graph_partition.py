import pytest
import numpy as np
import rrtd, prior_envs, solway_objective
import random
from graph_partition import *

def test_partition_count():
    def parti_count_for_fc_graph(lim):
        mdp = rrtd.Graph({
            i: range(lim)
            for i in range(lim)
        })
        return len(list(graphenum(mdp, edges=list(edges(mdp)))))

    # matches the Bell numbers https://oeis.org/A000110
    assert [parti_count_for_fc_graph(i) for i in range(9)] == [1, 1, 2, 5, 15, 52, 203, 877, 4140]

def test_assign_subgoals():
    random.seed(42)
    basemdp = solway_objective.SolwayGraph(prior_envs.f2c.adjacency)
    mdp = basemdp.for_task(0, 9)
    ol_mdp = SolwayPartitionOLMDP(mdp, {i: i//5 for i in range(10)}).for_task(0, 9)
    ol_mdp(mdp, ol_mdp.subgoals) # this is a hack and bummer. part of the complicated way this is integrated.
    assert solway_objective.assign_subgoals(ol_mdp, basemdp.vi_for_goal(9).policy) == [[0, 1, 4], 5, 6, 9]

def test_exit_map_from_partition():
    def check_res(m, p, expected):
        for s, value in m.items():
            assert sorted(expected[p[s]]) == sorted(value)

    p = {i: 0 if i <= 4 else 1 for i in range(10)}
    m = exit_map_from_partition(prior_envs.f2c, p)
    check_res(m, p, {0: [5], 1: [4]})

    p = {i: 0 if i <= 3 else 1 for i in range(10)}
    m = exit_map_from_partition(prior_envs.f2c, p)
    check_res(m, p, {0: [4], 1: [1, 3]})


@pytest.mark.integrationtest
def test_solway_f2c():
    random.seed(42)
    scores = solway_task_decomposition(prior_envs.f2c, rrtd.all_pairs_shortest_path_distribution(prior_envs.f2c), samples=1)

    row = max(scores, key=lambda row: row['value'])
    comp_to_node = {}
    for node, comp in row['parti'].items():
        comp_to_node.setdefault(comp, set()).add(node)
    assert sorted(list(comp_to_node.values())) == [set(range(5)), set(range(5, 10))]

@pytest.mark.integrationtest
def test_solway_f2c_only_entrance():
    scores = solway_task_decomposition(prior_envs.f2c, rrtd.all_pairs_shortest_path_distribution(prior_envs.f2c, remove_successor_tasks=False), samples=1, only_count_entrance_options=True)

    row = max(scores, key=lambda row: row['value'])
    print(row)
    comp_to_node = {}
    for node, comp in row['parti'].items():
        comp_to_node.setdefault(comp, set()).add(node)
    assert sorted(list(comp_to_node.values())) == [set(range(5)), set(range(5, 10))]

def test_match_R_lang():
    random.seed(42)
    nmdp = solway_objective.SolwayGraph(rrtd.adjacency_ssp(prior_envs.f2c))
    for parti, expected in [
        ([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], -226.35998306964657),
        ([0]*10, -243.89192808431991),
    ]:
        ol = SolwayPartitionOLMDP(nmdp, dict(enumerate(parti)))
        td = rrtd.all_pairs_shortest_path_distribution(nmdp, remove_trivial_tasks=True, remove_successor_tasks=False)
        actual = solway_objective.solway_phi(nmdp, td, ol.subgoals, SolwayOptionLevelMDP_class=ol, only_count_entrance_options=True)
        assert np.isclose(actual, expected)
