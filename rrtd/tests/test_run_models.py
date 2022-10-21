import rrtd
import run_models

def test_solwayr_partition_greedy_uniformexit():
    #mdp = rrtd.Graph({0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4, 6], 6: [5, 7], 7: [6]})
    mdp = rrtd.Graph({0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]})
    m = run_models.ModelsWithCustomTaskDistribution(rrtd.all_pairs_shortest_path_distribution(mdp, remove_successor_tasks=False))
    res = m.solwayr_partition_greedy_uniformexit(mdp)
    assert [b['partition'] for b in res['best']] == [[0, 0, 0, 3, 3, 3]]

    m = run_models.ModelsWithCustomTaskDistribution(rrtd.DictDistribution({
        rrtd.frozendict(start=i, goal=5): 1/5
        for i in range(5)
    }))
    res = m.solwayr_partition_greedy_uniformexit(mdp)
    assert [b['partition'] for b in res['best']] == [[0, 0, 0, 0, 0, 5]]
