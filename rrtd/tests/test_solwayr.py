import solwayr, rrtd, graph_partition, prior_envs
import random
import pytest
import numpy as np


def needs_rlang():
    try:
        solwayr.init()
    except Exception as e:
        pytest.skip(f'could not init solway: {e}')
        return


@pytest.mark.integrationtest
def test_solway_validation():
    needs_rlang()

    tests = [
        [0]*10,
        [0]*5+[1]*5,
    ] + random.choices(list(graph_partition.graphenum(prior_envs.f2c)), k=30)
    for p in tests:
        solwayr.validate(prior_envs.f2c, p)

def test_solway_model_random_seed():
    needs_rlang()

    mdp = prior_envs.f2c
    td = rrtd.all_pairs_shortest_path_distribution(mdp)
    partition = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    with solwayr.r_seed_ctx(12345):
        val1 = solwayr.SolwayModel(mdp, nsamples=1).logevidence(td, partition)
    with solwayr.r_seed_ctx(12345):
        val2 = solwayr.SolwayModel(mdp, nsamples=1).logevidence(td, partition)
    assert np.isclose(val1, val2)

def test_solway_model_samples():
    needs_rlang()

    mdp = prior_envs.f2c
    td = rrtd.all_pairs_shortest_path_distribution(mdp)
    partition = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    with solwayr.r_seed_ctx(42):
        m = solwayr.SolwayModel(mdp, nsamples=7)
        val1 = m.logevidence(td, partition)
        sgr1 = m.subgoal_rate(td, partition)
    with solwayr.r_seed_ctx(42):
        ms = [solwayr.SolwayModel(mdp, nsamples=1) for _ in range(7)]
        val2 = np.mean([m.logevidence(td, partition) for m in ms])
        sgr2 = np.mean([m.subgoal_rate(td, partition) for m in ms], axis=0)
    assert np.isclose(val1, val2)
    assert np.allclose(sgr1, sgr2)

def test_subgoal_rate_from_paths():
    def sg_rate_from_paths_uni(sps, partition):
        td = rrtd.DictDistribution.uniform([rrtd.frozendict(start=sp[0], goal=sp[-1]) for sp in sps])
        return solwayr.subgoal_rate_from_paths(sps, td, partition)
    def assert_allclosedict(a, b):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for k in a.keys():
            assert np.allclose(a[k], b[k]), k

    # Goal is selected when there are no partition crossings.
    sgr, sgr_dict = sg_rate_from_paths_uni([[0, 1, 2]], [0, 0, 0])
    assert np.allclose(sgr, np.array([0, 0, 1]))
    assert_allclosedict(sgr_dict, {(0, 2): np.array([0, 0, 1])})
    # SG is selected when there's a partition crossing.
    sgr, sgr_dict = sg_rate_from_paths_uni([[0, 1, 2]], [0, 1, 1])
    assert np.allclose(sgr, np.array([0, 1, 0]))
    assert_allclosedict(sgr_dict, {(0, 2): np.array([0, 1, 0])})
    # Multiple SGs, where one is the goal.
    sgr, sgr_dict = sg_rate_from_paths_uni([[0, 1, 2]], [0, 1, 2])
    assert np.allclose(sgr, np.array([0, 1/2, 1/2]))
    assert_allclosedict(sgr_dict, {(0, 2): np.array([0, 1/2, 1/2])})
    # This is a simple mixture of two determinstic cases.
    sgr, sgr_dict = sg_rate_from_paths_uni([[0, 1, 2], [1, 2]], [0, 1, 1])
    assert np.allclose(sgr, np.array([0, 1/2, 1/2]))
    assert_allclosedict(sgr_dict, {(0, 2): np.array([0, 1, 0]), (1, 2): np.array([0, 0, 1])})
    # A mixture that includes a stochastic case.
    sgr, sgr_dict = sg_rate_from_paths_uni([[0, 1, 2], [0, 1]], [0, 1, 2])
    assert np.allclose(sgr, np.array([0, 3/4, 1/4]))
    assert_allclosedict(sgr_dict, {(0, 2): np.array([0, 1/2, 1/2]), (0, 1): np.array([0, 1, 0])})
    # A mixture that doesn't involve goals at all.
    sgr, sgr_dict = sg_rate_from_paths_uni([[0, 1, 2, 3], [1, 2]], [0, 1, 2, 2])
    assert np.allclose(sgr, np.array([0, 1/4, 3/4, 0]))
    assert_allclosedict(sgr_dict, {(0, 3): np.array([0, 1/2, 1/2, 0]), (1, 2): np.array([0, 0, 1, 0])})
