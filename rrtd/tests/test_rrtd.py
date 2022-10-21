from rrtd import *
import prior_envs

f2c = prior_envs.f2c

def test_dist_prod():
    assert dist_prod(
        DictDistribution(a=1/4, b=3/4),
        DictDistribution(c=1/4, d=3/4),
    ).isclose(
        DictDistribution({
            ('a', 'c'): 1/16, ('a', 'd'): 3/16, ('b', 'c'): 3/16, ('b', 'd'): 9/16
        })
    )

def test_expectation():
    assert expectation(DictDistribution.uniform([0, 1])) == 0.5 # defaults to identity
    assert expectation(DictDistribution.uniform([0, 1]), lambda x: x) == 0.5
    assert expectation(DictDistribution({0: 1/4, 1: 3/4}), lambda x: x) == 0.75
    assert expectation(DictDistribution.uniform([0, 1, 2, 3]), lambda x: x**2) == (0+1+4+9)/4

def test_condition():
    assert dist_condition(dist_prod(Multinomial([0, 1]), Multinomial([0, 1])), lambda x: x[0]==1).isclose(
            Multinomial([(1, 0), (1, 1)]))
    assert dist_condition(Multinomial([0, 1]), lambda x: x==42).isclose(DictDistribution())

def test_vi_wrapper_caching():
    vi = value_iteration(f2c)
    assert vi(1, 0)['value'] == -1
    assert vi(4, 0)['value'] == -2
    assert list(vi.cache.keys()) == [0]
    assert vi(0, 1)['value'] == -1
    assert list(vi.cache.keys()) == [0, 1]

def test_option_level_rw():
    mdp = f2c
    assert mdp.__class__ is Graph
    start, goal = 0, 9
    alg = new_random_walk_algorithm(mdp)
    cost_sg4 = alg(start, 4)['value'] + alg(4, goal)['value']
    cost_sg6 = alg(start, 6)['value'] + alg(6, goal)['value']
    assert np.isclose(cost_sg4, alg(start, goal)['value'])
    assert cost_sg6 < alg(start, goal)['value']

    _val = lambda sg, fsu: value_iteration(
        OptionLevelMDP(mdp, alg, sg, force_subgoal_use=fsu))(start, goal)['value']

    assert np.isclose(_val([6], False), _val([4], False))
    assert _val([6], True) < _val([4], True)

    assert np.isclose(_val([4], True), _val([4], False))
    assert np.isclose(_val([4], True), cost_sg4)

    assert np.isclose(_val([6], True), cost_sg6)



def test_option_level_no_cost():
    mdp = f2c
    alg = new_random_walk_algorithm(mdp)
    D = floyd_warshall(mdp)

    for start in [0, 1, 2, 3]:
        for sg in [4, 5]:
            for goal in [6, 7, 8, 9]:
                _value = lambda **kw: value_iteration(OptionLevelMDP(mdp, alg, [sg], **kw))(start, goal)['value']
                assert _value(no_cost_subgoals=True) == alg(sg, goal)['value']
                assert _value(only_travel_cost_subgoals=True) == alg(sg, goal)['value'] - D[start, sg]

def test_binary_adjacency_ssp():
    A = binary_adjacency_ssp(f2c)
    assert (A == np.array([
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    ])).all()
    assert (A[np.eye(A.shape[0], dtype=bool)] == 0).all()
    assert (A == A.T).all()
