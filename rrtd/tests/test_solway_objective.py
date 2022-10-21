from solway_objective import *
import prior_envs

def test_symmetric_reward():
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    # Ensure rewards are symmetric
    assert mdp.reward(0, None, 1) == mdp.reward(1, None, 0)
    # Ensure they also have noise
    assert mdp.reward(0, None, 2) != mdp.reward(1, None, 0)


def test_option_level_mdp():
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    mdp2 = SolwayOptionLevelMDP(mdp, [4])
    assert np.isclose(mdp2.reward(0, ('option', 4), 4), -2, atol=1e-1)
    assert np.isclose(mdp2.reward(1, ('option', 4), 4), -1, atol=1e-1)
    assert mdp._cache_info_vi_for_goal['misses'] == 1
    assert mdp._cache_info_vi_for_goal['hits'] == 2

    assert np.isclose(mdp2.reward(0, ('option', 9), 9), -5, atol=1e-1)
    assert np.isclose(mdp2.reward(1, ('option', 9), 9), -4, atol=1e-1)
    assert mdp._cache_info_vi_for_goal['misses'] == 2
    assert mdp._cache_info_vi_for_goal['hits'] == 4

def test_assign_subgoals():
    random.seed(42)
    basemdp = SolwayGraph(prior_envs.f2c.adjacency)
    mdp = basemdp.for_task(0, 9)
    ol_mdp = SolwayOptionLevelMDP(mdp, [4]).for_task(0, 9)
    assert assign_subgoals(ol_mdp, basemdp.vi_for_goal(9).policy) == [[0, 1], 4, 5, 6, 9]

def test_solway_phi():
    random.seed(42)
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    td = rrtd.Multinomial([dict(start=0, goal=5), dict(start=1, goal=6)])
    # 4 * 3 * 3 * 4 & 4 * 1 (cached) * 4 * 4
    assert np.isclose(solway_phi(mdp, td, [4]), np.log(1/(
        # Path is 0, 1, 4, 5.
        # - At 0, we choose subgoal (costs 4)
        # - At 0, in subgoal to 4, we choose 1 (costs 3)
        # - At 1, in subgoal to 4, we choose 4 (costs 3)
        # - At 4, no longer in subgoal, we choose 5 (costs 4)
        (4*3) * 3 * 4 *
        # Path is 1, 4, 5, 6. Only has one step in subgoal, but avoids paying for that step because of caching.
        (4*1) * 4 * 4)))

    random.seed(42)
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    td = rrtd.Multinomial([dict(start=0, goal=5), dict(start=3, goal=6)])
    assert np.isclose(solway_phi(mdp, td, [4]), np.log(1/(
        (4*3) * 3 * 4 *
        (4*3) * 4 * 4)))

def test_solway_phi_only_count_entrance_options():
    random.seed(42)
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    td = rrtd.Multinomial([dict(start=0, goal=5), dict(start=1, goal=6)])
    # This is the same as first task in test_solway_phi, except at second task, after we decide to choose no subgoal.
    # Then, we pay the cheaper rate for actions, leaving options out of our consideration.
    assert np.isclose(solway_phi(mdp, td, [4], only_count_entrance_options=True), np.log(1/(
        (4*3) * 3 * 4 *
        (4*1) * 4 * 3)))

    random.seed(42)
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    td = rrtd.Multinomial([dict(start=0, goal=7), dict(start=1, goal=9)])
    # In this case, we have longer "last miles" to the goal from the subgoal; both 7 and 9 are 3 steps from the subgoal (4)
    assert np.isclose(solway_phi(mdp, td, [4], only_count_entrance_options=True), np.log(1/(
        (4*3) * 3 * 4 * 3 * 3 *
        (4*1) * 4 * 3 * 3)))

def test_task_decomposition():
    for name, mdp, expected in [
#        ('f2a', prior_envs.f2a, {3, 4, 8, 9, 13, 14}),
        ('f2c', prior_envs.f2c, {4, 5}),
#        ('f2d', prior_envs.f2d, {9}),
#        ('f2f', prior_envs.f2f, {3, 4, 10, 19, 20, 23}),
    ]:
        res = task_decomposition(mdp, rrtd.all_pairs_shortest_path_distribution(mdp), samples=3)
        assert res.argmax() in expected

def test_solway_phi_subgoal_rate():
    random.seed(42)
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    td = rrtd.Multinomial([dict(start=0, goal=5), dict(start=1, goal=6)])
    p, sgr = solway_phi(mdp, td, [4], compute_subgoal_rate=True)
    assert np.allclose(sgr, np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))

    # Now checking to see how distributions work
    # HACK HACK this is a weird case: how should we be handling the goal when it's also a subgoal?
    random.seed(42)
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    td = rrtd.Multinomial([dict(start=0, goal=4), dict(start=1, goal=6)])
    p, sgr = solway_phi(mdp, td, [4, 5], compute_subgoal_rate=True)
    assert np.allclose(sgr, np.array([0, 0, 0, 0, .75, .25, 0, 0, 0, 0]))

    # Now checking to see how distributions work, pt. 2
    # Another display of a goal that counts as an sg
    random.seed(42)
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    td = rrtd.Multinomial([dict(start=0, goal=5), dict(start=1, goal=6)])
    p, sgr = solway_phi(mdp, td, [4, 5], compute_subgoal_rate=True)
    assert np.allclose(sgr, np.array([0, 0, 0, 0, .5, .5, 0, 0, 0, 0]))

    # Now including the non-sg special case
    random.seed(42)
    mdp = SolwayGraph(prior_envs.f2c.adjacency)
    td = rrtd.Multinomial([dict(start=0, goal=5), dict(start=0, goal=1)])
    p, sgr = solway_phi(mdp, td, [4], compute_subgoal_rate=True)
    assert np.allclose(sgr, np.array([0, .5, 0, 0, .5, 0, 0, 0, 0, 0]))

    # This is a case where subgoal use isn't deterministic: the path can
    # either be 0, 1, 4 or 0, 3, 4. Our subgoal is only used in one case
    # then.
    random.seed(42)
    sgrs = np.zeros(len(mdp.state_list))
    for _ in range(4):
        mdp = SolwayGraph(prior_envs.f2c.adjacency)
        td = rrtd.Multinomial([dict(start=0, goal=4)])
        p, sgr = solway_phi(mdp, td, [1], compute_subgoal_rate=True)
        sgrs += sgr
    idxs = np.isin(np.arange(len(mdp.state_list)), [1, 4])
    assert (sgrs[idxs] > 0).all()
    assert (sgrs[~idxs] == 0).all()
