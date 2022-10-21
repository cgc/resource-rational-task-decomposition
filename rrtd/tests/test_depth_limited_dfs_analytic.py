import random
from re import A
import numpy as np
import scipy.stats
import prior_envs, depth_limited_dfs_analytic, search_algorithms, rrtd
from frozendict import frozendict

# some tiny MDPs for testing.
'''
twostage:

0--> 1--> 3
|    |--> 4
|--> 2--> 5
'''
twostage = rrtd.Graph.from_binary_adjacency(np.array([
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]))

'''
onestage:

0--> 1
|--> 2
'''
onestage = rrtd.Graph.from_binary_adjacency(np.array([
    [0, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
]))

dfs = depth_limited_dfs_analytic.recursive_dfs_exact

def test_dfs_exact():
    '''
    onestage, from 0 to 1
    '''
    for kw, expected in [
        (dict(), 2.5),
        (dict(add_goal_to_visited_count=False), 1.5),
    ]:
        assert dfs(onestage.for_task(0, 1), **kw)['plan_cost'] == expected
    '''
    twostage, from 0 to 3
    0,1,3 | Cost=3, P=1/4
    0,1,4 | Then 3, Cost=4, P=1/4
    0,2,4 | Then 5,backtrack,1,3, Cost=6, P=1/8
            | Then 5,backtrack,1,4,3, Cost=7, P=1/8
    0,2,5 | Then 4,backtrack,1,3, Cost=6, P=1/8
            | Then 4,backtrack,1,4,3, Cost=7, P=1/8
    '''
    for kw, expected in [
        (dict(), (3+4+6+7)/4),
        (dict(add_goal_to_visited_count=False), (3+4+6+7)/4 - 1),
    ]:
        assert dfs(twostage.for_task(0, 3), **kw)['plan_cost'] == expected
    '''
    twostage, from 0 to 1
    0,1 | Cost=2, P=1/2
    0,2 | Then 4,5,backtrack,1 Cost=5, P=1/4
        | Then 5,4,backtrack,1 Cost=5, P=1/4

    depth-limit=1
    0,1 | Cost=2, P=1/2
    0,2,limit,1 | Cost=3, P=1/2
    '''
    for kw, expected in [
        (dict(), (2 + 5)/2),
        (dict(depth_limit=2), (2 + 5)/2), # no change
        (dict(add_goal_to_visited_count=False), (2 + 5)/2 - 1),
        (dict(depth_limit=1, add_depth0_to_visited_count=True), (2 + 3)/2),
        (dict(depth_limit=1, add_depth0_to_visited_count=False), (2 + 2)/2),
    ]:
        assert dfs(twostage.for_task(0, 1), **kw)['plan_cost'] == expected

    res = dfs(prior_envs.f2c.for_task(0, 4))
    assert res['plan_cost'] == (
        # You visit the sides, nodes 1 or 3.
        2/3 * (
            # From those nodes, you either go to 4.
            1/2 * 3 +
            # Or you pass through the center node (2), taking the long path to 4.
            1/2 * 5
        # You visit the center, node 2.
        ) + 1/3 * (
            # From there, you can visit either side, and you'll take a 4-step path.
            4
        )
    )
    assert res['path_set_dist'].isclose(rrtd.DictDistribution({
        frozenset({0, 1}): 1/6,
        frozenset({0, 3}): 1/6,
        frozenset({0, 1, 2, 3}): 1/3,
        frozenset({0, 1, 2}): 1/6,
        frozenset({0, 3, 2}): 1/6,
    }))

    res = dfs(prior_envs.f2c.for_task(0, 2))
    assert res['plan_cost'] == (
        # From node=0
        1 +
        # node=2, p=1/2
        1/3 +
        # node=1 or 3, p=2/3
        2/3 * (
            # From node 1 or 3
            1 +
            # node=2, p=1/2
            1/2 +
            # node=4, p=1/2
            1/2 * (
                # node=2, with cost=3, from 4, 3 or 1, 2
                3 +
                # but p=1/2 you pass through 5; not going to
                # sum things here, but it takes 23 nodes in total
                # to visit the other half of the graph.
                1/2 * (
                    23
                )
            )
        )
    )
    assert res['path_set_dist'].isclose(rrtd.DictDistribution({
        frozenset({0}): 1/3,
        frozenset({0, 1}): 1/6,
        frozenset({0, 3}): 1/6,
        frozenset({0, 1, 3, 4}): 1/3,
    }))

def test_iddfs_exact():
    '''
    twostage, from 0 to 3
    depth=0
    0 | Cost=D0

    depth=1
    0,1,2 | Cost=1+2*D0, P=1/2
    0,2,1 | Cost=1+2*D0, P=1/2

    depth=2
    0,1,3 | Cost=2+G, P=1/4
    0,1,4,3 | Cost=2+D0+G, P=1/4
    0,2,{4|5},1,3 | Cost=3+2*D0+G, P=2 * 1/8
    0,2,{4|5},1,4,3 | Cost=3+3*D0+G, P=2 * 1/8
    '''
    D = rrtd.floyd_warshall(twostage)
    for kw, expected in [
        (dict(), 1 + 3 + (3+4+6+7)/4),
        (dict(add_goal_to_visited_count=False), 1 + 3 + (3+4+6+7)/4 - 1),
        # This is an important case for why add_depth0_to_visited_count should probably
        # rarely be false -- it excludes a lot of states that we would have otherwise counted.
        (dict(add_depth0_to_visited_count=False), 0 + 1 + (3+3+4+4)/4),
        (dict(add_depth0_to_visited_count=False, add_goal_to_visited_count=False), 0 + 1 + (3+3+4+4)/4 - 1),
    ]:
        s, g = 0, 3
        alg = depth_limited_dfs_analytic.iddfs_for_rrtd(twostage, recursive_dfs_exact_kwargs=kw)
        print(kw)
        assert -alg(s, g)['value'] == expected + D[s, g]

def test_sampled_dfs():
    mdp = prior_envs.f2c
    rng = random.Random(1337)
    fn = lambda *a, **k: search_algorithms.recursive_dfs(*a, **k, random=rng)

    sample_alg = rrtd.new_search_sampling_algorithm(mdp, value_fn=fn, samples=800)
    alg = depth_limited_dfs_analytic.recursive_dfs_for_rrtd(mdp)
    sample_res = sample_alg(0, 9)
    assert np.abs(alg(0, 9)['value'] - sample_res['value']) < 2 * sample_res['value_sem']

def test_sampled_iddfs():
    mdp = prior_envs.f2c
    rng = random.Random(1337)
    fn = lambda *a, **k: search_algorithms.iddfs(*a, **k, random=rng)

    sample_alg = rrtd.new_search_sampling_algorithm(mdp, value_fn=fn, samples=800)
    alg = depth_limited_dfs_analytic.iddfs_for_rrtd(mdp)
    sample_res = sample_alg(4, 6)
    assert np.abs(alg(4, 6)['value'] - sample_res['value']) < 2 * sample_res['value_sem']

def test_event_orderings():
    def assert_events_are_close(gen, expected):
        cts = {}
        for p, path, term in gen:
            key = frozenset(path), term
            cts.setdefault(key, 0)
            cts[key] += p

        assert cts.keys() == expected.keys()
        for k in cts.keys():
            assert np.isclose(cts[key], expected[key])

    for event_orderings in [
        depth_limited_dfs_analytic._event_orderings_subset_enum,
        depth_limited_dfs_analytic._event_orderings_permutations,
    ]:
        assert_events_are_close(event_orderings({0, 1, 2}, {0}), {
            (frozenset(), 0): 1/3,
            (frozenset([1]), 0): 1/6,
            (frozenset([2]), 0): 1/6,
            (frozenset([1, 2]), 0): 1/3,
        })

        assert_events_are_close(event_orderings({0, 1, 2}, {0, 1}), {
            (frozenset(), 0): 1/3,
            (frozenset(), 1): 1/3,
            (frozenset([2]), 0): 1/6,
            (frozenset([2]), 1): 1/6,
        })

        assert_events_are_close(event_orderings({0, 1, 2, 3}, {0, 1}), {
            (frozenset(), 0): 1/4,
            (frozenset(), 1): 1/4,
            (frozenset([2]), 0): 1/12,
            (frozenset([2]), 1): 1/12,
            (frozenset([3]), 0): 1/12,
            (frozenset([3]), 1): 1/12,
            (frozenset([2, 3]), 0): 2/24,
            (frozenset([2, 3]), 1): 2/24,
        })

def ppl_dfs_plancost(mdp, add_goal_to_visited_count=True, add_depth0_to_visited_count=True, depth_limit=None, random=random):
    import pythonland_ppl
    succ = mdp.successor_mapping
    ct = 0
    def dfs(path_set, state, depth_limit):
        nonlocal ct
        if mdp.is_terminal(state):
            if add_goal_to_visited_count:
                ct += 1
            return True
        if depth_limit == 0:
            if add_depth0_to_visited_count:
                ct += 1
            return False
        ct += 1
        for ns in pythonland_ppl.shuffled(succ[state] - path_set, random=random):
            if dfs(path_set | frozenset({state}), ns, depth_limit=None if depth_limit is None else depth_limit-1):
                return True
    terminal = dfs(frozenset(), mdp.initial_state(), depth_limit)
    return frozendict(
        terminal=terminal,
        plan_cost=ct,
    )

def ppl_iddfs_plancost(mdp, *, recursive_dfs_exact_kwargs={}, random=random):
    plan_cost = 0
    for distance in range(len(mdp.state_list) * 2):
        res = ppl_dfs_plancost(mdp, depth_limit=distance, **recursive_dfs_exact_kwargs, random=random)
        plan_cost += res['plan_cost']
        if res['terminal']:
            break
    return plan_cost


def test_dfs_with_ppl():
    import pythonland_ppl
    mdp = prior_envs.f2c
    D = rrtd.floyd_warshall(mdp)

    for s in mdp.state_list:
        for g in mdp.state_list[5:]:
            for kw1 in [
                # Since we have no depth_limit, we skip add_depth0
                dict(add_depth0_to_visited_count=False, add_goal_to_visited_count=True),
                dict(add_depth0_to_visited_count=False, add_goal_to_visited_count=False),
                # Avoid checking add_goal=False since it isn't reached with this depth
                dict(add_depth0_to_visited_count=True, add_goal_to_visited_count=True, depth_limit=D[s, g]-1),
                # Since we can reach the goal at this depth, we check add_goal
                dict(add_depth0_to_visited_count=True, add_goal_to_visited_count=True, depth_limit=D[s, g]),
                dict(add_depth0_to_visited_count=True, add_goal_to_visited_count=False, depth_limit=D[s, g]),
            ]:
                ppl_res = pythonland_ppl.ExecutionEnumerator.distribution(
                    lambda random=random: ppl_dfs_plancost(mdp.for_task(s, g), random=random, **kw1)['plan_cost'], deduplicate_traces=False)

                for kw in [
                    dict(event_orderings=depth_limited_dfs_analytic._event_orderings_subset_enum),
                    dict(event_orderings=depth_limited_dfs_analytic._event_orderings_permutations),
                ]:
                    res = depth_limited_dfs_analytic.recursive_dfs_exact(mdp.for_task(s, g), **kw, **kw1)
                    assert np.isclose(rrtd.expectation(rrtd.DictDistribution(ppl_res)), res['plan_cost'])

def test_iddfs_with_ppl():
    import pythonland_ppl

    def check_iddfs(mdp, ss, gs):
        D = rrtd.floyd_warshall(mdp)
        for kw1 in [
            dict(add_depth0_to_visited_count=True, add_goal_to_visited_count=True),
            dict(add_depth0_to_visited_count=True, add_goal_to_visited_count=False),
            dict(add_depth0_to_visited_count=False, add_goal_to_visited_count=True),
            dict(add_depth0_to_visited_count=False, add_goal_to_visited_count=False),
        ]:
            for kw in [
                dict(event_orderings=depth_limited_dfs_analytic._event_orderings_subset_enum),
                dict(event_orderings=depth_limited_dfs_analytic._event_orderings_permutations),
            ]:
                iddfs_alg = depth_limited_dfs_analytic.iddfs_for_rrtd(mdp, recursive_dfs_exact_kwargs=dict(**kw, **kw1))
                for s in ss:
                    for g in gs:
                        ppl_res = pythonland_ppl.ExecutionEnumerator.distribution(
                            lambda random=random: ppl_iddfs_plancost(mdp.for_task(s, g), random=random, recursive_dfs_exact_kwargs=kw1), deduplicate_traces=False)
                        plan_cost = rrtd.expectation(rrtd.DictDistribution(ppl_res))

                        res = iddfs_alg(s, g)
                        assert np.isclose(plan_cost + D[s, g], -res['value']), (s, g, plan_cost, res, D[s, g])

    check_iddfs(twostage, [0], range(1, len(twostage.state_list)))
    check_iddfs(prior_envs.f2c, [4], [6])
