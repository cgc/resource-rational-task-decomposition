import math, itertools
import rrtd
import numpy as np

def _event_orderings_permutations(events, terminal_set):
    assert any(e in terminal_set for e in events), 'Some event must be in the terminal set.'

    # We combine the results from all ordering of events
    p_perm = 1/math.factorial(len(events))

    for ordering in itertools.permutations(events):
        # For a given permutation of events, we identify the events that happen before a terminal
        # event is encountered.
        for terminal_idx, e in enumerate(ordering):
            if e in terminal_set:
                break
        yield p_perm, ordering[:terminal_idx], e


def _event_orderings_subset_enum(events, terminal_set):
    '''
    This attempts to optimize the above by avoiding enumerating all permutations and instead
    enumerating all subsets of non-terminal events, with appropriate weight.
    '''
    assert any(e in terminal_set for e in events), 'Some event must be in the terminal set.'

    nonterminal = events - terminal_set

    p_subset_len = 1
    for subset_len in range(len(nonterminal)+1):
        # The probability of each combination as the prefix of a permutation of the events. This is conditioned on
        # subset length, so sums to one for a given subset length.
        p_comb = 1/math.comb(len(nonterminal), subset_len)
        remaining = len(events) - subset_len
        # The probability of each individual terminal item
        p_terminal = 1/remaining

        for combination in itertools.combinations(nonterminal, subset_len):
            for terminal in terminal_set:
                yield p_subset_len * p_comb * p_terminal, combination, terminal

        # Update subset length as though we drew some non-terminal item
        p_subset_len *= (remaining-len(terminal_set))/remaining


def recursive_dfs_exact(
    mdp, *,
    debug=False,
    depth_limit=None,
    add_goal_to_visited_count=True,
    add_depth0_to_visited_count=True,
    event_orderings=_event_orderings_permutations,
):
    '''
    This function takes an expectation over all possible DFS execution traces.
    '''

    # Caching successors here.
    if hasattr(mdp, 'successor_mapping'):
        succ_map = mdp.successor_mapping
    else:
        succ_map = {
            s: frozenset({mdp.next_state(s, a) for a in mdp.actions(s)})
            for s in mdp.state_list}

    def _dfs(path_set, state, depth, space=''):
        if debug: print(space, f'dfs({set(path_set)}, curr={state})')

        # If we have reached a terminal state, we return this distribution over paths.
        if mdp.is_terminal(state):
            return 1. if add_goal_to_visited_count else 0., True, rrtd.DictDistribution.deterministic(path_set)

        # If we reach our depth limit, we end our search.
        if depth == 0:
            return 1. if add_depth0_to_visited_count else 0., False, None

        next_path_set = path_set | frozenset({state})
        # We consider successors that haven't been visited along our path
        succ = succ_map[state] - next_path_set

        # We recurse for each of these successors.
        res = {
            ns: _dfs(next_path_set, ns, depth=depth-1, space=space+' ')
            for ns in succ
        }
        if debug:
            for ns, v in res.items():
                print(space, f'from child {ns}: {v}')

        # If any are terminal, this means that we will also be terminal.
        terminal_set = {ns for ns, (_, terminal, _) in res.items() if terminal}

        # If none are terminal, then we can simply exit early without considering permutations of children.
        if not terminal_set:
            return 1+sum(cost for cost, _, _ in res.values()), False, None

        # We combine the results from our successors, considering every permutation.
        total = 0
        totalp = 0
        total_path_set_dist = rrtd.DictDistribution()
        for p_perm, nss, ns_term in event_orderings(succ, terminal_set):
            totalp += p_perm
            # For a given ordering, we sum the costs of successors until
            # we reach a successor that leads us to a terminal state.
            # We compute a distribution over path sets by combining the results
            # from successors that lead to terminal states.
            perm_total = 0
            for ns in nss:
                cost, terminal, path_set_dist = res[ns]
                perm_total += cost
                assert not terminal and path_set_dist is None

            # Handle terminal separately
            cost, terminal, path_set_dist = res[ns_term]
            perm_total += cost
            assert terminal
            total_path_set_dist = total_path_set_dist | path_set_dist * p_perm

            total += p_perm * perm_total
        assert np.isclose(totalp, 1)
        # We add one to the total cost here to account for this call.
        return 1+total, bool(terminal_set), total_path_set_dist

    # 2x is probably excessive but is correct
    dl = depth_limit if depth_limit is not None else 2*len(mdp.state_list)
    cost, terminal, path_set_dist = _dfs(frozenset(), mdp.initial_state(), dl)
    if terminal:
        assert path_set_dist is not None
        assert np.isclose(sum(path_set_dist.values()), 1)
    else:
        assert path_set_dist is None
    return dict(
        plan_cost=cost,
        path_set_dist=path_set_dist,
        terminal=terminal,
    )

def recursive_dfs_for_rrtd(mdp, add_goal_to_visited_count=True):
    # Precaching
    mdp.successor_mapping # This one is pretty specific to this algorithm
    mdp.transition_matrix
    mdp.reward_matrix

    def algorithm(s, g):
        res = recursive_dfs_exact(mdp.for_task(s, g), add_goal_to_visited_count=add_goal_to_visited_count)
        cost = res['plan_cost'] + rrtd.expectation(res['path_set_dist'], lambda ps: len(ps))
        return dict(value=-cost)
    return algorithm

def iddfs_for_rrtd(mdp, *, plan_cost_scale=1., recursive_dfs_exact_kwargs={}, cache_non_terminal_plan_cost=False):
    # Precaching
    mdp.successor_mapping # This one is pretty specific to this algorithm
    mdp.transition_matrix
    mdp.reward_matrix

    # Since IDDFS returns optimal paths, we can use this to precompute them.
    distance_matrix = rrtd.floyd_warshall(mdp)
    plan_cost_cache = {}
    def algorithm(s, g):
        plan_cost_cache.setdefault(s, {})

        distance = int(distance_matrix[s, g])
        plan_cost = 0
        for d in range(distance+1):
            if cache_non_terminal_plan_cost and d != distance and d in plan_cost_cache[s]:
                plan_cost += plan_cost_cache[s][d]
                continue
            res = recursive_dfs_exact(mdp.for_task(s, g), depth_limit=d, **recursive_dfs_exact_kwargs)
            plan_cost += res['plan_cost']
            if d == distance:
                # Assertions to make sure we found a plan with the expected depth limit.
                assert res['path_set_dist'] is not None
                assert all(len(ps)==distance for ps in res['path_set_dist'].support)
            else:
                # When we haven't reached the expected depth limit, we should have no solutions,
                # and the search cost should match previously computed amounts (in plan_cost_cache).
                assert res['path_set_dist'] is None
                if d in plan_cost_cache[s]:
                    assert res['plan_cost'] == plan_cost_cache[s][d]
                else:
                    plan_cost_cache[s][d] = res['plan_cost']

        total_cost = plan_cost_scale * plan_cost + distance
        return dict(value=-total_cost)
    return algorithm
