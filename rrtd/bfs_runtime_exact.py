#!/usr/bin/env python
# coding: utf-8

import rrtd
import itertools, math, collections, numpy as np
from frozendict import frozendict
from msdm.core.utils.funcutils import method_cache

def deduplicate_trace_deco(fn):
    cache = {}
    def wrapped(q, visited):
        #key = hash((tuple(q), visited.efficient_hash() if isinstance(visited, bitset) else frozenset(visited)))
        key = hash((tuple(q), frozenset(visited)))
        if key not in cache:
            cache[key] = fn(q, visited)
        return cache[key]
    return wrapped

def bfs_exact(mdp, *, debug=False, deduplicate_traces=False):
    succ_map = {
        s: [mdp.next_state(s, a) for a in mdp.actions(s)]
        for s in mdp.state_list
    }
    info = {'leaf_count': 0}
    def recur(q, visited):
        current = q.popleft()
        if debug: print('curr', current, 'q', q, 'visi', visited, mdp.is_terminal(current))
        assert current not in visited
        assert current not in q
        visited.add(current)

        if mdp.is_terminal(current):
            res = len(visited)
            info['leaf_count'] += 1
            if debug: print('hi res', res)
        else:
            succ = [ns for ns in succ_map[current] if (ns not in q and ns not in visited)]
            if succ:
                res = 0
                p_perm = 1/math.factorial(len(succ))
                for els in itertools.permutations(succ): # every ordering of successors that isn't in visited or q
                    if debug: print('perm', els)
                    q.extend(els)

                    subres = recur(q, visited)
                    res += p_perm * subres

                    # rewinding
                    for el in els[::-1]:
                        assert q.pop() == el
            else:
                res = recur(q, visited)

        # rewinding
        assert current in visited
        assert current not in q
        visited.remove(current)
        q.appendleft(current)

        return res

    if deduplicate_traces:
        recur = deduplicate_trace_deco(recur)

    #initial_visited = bitset(limit=len(mdp.state_list))
    initial_visited = set()
    return dict(
        expected_cost=recur(collections.deque([mdp.initial_state()]), initial_visited),
        info=info,
    )



# # One last optimization
# In this final attempt at optimizing, I'll try to have one single breadth-first pass for all goals.



def bfs_exact_allgoals(mdp, *, debug=False, deduplicate_traces=False):
    # We assume the states are indices into an array
    assert sorted(mdp.state_list) == list(range(len(mdp.state_list)))

    succ_map = {
        s: [mdp.next_state(s, a) for a in mdp.actions(s)]
        for s in mdp.state_list
    }

    num_states = len(mdp.state_list)

    # Return value for leaf nodes. It is read-only.
    CONST_ZEROS = np.zeros(num_states)
    CONST_ZEROS.flags.writeable = False

    def recur(q, visited):
        if not q:
            return CONST_ZEROS

        current = q.popleft()
        if debug: print('curr', current, 'q', q, 'visi', visited, mdp.is_terminal(current))
        assert current not in visited
        assert current not in q
        visited.add(current)

        res = np.zeros(num_states)
        res[current] = len(visited)

        succ = [ns for ns in succ_map[current] if (ns not in q and ns not in visited)]
        if succ:
            p_perm = 1/math.factorial(len(succ))
            for els in itertools.permutations(succ): # every ordering of successors that isn't in visited or q
                if debug: print('perm', els)
                q.extend(els)

                subres = recur(q, visited)
                res += p_perm * subres

                # rewinding
                for el in els[::-1]:
                    assert q.pop() == el
        else:
            res += recur(q, visited)

        # rewinding
        assert current in visited
        assert current not in q
        visited.remove(current)
        q.appendleft(current)

        return res

    if deduplicate_traces:
        recur = deduplicate_trace_deco(recur)

    #initial_visited = bitset(limit=len(mdp.state_list))
    initial_visited = set()
    return recur(collections.deque([mdp.initial_state()]), initial_visited)

def make_bfs_runtime_exact_algorithm(mdp, *, add_goal_to_visited_count=False, plan_cost_scale=1.):
    distance = rrtd.floyd_warshall(mdp)
    cache = {}
    def algorithm(s, g):
        if s not in cache:
            cache[s] = bfs_exact_allgoals(mdp.for_task(s, s), deduplicate_traces=True)
        adjust = 0 if add_goal_to_visited_count else -1
        plan_cost = adjust + cache[s][g]
        return dict(value=-(plan_cost_scale * plan_cost + distance[s, g]))
    return algorithm
