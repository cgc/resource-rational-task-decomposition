import collections
import random
import heapq
from frozendict import frozendict

def reconstruct_path(start, end, camefrom):
    p = [end]
    while p[-1] != start:
        p.append(camefrom[p[-1]])
    return p[::-1]

def bfs(mdp, *, shuffle_actions=True, visited_init=frozenset(), add_goal_to_visited_count=False, random=random):
    queue = collections.deque()
    queue.append(mdp.initial_state())
    visited = set(visited_init)
    camefrom = {}
    while queue:
        current = queue.popleft()
        if mdp.is_terminal(current):
            p = reconstruct_path(mdp.initial_state(), current, camefrom)
            return frozendict(
                visited=frozenset(visited),
                frontier=frozenset(queue),
                path=tuple(p),
                cost=len(p)-1,
                # API
                distance=len(p)-1,
                plan_cost=len(visited) + (1 if add_goal_to_visited_count else 0),
            )
        visited.add(current)
        actions = list(mdp.actions(current))
        if shuffle_actions:
            random.shuffle(actions)
        for a in actions:
            ns = mdp.next_state(current, a)
            if ns not in visited and ns not in queue:
                camefrom[ns] = current
                queue.append(ns)

def dfs(mdp, *, shuffle_actions=True, append_queue_entries=False, return_path=False, visited_init=frozenset(), add_goal_to_visited_count=False):
    '''
    append_queue_entries: When true, nodes are appended to the queue, even if
    they're in the queue already. This might use more memory, but can ensure
    a promising node encountered early (but not first) isn't ignored.
    '''
    visited = set(visited_init)
    queue = [mdp.initial_state()]
    distance = {mdp.initial_state(): 0}
    come_from = {}

    while queue:
        s = queue.pop()
        if s in visited:
            continue
        if mdp.is_terminal(s):
            r = dict(
                distance=distance[s],
                plan_cost=len(visited) + (1 if add_goal_to_visited_count else 0),
            )
            if return_path:
                r['path'] = tuple(reconstruct_path(mdp.initial_state(), s, come_from))
                # HACK
                r['visited'] = frozenset(visited)
            return r
        visited.add(s)
        actions = list(mdp.actions(s))
        if shuffle_actions:
            random.shuffle(actions)
        for a in actions:
            ns = mdp.next_state(s, a)
            if (append_queue_entries or ns not in queue) and ns not in visited:
                queue.append(ns)
                distance[ns] = distance[s] + 1
                if return_path:
                    come_from[ns] = s

def deterministic_search(
    mdp, *,
    shuffle_actions=True,
    visited_init=frozenset(),
    queue_order=None,
    heuristic_cost=None,
    algorithm=None,
    debug=False,
    random=random,
):
    cost_prioritization = True
    assert algorithm in (None, 'a*', 'dfs', 'bfs', 'ucs')
    if algorithm == 'a*':
        queue_order = queue_order or 'lifo'
        assert heuristic_cost is not None
    elif algorithm == 'ucs':
        # Slight preference for lifo since it should minimize heap usage.
        queue_order = queue_order or 'lifo'
    elif algorithm == 'dfs':
        queue_order = 'lifo'
        cost_prioritization = False
    elif algorithm == 'bfs':
        queue_order = 'fifo'
        cost_prioritization = False

    if cost_prioritization:
        def not_in_queue(next_state, f_next_state):
            return all(f_next_state < f for (f, _, g, state) in queue if state == next_state)
    else:
        def not_in_queue(next_state, f_next_state):
            return not any(state == next_state for (f, _, g, state) in queue)

    if not cost_prioritization:
        f_fn = lambda state, g: 0
    elif heuristic_cost is None:
        f_fn = lambda state, g: g
    else:
        f_fn = lambda state, g: g + heuristic_cost(state)

    assert queue_order in ('fifo', 'lifo')
    inc = +1 if queue_order == 'fifo' else -1

    queue_ct = 0
    queue = [] # [(f(state), queue #, g(state), state)]
    heapq.heappush(queue, (f_fn(mdp.initial_state(), 0), queue_ct, 0, mdp.initial_state()))
    queue_ct += inc

    visited = set(visited_init)
    camefrom = {}

    while queue:
        f, _, g, current = heapq.heappop(queue)
        if debug:
            print('Current node', current, 'f', f, 'g', g)
        if current in visited: # Need to do this since we might queue state twice for A* or DFS
            continue
        if mdp.is_terminal(current):
            if debug:
                for el in queue:
                    print(queue)
            path = reconstruct_path(mdp.initial_state(), current, camefrom)
            return frozendict(
                cost=g,
                visited=frozenset(visited),
                frontier=frozenset((el[-1] for el in queue)),
                path=tuple(path),
                # API
                distance=len(path)-1,
                plan_cost=len(visited),# HACK + (1 if add_goal_to_visited_count else 0),
            )

        visited.add(current)
        actions = list(mdp.actions(current))
        if shuffle_actions:
            random.shuffle(actions)
        for a in actions:
            ns = mdp.next_state(current, a)
            g_ns = g - mdp.reward(current, a, ns)
            f_ns = f_fn(ns, g_ns)

            if debug:
                print('\tSuccessors', ns, 'f', f_ns, 'g', g_ns, 'will add', ns not in visited and not_in_queue(ns, f_ns))
            if ns not in visited and not_in_queue(ns, f_ns):
                camefrom[ns] = current
                heapq.heappush(queue, (f_ns, queue_ct, g_ns, ns))
                queue_ct += inc

def recursive_dfs(mdp, *, debug=False, shuffle_actions=True, random=random):
    info = dict(call_count=0)
    next_states = {
        s: [mdp.next_state(s, a) for a in mdp.actions(s)]
        for s in mdp.state_list}

    current_path_set = set()

    def _dfs(state):
        if debug: print('\tstate', state)
        info['call_count'] += 1

        if mdp.is_terminal(state):
            return [state]

        current_path_set.add(state)

        if shuffle_actions:
            random.shuffle(next_states[state])

        for next_state in next_states[state]:
            # By default, we prevent cycles by retaining the current path.
            # This uses O(depth) memory, since the path can be at most that length.
            if next_state in current_path_set:
                continue
            result = _dfs(next_state)
            if result is not None:
                return [state]+result

        current_path_set.remove(state)
        return None

    result = _dfs(mdp.initial_state())
    if result is not None:
        return frozendict(
            path=result,
            # API
            distance=len(result)-1,
            plan_cost=info['call_count'],
        )


def iddfs(mdp, *, debug=False, shuffle_actions=True, depth_start=0, random=random, state_visit_cb=None):
    '''
    Iterative-deepening DFS
    https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
    '''
    info = dict(dls_call_count=0)
    next_states = {
        s: [mdp.next_state(s, a) for a in mdp.actions(s)]
        for s in mdp.state_list
    }

    current_path_set = set()

    def dls(state, depth):
        if debug: print('\tstate', state, 'depth', depth)
        info['dls_call_count'] += 1
        if state_visit_cb is not None:
            state_visit_cb(state)

        if depth == 0:
            found = [state] if mdp.is_terminal(state) else None
            return found, True

        current_path_set.add(state)

        any_remaining = False

        if shuffle_actions:
            random.shuffle(next_states[state])
        for next_state in next_states[state]:
            # By default, we prevent cycles by retaining the current path.
            # This uses O(depth) memory, since the path can be at most that length.
            if next_state in current_path_set:
                continue
            found, remaining = dls(next_state, depth-1)
            if found is not None:
                return [state]+found, True # does the true matter here?
            if remaining:
                any_remaining = True
        current_path_set.remove(state)
        return None, any_remaining

    for depth in range(depth_start, len(mdp.state_list)):
        if debug: print(f'depth limit {depth}')
        found, remaining = dls(mdp.initial_state(), depth)
        if found is not None:
            return frozendict(
                path=found,
                # API
                distance=len(found)-1,
                plan_cost=info['dls_call_count'],
            )
        elif not remaining:
            return

def iddfs_experimental(mdp, *, debug=False, shuffle_actions=True, avoid_one_step_backtrack=True, track_visited=False, prevent_cycles=False):
    '''
    Iterative-deepening DFS
    https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
    '''
    info = dict(dls_call_count=0)
    next_states = {
        s: [mdp.next_state(s, a) for a in mdp.actions(s)]
        for s in mdp.state_list
    }

    if track_visited:
        visited = {}

    def dls(prev_state, state, depth):
        if track_visited:
            visited[state] = depth
        if debug: print('\tstate', state, 'depth', depth)
        info['dls_call_count'] += 1
        if depth == 0:
            found = [state] if mdp.is_terminal(state) else None
            return found, True
        any_remaining = False

        if shuffle_actions:
            random.shuffle(next_states[state])

        for next_state in next_states[state]:
            if avoid_one_step_backtrack and next_state == prev_state:
                continue
            if track_visited and next_state in visited and visited[next_state] >= depth-1:
                continue
            found, remaining = dls(state, next_state, depth-1)
            if found is not None:
                return [state]+found, True
            if remaining:
                any_remaining = True
        return None, any_remaining

    if prevent_cycles:
        current_path_set = set()
        def deco(fn):
            def wrapped(prev_state, state, depth):
                if state in current_path_set:
                    return None, False # annoying way to do this? be nicer to continue in loop above
                current_path_set.add(state)
                rv = fn(prev_state, state, depth)
                current_path_set.remove(state)
                return rv
            return wrapped
        dls = deco(dls)

    for depth in range(len(mdp.state_list)):
        if track_visited:
            visited.clear() # We clear the set of visited nodes at each depth.
        if debug: print(f'depth limit {depth}')
        found, remaining = dls(None, mdp.initial_state(), depth)
        if found is not None:
            return frozendict(
                path=found,
                # API
                distance=len(found)-1,
                plan_cost=info['dls_call_count'],
            )
        elif not remaining:
            return
