import warnings
from msdm.algorithms import ValueIteration
import rrtd
import numpy as np

def enumerate_paths(mdp, policy, *, state=None):
    if state is None:
        state = mdp.initial_state()
    if mdp.is_terminal(state):
        yield (state,)
        return
    for action in policy.action_dist(state).support:
        ns = mdp.next_state(state, action)
        for ps in enumerate_paths(mdp, policy, state=ns):
            yield (state,) + ps

def make_path_generator_vi(mdp, task_distribution):
    policies_by_goal = {
        goal: ValueIteration().plan_on(mdp.for_task(goal, goal)).policy
        for goal in {t['goal'] for t in task_distribution.support}
    }
    def paths(task):
        mm = mdp.for_task(task['start'], task['goal'])
        yield from enumerate_paths(mm, policies_by_goal[task['goal']])
    return paths

def make_path_generator_distance(mdp, task_distribution):
    D = rrtd.floyd_warshall(mdp)

    def enump(D, mdp, s, g):
        if s == g:
            yield (g,)
            return

        # get list of next states
        nss = [mdp.next_state(s, a) for a in mdp.actions(s)]
        # get value of successor closest to goal (i.e. the best next option)
        mindist = min(D[ns, g] for ns in nss)
        # now iterate over neighbors, recursing on ones that are along optimal path.
        for ns in nss:
            if D[ns, g] != mindist:
                continue
            for p in enump(D, mdp, ns, g):
                yield (s,)+p
    return lambda t: enump(D, mdp, t['start'], t['goal'])

def betweenness_centrality(
    mdp, *,
    endpoints=False,
    include_start=False,
    include_goals=False,
    # Ways of configuring the task distribution
    task_distribution=None,
    remove_successor_tasks=False, remove_trivial_tasks=True,
    # Should the counts be normalized by all the possible paths the state could have appeared in?
    # Notably, when endpoints/include_start/include_goals are False, they will affect the term
    # used for normalization.
    normalized=True,
    make_path_generator=make_path_generator_distance,
):
    warnings.warn('deprecated', DeprecationWarning)
    def valid_for_task(task, s):
        '''
        it is true that:
        if endpoints: return True
        '''
        if s == t['start'] and not (endpoints or include_start):
            return False
        if s == t['goal'] and not (endpoints or include_goals):
            return False
        return True

    if task_distribution is None:
        task_distribution = rrtd.all_pairs_shortest_path_distribution(
            mdp, remove_successor_tasks=remove_successor_tasks, remove_trivial_tasks=remove_trivial_tasks)
    # Currently implementing uniform distributions only.
    # for non-uniform, I think we'll need to accumulate task probabilities & then weight presence on paths by task probability
    assert all(np.isclose(p, 1/len(task_distribution.support)) for event, p in task_distribution.items()), 'This function requires uniform distributions.'

    pathgen = make_path_generator(mdp, task_distribution)

    counts = np.zeros(len(mdp.state_list))
    if normalized:
        valid_task_count = np.zeros(len(mdp.state_list))

    for t, prob in task_distribution.items():
        paths = list(pathgen(t))
        for path in paths:
            for s in path:
                if valid_for_task(t, s):
                    counts[s] += 1/len(paths)

        if normalized:
            for s in mdp.state_list:
                if valid_for_task(t, s):
                    valid_task_count[s] += 1

    # to replicate normalized=False or normalized=True
    if normalized:
        return counts / valid_task_count
    else:
        return counts

def optimal_occupancy(
    mdp, *,
    task_distribution=None,
):
    warnings.warn('deprecated', DeprecationWarning)
    if task_distribution is None:
        task_distribution = rrtd.all_pairs_shortest_path_distribution(mdp, remove_successor_tasks=True)

    pathgen = make_path_generator_distance(mdp, task_distribution)

    counts = np.zeros(len(mdp.state_list))
    for task, task_prob in task_distribution.items():
        paths = list(pathgen(task))
        # We assume uniform over the paths
        p = task_prob/len(paths)
        for path in paths:
            for s in path:
                counts[s] += p
    return counts / counts.sum()
