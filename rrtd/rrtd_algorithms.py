from msdm.algorithms.valueiteration import ValueIteration
import hitting_time
import scipy.stats
import numpy as np

def new_value_iteration_algorithm(mdp, map_state_to_mdp=lambda x: x):
    '''
    We permit the optional specification of a state projection that is used when accessing the results object.
    '''
    vi = ValueIteration()
    cache = {}

    def algorithm(start, goal):
        if goal not in cache:
            # To solve the task for all goals, we set the start state to None to
            # signify the full distribution of states is required.
            arbitrary_state = start
            task_mdp = mdp.for_task(arbitrary_state, goal)
            cache[goal] = vi.plan_on(task_mdp)
            assert cache[goal].converged
        return dict(
            value=cache[goal].V[map_state_to_mdp(start)],
            result=cache[goal],
        )

    algorithm.cache = cache
    return algorithm

def new_random_walk_algorithm(mdp, *, hitting_time_fn=hitting_time.hitting_time_inv):
    from rrtd import binary_adjacency_ssp
    adjacency = binary_adjacency_ssp(mdp, terminal_absorbing=False)
    htcache = {}
    def algorithm(start, goal):
        if goal not in htcache:
            A = hitting_time.with_absorbing_state(adjacency, goal)
            P = A / hitting_time.outdegree(A)[:, None]
            htcache[goal] = hitting_time_fn(P)
        v = -htcache[goal][start]
        return dict(value=v)
    return algorithm

def new_random_walk_algorithm_spectral(mdp, *, eigenvector_limit=None):
    from rrtd import binary_adjacency_ssp
    '''
    eigenvector_limit is used to keep the highest-eigenvalue vectors.
    So, for a rank-1 approximation like Q-cut, you'd choose eigenvector_limit=2,
    since the top eigenvector (with value of 1) is not included in the computation
    of hitting time.
    '''
    adjacency = binary_adjacency_ssp(mdp, terminal_absorbing=False)
    evals, evecs = hitting_time.sorted_eig(hitting_time.lovasz_N(adjacency))

    if eigenvector_limit is not None:
        # HACK this is a brittle way to do this. -> instead prefer explicitly specifying range, or having arg to spectral access?
        # brittle in that it assumes about spectral accss
        evals = evals[:eigenvector_limit]

    def algorithm(start, goal):
        return dict(value=-hitting_time._spectral_access(adjacency, start, goal, evals, evecs))
    return algorithm


# This is an algorithm for any search process you can sample from. The expected API is
# a bit of an unintentional one, but hopefully somewhat clear.
def new_search_sampling_algorithm(mdp, *, value_fn, samples=1000):
    cache = {}
    sample_history = np.zeros(samples)
    def algorithm(start, goal):
        if (start, goal) not in cache:
            for i in range(samples):
                # HACK HACK figure out API here??? are start/goal args or in MDP?
                r = value_fn(mdp.for_task(start, goal))
                sample_history[i] = (r['plan_cost'] + r['distance'])
                #total_cost += (r['plan_cost'] + r['distance'])
            total_cost = np.mean(sample_history)
            #total_cost /= samples
            #print(start, goal, total_cost)

            cache[start, goal] = dict(
                value=-total_cost,
                value_sem=scipy.stats.sem(sample_history),
            )
        return cache[start, goal]
    return algorithm
