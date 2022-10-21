import numpy as np
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, DeterministicShortestPathProblem
from msdm.core.distributions import DictDistribution
from msdm.core.utils.funcutils import cached_property
import functools
from frozendict import frozendict
import warnings
import plotting
import tools
import rrtd_algorithms

# Some MDP classes we're going to work with.

import abc
class TaskInstanceMixin(DeterministicShortestPathProblem, abc.ABC):
    def __init__(self):
        self.is_task_instance = False

    @property
    @abc.abstractmethod
    def state_list(self):
        '''
        Classes that use this mixin should provide a list of states. This is useful for any kind
        of analysis of the task distribution that is independent of the start and terminal states.
        For example, this state_list is used to ensure that a solution to value iteration
        for some goal can be applied for any start state.
        '''
        pass

    def initial_state(self):
        if not self.is_task_instance:
            raise ValueError('Must specify start/goal with for_task before using task instance.')
        return self._initial_state

    def is_terminal(self, s):
        if not self.is_task_instance:
            return False
        return s == self.goal_state

    def for_task(self, start, goal):
        assert not getattr(self, 'is_task_instance', False), 'Cannot make task instance from a task instance.'
        import copy
        inst = copy.copy(self)
        inst._initial_state = start
        inst.goal_state = goal
        inst.is_task_instance = True
        return inst

    @classmethod
    def is_task_distribution(cls, mdp):
        '''
        A duck-typed definition to avoid issues with jupyter reloading.
        '''
        return callable(getattr(mdp, 'for_task'))

    @cached_property
    def successor_mapping(self):
        '''
        While not directly related to TaskInstanceMixin, this is an
        otherwise very convenient place to add this method, which is
        really a performance-motivated method to avoid actions()
        and next_state() while making successor membership checks fast (by using set).
        Only appropriate for DeterministicShortestPathProblem.

        Returns a dictionary mapping states to frozensets of successors.
        '''
        return frozendict({
            s: frozenset({self.next_state(s, a) for a in self.actions(s)})
            for s in self.state_list
        })

class Graph(TaskInstanceMixin, DeterministicShortestPathProblem, TabularMarkovDecisionProcess):
    def __init__(self, adjacency):
        super().__init__()
        self.adjacency = adjacency

    @property
    @functools.lru_cache(maxsize=None)
    def state_list(self):
        return sorted(self.adjacency.keys())

    def actions(self, s):
        return range(len(self.adjacency[s]))

    def next_state(self, s, a):
        return self.adjacency[s][a]

    def reward(self, s, a, ns):
        return -1

    def __repr__(self):
        return str(f'Graph({repr(self.adjacency)})')

    @classmethod
    def from_binary_adjacency(cls, binadj):
        assert len(binadj.shape) == 2
        return cls({
            idx: np.where(binary_ns)[0]
            for idx, binary_ns in enumerate(binadj)
        })

class OptionLevelMDP(TaskInstanceMixin, DeterministicShortestPathProblem, TabularMarkovDecisionProcess):
    def __init__(self, mdp, algorithm, subgoals, *, force_subgoal_use=False, no_cost_subgoals=False, only_travel_cost_subgoals=False):
        self.mdp = mdp
        self.algorithm = algorithm
        self.subgoals = list(subgoals)
        self.force_subgoal_use = force_subgoal_use
        self.no_cost_subgoals = no_cost_subgoals
        self.only_travel_cost_subgoals = only_travel_cost_subgoals
        if only_travel_cost_subgoals:
            self.distance_matrix = floyd_warshall(mdp)

    # Need to include this for VI and other algorithms to operate before we've made task instances.
    @property
    def state_list(self):
        return self.mdp.state_list

    def actions(self, s):
        if self.force_subgoal_use and s not in self.subgoals and self.subgoals:
            return self.subgoals
        return self.subgoals + [self.goal_state]

    def next_state(self, s, a):
        return a

    def reward(self, s, a, ns):
        # Since we define self-visits as automatically satisfied (hit time/reward/cost=0)
        # we need a way to disincentivize them. This helps avoid bugs like what affected
        # previous buggy simulation results with DFS.
        if s == a:
            return float('-inf')
        if self.no_cost_subgoals and a in self.subgoals:
            return 0
        if self.only_travel_cost_subgoals and a in self.subgoals:
            return -self.distance_matrix[s, a]
        return self.algorithm(s, a)['value']


def binary_adjacency_ssp(mdp, terminal_absorbing=False):
    adjacency = np.zeros((len(mdp.state_list), len(mdp.state_list)), dtype=bool)
    for s in mdp.state_list:
        if terminal_absorbing and mdp.is_terminal(s):
            adjacency[s, s] = 1
            continue
        for a in mdp.actions(s):
            adjacency[s, mdp.next_state(s, a)] = 1
    return adjacency

def adjacency_ssp(mdp, terminal_absorbing=False):
    adjacency = {}
    for s in mdp.state_list:
        if terminal_absorbing and mdp.is_terminal(s):
            adjacency[s] = [s]
            continue
        adjacency[s] = [mdp.next_state(s, a) for a in mdp.actions(s)]
    return adjacency


# Rendering routines

_state_arg_to_callable = plotting._state_arg_to_callable
edge_penwidth_map = plotting.edge_penwidth_map
plot_graph = plotting.plot_graph
display_graphs = plotting.display_graphs

# Distribution utilities

marginalize = tools.marginalize
dist_prod = tools.dist_prod
expectation = tools.expectation
dist_condition = tools.dist_condition
Multinomial_from_probability_dict = DictDistribution

# Backwards-compatibility
def Multinomial(support, *, probs=None):
    assert not isinstance(support, dict)
    if probs is None:
        return DictDistribution.uniform(support)
    else:
        return DictDistribution({s: p for s, p in zip(support, probs)})

# Algorithms

new_value_iteration_algorithm = rrtd_algorithms.new_value_iteration_algorithm
new_random_walk_algorithm = rrtd_algorithms.new_random_walk_algorithm
new_random_walk_algorithm_spectral = rrtd_algorithms.new_random_walk_algorithm_spectral
new_search_sampling_algorithm = rrtd_algorithms.new_search_sampling_algorithm

# some aliases for backwards compatibility
new_sampling = new_search_sampling_algorithm
value_iteration = new_value_iteration_algorithm

# Utilities

def floyd_warshall(mdp):
    '''
    Utility to compute distance matrix for an MDP.

    https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
    '''
    dist = np.full((len(mdp.state_list), len(mdp.state_list)), np.inf)
    for s in mdp.state_list:
        for a in mdp.actions(s):
            ns = mdp.next_state(s, a)
            dist[s, ns] = -mdp.reward(s, a, ns)
        dist[s, s] = 0
    for k in mdp.state_list:
        for i in mdp.state_list:
            for j in mdp.state_list:
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist

def all_pairs_shortest_path_distribution(mdp, remove_trivial_tasks=True, remove_successor_tasks=True):
    '''
    Trivial tasks are defined as tasks with the same start and end state.
    '''
    td = dist_prod(
        Multinomial([dict(start=s) for s in mdp.state_list]),
        Multinomial([dict(goal=s) for s in mdp.state_list]),
    )
    if remove_trivial_tasks:
        td = dist_condition(td, lambda task: task['start'] != task['goal'])
    if remove_successor_tasks:
        successors = {
            s: {mdp.next_state(s, a) for a in mdp.actions(s)}
            for s in mdp.state_list
        }
        td = dist_condition(td, lambda task: task['goal'] not in successors[task['start']])
    return td

# The big function

def task_decomposition(
    mdp, task_distribution, make_algorithm,
    *,
    make_option_level_algorithm=value_iteration,
    subgoal_sets=None,
    num_options=1,
    include_no_subgoals=True,
    cache_algorithm=True,
    tqdm=lambda x: x,
    OptionLevelMDP_kwargs={},
    OptionLevelMDP=OptionLevelMDP,
):
    # Pre-caching here.
    mdp.transition_matrix
    mdp.reward_matrix
    mdp.action_matrix

    assert TaskInstanceMixin.is_task_distribution(mdp)

    import itertools
    subgoal_sets = subgoal_sets or list(itertools.combinations(range(len(mdp.state_list)), num_options))
    if include_no_subgoals and all(subgoal_sets):
        subgoal_sets = list(subgoal_sets) + [set()]

    import functools
    # HACK HACK do we even need this caching if we are using VecVI?
    # HACK since we initialize a new mdp every time maybe?
    algorithm = make_algorithm(mdp)
    if cache_algorithm:
        algorithm = functools.lru_cache(maxsize=None)(algorithm)

    rv = []
    for subgoals in tqdm(subgoal_sets):
        ol_algorithm = make_option_level_algorithm(OptionLevelMDP(mdp, algorithm, subgoals, **OptionLevelMDP_kwargs))
        rv.append(dict(
            subgoals=subgoals,
            value=expectation(task_distribution, lambda task: ol_algorithm(task['start'], task['goal'])['value']),
        ))
    return sorted(rv, key=lambda row: -row['value'])

# Utilities

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def show_task_decomposition(mdp, res, *, invtemp=1, **kw):
    warnings.warn("show_task_decomposition is deprecated", DeprecationWarning)
    zz = res_to_arr(mdp, res)
    return plot_graph(mdp, z=softmax(zz*invtemp), **kw)

def res_to_arr(mdp, res):
    arr = np.zeros(len(mdp.state_list))
    for el in res:
        if not el['subgoals']:
            continue
        assert len(el['subgoals']) == 1
        arr[el['subgoals'][0]] = el['value']
    return arr

def argmaxes(arr):
    '''
    While an argmax returns the index of the first or some arbitrary largest element, this method
    returns the indices of those that are close to the max value.

    >>> argmaxes([1, 2, 3, 3, 3.0000000000001, -4]).tolist()
    [2, 3, 4]
    '''
    _max = np.max(arr)
    return np.where(np.isclose(_max, arr))[0]
