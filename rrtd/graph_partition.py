import rrtd, solway_objective, hitting_time
import numpy as np
import shannon
import scipy.special

def ordpair(a, b):
    '''
    Given two elements, returns them in a tuple in sorted order.

    >>> assert ordpair(3, 4) == (3, 4)
    >>> assert ordpair(4, 3) == (3, 4)
    '''
    if a < b:
        return a, b
    return b, a

def edges(g):
    '''
    >>> g0 = rrtd.Graph({ 0: [1, 2], 1: [0, 2], 2: [0, 1] })
    >>> assert edges(g0) == {(0, 1), (0, 2), (1, 2)}
    '''
    return {
        ordpair(s, g.next_state(s, a))
        for s in g.state_list
        for a in g.actions(s)
    }


def graphenum(g, *, edges=None, mapped=None, forbidden=None):
    '''
    Enumerate all connected partitions of a graph.

    Implementation vaguely inspired by recent publications that try to make graph partition enumeration
    efficient by using BDD/ZDDs[1,2]. Notably, we don't take on the same constraints these papers make,
    so this just simply enumerates all partitions.

    - Kawahara et al. (2017) Generating All Patterns of Graph Partitions Within a Disparity Bound
    - Nakahata et al. (2018) Enumerating Graph Partitions Without Too Small Connected Components Using
      Zero-suppressed Binary and Ternary Decision Diagrams

    >>> g0 = rrtd.Graph({ 0: [1, 2], 1: [0, 2], 2: [0, 1] })
    >>> list(graphenum(g0, edges=list(edges(g0))))
    [{0: 0, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 1}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 0, 2: 2}, {0: 0, 1: 0, 2: 0}]
    '''
    if edges is None:
        # Iterating over edges by degree seems efficient in some tests -- order doesn't matter for correctness.
        edges = degree_heuristic_edge_order(g)
    if mapped is None:
        mapped = {s: s for s in g.state_list}
        forbidden = frozenset()
    if not edges:
        yield mapped
        return

    # The algorithm iterates over edges. At every step, we consider whether to merge the partitions of
    # the nodes connected by this edge.
    edge = edges[0]
    comp_pair = ordpair(mapped[edge[0]], mapped[edge[1]])

    # do not merge at this edge
    yield from graphenum(g, edges=edges[1:], mapped=mapped, forbidden=forbidden | {comp_pair})

    # merge at this edge
    c0, c1 = comp_pair
    if c0 != c1 and comp_pair not in forbidden:
        # to merge at this edge, we have to remap references from c1 to c0
        mapped = {state: c0 if c == c1 else c for state, c in mapped.items()}
        forbidden = frozenset({
            ordpair(
                c0 if a == c1 else a,
                c0 if b == c1 else b,
            )
            for a, b in forbidden
        })
        yield from graphenum(g, edges=edges[1:], mapped=mapped, forbidden=forbidden)

def fiedler_heuristic_edge_order(g):
    f = hitting_time.fiedler(g, graph_laplacian_fn=hitting_time.symmetric_normalized_graph_laplacian)
    es = list(edges(g))
    return sorted(es, key=lambda pair: abs((f[pair[0]]+f[pair[1]])/2))

def degree_heuristic_edge_order(g):
    deg = {s: len(g.actions(s)) for s in g.state_list}
    es = list(edges(g))
    return sorted(es, key=lambda pair: deg[pair[0]]+deg[pair[1]], reverse=True)

# Now code that's more solway-specific.

def exit_map_from_partition(mdp, partition):
    # initialize
    c_to_exit = {partition[s]: set() for s in mdp.state_list}
    # for every state
    for s in mdp.state_list:
        for a in mdp.actions(s):
            # see if neighbor is in the same group
            ns = mdp.next_state(s, a)
            if partition[s] != partition[ns]:
                # if not, it's an exit!
                c_to_exit[partition[s]].add(ns)
    return {
        s: list(c_to_exit[partition[s]])
        for s in mdp.state_list
    }

class SolwayPartitionOLMDP(solway_objective.SolwayOptionLevelMDP):
    '''
    This whole class is a bit of a hack meant to shim into the interface we have for solway decomposition.
    '''
    def __init__(self, mdp, partition):
        # NOTE: we don't save the mdp here. we wait for __call__ below
        self.exits = exit_map_from_partition(mdp, partition)
        subgoals = set.union(*[{o for _, o in self._options(s)} for s in mdp.state_list])
        super().__init__(None, subgoals)
    def __call__(self, mdp, subgoals):
        assert self.mdp is None
        assert subgoals == self.subgoals
        super().__init__(mdp, subgoals)
        return self
    def _options(self, s):
        return [('option', s) for s in self.exits[s]]

def solway_task_decomposition(
    orig_mdp, td, *, tqdm=lambda x, **kw: x, samples=10, resfmt=False, average=False, only_count_entrance_options=True,
    compute_subgoal_rate=False,
    compute_subgoal_rate_per_partition=False,
    exclude_trivial_partition=False,
):
    adjacency = rrtd.adjacency_ssp(orig_mdp)
    partis = list(graphenum(orig_mdp))
    if exclude_trivial_partition:
        partis = [p for p in partis if not all(v==0 for v in p.values())]
    total = np.zeros(len(partis))
    if compute_subgoal_rate or compute_subgoal_rate_per_partition:
        parti_sgr = np.zeros((len(partis), len(orig_mdp.state_list)))

    def solway_phi(mdp, parti, **kw):
        ol = SolwayPartitionOLMDP(mdp, parti)
        return solway_objective.solway_phi(mdp, td, ol.subgoals, SolwayOptionLevelMDP_class=ol, only_count_entrance_options=only_count_entrance_options, **kw)

    for _ in tqdm(range(samples)):
        # sample a new MDP (since weights have random noise for tie-breaking)
        mdp = solway_objective.SolwayGraph(adjacency)
        #solway_phi(mdp, td, [sg], SolwayOptionLevelMDP_class=SolwayOptionLevelMDP_class)
        if compute_subgoal_rate or compute_subgoal_rate_per_partition:
            for idx, parti in enumerate(tqdm(partis, leave=False)):
                score, sgr = solway_phi(mdp, parti, compute_subgoal_rate=True)
                total[idx] += score/samples
                parti_sgr[idx] += sgr/samples
            continue
        total += np.array([solway_phi(mdp, parti) for parti in tqdm(partis, leave=False)]) / samples
    if average:
        total /= len(td)
    res = [dict(parti=parti, value=t) for parti, t in zip(partis, total)]
    if compute_subgoal_rate:
        return res, scipy.special.softmax(total)@parti_sgr
    if compute_subgoal_rate_per_partition:
        return res, parti_sgr
    return res

def pyp_log_prior(partition, *, a=0, b=1):
    '''
    >>> assert np.isclose(1/2, np.exp(pyp_log_prior({0: 0, 1: 0})))
    >>> assert np.isclose(1/2, np.exp(pyp_log_prior({0: 0, 1: 1})))
    >>> assert np.isclose(1/3, np.exp(pyp_log_prior({0: 0, 1: 0, 2:0})))
    >>> assert np.isclose(1/6, np.exp(pyp_log_prior({0: 0, 1: 1, 2:0})))
    >>> assert np.isclose(1/6, np.exp(pyp_log_prior({0: 0, 1: 1, 2:2})))
    '''
    # a is the extra parameter from a PYP
    # b is what is usually called alpha in a CRP
    nk = [0]*len(partition) # assuming tables are bounded by # of elements
    n = 0
    m = 0
    logp = 0
    for _, k in partition.items():
        if nk[k] == 0:
            # new group!
            logp += np.log((m*a+b)/(n+b))
            m += 1
        else:
            logp += np.log((nk[k]-a)/(n+b))
        nk[k] += 1
        n += 1
    return logp

def all_partitions_have_non_exit(mdp, partition):
    '''
    >>> graph, p = rrtd.Graph({0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}), {0: 0, 1: 0, 2: 1, 3: 1}
    >>> assert all_partitions_have_non_exit(graph, p)
    >>> assert not all_partitions_have_non_exit(graph, {0: 0, 1: 0, 2: 0, 3: 1})
    '''
    exit_map = exit_map_from_partition(mdp, partition)
    exits = {s for vals in exit_map.values() for s in vals}
    c_to_s = {}
    for state, c in partition.items():
        c_to_s.setdefault(c, []).append(state)
    return all(
        any(s not in exits for s in states)
        for c, states in c_to_s.items()
    )

def subgoal_choice_binary_exit(mdp, partition):
    '''
    >>> graph, p = rrtd.Graph({0: [1], 1: [0, 2], 2: [1]}), {0: 0, 1: 0, 2: 1}
    >>> assert np.allclose(subgoal_choice_binary_exit(graph, p), np.array([0, 1, 1])), subgoal_choice_uniform_exit(graph, p)
    >>> assert np.allclose(subgoal_choice_binary_exit(graph, {0: 0, 1: 0, 2: 0}), np.array([0, 0, 0]))
    '''
    exit_map = exit_map_from_partition(mdp, partition)
    exits = {s for vals in exit_map.values() for s in vals}
    return np.array([1 if s in exits else 0 for s in range(len(mdp.state_list))])

def subgoal_choice_uniform_exit(mdp, partition):
    '''
    >>> graph, p = rrtd.Graph({0: [1], 1: [0, 2], 2: [1]}), {0: 0, 1: 0, 2: 1}
    >>> assert np.allclose(subgoal_choice_uniform_exit(graph, p), np.array([0, 1/2, 1/2]))
    >>> assert np.allclose(subgoal_choice_uniform_exit(graph, {0: 0, 1: 0, 2: 0}), np.array([1/3, 1/3, 1/3]))
    '''
    bin_exit = subgoal_choice_binary_exit(mdp, partition)
    total = bin_exit.sum()
    if not total:
        # return uniform
        return np.ones(len(mdp.state_list)) / len(mdp.state_list)
    return bin_exit / total

def subgoal_choice_uniform_exit_after_margin_state(mdp, partition):
    '''
    >>> graph, p = rrtd.Graph({0: [1], 1: [0, 2], 2: [1]}), {0: 0, 1: 0, 2: 1}
    >>> assert np.allclose(subgoal_choice_uniform_exit_after_margin_state(graph, p), np.array([0, 1/3, 2/3]))
    '''
    # now trying something like: assume a uniform over states, then a uniform over exits
    exit_map = exit_map_from_partition(mdp, partition)
    res = np.zeros(len(mdp.state_list))
    for s, exits in exit_map.items():
        for ex in exits:
            res[ex] += 1/(len(exits)*len(res))
        if not exits:
            res += 1/(len(res)*len(res)) # a uniform if no exits
    assert np.isclose(res.sum(), 1)
    return res

def marginalize_decomp(
    mdp, scores,
    *,
    exclude_trivial_partition=False,
    uniform_exit=True,
    uniform_exit_margin_state=False,
    pyp_prior=False,
    normalize_logp=False,
    inverse_temperature=1.,
):
    '''
    This was an attempt to marginalize over decompositions to find a relation to our subgoal
    algo. While f2c works, a simple balloon didn't, in that marginalizing over all partitions
    (weighted by softmax of value) gave a different distribution than our algo.
    '''
    if uniform_exit:
        partition_exit_dist = subgoal_choice_uniform_exit

    if uniform_exit_margin_state:
        partition_exit_dist = subgoal_choice_uniform_exit_after_margin_state

    # Another generative model is: for each task, sample a partition. you use the partition.
    # If you marginalize out tasks and partitions, that would give you a somewhat reasonable
    # distribution over subgoals. Would have to resample tasks.

    if exclude_trivial_partition:
        # Filter out the no-sg result.
        scores = [s for s in scores if not all(v==0 for v in s['parti'].values())]
    logp = np.array([s['value'] for s in scores])
    if pyp_prior:
        logp = np.array(logp)
        logprior = np.array([pyp_log_prior(s['parti']) for s in scores])
        if normalize_logp:
            logp_std = np.std(logp)
            if logp_std != 0:
                logp = (logp - np.mean(logp)) / logp_std
            logprior_std = np.std(logprior)
            if logprior_std != 0:
                logprior = (logprior - np.mean(logprior)) / logprior_std
        logp = logp + logprior
    # Take the softmax
    p = shannon.softmax(inverse_temperature*logp)
    # Map each partition to a dist
    onehots = np.stack([
        partition_exit_dist(mdp, s['parti'])
        for s in scores
    ])
    assert np.allclose(onehots.sum(1), np.ones(len(scores)))
    return p@onehots
