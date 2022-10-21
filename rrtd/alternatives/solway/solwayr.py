import os
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro
import numpy as np
import solway_objective, graph_partition, rrtd
import contextlib

currdir = os.path.dirname(os.path.abspath(__file__))

INIT = False
def init():
    global INIT
    if INIT:
        return
    originalwd = ro.r.getwd()
    solwaydir = os.path.join(currdir, 'solway-obh')
    ro.r.setwd(solwaydir)
    ro.r(f'source("optimal_options.R")')
    ro.r('''
    solwayEvidenceAndMDP = function(adj, membership, ii=FALSE) {
        g = graph_from_adjacency_matrix(adj, mode='undirected')
        path_data <- get_random_consistent_shortest_paths(g)
        sp = path_data$shortest_paths
        if (ii) { # We use this to index into the task distribution.
            sp = sp[ii]
        }
        list(
            shortest_paths=sp,
            noise_adj=as_adjacency_matrix(path_data$flat_G, attr='weight', sparse=FALSE),
            score=options_model_selection_objective(g, membership, sp))
    }
    ''')
    ro.r.setwd(originalwd)
    INIT = True

def r_with_vars(rcode, **kw):
    '''
    This takes some R code and variables used in the R code (as kwargs)
    and runs it by dynamically making an appropriate function. Assumes
    that the variables you are handling are R objects.
    '''
    keys, values = zip(*kw.items())
    return ro.r('function('+','.join(keys)+') {'+rcode+'}')(*values)

@contextlib.contextmanager
def r_seed_ctx(seed):
    # based on http://www.cookbook-r.com/Numbers/Saving_the_state_of_the_random_number_generator/
    orig_seed = r_with_vars('''
    if (exists(".Random.seed", .GlobalEnv)) {
        oldseed <- .GlobalEnv$.Random.seed
    } else {
        oldseed <- NULL
    }

    set.seed(seed)

    oldseed
    ''', seed=seed)
    try:
        yield
    finally:
        r_with_vars('''
        if (!is.null(oldseed)) {
            .GlobalEnv$.Random.seed <- oldseed
        } else {
            rm(".Random.seed", envir = .GlobalEnv)
        }
        ''', oldseed=orig_seed)

def solwayEvidenceAndMDP(mdp, partition, **kw):
    init()
    return wrapped_r(ro.r.solwayEvidenceAndMDP)(rrtd.binary_adjacency_ssp(mdp), np.array(partition), **kw)

def r2py(v, lim=5):
    # without recursion limit, igraph kind of messes us up
    if lim == 0:
        return v
    if isinstance(v, (ro.Array, ro.Matrix)):
        return ro.numpy2ri.rpy2py(v)
    try:
        pairs = [(key, r2py(val, lim=lim-1)) for key, val in v.items()]
        if all(k for k, v in pairs):
            return dict(pairs)
        elif all(not k for k, v in pairs):
            return [v for k, v in pairs]
        else:
            return pairs
    except AttributeError:
        return v

def wrapped_r(fn):
    '''
    Handy wrapper to call an rpy2 function without needing to deal with wrapping code.
    >>> assert np.all(wrapped_r(ro.r.c)(3, 4) == [3, 4])
    >>> assert wrapped_r(ro.r.c)(x=3, f=4) == dict(x=3, f=4)
    >>> assert wrapped_r(ro.r.c)(3, f=4) == [('', 3), ('f', 4)]
    >>> assert np.all(wrapped_r(ro.r.list)(x=np.array([3, 4]))['x'] == np.array([3, 4]))
    '''
    def wrapped(*args, **kwargs):
        def convarg(arg):
            return ro.numpy2ri.py2rpy(arg) if isinstance(arg, np.ndarray) else arg
        v = fn(
            *[convarg(arg) for arg in args],
            **{k: convarg(arg) for k, arg in kwargs.items()},
        )
        return r2py(v)
    return wrapped


def nmdp_from_R(mdp, res, *, validate_shortest_paths=True):
    '''
    This converts the MDP with noise on edge rewards from R into
    something we can work with. Also validates that we find same shortest
    paths as R.
    '''
    noise_adj = res['noise_adj']
    shortest_paths = res['shortest_paths']

    # Fill rewards in.
    nmdp = solway_objective.SolwayGraph(rrtd.adjacency_ssp(mdp))
    for (s, ns) in np.ndindex(noise_adj.shape):
        vv = -noise_adj[s, ns]
        if vv != 0:
            nmdp._rewards[s, ns] = vv
            # This one is almost not necessary
            assert nmdp.reward(s, None, ns) == vv

    # Check that we recover the same matrix.
    assert np.allclose(-noise_adj, np.array([
        [
            nmdp.reward(s, None, ns) if any(mdp.next_state(s, a)==ns for a in mdp.actions(s)) else 0
            for ns in mdp.state_list
        ] for s in mdp.state_list]))

    # verifying our shortest paths match
    def policy_trajectory(policy, start, goal):
        s = start
        yield s
        while s != goal:
            s = nmdp.next_state(s, poli.action(s))
            yield s
    if shortest_paths is not None:
        for p in shortest_paths:
            p = [s-1 for s in p]
            poli = nmdp.vi_for_goal(p[-1]).policy
            assert p == list(policy_trajectory(poli, p[0], p[-1]))

    return nmdp

def validate(mdp, partition):
    if isinstance(partition, dict):
        partition = [partition[i] for i in range(len(partition))]
    # First compute for their version.
    res = solwayEvidenceAndMDP(mdp, partition)
    nmdp = nmdp_from_R(mdp, res)
    # Now computing it for our version of the code.
    # Critically, we have to include successor tasks & only count options at entrances
    ol = graph_partition.SolwayPartitionOLMDP(nmdp, dict(enumerate(partition)))
    td = rrtd.all_pairs_shortest_path_distribution(
        mdp, remove_trivial_tasks=True, remove_successor_tasks=False)
    ours = solway_objective.solway_phi(
        nmdp, td, ol.subgoals, SolwayOptionLevelMDP_class=ol, only_count_entrance_options=True)
    assert np.isclose(res['score'][0], ours), (mdp, partition)

def maxes(l, key=lambda x: x):
    '''
    >>> assert maxes([3, 4, 5, 5]) == [5, 5]
    >>> assert maxes([{'f':3}, {'f': 4}, {'f': 4}], key=lambda x: x['f']) == [{'f': 4}, {'f': 4}]
    '''
    maxel = max(l, key=key)
    return [item for item in l if np.isclose(key(item), key(maxel))]

def subgoal_rate_from_paths(shortest_paths, td, partition):
    totalp = 0
    sgr = np.zeros(len(partition))
    sgr_dict = {}
    for p in shortest_paths:
        sgs = []
        # On this path, we consider each pair of states to identify partition crossings.
        for i in range(1, len(p)):
            s = p[i - 1]
            ns = p[i]
            # When the partition of a state doesn't match the subsequent state, that subsequent
            # state is an exit of the partition. We mark it as a subgoal.
            if partition[s] != partition[ns]:
                sgs.append(ns)

        # We handle the case of no subgoals in a special way; we let the goal be the
        # singular "subgoal".
        if len(sgs) == 0:
            sgs = [p[-1]]

        prob = td.prob(rrtd.frozendict(start=p[0], goal=p[-1]))
        assert prob > 0, 'All items in the shortest path list must have non-zero probability in the task distribution.'
        totalp += prob
        # We add the subgoals used in the paths, weighted by 1) the probability of the task
        # and 2) the probability of the subgoal in a particular task. We let the latter be a
        # uniform over all subgoals used.
        dist = np.zeros(sgr.shape)
        for sg in sgs:
            dist[sg] += 1/len(sgs)
        sgr += prob * dist
        sgr_dict[p[0], p[-1]] = dist

    assert np.isclose(totalp, 1), 'Sanity check that task distribution sums to 1, also checks that it is a subset of shortest paths.'
    assert np.isclose(sgr.sum(), 1)
    return sgr, sgr_dict


class SolwayModel(object):
    '''
    Make this class-based to hold the sampled path sets we marginalize out.
    '''
    def __init__(self, mdp, *, nsamples=5):
        init()

        adj = ro.numpy2ri.py2rpy(rrtd.binary_adjacency_ssp(mdp))
        self.g = ro.r.graph_from_adjacency_matrix(adj, mode='undirected')

        self.sps_samples = []
        self.__path_data_ref = []

        for _ in range(nsamples):
            path_data = ro.r.get_random_consistent_shortest_paths(self.g)
            # Holding onto these references to avoid GC issues
            self.__path_data_ref.append(path_data)

            sps = path_data.rx2('shortest_paths')
            # Make versions that are 0-based instead of 1-based
            sps = [
                dict(
                    pytask=rrtd.frozendict(start=p[0]-1, goal=p[-1]-1),
                    pypath=[s-1 for s in p],
                    rpath=p,
                )
                for p in sps
            ]
            self.sps_samples.append(sps)

    def logevidence(self, td, partition):
        r = 0
        for sps in self.sps_samples:
            # Filter based on task distribution
            r_sps = [o['rpath'] for o in sps if td.prob(o['pytask']) > 0]
            assert len(r_sps) == len(td), 'Task distribution should be a strict subset of the shortest paths computed in R'
            logevidence = ro.r.options_model_selection_objective(self.g, partition, r_sps)
            r += logevidence[0] / len(self.sps_samples)
        return r

    def subgoal_rate(self, td, partition):
        r = np.zeros(len(partition))
        for sps in self.sps_samples:
            # Filter based on task distribution
            py_sps = [o['pypath'] for o in sps if td.prob(o['pytask']) > 0]
            assert len(py_sps) == len(td), 'Task distribution should be a strict subset of the shortest paths computed in R'
            sgr, sgr_dict = subgoal_rate_from_paths(py_sps, td, partition)
            r += sgr / len(self.sps_samples)
        assert np.isclose(r.sum(), 1)
        return r

def optimal_task_decomposition_subgoal_rate(
    mdp, td, *,
    exclude_trivial_partition=False,
    nsamples=1,
    seed=None,
    tqdm=lambda x: x,
):
    if seed is None:
        seed = ro.r('NULL')
    with r_seed_ctx(seed):
        m = SolwayModel(mdp, nsamples=nsamples)

    partitions = list(graph_partition.graphenum(mdp))
    if exclude_trivial_partition:
        partitions = [
            p
            for p in partitions
            if set(p.values()) != {0}
        ]

    best = maxes([
        dict(
            partition=p,
            value=m.logevidence(td, [p[s] for s in range(len(p))]),
        )
        for p in tqdm(partitions)
    ], key=lambda d: d['value'])

    sgr = np.mean([
        m.subgoal_rate(td, b['partition'])
        for b in best
    ], axis=0)

    return dict(
        scores=np.log(sgr),
        best=best,
    )
