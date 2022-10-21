import numpy as np

def hitting_time_inv(P, *, return_sr=False):
    '''
    >>> assert np.allclose(hitting_time_inv(with_absorbing_state(counter, 3)), np.array([12, 10, 6, 0]))
    '''
    absorb = np.isclose(np.diag(P), 1)
    absorb_idx = np.where(absorb)[0]
    transient_idx = np.where(~absorb)[0]
    Q = np.delete(np.delete(P, absorb_idx, 0), absorb_idx, 1)
    R = np.delete(np.delete(P, absorb_idx, 0), transient_idx, 1)
    assert Q.shape == (len(transient_idx), len(transient_idx))
    assert R.shape == (len(transient_idx), len(absorb_idx))
    sr = np.linalg.pinv(np.eye(len(transient_idx)) - Q)
    if return_sr:
        fullsr = np.zeros(P.shape)
        for sri, fullsri in enumerate(transient_idx):
            fullsr[fullsri, transient_idx] = sr[sri]
        return fullsr
    # H = sr @ R # this is a hitting probability
    h = sr.sum(1)
    assert h.shape == (len(transient_idx),)
    h_ = np.zeros(P.shape[0])
    h_[transient_idx] = h
    return h_

def hitting_time_iter(P, niter=1000):
    '''
    >>> assert np.allclose(hitting_time_iter(with_absorbing_state(counter, 3)), np.array([12, 10, 6, 0]))
    '''
    absorb = np.isclose(np.diag(P), 1)
    transient_idx = np.where(~absorb)[0]
    hittings = []
    for i in transient_idx:
        s = np.zeros(P.shape[0])
        s[i] = 1
        hitting = 0
        for _ in range(niter):
            hitting += s[~absorb].sum()
            s = s@P
        hittings.append(hitting)
    h_ = np.zeros(P.shape[0])
    h_[transient_idx] = np.array(hittings)
    return h_

def lovasz_N(adjacency):
    assert not np.allclose(adjacency.sum(axis=1), np.ones(adjacency.shape[0])), 'should not be a probability distribution'
    '''
    Compute `N` from Lovasz 1993. Related to Symmetric GL = I - N

    Top of Page 15 in Lovasz 1993

    Notes:
    - Lovasz's D is diag(1/outdegree(node))
    '''
    Dinv = np.diag(1/outdegree(adjacency))
    return (Dinv**(1/2)) @ adjacency @ (Dinv**(1/2))

# TODO: make sure np.linalg.eigh makes sense
def sorted_eig(mat, *, eigfn=np.linalg.eigh, ascending=False):
    '''
    Returns eigendecomposition of matrix, ensuring that
    vectors and values are sorted in descending order by eigenvalue.
    '''
    if eigfn == np.linalg.eigh:
        assert np.allclose(mat, mat.T), 'assert symmetry when using eigh'
    evals, evecs = eigfn(mat)
    sortidx = np.argsort(evals if ascending else -evals)
    sort_evals, sort_evecs = evals[sortidx], evecs[:, sortidx]
    return sort_evals, sort_evecs

def _spectral_access(adjacency, s, t, evals, evecs):
    '''
    This private function computes the access time from start state s to t
    using the sorted spectra (evals and evecs) of the graph with supplied
    adjacency matrix.
    '''
    assert np.allclose(evals[0], 1), evals

    # The first seems like the right way to handle self-loops
    #m = (adjacency.sum() + np.diag(adjacency).sum()) / 2
    # This seems slightly insane (since self-loops only count for 0.5????) but seems to be correct
    m = adjacency.sum()/2
    d = outdegree(adjacency)

    # This is a sanity check
    analytic_stationary = np.sqrt(d / (2 * m))
    assert (
        np.allclose(evecs[:, 0], analytic_stationary) or
        np.allclose(-evecs[:, 0], analytic_stationary)
    ), (evecs[:, 0], np.sqrt(d))

    tot = 0
    for k in range(1, len(evals)):
        vk = evecs[:, k]
        tot += 1/(1-evals[k]) * (
            vk[t]**2/d[t] -
            vk[s]*vk[t]/np.sqrt(d[s]*d[t])
        )
    return 2 * m * tot

def hitting_time_spectral(adjacency, terminal, eig_kwargs={}):
    '''
    >>> assert np.allclose(hitting_time_spectral(counterA, 3), hitting_time_inv(with_absorbing_state(counter, 3)))
    >>> assert np.allclose(hitting_time_spectral(counterA, 3), hitting_time_iter(with_absorbing_state(counter, 3)))
    '''
    evals, evecs = sorted_eig(lovasz_N(adjacency), **eig_kwargs)

    rv = []
    for s in range(adjacency.shape[0]):
        rv.append(_spectral_access(adjacency, s, terminal, evals, evecs))

    return np.array(rv)

def with_absorbing_state(P, idx):
    P = np.copy(P)
    P[idx, :] = 0
    P[idx, idx] = 1
    return P

def outdegree(adjacency):
    '''
    >>> assert (outdegree(counterA) == np.array([2, 2, 2, 2])).all()
    >>> assert (outdegree(with_absorbing_state(counterA, 3)) == np.array([2, 2, 2, 1])).all()
    >>> assert (outdegree(oneway_adjacency) == np.array([1, 2])).all()
    '''
    return adjacency.sum(1)

def indegree(adjacency):
    '''
    >>> assert (indegree(counterA) == np.array([2, 2, 2, 2])).all()
    >>> assert (indegree(with_absorbing_state(counterA, 3)) == np.array([2, 2, 1, 2])).all()
    >>> assert (indegree(oneway_adjacency) == np.array([2, 1])).all()
    '''
    return adjacency.sum(0)

# Older spectral code

def graph_laplacian(mdp):
    '''
    https://en.wikipedia.org/wiki/Laplacian_matrix#Laplacian_matrix_for_simple_graphs
    '''
    W = np.zeros(((len(mdp.state_list)), len(mdp.state_list)))
    for s in mdp.state_list:
        for a in mdp.actions(s):
            ns = mdp.next_state(s, a)
            assert s != ns, 'mdp should not have self-loops'
            W[s, ns] = 1
    assert (W == W.T).all(), 'assuming undirected graphs'
    D = np.diag(W.sum(0))
    return D - W

def symmetric_normalized_graph_laplacian(mdp):
    '''
    https://en.wikipedia.org/wiki/Laplacian_matrix#Laplacian_matrix_for_simple_graphs
    '''
    W = np.zeros(((len(mdp.state_list)), len(mdp.state_list)))
    for s in mdp.state_list:
        for a in mdp.actions(s):
            ns = mdp.next_state(s, a)
            assert s != ns, 'mdp should not have self-loops'
            W[s, ns] = 1
    assert (W == W.T).all(), 'assuming undirected graphs'
    Dinv = np.diag(1/W.sum(0))
    sqrtDinv = np.sqrt(Dinv)
    return np.eye(W.shape[0]) - sqrtDinv @ W @ sqrtDinv

def fiedler(mdp, *, idx=-2, graph_laplacian_fn=graph_laplacian):
    assert graph_laplacian_fn != graph_laplacian
    gl = graph_laplacian_fn(mdp)
    sort_evals, sort_evecs = sorted_eig(gl)
    return sort_evecs[:, idx]

def qcut_decomposition_OLD_default_symmetricGL(mdp, *, graph_laplacian_fn=symmetric_normalized_graph_laplacian):
    assert graph_laplacian_fn != graph_laplacian
    # elements closer to center should be prioritized
    scores = -(fiedler(mdp, graph_laplacian_fn=graph_laplacian_fn)**2)
    return sorted([
        dict(subgoals=(s,), value=v)
        for s, v in enumerate(scores)
    ], key=lambda row: -row['value'])

def qcut_decomposition_equivalent_evals_OLD(mdp, *, debug=False):
    gl = graph_laplacian(mdp)
    sort_evals, sort_evecs = sorted_eig(gl)

    # First, we look for all vectors with same eigenvalue; we take their summed scores at the end
    fiedler = idx = len(mdp.state_list)-2
    while 0 <= idx:
        if not np.isclose(sort_evals[fiedler], sort_evals[idx]):
            break
        idx -= 1
    equividx = range(idx+1, fiedler+1)
    if debug: print(equividx)

    # We just add the scores for all vectors.
    scores = np.zeros(len(mdp.state_list))
    for idx in equividx:
        vec = sort_evecs[:, idx]
        if debug: print(sort_evals[idx], vec, -(vec**2))
        # elements closer to center should be prioritized
        scores += -(vec**2)

    return sorted([
        dict(subgoals=(s,), value=v)
        for s, v in enumerate(scores)
    ], key=lambda row: -row['value'])

# Definiing for tests.

counterA = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
])
counter = counterA/outdegree(counterA)[:, None]
oneway_adjacency = np.array([
    [1, 0],
    [1, 1],
])

# Utilities to fix issues for graphs with many eigenvectors of same eigenvalue

def partition_close(arr):
    '''
    Groups the elements of the array by floating point closeness, returning
    an identifier for the group id.
    '''
    partitions = {}
    rv = []
    for el in arr:
        for parti_el, parti in partitions.items():
            # If the element is close to that of an existing group, we
            # add it to that one.
            if np.isclose(parti_el, el):
                rv.append(parti)
                break
        else:
            # Otherwise, we make a new group.
            parti = len(partitions)
            partitions[el] = parti
            rv.append(parti)
    return np.array(rv)

def weight_for_limited_evals(sorted_evals, limit):
    '''
    When trying to use a subset of eigenvalues/vectors, it's important to avoid
    errors due to the order of eigenvalues. This function returns weights on
    eigenvalues/vectors to use when trying to avoid the impact of order.
    '''
    assert limit >= 1
    equiv = partition_close(sorted_evals)

    # We first assign full weight to everything before the limit.
    w = np.zeros(sorted_evals.shape)
    w[:limit] = 1

    # We identify the final class before the limit.
    limited_equiv = equiv[:limit]
    final_class = limited_equiv[-1]
    # We count the total number in the class, both in the entire array...
    class_ct = (equiv == final_class).sum()
    # ... and before the limit.
    class_ct_before_limit = (limited_equiv == final_class).sum()
    # We have uniform weight over all equivalent to final class before the limit.
    start = limit-class_ct_before_limit
    w[start:start+class_ct] = class_ct_before_limit / class_ct

    assert np.isclose(np.sum(w), limit), (sorted_evals, equiv, limit, final_class, w)
    return w


def make_limited_eigensystem_sum(sort_evals, sort_evecs, *, limit=None):
    '''
    We use this function to sum the contribution of eigenvectors in a way that accounts for "ties"
    in eigenvalues. Eigenvectors with matching eigenvalues are non-unique, so this feels like the
    simplest way to avoid any influence from particularities of the decomposition algorithm or
    numerical issues.

    The basic strategy is to first compute weights that sum to the rank, but distribute weight
    among eigenvectors with equal eigenvalue. Given these weights, we can perform a weighted
    sum over functions that take eigenvectors and return values.
    '''
    if limit is None:
        limit = len(sort_evals)
    w = weight_for_limited_evals(sort_evals, limit)

    def summer(fn):
        tot = 0
        for i in range(len(sort_evals)):
            if w[i] == 0:
                continue
            tot += w[i] * fn(i, sort_evals[i], sort_evecs[:, i])
        return tot
    return summer

def new_random_walk_spectral_algorithm(mdp, *, limit=None):
    import rrtd
    A = rrtd.binary_adjacency_ssp(mdp)
    assert (A == A.T).all(), 'must be undirected'
    d = A.sum(0)
    m = A.sum() / 2
    n = A.shape[0]
    N = lovasz_N(A)
    if limit is None:
        limit = n
    assert limit >= 1

    sort_evals, sort_evecs = sorted_eig(N)
    summer = make_limited_eigensystem_sum(sort_evals, sort_evecs, limit=limit)

    def eig_fn(s, z, i, eval_, evec):
        if i == 0:
            return 0
        return 2 * m / (1 - eval_) * (evec[z]**2/d[z] - evec[s]*evec[z]/np.sqrt(d[s]*d[z]))

    def alg(s, z):
        cost = summer(fn=lambda *args: eig_fn(s, z, *args))
        return dict(value=-cost)
    return alg


def qcut_decomposition(mdp):
    import rrtd
    A = rrtd.binary_adjacency_ssp(mdp)
    assert (A == A.T).all(), 'must be undirected'
    N = lovasz_N(A)

    sort_evals, sort_evecs = sorted_eig(N)

    def eig_fn(i, eval_, evec):
        if i == 0:
            return 0
        # elements closer to center should be prioritized
        return -evec**2
    values = make_limited_eigensystem_sum(sort_evals, sort_evecs, limit=2)(fn=eig_fn)

    return sorted([
        dict(subgoals=(s,), value=v)
        for s, v in enumerate(values)
    ], key=lambda row: -row['value'])

def spectral_gap(mdp):
    import rrtd
    sort_evals, sort_evecs = sorted_eig(lovasz_N(rrtd.binary_adjacency_ssp(mdp)))
    assert np.isclose(sort_evals[0], 1)
    return sort_evals[0] - sort_evals[1]
