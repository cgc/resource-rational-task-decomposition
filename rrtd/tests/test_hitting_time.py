import hitting_time, prior_envs, rrtd
import numpy as np
import scipy.optimize
import prior_envs
import automated_design
import conditional

def test_fiedler_matches_opti():
    mdp = prior_envs.f2c

    fiedler = hitting_time.fiedler(mdp, graph_laplacian_fn=hitting_time.symmetric_normalized_graph_laplacian)

    # Relating fiedler to the optimization criteria folks mention
    # from Shi & Malik 2000 "Normalized cuts and image segmentation"
    # and Estrada & Higham 2010 "Network Properties Revealed through Matrix Functions"
    binary = rrtd.binary_adjacency_ssp(mdp)
    d = hitting_time.outdegree(binary)

    # Handling conversion of optimization parameter
    def conv(x):
        # First, we add the last element to satisfy sum(x)=0
        x = np.append(x, -x.sum())
        # Now we normalize
        x = x / np.linalg.norm(x)
        return x

    def fn(x):
        x = conv(x)
        pairwisediff = x[:, None] - x[None, :]
        return np.sum((binary * pairwisediff)**2) / (x**2 * d).sum()

    np.random.seed(43)
    res = scipy.optimize.minimize(fn, np.random.uniform(-1, 1, size=len(mdp.state_list)-1))
    assert res.success

    # depends on seed! otherwise could be negative
    assert (
        np.allclose(conv(res.x), fiedler) or
        np.allclose(conv(res.x), -fiedler)
    )

def test_hitting_time():
    # Figure 2c
    solway = np.array([
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    ])
    solway_random = solway/hitting_time.outdegree(solway)[:, None]

    # a geometric, for testing
    p = 1/17
    geom = np.array([
        [1-p, p],
        [0, 1],
    ])

    r_iter = hitting_time.hitting_time_iter(geom)
    r_inv = hitting_time.hitting_time_inv(geom)
    assert np.allclose(r_iter, np.array([17, 0]))
    assert np.allclose(r_iter, r_inv)

    solway_goal = 9
    r_iter = hitting_time.hitting_time_iter(hitting_time.with_absorbing_state(solway_random, solway_goal))
    r_inv = hitting_time.hitting_time_inv(hitting_time.with_absorbing_state(solway_random, solway_goal))
    r_spectral = hitting_time.hitting_time_spectral(solway, solway_goal)
    assert np.allclose(r_iter, r_inv)
    assert np.allclose(r_iter, r_spectral)

    # Adding these to capture numerical issues I'm running into on M1 Pro
    r_spectral_non_eigh1 = hitting_time.hitting_time_spectral(solway, solway_goal, eig_kwargs=dict(eigfn=np.linalg.eig))
    r_spectral_non_eigh2 = hitting_time.hitting_time_spectral(solway, solway_goal, eig_kwargs=dict(eigfn=scipy.linalg.eig))
    if conditional.is_arm64_mac():
        assert not np.allclose(r_iter, r_spectral_non_eigh1)
        assert not np.allclose(r_iter, r_spectral_non_eigh2)
    else:
        assert np.allclose(r_iter, r_spectral_non_eigh1)
        assert np.allclose(r_iter, r_spectral_non_eigh2)

def test_partition_close():
    eps = 1e-8
    assert (hitting_time.partition_close(np.array([7, 1-eps, 3, 1, 3+eps, 7-eps])) == np.array([0, 1, 2, 1, 2, 0])).all()

def test_weight_for_limited_evals():
    evals = np.array([0, 0, 0, 1, 1])
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 1), np.array([1/3, 1/3, 1/3, 0, 0]))
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 2), np.array([2/3, 2/3, 2/3, 0, 0]))
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 3), np.array([1, 1, 1, 0, 0]))
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 4), np.array([1, 1, 1, 1/2, 1/2]))
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 5), np.array([1, 1, 1, 1, 1]))

    evals = np.array([0, 0, 1, 2, 2])
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 1), np.array([1/2, 1/2, 0, 0, 0]))
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 2), np.array([1, 1, 0, 0, 0]))
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 3), np.array([1, 1, 1, 0, 0]))
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 4), np.array([1, 1, 1, 1/2, 1/2]))
    assert np.allclose(hitting_time.weight_for_limited_evals(evals, 5), np.array([1, 1, 1, 1, 1]))

def test_limited_eigensystem_sum():
    sort_evals, sort_evecs = np.array([1/2, 1, 1, 1.5]), np.array([
        [1, 6, -1, 2],
        [3, 5, 2, 3],
    ])
    def _check(exp, *, limit=None):
        return np.allclose(
            hitting_time.make_limited_eigensystem_sum(sort_evals, sort_evecs, limit=limit)(fn=lambda i, eval, evec: eval * evec),
            exp)
    assert _check([1/2, 3/2], limit=1)
    assert _check([1/2+(6-1)/2, 3/2+(5+2)/2], limit=2)
    assert _check([1/2+(6-1), 3/2+(5+2)], limit=3)
    assert _check([1/2+(6-1)+2*1.5, 3/2+(5+2)+3*1.5], limit=4)

def test_new_random_walk_spectral_algorithm():
    for g in prior_envs.experiment_mdps + [prior_envs.f2c]:
        rw_alg = rrtd.new_random_walk_algorithm(g)
        new_rw = hitting_time.new_random_walk_spectral_algorithm(g, limit=len(g.state_list))
        for s in g.state_list:
            for z in g.state_list:
                old_v = rw_alg(s, z)
                v = new_rw(s, z)
                assert np.isclose(v['value'], old_v['value'])

def test_new_random_walk_spectral_algorithm_limited():
    regular_graphs = [
        "G?qa`_", "G?zTb_", "G?~vf_", "GCXmd_", "GCY^B_", "GCZJd_", "GCrb`o", "GCzvbo", "GEnbvG",
        "GEnfbW", "GFzvvW", "GQyurg", "GQzTrg", "GQ~vvg", "GUzvrw", "G]~v~w", "G~~~~{"
    ]
    tested = 0
    non_uniform_predictions = 0
    for g6 in regular_graphs:
        g = automated_design.parse_g6(g6)
        sort_evals, sort_evecs = hitting_time.sorted_eig(hitting_time.lovasz_N(rrtd.binary_adjacency_ssp(g)))
        equiv = hitting_time.partition_close(sort_evals)
        # We're skipping graphs where the 2nd eigenvector isn't unique.
        # In general, these regular graphs with non-unique 2nd eigenvectors tend to have no bottlenecks.
        if equiv[1] == equiv[2]:
            continue
        tested += 1

        d = len(g.actions(0))
        n = len(g.state_list)
        m = d * n / 2

        rw_full = hitting_time.new_random_walk_spectral_algorithm(g, limit=len(g.state_list))
        rw_rank1 = hitting_time.new_random_walk_spectral_algorithm(g, limit=2)
        def alg_cost(z, rw):
            return 1/n*sum(rw(s, z)['value'] + rw(z, s)['value'] for s in g.state_list)
        def spectral_cost(z, sl):
            # From appendix
            # return -2 * m / (d*n) * (1 / (1 - sort_evals[sl])) @ (n*sort_evecs[z, sl]**2 + 1)
            # Simplified previous line by removing $n$.
            return -2 * m / d * (1 / (1 - sort_evals[sl])) @ (sort_evecs[z, sl]**2 + 1/n)
        predictions = np.zeros(len(g.state_list))
        for z in g.state_list:
            alg_cost_full = alg_cost(z, rw_full)
            alg_cost_rank1 = alg_cost(z, rw_rank1)
            spectral_full = spectral_cost(z, slice(1, None))
            spectral_rank1 = spectral_cost(z, slice(1, 2))
            assert np.isclose(alg_cost_full, spectral_full)
            assert np.isclose(alg_cost_rank1, spectral_rank1)
            predictions[z] = spectral_rank1
        # This checks that a simplified expression for rank 1 matches QCut.
        qcut = rrtd.res_to_arr(g, hitting_time.qcut_decomposition(g))
        assert np.allclose(
            -2 * m / d * (1 / (1 - sort_evals[1])) * (-qcut + 1/n),
            predictions,
        )
        if not np.allclose(qcut[0], qcut):
            # Since QCut ~ RRTD-RW-rank1, they should have correlation of 1.
            assert np.isclose(np.corrcoef(predictions, qcut)[0, 1], 1)
            non_uniform_predictions += 1

    assert tested == 8
    assert non_uniform_predictions == 6

def test_qcut():
    def _check(mdp, exp, fn):
        res = rrtd.res_to_arr(mdp, fn(mdp))
        assert np.allclose(hitting_time.partition_close(res), exp), (res, exp)

    mdp = prior_envs.f2c
    _check(mdp, np.array([0, 1, 0, 1, 2, 2, 1, 0, 1, 0]), hitting_time.qcut_decomposition)
    _check(mdp, np.array([0, 1, 0, 1, 2, 2, 1, 0, 1, 0]), hitting_time.qcut_decomposition_OLD_default_symmetricGL)

    # This is the ring graph (a chain with ends connected)
    mdp = automated_design.parse_g6('G?qa`_')
    _check(mdp, np.array([0]*8), hitting_time.qcut_decomposition)
    # this answer is entirely dependent on noise from eigendecomposition
    _check(mdp, np.array([0, 1, 1, 0, 2, 3, 3, 2]), hitting_time.qcut_decomposition_OLD_default_symmetricGL)

    # This is the complete graph
    mdp = automated_design.parse_g6('G~~~~{')
    _check(mdp, np.array([0]*8), hitting_time.qcut_decomposition)
    # this answer is entirely dependent on noise from eigendecomposition
    _check(mdp, np.array([0, 1, 2, 3, 4, 5, 6, 7]), hitting_time.qcut_decomposition_OLD_default_symmetricGL)

def test_qcut_matches_old_for_unambiguous_graphs():
    # These are the first graphs where all eigenvalues are unique
    unambiguous_graphs = [
        'G?`@Fo', 'G?`@eS', 'G?`@f?', 'G?`@fO', 'G?`@fS', 'G?`@fc', 'G?`@fo', 'G?`@fs', 'G?`CV_',
        'G?`CVo', 'G?`CVw', 'G?`DB_', 'G?`DBg', 'G?`DBo', 'G?`DBw', 'G?`DE_', 'G?`DEc', 'G?`DEg',
        'G?`DEk', 'G?`DEo', 'G?`DEw']
    for g6 in unambiguous_graphs:
        g = automated_design.parse_g6(g6)
        sort_evals, sort_evecs = hitting_time.sorted_eig(hitting_time.lovasz_N(rrtd.binary_adjacency_ssp(g)))
        assert hitting_time.partition_close(sort_evals)[-1] == 7
        new_res = rrtd.res_to_arr(g, hitting_time.qcut_decomposition(g))
        old_res = rrtd.res_to_arr(g, hitting_time.qcut_decomposition_OLD_default_symmetricGL(g))
        print(g6)
        print(sort_evals)
        assert np.allclose(new_res, old_res)
