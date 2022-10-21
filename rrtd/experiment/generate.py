import rrtd, solway_objective
import numpy as np
import math, random, itertools, functools, collections, collections.abc

def shuffled(seq, random=random):
    '''
    This function returns a shuffled copy of the supplied sequence.
    '''
    seq = list(seq) # make a copy
    random.shuffle(seq) # shuffle inplace
    return seq # return the shuffled copy

def block_rand(population, k, random=random):
    '''
    This implements block randomization for k samples.
    It is a blocked version of `random.sample`, so that we sample
    from the population without replacement, then replenish.
    We repeat this until we've returned the desired number of samples.

    >>> r = random.Random(42)
    >>> expected = [4] * 4 + [5] * 6
    >>> for _ in range(5):
    ...     assert sorted(collections.Counter(block_rand(range(10), 46, random=r)).values()) == expected
    '''
    quot, rem = divmod(k, len(population))
    return (
        # We shuffle as many times as k divides the population
        sum([shuffled(population, random=random) for _ in range(quot)], []) +
        # And sample the remainder
        random.sample(population, rem))

def _normalize_sequence(values):
    '''
    standard normalization to the unit interval [0, 1].
    '''
    min_, max_ = min(values), max(values)
    return [(v-min_)/(max_-min_) for v in values]

def svd_coordinates(pos, horizontal=True):
    '''
    This seems a bit silly, but is handy to avoid some of the randomness of graphviz layouts.
    '''
    pos = np.array(pos)
    pos = pos - np.mean(pos, axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(pos, full_matrices=False)
    sortscale = -1 if horizontal else +1
    si = np.argsort(sortscale*s)
    return u[:, si]*s[si]

def graphviz_coordinates(g, *, layout='neato', normalize=True, plot_graph_kwargs={}):
    '''
    coordinates are rescaled so they are in [0, 1]
    '''
    from bs4 import BeautifulSoup
    assert sorted(g.state_list) == list(range(len(g.state_list))), 'Must be a simple state space.'
    svg = BeautifulSoup(rrtd.plot_graph(g, layout=layout, **plot_graph_kwargs)._repr_image_svg_xml(), features='html.parser')
    rv = [None] * len(g.state_list)
    node_elements = svg.select('.node')
    assert len(node_elements) == len(g.state_list)
    for el in node_elements:
        title, = el.select('title')
        node = int(title.text)
        assert node in g.state_list
        circle, = el.select('ellipse')
        rv[node] = (
            float(circle['cx']),
            float(circle['cy']),
        )

    if normalize:
        return normalized_xy_coords(rv)
    return rv

def normalized_xy_coords(xy, horizontal=True):
    '''
    Input is a list of x, y pairs.
    '''
    def _minus_min(seq):
        mins = min(seq)
        return [val-mins for val in seq]
    # Subtracting min here first so that we can normalize the two together
    xs = _minus_min([x for x, y in xy])
    ys = _minus_min([y for x, y in xy])

    # normalize them together. Note that this only does scaling
    # since we already subtracted min.
    normed = _normalize_sequence(xs+ys)
    xs, ys = normed[:len(xs)], normed[len(xs):]

    y_larger_range = max(ys)-min(ys) > max(xs)-min(xs)
    if (
        # if horizontal and y is larger, then swap
        (horizontal and y_larger_range) or
        # if vertical and x is larger, then swap
        (not horizontal and not y_larger_range)
    ):
        xs, ys = ys, xs

    return [(x, y) for x, y in zip(xs, ys)]


def generate_circle_orderings(g, *, prohibit_succ_dist=2, rotation_invariant=False):
    '''
    >>> import rrtd
    >>> assert len(list(generate_circle_orderings(rrtd.Graph({ 0: [1], 1: [0], 2: [3], 3: [2] }), prohibit_succ_dist=1))) == 8
    >>> assert len(list(generate_circle_orderings(rrtd.Graph({ 0: [1], 1: [0], 2: [3], 3: [2] }), prohibit_succ_dist=1, rotation_invariant=True))) == 2
    '''
    num_states = len(g.state_list)
    adj = g.adjacency # HACK
    assert sorted(g.state_list) == list(range(len(g.state_list))), 'Must be a simple state space.'
    #degree = {k: len(v) for k, v in adj.items()}
    nodes_degree_desc = sorted(g.state_list, key=lambda s: len(adj[s]), reverse=True)

    def recur(order):
        '''
        order maps circle index to state
        '''
        if len(order) == num_states:
            ls = [None]*num_states
            for idx, state in order.items():
                ls[idx] = state
            yield tuple(ls)
            return
        # HACK we choose by highest degree first??
        #si = max(range(num_states), key=lambda si: degree[si] if si not in order.values() else float('-inf'))
        si = nodes_degree_desc[len(order)] # sort of a hack?
        succ = adj[si]
        if rotation_invariant and len(order) == 0:
            # we fix the starting place to be invariant to rotation
            valid_idxs = [0]
        else:
            valid_idxs = [
                idx
                for idx in range(num_states)
                # Needs to be unassigned
                if idx not in order
                # And needs successors to be distance `prohibit_succ_dist` at least.
                if all(
                    (
                        order.get((idx-dist) % num_states) not in succ and
                        order.get((idx+dist) % num_states) not in succ
                    )
                    # We don't need to check for 0
                    for dist in range(1, prohibit_succ_dist+1)
                )
            ]
        for idx in valid_idxs:
            order[idx] = si
            yield from recur(order)
            del order[idx]
    yield from recur({})

def coordinates_for_circle_order(order):
    '''
    order: array mapping index of position on circle -> state
    '''
    assert sorted(order) == list(range(len(order))), 'State space should be a sequence of numbers'
    return [
        (
            (math.cos(2*math.pi/len(order)*order.index(s)) + 1)/2,
            (math.sin(2*math.pi/len(order)*order.index(s)) + 1)/2,
        )
        for s in range(len(order))
    ]

def _validate_and_count_experiment(experiment, assignment_spec):
    '''
    This is a helper function we use in experiment generation to validate
    a factored representation of an experiment and return the counts needed
    at each site where an assignment to a factor must happen.
    - Validate that types are correct (factors are lists; non-factor parents of factors are not lists)
    - Validate that counts are the same across many branches (child factors with same name should all have same number of levels.)

    >>> _validate_and_count_experiment(dict(f=[dict(g=[3, 4, 5])]*7, h=[dict(j=3)]), ['f', 'f.g'])
    {'f': 7, 'f.g': 3}
    >>> _validate_and_count_experiment(dict(f=[dict(g=[3, 4, 5])]*7), ['f.g'])
    Traceback (most recent call last):
        ...
    AssertionError: No support for configuring factorial assignment inside sequences. At "f".
    >>> _validate_and_count_experiment(dict(), ['f'])
    Traceback (most recent call last):
        ...
    AssertionError: Missing key "f".
    >>> _validate_and_count_experiment(dict(f=3), ['f'])
    Traceback (most recent call last):
        ...
    AssertionError: Expected a sequence at "f".
    >>> _validate_and_count_experiment(dict(f=[{'g':[0]}, {'g':[1, 2]}]), ['f', 'f.g'])
    Traceback (most recent call last):
        ...
    AssertionError: Found 2 options at "f.g", but expected 1.
    '''
    counts = {}

    def _recur(exp, spec, *, prefix=''):
        # first gather the keys by their current level of key
        subconf = {}
        for keys in spec:
            subconf.setdefault(keys[0], []).append(keys[1:])

        for curr_key, subkeys in subconf.items():
            full_curr_key = f'{prefix}{curr_key}'
            errkey = f'"{full_curr_key}"'
            assert curr_key in exp, f'Missing key {errkey}.'

            isseq = isinstance(exp[curr_key], collections.abc.Sequence)

            # First, we see if this key is a factor.
            if () in subkeys:
                # If it is, we remove it & assert on type.
                subkeys.remove(())
                assert isseq, f'Expected a sequence at {errkey}.'

                curr_len = len(exp[curr_key])
                if full_curr_key not in counts:
                    # If not yet encountered, we record the number of values for this key.
                    counts[full_curr_key] = curr_len
                # We ensure the count is the same at sibling keys.
                assert curr_len == counts[full_curr_key], f'Found {curr_len} options at {errkey}, but expected {counts[full_curr_key]}.'

                # Finally, we recurse for each possible value this key might take.
                for value in exp[curr_key]:
                    _recur(value, subkeys, prefix=prefix+curr_key+'.')
            else:
                # If this key isn't factor but we have made it here, that means we
                # have children that are factors. We recurse.
                assert not isseq, f'No support for configuring factorial assignment inside sequences. At {errkey}.'
                _recur(exp[curr_key], subkeys, prefix=prefix+curr_key+'.')

    _recur(experiment, [tuple(key.split('.')) for key in assignment_spec])
    return counts


def sample_factor_assignment(factored_config, *, counterbalance=[], sample=[], count, random=random):
    '''
    >>> list(sorted(sample_factor_assignment(dict(f=range(10)), counterbalance=['f'], count=10, random=random.Random(42))['f'])) # all items enumerated
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> a = sample_factor_assignment(dict(f=range(2), g=range(2)), counterbalance=['f', 'g'], count=4, random=random.Random(42)) # we enumerate the cross product
    >>> list(sorted(zip(a['f'], a['g'])))
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> list(sorted(sample_factor_assignment(dict(f=range(10)), sample=['f'], count=10, random=random.Random(42))['f'])) # just random
    [0, 0, 1, 2, 3, 3, 8, 8, 8, 9]
    '''
    assert not (set(counterbalance) & set(sample)), 'should have disjoint set of things for counterbalancing & sampling'
    counts = _validate_and_count_experiment(factored_config, counterbalance+sample)

    # We do column-based storage to avoid repeating keys
    res = {k: [] for k in counterbalance+sample}

    # We count every combination of the values for the counterbalanced keys
    cb_ct = {key: 0 for key in itertools.product(*[range(counts[cb]) for cb in counterbalance])}
    # This should equal the product of the counts of the keys, by the definition of itertools.product()
    assert len(cb_ct) == functools.reduce(lambda a, b: a*b, [counts[cb] for cb in counterbalance], 1)

    # We generate all of our samples.
    for _ in range(count):
        # First, we counterbalance.

        # 1. Find minimum value based on current counts
        minval = min(cb_ct.values())
        # 2. Sample from items with minimum value
        sampled = random.choice([key for key, ct in cb_ct.items() if ct == minval])
        # 3. Add to our count to factor into future counterbalancing
        cb_ct[sampled] += 1

        # 4. Assign values into column-wise storage.
        for key, sampled_idx in zip(counterbalance, sampled):
            assert 0 <= sampled_idx < counts[key], 'This should only happen if the application is incorrectly matching indices and keys'
            res[key].append(sampled_idx)

        # For other keys, we just sample!
        for key in sample:
            res[key].append(random.choice(range(counts[key])))

    # Now, some simple checks.
    empirical_ct = collections.Counter([tuple([res[key][idx] for key in counterbalance]) for idx in range(count)])
    # When we count again, ensure the values match. This is a sanity check for the application logic.
    for key, ct in cb_ct.items():
        assert empirical_ct[key] == ct

    # We ensure the counterbalancing has worked.
    minval = min(cb_ct.values())
    maxval = max(cb_ct.values())
    assert maxval - minval in (0, 1), 'Counterbalancing max and min values should be equal or off by one.'
    assert minval == math.floor(count / len(cb_ct)), 'Counterbalancing min should be floor of expectation.'
    assert maxval == math.ceil(count / len(cb_ct)), 'Counterbalancing max should be ceiling of expectation.'

    return res


def allrotflip(order):
    '''
    Given a circular ordering, this returns all rotations and flips of the circle.
    order: maps circle position to state

    >>> allrotflip([2, 3, 0])
    [[2, 3, 0], [3, 0, 2], [0, 2, 3], [2, 0, 3], [3, 2, 0], [0, 3, 2]]
    '''
    rots = [
        order[i:] + order[:i]
        for i in range(len(order))
    ]
    flips = [r[::-1] for r in rots]
    # HACK we reorder the flips so that we can have better visual matches between the ordering of the flips & rots
    flips = flips[1:] + [flips[0]]
    for r, f in zip(rots, flips):
        assert r[0] == f[0], ''
    return rots + flips
    # HACK old code
    return [
        rot[::flip]
        for rot in [
            order[i:] + order[:i]
            for i in range(len(order))
        ]
        for flip in [+1, -1]
    ]

def correlate_distance_and_coordinates(distance_matrix, *, circle_order=None, coordinates=None):
    '''
    Correlates the distance matrix from a circle order with a distance matrix.
    >>> line_dist, line_coord = np.array([ [0, 2, 4], [2, 0, 2], [4, 2, 0] ]), [(0, 0), (0, 2), (0, 4)]
    >>> assert np.isclose(correlate_distance_and_coordinates(line_dist, coordinates=line_coord), 1)
    '''
    if coordinates is None:
        coordinates = coordinates_for_circle_order(circle_order)
    else:
        assert circle_order is None

    coordinates = np.array(coordinates)
    assert coordinates.shape == (distance_matrix.shape[0], 2)
    c_d = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    # This can pretty dramatically change things
    task_dist = (distance_matrix != 0) & (distance_matrix != 1)
    return np.corrcoef(distance_matrix[task_dist].flatten(), c_d[task_dist].flatten())[0, 1]

def plot_coordinates_relative_to_null(
    mdp, *,
    circle_order=None, circle_orders=None, samples=2000, return_quantiles=False,
):
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    if circle_order is not None:
        assert circle_orders is None
        circle_orders = [circle_order]

    d = rrtd.floyd_warshall(mdp)
    corrs = [correlate_distance_and_coordinates(d, circle_order=co) for co in circle_orders]

    null = [
        correlate_distance_and_coordinates(d, circle_order=shuffled(mdp.state_list))
        for _ in tqdm(range(samples), leave=False)
    ]

    f, ax = plt.subplots(figsize=(4, 3))
    plt.hist(null, bins=30, alpha=0.3)
    for c in corrs:
        plt.axvline(c, c='k')
    ax.set(title='Histogram of correlation of geodesic and euclidean embeddings\nSampled embeddings are lines')

    if return_quantiles:
        qs = [np.sum(c<null)/samples for c in corrs]
        if circle_order is None:
            return qs
        else:
            return qs[0]
