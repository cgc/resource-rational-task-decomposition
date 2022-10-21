from pythonland_ppl import *
import pytest

def test_basic():
    def tree_to_vals(node):
        return [node.get('value', '<root>'), node['complete'], [tree_to_vals(c) for c in node['children']] if 'children' in node else None]

    ee = ExecutionEnumerator(deduplicate_traces=False)
    assert tree_to_vals(ee.root) == ['<root>', False, None]

    assert ee.choice([0, 1]) == 0
    assert tree_to_vals(ee.root) == ['<root>', False, [[0, False, None], [1, False, None]]]

    assert ee.choice(['a', 'b']) == 'a'
    ee._reset()
    assert tree_to_vals(ee.root) == ['<root>', False, [[0, False, [['a', True, None], ['b', False, None]]], [1, False, None]]]

    # should shout for incorrect data
    with pytest.raises(ValueError) as e:
        ee.choice(['bad', 'data'])
    assert 'Unexpected choice argument' in str(e)

    assert ee.choice([0, 1]) == 0
    assert ee.choice(['a', 'b']) == 'b'
    ee._reset()
    assert tree_to_vals(ee.root) == ['<root>', False, [[0, True, [['a', True, None], ['b', True, None]]], [1, False, None]]]

    assert ee.choice([0, 1]) == 1
    assert ee.choice(['c', 'd']) == 'c'
    ee._reset()
    assert tree_to_vals(ee.root) == ['<root>', False, [[0, True, [['a', True, None], ['b', True, None]]], [1, False, [['c', True, None], ['d', False, None]]]]]

    assert ee.choice([0, 1]) == 1
    assert ee.choice(['c', 'd']) == 'd'
    ee._reset()
    assert tree_to_vals(ee.root) == ['<root>', True, [[0, True, [['a', True, None], ['b', True, None]]], [1, True, [['c', True, None], ['d', True, None]]]]]

    # should shout saying it's done
    with pytest.raises(ValueError) as e:
        ee.choice([0, 1])
    assert 'Invalid execution: All children have been marked complete.' in str(e)

def test_deduplicate_traces():
    def state_collapse(random=random, lim=3):
        x = 0
        for _ in range(lim):
            x += random.choice([0, 1])
        return x

    for limit in [3, 4, 5, 6]:
        res = ExecutionEnumerator.distribution(lambda **kw: state_collapse(lim=limit, **kw), deduplicate_traces=False)
        res2 = ExecutionEnumerator.distribution(lambda **kw: state_collapse(lim=limit, **kw), deduplicate_traces=True)
        assert dist_is_close(res, res2)
    # HACK should also assert # of traces is smaller

def test_nested_functions():
    def program(random=random):
        def something():
            return random.choice([0, 1])
        total = 0
        for _ in range(2):
            total += something()
        return total
    expected = {0: 0.25, 1: 0.5, 2: 0.25}
    assert ExecutionEnumerator.distribution(lambda **kw: program(**kw), deduplicate_traces=False) == expected
    assert ExecutionEnumerator.distribution(lambda **kw: program(**kw), deduplicate_traces=True) == expected

def test_choice_p():
    def program(random=random):
        v = 0
        v += random.choice([0, 1], p=[0.1, 0.9])
        v += random.choice([0, 1], p=[0.1, 0.9])
        return v
    expected = {0: 0.1**2, 1: 0.1*0.9*2, 2: 0.9**2}
    assert dist_is_close(expected, ExecutionEnumerator.distribution(program, deduplicate_traces=False))
    assert dist_is_close(expected, ExecutionEnumerator.distribution(program, deduplicate_traces=True))

def test_ambiguous_sampling_statements():
    def program(random=random):
        return random.choice([0, 1], p=[0.1, 0.9]) + random.choice([0, 1], p=[0.1, 0.9])
    expected = {0: 0.1**2, 1: 0.1*0.9*2, 2: 0.9**2}
    assert dist_is_close(expected, ExecutionEnumerator.distribution(program, deduplicate_traces=False))
    with pytest.raises(Exception) as err:
        ExecutionEnumerator.distribution(program, deduplicate_traces=True)
    assert 'Program with ambiguous sampling statements' in str(err)

def test_verybad():
    def program(random=random):
        return (
            random.choice([0, 1], p=[0.1, 0.9]) +
            random.choice([0, 1], p=[0.1, 0.9])
        )
    expected = {0: 0.1**2, 1: 0.1*0.9*2, 2: 0.9**2}
    assert dist_is_close(expected, ExecutionEnumerator.distribution(program, deduplicate_traces=False))
    # HACK this is very bad. can we detect this???
    # still at a loss here. can't just look for choice calls on the same line, since they could be inside of functions
    # could potentially parse then unparse the ast which would get rid of this issue, but would have to
    # do that recursively?
    # wonder if the best bet is tacking on a counter
    with pytest.raises(AssertionError):
        assert dist_is_close(expected, ExecutionEnumerator.distribution(program, deduplicate_traces=True, debug=True))
    with pytest.raises(ValueError) as err:
        # HACK if True is just fixing an indent issue
        validate_repeated_choice_call('if True:\n'+inspect.getsource(program))
    assert 'Found repeated call to random.choice() in the same statement' in str(err)

def test_verybad2():
    '''
    This unfortunately will be hard to change without modifying the contract; we'd need for all function calls to
    '''
    def subfn(random=random):
        return random.choice([0, 1], p=[0.1, 0.9])
    def program(random=random):
        return (
            subfn(random=random) +
            subfn(random=random)
        )
    expected = {0: 0.1**2, 1: 0.1*0.9*2, 2: 0.9**2}
    assert dist_is_close(expected, ExecutionEnumerator.distribution(program, deduplicate_traces=False))
    with pytest.raises(AssertionError):
        assert dist_is_close(expected, ExecutionEnumerator.distribution(program, deduplicate_traces=True, debug=True))
    # sad...
    validate_repeated_choice_call('if True:\n'+inspect.getsource(program))

def test_verybad3():
    '''
    This one is a bummer because we actually _could_ deduplicate these since they are two distinct statements
    and result in differences to the environment (b/c of the assignment).
    '''
    def program(random=random):
        x = 0
        x += random.choice([0, 1], p=[0.1, 0.9]); x += random.choice([0, 1], p=[0.1, 0.9])
        return x
    expected = {0: 0.1**2, 1: 0.1*0.9*2, 2: 0.9**2}
    assert dist_is_close(expected, ExecutionEnumerator.distribution(program, deduplicate_traces=False))
    with pytest.raises(Exception) as err:
        assert dist_is_close(expected, ExecutionEnumerator.distribution(program, deduplicate_traces=True, debug=True))
    assert 'Program with ambiguous sampling statements' in str(err)

def test_inference():
    def program(random=random):
        return random.choice([0, 1], logp=[-1, 0])
    import scipy.special
    expected = {i: v for i, v in enumerate(scipy.special.softmax([-1, 0]))}
    assert dist_is_close(expected,  ExecutionEnumerator.distribution(program, deduplicate_traces=False, normalize=True))
    assert dist_is_close(expected,  ExecutionEnumerator.distribution(program, deduplicate_traces=True, normalize=True))

def test_shuffled():
    def program(random=random):
        res = tuple()
        for el in shuffled([0, 1, 2], random=random):
            res += (el,)
        return res
    import itertools
    perms = list(itertools.permutations(range(3)))
    expected = {p: 1/len(perms) for p in perms}
    assert dist_is_close(expected, ExecutionEnumerator.distribution(lambda **kw: program(**kw), deduplicate_traces=False))
    assert dist_is_close(expected, ExecutionEnumerator.distribution(lambda **kw: program(**kw), deduplicate_traces=True))

def test_shuffle():
    def program(random=random):
        arr = [0, 1, 2, 3]
        random.shuffle(arr)
        return tuple(arr)
    import itertools
    perms = list(itertools.permutations(range(4)))
    expected = {p: 1/len(perms) for p in perms}
    assert dist_is_close(expected, ExecutionEnumerator.distribution(lambda **kw: program(**kw), deduplicate_traces=False))
    assert dist_is_close(expected, ExecutionEnumerator.distribution(lambda **kw: program(**kw), deduplicate_traces=True))

def test_parents_different_depth():
    '''
    We're testing a funny bug here. Trying to find cases where our breadth-first way
    of calculating probabilities will fail when a node has been deduplicated and has
    one parent at a lower depth than the others. In this example, you can reach the state
    where s=2 through either the state where s=0 (initial state) or when s=1 (which is a sibling of s=2).
    We do detect these can be deduplicated (the choice statement is same line, and only variable is s).
    The bug we run into is that when we try to compute the probability of the s=2 node, we
    haven't yet computed the probability of the s=1 node; this depends critically on the order
    of the elements in the sample statement for s=0.
    '''
    def weird_test(random=random):
        s = 0
        while s != 2:
            s = random.choice({
                0: [2, 1],
                1: [2],
            }[s])
        random.choice([None]) # a dummy choice here to run through the deduplication logic.
        return
    ExecutionEnumerator.distribution(weird_test, deduplicate_traces=True, normalize=True)

def _adj_to_list(A):
    return ([
        np.where(row)[0].tolist()
        for row in A
    ])
'''
A line world of length 4:
0123
'''
A = _adj_to_list(np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
]))
'''
a 2x3 grid:
012
345
'''
Agrid = _adj_to_list(np.array([
    [1, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 1],
]))

def test_bfs():
    assert dist_is_close(ExecutionEnumerator.distribution(lambda **kw: bfs(A, 0, 1, **kw)), {2: 1})
    assert dist_is_close(ExecutionEnumerator.distribution(lambda **kw: bfs(A, 1, 0, **kw)), {2: 0.5, 3: 0.5})
    # This is a pretty critical case; we don't have a uniform over elements at some depth.
    # In this case, the depth 1 nodes (1, 3) are more likely to queue up 4 ahead of 2 (the depth 2 nodes).
    assert dist_is_close(ExecutionEnumerator.distribution(lambda **kw: bfs(Agrid, 0, 4, **kw)), {4: 0.75, 5: 0.25})
