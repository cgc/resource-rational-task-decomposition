import inspect
import random
import collections
import math
import numpy as np
import ast

# NOTE: This PPL was a small experiment, partly developed to make sure expectations over search algorithm executions behaved as expected.

# # Another design for a PPL
#
# In other designs for PPLs that we've worked on, we've tried different methods of intercepting & manipulating calls to random sampling. My early AST-based version transformed sampling sites to a generator-based format, not unlike that of early PyMC4 designs (links [here](https://eigenfoo.xyz/manipulating-python-asts/) and [here](https://github.com/eigenfoo/random/blob/master/python/ast-hiding-yield/00-prototype/hiding-yield.ipynb)). Another design that Mark has proposed steps the Python interpreter through a program (I think?) replacing the AST at sampling sites.
#
# Here, I propose a design that doesn't manipulate Python programs; it only requires that a reference to the sampling function can be provided. In the below, I see how far I can get with this idea by providing a complex reference to replace the `random.choice` function.
#
# The below `ExecutionEnumerator` has many roles; it implements a subset of the `random` module (only the `choice` function), it tracks the history of samplings, and it uses this history of samplings to visit all possible sampling orderings. The samplings are maintained in a tree, where every node corresponds to a sampled value (key `value`) and children are future samplings (key `choice`). It can thus compute a distribution over return values for a function that samples using `random.choice`, exposed in the `ExecutionEnumerator.distribution` function.
#
# First, I write a function that makes a key for a stack trace and local variables. Under the assumption that there is at most one sampling site per line of the program, this fully describes the state of the program without requiring reference to prior sampled choices. By combining traces with matching keys, we can dramatically reduce the size of the tree of all samplings, leaving us with a directed acyclic graph.

# # Things the current design does not handle
#
# - lines with same distribution samplings, like `x = random([0, 1]) + random([0, 1])`. This is currently detected and throws a "ambiguous sampling statements" error. Can be fixed by annotating all `random.choice` callsites with some identifier.
# - loops with the same variables.

def get_trace_key(uptofn='_run'):
    '''
    This is optimized to avoid getframeinfo, a performance bottleneck from profiles. repr(f_locals) is still pretty costly.
    This also supports nested function invocation by identifing machine state through stack/variables; it walks up the stack
    until it finds the `_run` function used to invoke the stochastic program.
    '''
    def frame_key(frame):
        return (frame.f_code.co_filename, str(frame.f_lineno), frame.f_code.co_name, repr(frame.f_locals))
    import inspect
    caller = inspect.currentframe().f_back.f_back
    # TODO assert that .f_back's name is `choice`?
    callers = []
    while caller.f_code.co_name != uptofn:
        callers.append(caller)
        caller = caller.f_back
    return tuple(frame_key(c) for c in callers)


class EarlyTerminationClonedTrace(Exception):
    '''
    This error is used to terminate a trace execution when it has a matching trace key
    # (file + line number + local variables) to another trace, signalling that the probability
    # from another trace can be reused.
    '''
    pass

import weakref

class ExecutionEnumerator(object):
    def __init__(self, *, deduplicate_traces=True, debug=False):
        # In the tree of assignments, we require the ability to transition from parent to child, as well as from child to parent.
        # To avoid memory leaks, we require that one of these direction of references be a weak reference.
        # To have weakrefs to dictionaries (the structure we use for nodes), we need to subclass dict.
        class Dict(dict):
            pass
        self.dict_class = Dict

        # The root node!
        self.root = self.dict_class(complete=False, parents=[], logp=0, summed_logp=0)

        self.current_node = self.root
        self.leaf_nodes = []
        self.key_to_node = {}

        # When this is true, we use the parent's call details (file and line number) and parent's locals to deduplicate traces.
        self.deduplicate_traces = deduplicate_traces

        self.debug = debug

    #def __repr__(self): TODO should ensure thi is never implemented properly as it would affect trace keys... or should proxy more smartly to a fake `random`?

    def _reset(self):
        # HACK This could inadvertently mark root node complete if called early on.
        self._mark_parents_complete(self.current_node)
        self.current_node = self.root

    def _mark_parents_complete(self, node):
        '''
        This method marks the node complete if all it's children are complete, then recursively
        considers all parents of the node.
        '''
        if node['complete']:
            return
        if 'children' not in node:
            node['complete'] = True
        elif all(c['complete'] for c in node['children']):
            node['complete'] = True

        if node['complete']:
            for parent_ref in node['parents']:
                self._mark_parents_complete(parent_ref())

    def shuffle(self, arr):
        # NOTE wonder if this isn't ideal since it doesn't really expose the stream?
        # let's do the fisher-yates
        # https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        # We assume the array can be modified in-place.
        for i in range(len(arr)-1, 0, -1): # for i from n−1 downto 1 do
            j = self.choice(range(i+1)) # j ← random integer such that 0 ≤ j ≤ i
            arr[j], arr[i] = arr[i], arr[j] # exchange a[j] and a[i]

    def choice(self, arr, *, p=None, logp=None):
        if not arr:
            raise IndexError('Cannot choose from an empty sequence')
        # coerce to list
        arr = list(arr)
        # Argument handling. Everything in this functions uses logp, particularly because that makes adding scores easy.
        if logp is None:
            logp = [math.log(1/len(arr) if p is None else p[idx]) for idx in range(len(arr))]
        p = None

        if 'children' in self.current_node:
            # If children has been set, ensure it is consistent with the distribution we are currently sampling from.
            sfn = self.current_node['sampling_fn']
            assert sfn[0] == 'choice'
            if arr != sfn[1]:
                raise ValueError('Unexpected choice argument. Expected {} but found {}'.format(arr, sfn[1]))
            if logp != sfn[2]:
                raise ValueError('Unexpected logp argument. Expected {} but found {}'.format(logp, sfn[2]))
        else:
            if self.deduplicate_traces:
                # First we see if another callsite is like this one
                key = hash(get_trace_key())
                self.current_node['trace_key'] = key

                assert all(key != ref()['trace_key'] for ref in self.current_node['parents']), 'Program with ambiguous sampling statements.'

                if key in self.key_to_node:
                    # We copy relevant properties from the node with matching trace key.
                    similar = self.key_to_node[key]
                    self.current_node['children'] = similar['children'] # This intentionally does not perform a deep copy.
                    self.current_node['sampling_fn'] = similar['sampling_fn']

                    # And set this node as a parent to children.
                    for c in self.current_node['children']:
                        c['parents'].append(weakref.ref(self.current_node))

                    # this is true since we do depth-first executions, and focus on not-yet-complete leaf nodes.
                    assert similar['complete'], 'A node similar to the current node was not complete. If execution order is depth-first, this may indicate variables are being overwritten.'

                    # HACK we leave this out since _mark_parents_complete is incorrect if this is set here.
                    # self.current_node['complete'] = True

                    raise EarlyTerminationClonedTrace()
                else:
                    # If no other node matches, we add this node to the dictionary
                    self.key_to_node[key] = self.current_node

            # Initialize variable of children.
            self.current_node['children'] = [None] * len(arr)
            for idx, c in enumerate(arr):
                self.current_node['children'][idx] = self.dict_class(
                    value=c,
                    parents=[weakref.ref(self.current_node)],
                    complete=False,
                    logp=logp[idx],
                )
                if not self.deduplicate_traces:
                    # Should be stressed that summed_logp is not correct when deduplicate_traces is true.
                    self.current_node['children'][idx]['summed_logp'] = self.current_node['summed_logp'] + logp[idx]
            # Capture sampling arg
            self.current_node['sampling_fn'] = ('choice', arr, logp)

        # Find a child that hasn't been completed yet.
        for item in self.current_node['children']:
            if not item.get('complete'):
                break
        else:
            raise ValueError('Invalid execution: All children have been marked complete.')

        # Maintain a reference to the current node; this will be the parent at the next sampling.
        self.current_node = item
        return item['value']

    def _run(self, fn):
        '''
        Run function with this execution enumerator replacing the `random` module, saving return value in last sampled node.
        '''
        try:
            rv = fn(random=self)
            self.current_node['return_value'] = rv
            self.leaf_nodes.append(self.current_node)
        except EarlyTerminationClonedTrace:
            pass
        self._reset()

    @classmethod
    def distribution(cls, fn, *, debug=False, deduplicate_traces=True, normalize=False):
        '''
        This function evaluates all possible sequences of samplings
        from the function, returning a distribution over possible return values.

        >>> ExecutionEnumerator.distribution(lambda random=random: random.choice([0, 1]))
        {0: 0.5, 1: 0.5}
        >>> ExecutionEnumerator.distribution(lambda random=random: random.choice([0, 1], p=[0.25, 0.75]))
        {0: 0.25, 1: 0.75}
        >>> ExecutionEnumerator.distribution(lambda random=random: random.choice([0, 1]) + random.choice([0, 1]), deduplicate_traces=False)
        {0: 0.25, 1: 0.5, 2: 0.25}
        '''
        if debug:
            import time
            st = time.time()
        ee = cls(deduplicate_traces=deduplicate_traces, debug=debug)
        while not ee.root['complete']:
            ee._run(fn)

        if deduplicate_traces:
            _, probs = compute_probabilities_breadth_first(ee.root)
        else:
            # Since another breadth-first traversal is pretty slow when you have duplicate traces,
            # we keep both the probability for a node alone (as `logp`), as well as combined with
            # parent sampling sites (as `summed_logp`).
            probs = {}
            for item in ee.leaf_nodes:
                probs.setdefault(item['return_value'], 0)
                probs[item['return_value']] += math.exp(item['summed_logp'])
        # This should be used when doing inference.
        if normalize:
            z = sum(probs.values())
            probs = {e: p/z for e, p in probs.items()}
        else:
            assert np.isclose(sum(probs.values()), 1)
        if debug:
            print('debug, # leafs', len(ee.leaf_nodes))
            print('seconds', time.time() - st)
        return probs

def compute_probabilities_breadth_first(root):
    node_to_prob = {} # cumulative probability for each node
    q = collections.deque([root])
    rv_to_prob = {} # for final result
    while q:
        node = q.popleft()

        if node['parents']:
            parent_p = sum(node_to_prob[id(ref())] for ref in node['parents'])
        else:
            # for root
            parent_p = 1
        node_to_prob[id(node)] = parent_p * math.exp(node['logp'])

        if 'children' in node:
            for c in node['children']:
                def is_queued_or_processed(n):
                    return (
                        # is processed
                        id(n) in node_to_prob or
                        # is queued
                        n in q
                    )
                if (
                    not is_queued_or_processed(c) and
                    # This ensures a topological ordering over nodes, in case any parents of a node haven't been encountered yet.
                    all(is_queued_or_processed(ref()) for ref in c['parents'])
                ):
                    q.append(c)
                '''
                if id(c) not in node_to_prob and c not in q and all(
                    # topo??
                    id(ref()) in node_to_prob or ref() in q
                    for ref in c['parents']
                ):
                    q.append(c)
                '''
        else:
            # A leaf node!
            rv_to_prob.setdefault(node['return_value'], 0)
            rv_to_prob[node['return_value']] += node_to_prob[id(node)]

    return node_to_prob, rv_to_prob


def shuffled(arr, random=random):
    '''
    Since we only replace `random.choice`, this routine implements shuffling by using `random.choice`.
    Since this is a function that will be called by the main search routine, it requires that our
    algorithm for deduplication correctly handle multiple entries in the stack.
    '''
    arr = list(arr) # make a copy so we can modify.
    while arr:
        if len(arr) == 1:
            value = arr[0]
        else:
            value = random.choice(arr)
        yield value
        arr.remove(value)

def bfs(succ_map, start, goal, random=random, depth_first=False, debug=False):
    if debug: print('start')
    q = collections.deque()
    q.append(start)
    visited = set()
    while q:
        if depth_first:
            node = q.pop()
        else:
            node = q.popleft()
        if debug: print('node', node, 'visited', visited, 'q', q)
        visited.add(node)
        if node == goal:
            if debug: print('done')
            return len(visited)
        for next_node in shuffled(succ_map[node], random=random):
            if next_node not in visited and next_node not in q:
                q.append(next_node)
            if debug: print('\tnext', next_node, q)

def bfs_search_algorithm(mdp, random=random, depth_first=False, debug=False):
    import search_algorithms
    from frozendict import frozendict
    '''
    A version tuned for sure here.
    '''
    if debug: print('start')
    q = collections.deque()
    q.append(mdp.initial_state())
    visited = set()
    camefrom = {}
    while q:
        if depth_first:
            node = q.pop()
        else:
            node = q.popleft()
        if debug: print('node', node, 'visited', visited, 'q', q)
        visited.add(node)
        if mdp.is_terminal(node):
            if debug: print('done')
            p = search_algorithms.reconstruct_path(mdp.initial_state(), node, camefrom)
            return frozendict(
                visited=frozenset(visited),
                frontier=frozenset(q),
                path=tuple(p),
                cost=len(p)-1,
                # API
                distance=len(p)-1,
                plan_cost=len(visited),
            )
        succ = [mdp.next_state(node, a) for a in mdp.actions(node)]
        for next_node in shuffled(succ, random=random):
            if next_node not in visited and next_node not in q:
                camefrom[next_node] = node
                q.append(next_node)
            if debug: print('\tnext', next_node, q)

import rrtd

def search_cost(mdp, s, g, *, depth_first=False, ExecutionEnumerator_kwargs={}):
    A = rrtd.adjacency_ssp(mdp, terminal_absorbing=False)
    dist = ExecutionEnumerator.distribution(lambda random=random: bfs(A, s, g, random=random, depth_first=depth_first), **ExecutionEnumerator_kwargs)
    return rrtd.expectation(rrtd.Multinomial_from_probability_dict(dist))

def search_cost_algorithm(mdp):
    distance = rrtd.floyd_warshall(mdp) # HACK this isn't really right for DFS
    A = rrtd.adjacency_ssp(mdp, terminal_absorbing=False)
    # HACK
    repr(A)
    def algorithm(s, g):
        dist = ExecutionEnumerator.distribution(lambda random=random: bfs(A, s, g, random=random, depth_first=False), deduplicate_traces=True)
        plan_cost = rrtd.expectation(rrtd.Multinomial_from_probability_dict(dist))
        return dict(value=-(distance[s, g] + plan_cost))
    return algorithm

def new_search_algorithm(mdp, *, search_fn):
    '''
    Yeah I know the name is really very similar to search_cost_algorithm... This expects a function with the naming conventions of those
    in search_algorithms. Those functions should also expect a `random` kwarg.
    '''
    def algorithm(s, g):
        dist = ExecutionEnumerator.distribution(lambda random=random: search_fn(mdp.for_task(s, g), random=random), deduplicate_traces=True)
        cost = rrtd.expectation(rrtd.DictDistribution(dist), lambda r: r['plan_cost'] + r['distance'])
        return dict(value=-cost, dist=dist)
    return algorithm


def dist_is_close(a, b):
    '''
    # obviously
    >>> assert dist_is_close({'f': 3}, {'f': 3})
    >>> assert not dist_is_close({'f': 3}, {'f': 3.1})

    # and this is the really one
    >>> off = {'f': 3.0000001}
    >>> assert {'f': 3} != off
    >>> assert dist_is_close({'f': 3}, off)
    '''
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        if not np.allclose(a[k], b[k]):
            return False
    return True

def validate_repeated_choice_call(code_string):
    '''
    This function detects cases where a random.choice() call happens twice in the same line
    '''
    class ValidateVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_statement = None
            self.call_set = set()
        def visit(self, n):
            # We keep track of most recent statement visited.
            # Technically this is a larger class of statements than we need.
            # For example, function def doesn't matter since we'll run this
            # function again for every line in the function.
            is_statement = n.__class__ in [
                ast.FunctionDef, ast.AsyncFunctionDef,
                ast.ClassDef, ast.Return,
                ast.Delete, ast.Assign, ast.AugAssign, ast.AnnAssign,
                ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith,
                ast.Raise, ast.Try, ast.Assert,
                ast.Import, ast.ImportFrom,
                ast.Global, ast.Nonlocal, ast.Expr,
                ast.Pass, ast.Break, ast.Continue
            ]
            # If this is a statement, we set it as the current statement.
            if is_statement:
                prev = self.current_statement
                self.current_statement = n
            super().visit(n)
            # Once we return from recursively visiting child nodes, we set our
            # statement to the previous value.
            if is_statement:
                self.current_statement = prev

        def visit_Call(self, n):
            if (
                isinstance(n.func, ast.Attribute) and
                isinstance(n.func.value, ast.Name) and n.func.value.id == 'random' and
                isinstance(n.func.attr, str) and n.func.attr == 'choice'
            ):
                # We use ast.dump() here as a way to compare two expressions in the same statement.
                # We could alternatively write a recursive equality check that avoids checking
                # lineno/colnumber attributes.
                key = (self.current_statement, ast.dump(n))
                if key in self.call_set:
                    raise ValueError(f'Line {self.current_statement.lineno}. Found repeated call to random.choice() in the same statement.')
                self.call_set.add(key)

    tree = ast.parse(code_string, mode='exec')
    ValidateVisitor().visit(tree)
