from msdm.core.distributions import DictDistribution
import itertools
from frozendict import frozendict

def chunks(lst, n):
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def roundrobin(*iterables):
    # https://docs.python.org/3/library/itertools.html
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


# Distribution utilities
# Most of these are included in more recent versions of msdm

def marginalize(dist, proj):
    '''
    >>> assert marginalize(DictDistribution.uniform(range(3)), lambda e: (e % 2) == 0).isclose(DictDistribution({False: 1/3, True: 2/3}))
    '''
    import collections
    rv = collections.defaultdict(float)
    for e, p in dist.items():
        rv[proj(e)] += p
    return DictDistribution(rv)

def dist_prod(a, b):
    '''
    A function to take the product of two distributions
    '''
    rv = {}
    for supa, proba in a.items():
        for supb, probb in b.items():
            key = supa, supb
            dict_classes = (dict, frozendict)
            if isinstance(supa, dict_classes) and isinstance(supb, dict_classes):
                # Combine dictionaries instead of making them into tuples
                key = frozendict(supa, **supb)
            rv[key] = proba*probb
    return DictDistribution(rv)

def expectation(distribution, fn=lambda value: value):
    '''
    Take an expectation over a distribution. Defaults to the identity function.
    '''
    return sum(p * fn(event) for event, p in distribution.items())


def dist_condition(distribution, fn):
    newdist = DictDistribution({
        (event, p)
        for event, p in distribution.items()
        if fn(event)
    })
    if not len(newdist):
        return newdist
    return newdist * (1 / sum(newdist.probs))

